"""Mamba2 with SSM hidden state extraction for track parameter regression.

This module provides wrappers around the Mamba2 SSM that expose the final
recurrent hidden state from the selective state space model.  The standard
Mamba2 forward path does not return the SSM state; these wrappers expose
``return_final_states=True`` from the underlying triton kernels so that
the state can be used as a learned track summary for downstream regression.

Only the **last encoder layer** should request the state to avoid unnecessary
IO overhead.  The bidirectional encoder therefore stacks ordinary Mamba2
layers for all but the final layer, and uses :class:`Mamba2WithState` there.

Design for padded batches
------------------------
Variable-length hit sequences are collated into padded tensors of shape
``(B, max_L, D)`` with a boolean ``hit_valid`` mask.  Each batch element
contains exactly one track, so the Mamba-2 kernels process ``B``
independent sequences in parallel.  The backward direction is obtained
by a simple ``torch.flip`` along the sequence dimension.

For the forward kernels ``seq_idx`` is left as ``None`` (one sequence
per batch element).  This approach enables proper mini-batch gradient
descent, in contrast to the alternative *packing* strategy where all
tracks would be concatenated into a single ``batch=1`` tensor.

References
----------
- Mamba-2: arXiv 2405.21060
- Vision Mamba (bidirectional): arXiv 2401.09417
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba2 = nn.Module  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Mamba2 wrapper that exposes the final SSM hidden state
# ---------------------------------------------------------------------------


@dataclass
class Mamba2Output:
    """Container returned by :class:`Mamba2WithState`.

    Attributes
    ----------
    output : Tensor
        Sequence output, same shape as input ``(B, L, D)``.
    final_state : Tensor | None
        Final SSM hidden state of shape ``(B, nheads, headdim, d_state)``.
        Only populated when ``return_state=True``.
    """

    output: Tensor
    final_state: Tensor | None = None


class Mamba2WithState(Mamba2):  # type: ignore[misc]
    """Subclass of :class:`mamba_ssm.modules.mamba2.Mamba2` that can
    optionally return the final SSM recurrent state.

    The upstream ``Mamba2.forward`` does not expose ``return_final_states``
    on the fused memory-efficient path.  This subclass overrides ``forward``
    to inject that flag into the underlying triton kernels when
    ``return_state=True`` is requested.

    All weight initialisation, projections, and other logic are inherited
    from upstream ``Mamba2`` — only the forward path is customised.

    Forked from mamba-ssm==2.3.0.  If you upgrade mamba-ssm, check for
    upstream changes to ``Mamba2.forward`` and update accordingly.
    """

    def __init__(self, *args, **kwargs):
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required.  Install with: pip install mamba-ssm[causal-conv1d]"
            )
        super().__init__(*args, **kwargs)

    # -- state dimension helpers -------------------------------------------

    @property
    def state_dim(self) -> int:
        """Flat dimension of the final SSM state: ``nheads * headdim * d_state``."""
        return self.nheads * self.headdim * self.d_state

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        u: Tensor,
        seq_idx: Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> Mamba2Output:
        """Forward pass with optional SSM state return.

        Parameters
        ----------
        u : Tensor
            Input of shape ``(B, L, D)``.
        seq_idx : Tensor | None
            Optional per-token sequence index, shape ``(B, L)``.
            Not used with padded batches (defaults to ``None``).
        return_state : bool
            If ``True``, also return the final SSM hidden state.

        Returns
        -------
        Mamba2Output
            Named container with ``.output`` and ``.final_state``.
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else {"dt_limit": self.dt_limit}
        )

        if self.use_mem_eff_path:
            result = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                return_final_states=return_state,
                **dt_limit_kwargs,
            )
            if return_state:
                out, final_states = result
            else:
                out = result
                final_states = None

        else:
            # ---------- fallback (non-fused) path ----------
            d_mlp = (
                zxbcdt.shape[-1]
                - 2 * self.d_ssm
                - 2 * self.ngroups * self.d_state
                - self.nheads
            ) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [
                    d_mlp,
                    d_mlp,
                    self.d_ssm,
                    self.d_ssm + 2 * self.ngroups * self.d_state,
                    self.nheads,
                ],
                dim=-1,
            )

            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires causal_conv1d"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : -(self.d_conv - 1)]
                )
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)

            x, B, C = torch.split(
                xBC,
                [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                dim=-1,
            )

            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                return_final_states=return_state,
                **dt_limit_kwargs,
            )

            if return_state:
                y, final_states = y
            else:
                final_states = None

            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            out = self.out_proj(y)

        return Mamba2Output(output=out, final_state=final_states)


# ---------------------------------------------------------------------------
# Bidirectional layer / encoder using Mamba2WithState
# ---------------------------------------------------------------------------


class BidirectionalMambaLayer(nn.Module):
    """Single bidirectional Mamba layer with gated merge (Vision Mamba style).

    This is a *plain* layer used for all but the last encoder layer.  It does
    **not** return hidden states — only the combined sequence output.

    Parameters
    ----------
    dim, d_state, d_conv, expand, headdim, ngroups, chunk_size, norm, dropout
        See :class:`Mamba2WithState` and the original reference code.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.norm = nn.RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        # Use standard Mamba2 (no state extraction needed for intermediate layers)
        mamba_kwargs = {
            "d_model": dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
            "ngroups": ngroups,
            "chunk_size": chunk_size,
        }
        self.forward_mamba = Mamba2(**mamba_kwargs)
        self.backward_mamba = Mamba2(**mamba_kwargs)

        self.gate = nn.Linear(dim, dim)
        self.gate_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        seq_idx: Tensor | None = None,
        flip_indices: Tensor | None = None,
        lens: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Padded ``(B, L, D)`` *or* packed ``(1, total_L, D)``.
        seq_idx : Tensor | None
            Per-token segment index, shape ``(1, total_L)`` int32. Only
            used in packed mode (resets Mamba-2 SSD state at boundaries).
        flip_indices : Tensor | None
            ``(total_L,)`` long gather index that reverses each segment of
            the packed sequence in place, OR ``(B, L)`` per-row indices for
            the padded-static path (valid-prefix flip). When provided, the
            backward direction uses this flip instead of the default
            global ``torch.flip`` along dim=1. Same indices flip and
            un-flip (segment/prefix reversal is its own inverse).
        """
        skip = x
        x_norm = self.norm(x).contiguous()

        if (
            cu_seqlens is not None
            and getattr(self, "_packed_fused", False)
        ):
            # v5p: packed-stream fused path — no pad rows anywhere.
            from .mamba_short import fused_bidi_scan_packed

            x_fwd, x_bwd = fused_bidi_scan_packed(self, x_norm, cu_seqlens)
        elif (
            flip_indices is not None
            and flip_indices.dim() == 2
            and hasattr(self, "_fused_in_w")
        ):
            # V4.1 fused path: one in_proj GEMM for both directions, the
            # backward flip done in-kernel — no gathers, no separate GEMMs.
            from .mamba_short import fused_bidi_scan

            x_fwd, x_bwd = fused_bidi_scan(self, x_norm, flip_indices, lens)
        else:
            x_fwd = self.forward_mamba(x_norm, seq_idx=seq_idx)
            if flip_indices is None:
                # Padded path — global flip suffices (one track per batch row).
                x_bwd_in = torch.flip(x_norm, dims=[1]).contiguous()
                x_bwd_out = self.backward_mamba(x_bwd_in, seq_idx=seq_idx)
                x_bwd = torch.flip(x_bwd_out, dims=[1])
            else:
                # Packed (1-D indices) or padded-static (2-D per-row indices) —
                # segment-/prefix-wise flip via gather.
                if flip_indices.dim() == 1:
                    gather_idx = flip_indices.unsqueeze(0).unsqueeze(-1).expand_as(x_norm)
                else:
                    gather_idx = flip_indices.unsqueeze(-1).expand_as(x_norm)
                x_bwd_in = torch.gather(x_norm, 1, gather_idx).contiguous()
                x_bwd_out = self.backward_mamba(x_bwd_in, seq_idx=seq_idx)
                x_bwd = torch.gather(x_bwd_out, 1, gather_idx)

        gate = self.gate_activation(self.gate(x_norm))
        x_combined = gate * x_fwd + (1 - gate) * x_bwd

        return skip + self.dropout(x_combined)


class BidirectionalMambaStateLayer(nn.Module):
    """Final bidirectional Mamba layer that also extracts final SSM states.

    Identical to :class:`BidirectionalMambaLayer` but uses
    :class:`Mamba2WithState` so that the recurrent hidden states can be
    returned for regression.

    Parameters
    ----------
    dim, d_state, d_conv, expand, headdim, ngroups, chunk_size, norm, dropout
        See :class:`BidirectionalMambaLayer`.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.headdim = headdim
        self.expand = expand

        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.norm = nn.RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        mamba_kwargs = {
            "d_model": dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
            "ngroups": ngroups,
            "chunk_size": chunk_size,
        }
        self.forward_mamba = Mamba2WithState(**mamba_kwargs)
        self.backward_mamba = Mamba2WithState(**mamba_kwargs)

        self.gate = nn.Linear(dim, dim)
        self.gate_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def state_dim(self) -> int:
        """Concatenated forward + backward flat state dimension."""
        return self.forward_mamba.state_dim + self.backward_mamba.state_dim

    def forward(
        self,
        x: Tensor,
        seq_idx: Tensor | None = None,
        *,
        return_state: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            ``(output, hidden_state)`` where ``hidden_state`` has shape
            ``(B, 2 * nheads * headdim * d_state)`` when ``return_state=True``.
        """
        skip = x
        x_norm = self.norm(x).contiguous()

        fwd_out = self.forward_mamba(x_norm, seq_idx=seq_idx, return_state=return_state)
        x_fwd = fwd_out.output

        # Reverse for backward SSM — simple flip works for padded batches
        x_bwd_in = torch.flip(x_norm, dims=[1]).contiguous()
        bwd_out = self.backward_mamba(x_bwd_in, seq_idx=seq_idx, return_state=return_state)
        x_bwd = torch.flip(bwd_out.output, dims=[1])

        gate = self.gate_activation(self.gate(x_norm))
        x_combined = gate * x_fwd + (1 - gate) * x_bwd
        output = skip + self.dropout(x_combined)

        hidden_state = None
        if return_state and fwd_out.final_state is not None and bwd_out.final_state is not None:
            # Each final_state: (B, nheads, headdim, d_state)  →  (B, nheads*headdim*d_state)
            h_fwd = fwd_out.final_state.flatten(1)
            h_bwd = bwd_out.final_state.flatten(1)
            hidden_state = torch.cat([h_fwd, h_bwd], dim=-1)

        return output, hidden_state


class BidirectionalMambaEncoder(nn.Module):
    """Bidirectional Mamba-2 encoder with hidden state extraction.

    Stacks ``num_layers - 1`` ordinary :class:`BidirectionalMambaLayer` layers
    followed by a single :class:`BidirectionalMambaStateLayer` that returns
    the concatenated forward + backward SSM hidden states from the final layer.

    The hidden state is used directly as the track-level representation for
    regression of the perigee track parameters.

    Parameters
    ----------
    num_layers : int
        Total number of bidirectional layers (must be >= 1).
    dim : int
        Model / input dimension.
    d_state : int
        SSM state expansion factor.
    d_conv : int
        Local convolution width.
    expand : int
        Block expansion factor.
    headdim : int
        Head dimension for Mamba-2.
    ngroups : int
        Number of head groups for B/C projections.
    chunk_size : int
        Chunk size for the SSD kernel.
    norm : str
        Normalisation type (``'LayerNorm'`` or ``'RMSNorm'``).
    dropout : float
        Dropout rate inside each layer.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1, "Need at least one layer"

        self.num_layers = num_layers
        self.dim = dim

        common = dict(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            norm=norm,
            dropout=dropout,
        )

        # Intermediate layers (no state extraction)
        self.layers = nn.ModuleList(
            [BidirectionalMambaLayer(**common) for _ in range(num_layers - 1)]
        )

        # Final layer with state extraction
        self.final_layer = BidirectionalMambaStateLayer(**common)

        # Post-encoder normalisation
        if norm == "LayerNorm":
            self.final_norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.final_norm = nn.RMSNorm(dim)
        else:
            self.final_norm = nn.Identity()

    @property
    def state_dim(self) -> int:
        """Total flat hidden state dimension (forward + backward)."""
        return self.final_layer.state_dim

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        seq_idx: Tensor | None = None,
        **kwargs,  # noqa: ARG002  — compatibility with transformer encoder API
    ) -> tuple[Tensor, Tensor]:
        """Encode a sequence and return the hidden state summary.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, N, D)``.
        x_sort_value : Tensor | None
            Values to sort tokens by (e.g. ``s`` — distance from IP).
            Tokens are sorted before processing and un-sorted after.
        seq_idx : Tensor | None
            Optional per-token sequence index (unused for padded batches;
            defaults to ``None``).

        Returns
        -------
        tuple[Tensor, Tensor]
            - Sequence output ``(B, N, D)`` (in original token order).
            - Hidden state ``(B, state_dim)``.
        """
        # Optional sort (e.g. by distance from IP for proper sequencing)
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x)).contiguous()

        # Intermediate layers (plain bidirectional, no state)
        for layer in self.layers:
            x = layer(x, seq_idx=seq_idx)

        # Final layer with state extraction
        x, hidden_state = self.final_layer(x, seq_idx=seq_idx, return_state=True)

        # Post-norm on sequence output (kept for API compatibility)
        x = self.final_norm(x)

        # Un-sort back to original order
        if x_sort_value is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            x = torch.gather(x, -2, x_unsort_idx.unsqueeze(-1).expand_as(x))

        assert hidden_state is not None, "Final layer should always return hidden state"

        # Tie sequence output into hidden_state graph so all parameters
        # participate in the backward pass (avoids DDP unused-parameter errors).
        # The 0-valued addition is a no-op numerically.  The .float() prevents
        # bf16 overflow on the sum (long sequences can exceed bf16 range).
        hidden_state = hidden_state + 0.0 * x.float().sum()

        return x, hidden_state
