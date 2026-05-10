"""Hybrid Muon + AdamW optimizer.

Muon (Jordan 2024, https://kellerjordan.github.io/posts/muon/) orthogonalizes
the momentum buffer via a Newton-Schulz iteration and applies the resulting
update to 2-D matrix parameters.  AdamW handles everything else (biases,
norms, 1-D SSM scalars, embeddings, the final output head).

This file vendors a single-optimizer implementation of the hybrid so that
Lightning's automatic-optimization path still works (one optimizer, one
scheduler).  Parameter routing is controlled by the per-group
``use_muon`` flag, which the caller sets when constructing the optimizer.

References
----------
- Muon blog post (Keller Jordan, 2024): https://kellerjordan.github.io/posts/muon/
- Kimi-K2 paper (Muon at scale, 2025): https://arxiv.org/abs/2502.16982
- MIT-licensed reference: https://github.com/KellerJordan/Muon
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute an orthogonalized version of ``G`` via the quintic Newton-Schulz
    iteration with coefficients (a, b, c) = (3.4445, -4.7750, 2.0315).

    Requires ``G.ndim == 2``.  The iteration converges to the matrix whose
    singular values are all ≈ 1 (i.e. the "zeroth power" U V^T of G = U S V^T).
    The output is always cast back to ``G.dtype``.

    The compute dtype controls the working precision *inside* the iteration
    only — it never affects the model weights, the forward pass, the loss, or
    the gradients, which remain at whatever dtype the training loop uses.
    The Muon reference implementation uses bfloat16 for speed, but for
    precision-sensitive regression tasks we default to ``G.dtype`` (typically
    fp32) so the orthogonalization preserves the full gradient precision.
    On A100/H100 fp32 matmul still uses TF32 tensor cores, so the speed
    penalty vs bfloat16 is small.
    """
    assert G.ndim == 2, f"Muon NS expects 2-D input, got shape {tuple(G.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315
    dtype_for_compute = compute_dtype if compute_dtype is not None else G.dtype
    X = G.to(dtype_for_compute)
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class MuonHybrid(Optimizer):
    """Muon for 2-D matrix params, AdamW for the rest, dispatched per group.

    Expects ``param_groups`` to be a list of dicts, each with a boolean
    ``use_muon`` key.  Groups with ``use_muon=True`` use the Muon update;
    others use AdamW.  All groups support standard AdamW-style ``lr`` and
    ``weight_decay`` kwargs; the ``betas`` / ``eps`` kwargs apply only to
    AdamW groups, and ``momentum`` / ``nesterov`` / ``ns_steps`` only to
    Muon groups.

    The Muon update is rescaled by ``0.2 * sqrt(max(fan_in, fan_out))`` so
    that the effective per-parameter step matches the AdamW-style normalized
    magnitude (Keller Jordan, 2024).  This means Muon's ``lr`` can be set to
    roughly the same order as AdamW's ``lr`` — typically 1–3× higher.
    """

    def __init__(
        self,
        param_groups: Iterable[dict[str, Any]],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_dtype: torch.dtype | None = None,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_dtype=ns_dtype,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_muon=False,
        )
        super().__init__(list(param_groups), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        momentum = float(group["momentum"])
        nesterov = bool(group["nesterov"])
        ns_steps = int(group["ns_steps"])
        ns_dtype = group.get("ns_dtype", None)
        wd = float(group["weight_decay"])

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim != 2:
                raise RuntimeError(
                    f"MuonHybrid: parameter in a use_muon=True group has "
                    f"ndim={g.ndim} (shape {tuple(g.shape)}); Muon requires 2-D."
                )

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            update_source = g.add(buf, alpha=momentum) if nesterov else buf
            update = _zeropower_via_newtonschulz5(
                update_source, steps=ns_steps, compute_dtype=ns_dtype
            )

            # Scale so the per-param effective step matches AdamW-style magnitude.
            scale = 0.2 * max(update.size(0), update.size(1)) ** 0.5

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group["weight_decay"])

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            denom = (v / bc2).sqrt_().add_(eps)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(m / bc1, denom, value=-lr)


def split_params_for_muon(
    model: torch.nn.Module,
    muon_lr: float,
    muon_weight_decay: float,
    adamw_lr: float,
    adamw_weight_decay: float,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    excluded_prefixes: tuple[str, ...] = ("input_net.", "output_head.", "pool_head."),
) -> list[dict[str, Any]]:
    """Build param_groups routing 2-D interior matrices to Muon, everything
    else to AdamW.

    Parameters whose fully-qualified name begins with one of
    ``excluded_prefixes`` stay on AdamW regardless of ndim, matching the
    "input embedding / output head / scalar params on AdamW" convention from
    the Muon literature.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        excluded = any(name.startswith(pre) for pre in excluded_prefixes)
        if p.ndim == 2 and not excluded:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": muon_lr,
            "weight_decay": muon_weight_decay,
        },
        {
            "params": adamw_params,
            "use_muon": False,
            "lr": adamw_lr,
            "weight_decay": adamw_weight_decay,
            "betas": adamw_betas,
        },
    ]
