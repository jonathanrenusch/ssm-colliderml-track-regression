"""Largely inspired by / copied from https://github.com/samvanstroud/hepattn."""

import math
import re
from collections import defaultdict

from lightning import Callback, LightningModule, Trainer


# Parameter name patterns used to group grads into logical submodules.
# Order matters — the first matching pattern wins. The regexes are compiled
# once at import time. The resulting label is what appears in the metric key
# ``grad/<label>/{norm,avg_abs,max_abs}``.
#
# Encoder layers are matched dynamically so the logger works for any depth —
# the ``encoder_layer_<i>`` label carries the layer index. Everything outside
# these patterns is aggregated under ``other``.
_LAYER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:model\.)?input_net\."),       "input_net"),
    # d0_* patterns must come before output_head/pool_head patterns so the
    # more specific name wins.
    (re.compile(r"^(?:model\.)?d0_output_head\."),  "d0_output_head"),
    (re.compile(r"^(?:model\.)?d0_pool_head\."),    "d0_pool_head"),
    (re.compile(r"^(?:model\.)?output_head\."),     "output_head"),
    (re.compile(r"^(?:model\.)?pool_head\."),       "pool_head"),
    (re.compile(r"^(?:model\.)?fwd_head\."),        "fwd_head"),
    (re.compile(r"^(?:model\.)?bwd_head\."),        "bwd_head"),
]
_ENCODER_LAYER_RE = re.compile(r"^(?:model\.)?encoder\.layers\.(\d+)\.")
_ENCODER_OTHER_RE = re.compile(r"^(?:model\.)?encoder\.")


def _group_for(name: str) -> str:
    """Map a parameter name to a submodule group label."""
    for pat, label in _LAYER_PATTERNS:
        if pat.match(name):
            return label
    m = _ENCODER_LAYER_RE.match(name)
    if m is not None:
        return f"encoder_layer_{int(m.group(1)):02d}"
    if _ENCODER_OTHER_RE.match(name):
        return "encoder_other"
    return "other"


class GradientLoggerCallback(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_parameter_stats: bool = False,
        log_layer_stats: bool = True,
        log_output_head_per_dim: bool = False,
        output_head_param_groups: dict[str, list[int]] | None = None,
    ):
        """Log gradient statistics during training.

        Always logs the three global metrics (`grad/global_norm`,
        `grad/avg_abs`, `grad/max_abs`). When ``log_layer_stats`` is set
        (default), the same three metrics are also logged under
        ``grad/<group>/...`` for each model submodule group:

        - ``input_net``, ``output_head``, ``pool_head``, ``fwd_head``,
          ``bwd_head`` — the Dense heads around the encoder.
        - ``encoder_layer_00``, ``encoder_layer_01``, ... — one group per
          encoder layer (matched via ``encoder.layers.<i>.*``, works for any
          depth).
        - ``encoder_other`` — any encoder-level parameter outside
          ``encoder.layers`` (e.g. trunk-level norms, CLS tokens).
        - ``other`` — anything that did not match the above.

        Args:
            log_every_n_steps: frequency of logging. ``0`` disables entirely.
            log_parameter_stats: also log norm+std per individual parameter
                tensor. Off by default — produces a very large number of
                series.
            log_layer_stats: aggregate grads per submodule group as described
                above. On by default so deeper networks come with per-layer
                visibility out of the box.
            log_output_head_per_dim: also log per-row gradient ``max_abs`` of
                the output_head's final readout Linear weight (one series
                per output dim). The readout is identified as the 2-D weight
                in the ``output_head`` namespace with the smallest first
                dim (= num_outputs). Off by default; useful for tracing
                which output dim is responsible for late-training gradient
                spikes.
            output_head_param_groups: optional mapping ``{label: [start,
                end]}`` (half-open row index slices over the readout) used
                to aggregate per-row stats into per-parameter series logged
                under ``grad/output_head/<label>/{max_abs,norm}``. Only
                consulted when ``log_output_head_per_dim`` is True.
        """
        self.log_every_n_steps = log_every_n_steps
        self.log_parameter_stats = log_parameter_stats
        self.log_layer_stats = log_layer_stats
        self.log_output_head_per_dim = log_output_head_per_dim
        # Optional mapping {param_name: [start, end]} (half-open) over the
        # output_head readout's row index. Logged as
        # ``grad/output_head/<param>/{max_abs,norm}`` aggregated over rows
        # in the slice.
        self.output_head_param_groups = output_head_param_groups
        self._sync_dist = False

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        self._sync_dist = len(trainer.device_ids) > 1

    def _find_output_head_readout(self, pl_module):
        """Locate the output_head's final readout Linear weight (the one with
        the smallest first-dim 2D weight under the output_head namespace).
        Returns (param, readout_first_dim) or (None, None) if not found.
        """
        readout = None
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            # Match top-level output_head (not d0_output_head, which has its
            # own grouping).
            if not (re.match(r"^(?:model\.)?output_head\.", name)):
                continue
            if param.dim() != 2:
                continue
            # The final readout has the *smallest* first dim
            # (num_outputs, e.g. 30 for Q7), the hidden Linears have
            # first-dim = hidden_size which is larger.
            if readout is None or param.shape[0] < readout.shape[0]:
                readout = param
        return readout

    def _log_output_head_per_dim(self, pl_module):
        readout = self._find_output_head_readout(pl_module)
        if readout is None or readout.grad is None:
            return
        grad_abs = readout.grad.detach().abs()
        per_row_max = grad_abs.amax(dim=1)  # (num_outputs,)
        per_row_norm = readout.grad.detach().norm(2, dim=1)  # (num_outputs,)
        n_out = per_row_max.numel()

        # Always log per-row max_abs (one series per output dim).
        for i in range(n_out):
            pl_module.log(
                f"grad/output_head/dim_{i:02d}/max_abs", per_row_max[i].item(),
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )

        # If a parameter-group mapping is supplied, also aggregate over each
        # group (max of max_abs, sum of norm-squared as group norm).
        if self.output_head_param_groups is not None:
            for label, span in self.output_head_param_groups.items():
                if len(span) != 2:
                    continue
                start, end = int(span[0]), int(span[1])
                if start < 0 or end > n_out or start >= end:
                    continue
                rows_max = per_row_max[start:end].max().item()
                rows_norm_sq = (per_row_norm[start:end] ** 2).sum().item()
                pl_module.log(
                    f"grad/output_head/{label}/max_abs", rows_max,
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )
                pl_module.log(
                    f"grad/output_head/{label}/norm", math.sqrt(rows_norm_sq),
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )

    def on_after_backward(self, trainer, pl_module):
        if self.log_every_n_steps <= 0:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_sq_norm = 0.0
        total_abs = 0.0
        total_params = 0
        max_abs = 0.0

        # Per-group accumulators.
        group_sq_norm: dict[str, float] = defaultdict(float)
        group_abs:     dict[str, float] = defaultdict(float)
        group_params:  dict[str, int]   = defaultdict(int)
        group_max:     dict[str, float] = defaultdict(float)

        for name, param in pl_module.named_parameters():
            grad = param.grad
            if grad is None:
                continue

            grad_detached = grad.detach()
            param_norm = grad_detached.norm(2).item()
            abs_grad = grad_detached.abs()
            abs_sum = abs_grad.sum().item()
            abs_max = abs_grad.max().item()
            numel = grad_detached.numel()

            total_sq_norm += param_norm ** 2
            total_abs += abs_sum
            total_params += numel
            max_abs = max(max_abs, abs_max)

            if self.log_layer_stats:
                g = _group_for(name)
                group_sq_norm[g] += param_norm ** 2
                group_abs[g]     += abs_sum
                group_params[g]  += numel
                if abs_max > group_max[g]:
                    group_max[g] = abs_max

            if self.log_parameter_stats:
                pl_module.log(
                    f"grad/{name}/norm", param_norm,
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )
                pl_module.log(
                    f"grad/{name}/std", grad_detached.std().item(),
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )

        if total_params == 0:
            return

        pl_module.log(
            "grad/global_norm", math.sqrt(total_sq_norm),
            on_step=True, on_epoch=False, logger=True,
            sync_dist=self._sync_dist,
        )
        pl_module.log(
            "grad/avg_abs", total_abs / total_params,
            on_step=True, on_epoch=False, logger=True,
            sync_dist=self._sync_dist,
        )
        pl_module.log(
            "grad/max_abs", max_abs,
            on_step=True, on_epoch=False, logger=True,
            sync_dist=self._sync_dist,
        )

        if self.log_output_head_per_dim:
            self._log_output_head_per_dim(pl_module)

        if not self.log_layer_stats:
            return

        for g, n in group_params.items():
            if n == 0:
                continue
            pl_module.log(
                f"grad/{g}/norm", math.sqrt(group_sq_norm[g]),
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )
            pl_module.log(
                f"grad/{g}/avg_abs", group_abs[g] / n,
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )
            pl_module.log(
                f"grad/{g}/max_abs", group_max[g],
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )
