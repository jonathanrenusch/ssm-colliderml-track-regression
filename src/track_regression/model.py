"""Seed-guided track parameter regressor and its Lightning wrapper.

:class:`TrackParameterRegressor` maps the packed hits of a batch of tracks to
the raw outputs of the five quantile heads:

1. per-hit features are min-max normalised to [0, 1] and expanded by a
   multi-scale Fourier encoding ``sin/cos(x / base^n)``;
2. a dense ``input_net`` embeds them to ``dim``;
3. the sequence encoder (minGRU, Mamba-2, Transformer or the diagonal RNN)
   returns a pooled per-track vector;
4. ``pool_head`` and ``output_head`` (dense) produce the quantile channels,
   interpreted by :class:`~track_regression.losses.TrackParameterLoss` as
   corrections to the analytic seed.

The 15 input features are the 12 measured hit features plus three residuals
to the seed helix.  Training batches carry all 15 (computed in the data
loader); at inference the batch carries only the 12 measured features and the
forward pass computes the seed and the residuals on the GPU (float64), as in
deployment, and returns the seed as ``out["seed"]``.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from lightning import LightningModule
from lion_pytorch import Lion
from torch import Tensor, nn

from track_regression.dense import Dense
from track_regression.layout import fused_kernels_enabled
from track_regression.losses import TrackParameterLoss
from track_regression.muon import MuonHybrid, split_params_for_muon

PARAMS = ("d0", "z0", "phi", "theta", "qop")

# RMSNorm on fp16 activations under autocast (fp32 weight) falls back to the
# unfused kernel, which torch reports once per process; harmless.
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")


def fourier_encode(x: Tensor, fourier_scales: list[int], fourier_base: int) -> Tensor:
    """``(*, D)`` -> ``(*, 2 * len(fourier_scales) * D)``: ``[sin(x / b^n)..., cos(x / b^n)...]``."""
    sin = [torch.sin(x / (fourier_base**n)) for n in fourier_scales]
    cos = [torch.cos(x / (fourier_base**n)) for n in fourier_scales]
    return torch.cat(sin + cos, dim=-1)


class TrackParameterRegressor(nn.Module):
    """Input embedding -> sequence encoder -> quantile heads.

    Parameters
    ----------
    input_dim : int
        Per-hit features (15 = 12 measured + 3 seed residuals).
    dim : int
        Embedding width fed to the encoder.
    encoder : nn.Module
        ``encoder(x, cu_seqlens=..., seq_idx=...) -> (hit_output, pooled)`` with a
        ``pool_dim`` attribute.
    loss_module : TrackParameterLoss
        The quantile heads' loss; fixes the number of outputs.
    input_net_hidden_layers, pool_head_hidden_layers, output_head_hidden_layers : list[int]
        Hidden widths of the dense networks.
    pool_head_dim : int
        Output width of ``pool_head``.
    fourier_scales, fourier_base
        Fourier encoding of the normalised features.
    norm_min, norm_max : list[float]
        Per-feature min-max normalisation bounds.
    encoder_autocast_dtype : str
        Autocast dtype of the encoder only (``float32`` for training; the
        paper's inference runs the encoder in ``float16``).
    output_head_init_scale : float
        Scale of the last output layer's initial weights (predictions start near
        the seed).
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        encoder: nn.Module,
        loss_module: TrackParameterLoss,
        input_net_hidden_layers: list[int],
        pool_head_dim: int,
        pool_head_hidden_layers: list[int],
        output_head_hidden_layers: list[int],
        fourier_scales: list[int],
        fourier_base: int,
        norm_min: list[float],
        norm_max: list[float],
        encoder_autocast_dtype: Literal["float32", "float16", "bfloat16"] = "float32",
        output_head_init_scale: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.fourier_scales = list(fourier_scales)
        self.fourier_base = fourier_base
        self.encoder_autocast_dtype = getattr(torch, encoder_autocast_dtype)
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))
        self.input_net = Dense(input_dim * 2 * len(self.fourier_scales), dim, input_net_hidden_layers)
        self.encoder = encoder
        self.loss_module = loss_module
        self.pool_head = Dense(encoder.pool_dim, pool_head_dim, pool_head_hidden_layers)
        self.output_head = Dense(pool_head_dim, loss_module.total_outputs, output_head_hidden_layers)
        last = self.output_head.net[-1]
        with torch.no_grad():
            last.weight.mul_(output_head_init_scale)
            last.bias.zero_()

    def _frontend(self, x: Tensor) -> Tensor:
        """Min-max normalisation -> Fourier encoding -> input_net."""
        x = (x - self.norm_min) / (self.norm_max - self.norm_min).clamp(min=1e-8)
        return self.input_net(fourier_encode(x, self.fourier_scales, self.fourier_base))

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """``inputs``: packed batch (``hit_features`` (1, T, F), ``cu_seqlens``, ``seq_idx``).

        Returns ``{"pred": (B, n_outputs)}``, plus ``"seed"`` (B, 5) when the
        seed was computed here (inference batches with the 12 measured features).
        """
        x, cu_seqlens = inputs["hit_features"], inputs["cu_seqlens"]
        out: dict[str, Tensor] = {}
        if not self.training and x.shape[-1] == self.input_dim - 3:
            from track_regression.seed_torch import gpu_seed_features

            seed, res = gpu_seed_features(x[0], cu_seqlens, max_len=20)
            x = torch.cat([x[0], res], dim=1).unsqueeze(0)
            out["seed"] = seed

        if x.is_cuda and not self.training and fused_kernels_enabled():
            # compiled front end at inference (fuses the 32 sin/cos kernels + cat)
            fe = getattr(self, "_compiled_frontend", None)
            if fe is None:
                fe = self._compiled_frontend = torch.compile(self._frontend, dynamic=True)
            x = fe(x)
        else:
            x = self._frontend(x)

        with torch.autocast("cuda", dtype=self.encoder_autocast_dtype,
                            enabled=x.is_cuda and self.encoder_autocast_dtype != torch.float32):
            _, pooled = self.encoder(x, cu_seqlens=cu_seqlens, seq_idx=inputs.get("seq_idx"))
        pooled = pooled.float()
        out["pred"] = self.output_head(self.pool_head(pooled))
        return out


class TrackRegressionWrapper(LightningModule):
    """Lightning wrapper: optimiser, schedule, train / val / test steps.

    Parameters
    ----------
    model : TrackParameterRegressor
    lrs_config : dict
        ``initial``, ``max``, ``end``, ``pct_start``, ``weight_decay`` and
        ``schedule`` (``onecycle`` or ``wsd``; WSD also reads ``decay_pct``).
        MuonHybrid additionally reads ``muon_max``, ``muon_weight_decay``,
        ``muon_momentum``, ``muon_ns_steps`` and the AdamW ``betas``.
    optimizer : str
        ``Lion`` (stage 1) or ``MuonHybrid`` (stage 2).
    pretrained_ckpt_path : str | None
        Initialise the model weights from a checkpoint (fresh optimiser and
        schedule) -- the stage-2 fine-tune starts from the stage-1 ``last.ckpt``.
    """

    def __init__(
        self,
        model: nn.Module,
        lrs_config: dict[str, Any],
        optimizer: Literal["Lion", "MuonHybrid"] = "Lion",
        pretrained_ckpt_path: str | None = None,
        train_metrics_every_n_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.lrs_config = lrs_config
        self.opt_name = optimizer
        self.train_metrics_every_n_steps = int(train_metrics_every_n_steps)
        if pretrained_ckpt_path is not None:
            ckpt = torch.load(os.path.expanduser(pretrained_ckpt_path), map_location="cpu", weights_only=False)
            state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            self.model.load_state_dict(state, strict=True)
            print(f"[fine-tune] loaded model weights from {pretrained_ckpt_path}")

    def setup(self, stage: str) -> None:
        if stage == "fit" and getattr(self.trainer, "is_global_zero", True):
            n = sum(p.numel() for p in self.parameters())
            print(f"[model] {type(self.model.encoder).__name__}: {n / 1e6:.3f} M parameters")

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    @staticmethod
    def _anchors(outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Seed anchors for ``predict_physical``: the GPU seed of the forward when
        it computed one (inference), else the data loader's ``seed_<param>``."""
        seed = outputs.get("seed")
        if seed is None:
            return targets
        return {f"seed_{n}": seed[:, i] for i, n in enumerate(PARAMS)}

    # -- metrics ------------------------------------------------------------

    _UNITS: dict[str, tuple[str, float]] = {
        "d0": ("[mm]", 1.0), "z0": ("[mm]", 1.0), "phi": ("[mrad]", 1000.0),
        "theta": ("[mrad]", 1000.0), "qop": ("[1/GeV]", 1.0),
    }

    def _log_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        """Residual MAE / std per parameter; on val also collect the residuals
        for the iterative-3-sigma RMSE of :meth:`on_validation_epoch_end`."""
        sync = stage != "train"
        for name in PARAMS:
            r = preds[name] - targets[name]
            if name == "phi":
                r = torch.remainder(r + math.pi, 2.0 * math.pi) - math.pi
            unit, scale = self._UNITS[name]
            self.log(f"{stage}/{name}/mae", r.abs().mean(), sync_dist=sync)
            self.log(f"{stage}/{name}/precision {unit}", r.std() * scale, sync_dist=sync)
            if stage == "val":
                self._val_residuals.setdefault(name, []).append(r.detach().float().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_residuals: dict[str, list[Tensor]] = {}

    def on_validation_epoch_end(self) -> None:
        """Iterative-3-sigma-clipped RMSE of the residuals pooled over the whole
        validation set and all ranks (the estimator quoted in the paper)."""
        from track_regression.eval_utils import iterative_rms_convergence

        local = {n: (torch.cat(v).numpy() if v else None) for n, v in self._val_residuals.items()}
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered: list = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            local = {n: np.concatenate([g[n] for g in gathered if g.get(n) is not None]) for n in local}
        self._val_residuals = {}
        summary = []
        for n, r in local.items():
            if r is None or len(r) < 2:
                continue
            unit, scale = self._UNITS[n]
            cut = iterative_rms_convergence(r)
            rms3s, tail = float(cut["rms"]) * scale, 1.0 - float(cut["n_kept"]) / len(r)
            self.log(f"val/{n}/ssm_rms3s {unit}", rms3s, rank_zero_only=True, sync_dist=False)
            self.log(f"val/{n}/ssm_tailfrac", tail, rank_zero_only=True, sync_dist=False)
            summary.append(f"{n} {rms3s:.4g} {unit.strip('[]')} (clipped {100 * tail:.2f} %)")
        if summary and self.trainer.is_global_zero:
            print(f"[val epoch {self.current_epoch}] iter-3sigma RMSE: " + " | ".join(summary), flush=True)

    # -- steps --------------------------------------------------------------

    def _shared_step(self, batch, stage: str):
        inputs, targets = batch
        outputs = self.model(inputs)
        losses = self.model.loss_module(outputs["pred"], targets)
        self.log(f"{stage}/total", losses["total"], sync_dist=stage != "train", prog_bar=True)
        if stage != "train" or self.global_step % self.train_metrics_every_n_steps == 0:
            for name in PARAMS:
                self.log(f"{stage}/{name}", losses[name], sync_dist=stage != "train")
            preds = self.model.loss_module.predict_physical(outputs["pred"], self._anchors(outputs, targets))
            self._log_metrics(preds, targets, stage)
        return outputs, losses

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")[1]["total"]

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")[1]["total"]

    # -- optimiser / schedule -----------------------------------------------

    def configure_optimizers(self):
        c = self.lrs_config
        if self.opt_name == "MuonHybrid":
            # Muon for the 2-D interior matrices, AdamW for the rest (norms,
            # biases, SSM scalars, input_net / pool_head / output_head); each
            # group starts at its own peak LR and the schedule scales both.
            groups = split_params_for_muon(
                self.model, muon_lr=float(c["muon_max"]),
                muon_weight_decay=float(c.get("muon_weight_decay", c["weight_decay"])),
                adamw_lr=float(c["max"]), adamw_weight_decay=float(c["weight_decay"]),
                adamw_betas=tuple(c.get("betas", (0.9, 0.95))))
            opt = MuonHybrid(groups, lr=float(c["max"]), momentum=float(c.get("muon_momentum", 0.95)),
                             ns_steps=int(c.get("muon_ns_steps", 5)))
        elif self.opt_name == "Lion":
            opt = Lion(self.model.parameters(), lr=c["initial"], weight_decay=c["weight_decay"], use_triton=True)
        else:
            raise ValueError(f"unknown optimizer {self.opt_name!r}")

        total = self.trainer.estimated_stepping_batches
        if c.get("schedule", "onecycle") == "wsd":
            # Warmup-Stable-Decay: linear warm-up initial -> max, constant, cosine decay -> end
            warmup = int(float(c["pct_start"]) * total)
            decay_start = int((1.0 - float(c["decay_pct"])) * total)
            if len(opt.param_groups) == 1:
                opt.param_groups[0]["lr"] = c["max"]
            sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=float(c["initial"]) / float(c["max"]),
                                                  end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=decay_start - warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - decay_start, eta_min=float(c["end"])),
            ], milestones=[warmup, decay_start])
        else:
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=c["max"], total_steps=total, div_factor=c["max"] / c["initial"],
                final_div_factor=c["initial"] / c["end"], pct_start=float(c["pct_start"]))
        return [opt], [{"scheduler": sch, "interval": "step"}]
