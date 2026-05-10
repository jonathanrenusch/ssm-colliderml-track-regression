"""Build the per-run reproducibility bundle.

Layout:
    <output_root>/<nicename>/
        config.yaml             (copied)
        metadata.yaml           (run id, ckpt path, dataset, d0_source_run_id, ablation_axes)
        best.ckpt               -> ckpts/<best>.ckpt   (symlink)
        test_predictions.h5     -> <best>__test_predictions.h5  (symlink)
        d0_override.h5          -> ... (symlink, only if --d0-run-id used)
        plots/
        stats.txt
        stats.json
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from . import COMET_OFFLINE_ROOT, PAPER_PLOTS_ROOT


def _resolve_best_ckpt_and_h5(run_dir: Path) -> tuple[Path, Path]:
    """Pick the best (top-k=1) ckpt and its matching predictions h5.

    Convention: top-k=1 saved ckpts are named ``epoch=NNN-val_total=X.YYY.ckpt``;
    the ``last.ckpt`` is also present but we prefer the val-best one when it exists.
    """
    ckpts_dir = run_dir / "ckpts"
    val_ckpts = sorted(ckpts_dir.glob("epoch=*-val_total=*.ckpt"))
    # Prefer a val-best ckpt whose matching h5 already exists (handles the
    # case where the user inferred an older epoch before a newer checkpoint
    # got saved). Fall back to last.ckpt + last__*.h5, then to first val_ckpt.
    for c in val_ckpts:
        h5 = run_dir / f"{c.stem}__test_predictions.h5"
        if h5.exists():
            return c, h5
    last_h5 = run_dir / "last__test_predictions.h5"
    if last_h5.exists():
        return ckpts_dir / "last.ckpt", last_h5
    # Last resort: any *__test_predictions.h5 in the run dir, paired with the
    # ckpt whose stem matches.
    any_h5s = sorted(run_dir.glob("*__test_predictions.h5"))
    if any_h5s:
        h5 = any_h5s[0]
        ckpt_stem = h5.name[: -len("__test_predictions.h5")]
        ckpt = ckpts_dir / f"{ckpt_stem}.ckpt"
        if not ckpt.exists():
            ckpt = ckpts_dir / "last.ckpt"
        return ckpt, h5
    if val_ckpts:
        c = val_ckpts[0]
        return c, run_dir / f"{c.stem}__test_predictions.h5"
    return ckpts_dir / "last.ckpt", last_h5


def create(
    run_id: str,
    nicename: str,
    *,
    d0_run_id: str | None = None,
    ablation_axes: list[str] | None = None,
    output_root: Path = PAPER_PLOTS_ROOT,
) -> dict:
    """Create the bundle directory and return a dict of resolved paths.

    Idempotent: re-running rebuilds symlinks and metadata but does not delete plots.
    """
    output_root = Path(output_root)
    bundle_dir = output_root / nicename
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "plots").mkdir(exist_ok=True)

    run_dir = COMET_OFFLINE_ROOT / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Comet run dir not found: {run_dir}")

    # copy config + metadata
    src_cfg = run_dir / "config.yaml"
    if src_cfg.exists():
        shutil.copy2(src_cfg, bundle_dir / "config.yaml")
    src_meta = run_dir / "metadata.yaml"
    if src_meta.exists():
        shutil.copy2(src_meta, bundle_dir / "source_metadata.yaml")

    # symlink best ckpt + predictions h5
    best_ckpt, h5 = _resolve_best_ckpt_and_h5(run_dir)
    _symlink(best_ckpt, bundle_dir / "best.ckpt")
    if h5.exists():
        _symlink(h5, bundle_dir / "test_predictions.h5")

    # optional d0 override
    d0_h5 = None
    if d0_run_id:
        d0_run_dir = COMET_OFFLINE_ROOT / d0_run_id
        _, d0_h5 = _resolve_best_ckpt_and_h5(d0_run_dir)
        if d0_h5 and d0_h5.exists():
            _symlink(d0_h5, bundle_dir / "d0_override.h5")

    # write our own metadata.yaml capturing what's in the bundle
    meta = {
        "nicename": nicename,
        "run_id": run_id,
        "ckpt_path": str(best_ckpt),
        "predictions_h5": str(h5) if h5.exists() else None,
        "d0_source_run_id": d0_run_id,
        "d0_predictions_h5": str(d0_h5) if d0_h5 and d0_h5.exists() else None,
        "ablation_axes": list(ablation_axes or []),
    }
    with open(bundle_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return {
        "bundle_dir": bundle_dir,
        "plots_dir": bundle_dir / "plots",
        "ckpt": best_ckpt,
        "predictions_h5": h5 if h5.exists() else None,
        "d0_predictions_h5": d0_h5 if d0_h5 and d0_h5.exists() else None,
        "config_yaml": bundle_dir / "config.yaml",
        "metadata": meta,
    }


def _symlink(src: Path, dst: Path) -> None:
    """Create or replace a symlink dst -> src (absolute)."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
