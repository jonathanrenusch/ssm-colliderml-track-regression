"""Re-render selected plot modules for an existing bundle.

Skips inference, bundle creation, stats — purely re-runs the chosen plot
module(s) against the bundle's predictions h5.  Use after editing a plot
module when you don't want to pay the bootstrap cost again.

Usage:
    python -m track_regression.paper_plots.replot \
        --nicename <bundle-dir-name> \
        [--module rms_vs_eta residual_hist target_vs_pred ...] \
        [--data-dir /scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune] \
        [--output-root ${PAPER_PLOTS_ROOT}]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import DATA_DIR, PAPER_PLOTS_ROOT, apply_paper_style
from . import predictions as pr
from .plots import (
    heatmap_pred_vs_truth,
    pt_distribution,
    residual_hist,
    residual_vs_pt,
    rms_vs_eta,
    target_vs_pred,
)

MODULES = {
    "target_vs_pred":         target_vs_pred,
    "rms_vs_eta":             rms_vs_eta,
    "heatmap_pred_vs_truth":  heatmap_pred_vs_truth,
    "residual_vs_pt":         residual_vs_pt,
    "residual_hist":          residual_hist,
    "pt_distribution":        pt_distribution,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nicename", required=True)
    p.add_argument("--module", nargs="+", default=list(MODULES.keys()),
                   choices=list(MODULES.keys()))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--output-root", default=str(PAPER_PLOTS_ROOT))
    args = p.parse_args(argv)

    apply_paper_style()
    bundle = Path(args.output_root) / args.nicename
    if not bundle.is_dir():
        raise SystemExit(f"bundle not found: {bundle}")

    meta = yaml.safe_load((bundle / "metadata.yaml").read_text())
    main_h5 = Path(meta["predictions_h5"])
    d0_h5 = (Path(meta["d0_predictions_h5"])
             if meta.get("d0_predictions_h5") else None)

    print(f"[replot] {args.nicename}")
    print(f"[replot]  preds = {main_h5.name}")
    print(f"[replot]  d0    = {d0_h5.name if d0_h5 else 'none'}")
    data = pr.load_predictions_with_d0_override(main_h5, d0_h5)
    res = pr.build_dm_residuals(data, data_dir=Path(args.data_dir))
    print(f"[replot]  DM N  = {res['count']:,}")

    for mod_name in args.module:
        print(f"[replot]  -> {mod_name}")
        MODULES[mod_name].make(res, bundle / "plots")
    print("[replot] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
