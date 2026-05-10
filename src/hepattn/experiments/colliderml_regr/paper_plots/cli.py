"""Unified paper-plot pipeline CLI.

Usage:
    python -m hepattn.experiments.colliderml_regr.paper_plots.cli \
        --run-id 7972d00dcde44bb199bfdf4c870587a5 \
        --nicename ssmcls_q7_p0pretrain_zeroshot_7972d00d_ep49 \
        [--d0-run-id da4a769796454b0f961eb9d3839094a1] \
        [--ablation-axes pooling scaling] \
        [--gpu 0] \
        [--bootstrap-n 200] \
        [--skip-inference] \
        [--skip-aggregate]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import DATA_DIR, PAPER_PLOTS_ROOT, apply_paper_style
from . import aggregate as agg
from . import bundle as bnd
from . import inference as infer
from . import predictions as pr
from . import stats as st
from .plots import (
    heatmap_pred_vs_truth,
    residual_hist,
    residual_vs_pt,
    rms_vs_eta,
    target_vs_pred,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--nicename", required=True)
    p.add_argument("--d0-run-id", default=None)
    p.add_argument("--ablation-axes", nargs="*", default=[])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bootstrap-n", type=int, default=200)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--output-root", default=str(PAPER_PLOTS_ROOT))
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true")
    p.add_argument("--skip-plots", action="store_true",
                   help="useful for stats-only smoke check")
    args = p.parse_args(argv)

    apply_paper_style()
    t0 = time.time()
    output_root = Path(args.output_root)
    data_dir = Path(args.data_dir)

    print(f"[pipeline] {args.nicename} | run={args.run_id}")
    print(f"[pipeline] output -> {output_root / args.nicename}")

    # 1. ensure inference (no-op if h5 exists; polls if a user job is running)
    main_h5 = infer.ensure_predictions(args.run_id, gpu=args.gpu, data_dir=data_dir,
                                       skip_inference=args.skip_inference)
    d0_h5 = None
    if args.d0_run_id:
        d0_h5 = infer.ensure_predictions(args.d0_run_id, gpu=args.gpu, data_dir=data_dir,
                                         skip_inference=args.skip_inference)

    # 2. bundle (copies config + symlinks ckpt + h5)
    info = bnd.create(args.run_id, args.nicename,
                      d0_run_id=args.d0_run_id,
                      ablation_axes=args.ablation_axes,
                      output_root=output_root)
    bundle_dir = info["bundle_dir"]
    plots_dir = info["plots_dir"]

    # 3. load preds + build DM residuals
    print(f"[pipeline] loading predictions: {main_h5}")
    data = pr.load_predictions_with_d0_override(main_h5, d0_h5)
    print(f"[pipeline] building DM residuals against {data_dir}")
    res = pr.build_dm_residuals(data, data_dir=data_dir)
    print(f"[pipeline] DM count = {res['count']:,}")

    # 4. stats (bootstrap σ for raw std / IQR / iter-3σ on SSM + CKF + ratio)
    print(f"[pipeline] bootstrap stats (n={args.bootstrap_n})…")
    stats = st.compute_stats(res, n_boot=args.bootstrap_n, seed=0)
    st.write_stats(stats, bundle_dir)

    # 5. plots (each module saves PDF + PNG)
    if not args.skip_plots:
        print("[pipeline] target_vs_pred…")
        target_vs_pred.make(res, plots_dir)
        print("[pipeline] rms_vs_eta…")
        rms_vs_eta.make(res, plots_dir, n_boot=min(50, args.bootstrap_n))
        print("[pipeline] heatmap_pred_vs_truth…")
        heatmap_pred_vs_truth.make(res, plots_dir)
        print("[pipeline] residual_vs_pt panels…")
        residual_vs_pt.make(res, plots_dir)
        print("[pipeline] residual_hist…")
        residual_hist.make(res, plots_dir)

    # 6. aggregator (cross-run tables)
    if not args.skip_aggregate:
        print("[pipeline] aggregator…")
        agg.run(output_root)

    print(f"[pipeline] DONE in {time.time() - t0:.1f}s -> {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
