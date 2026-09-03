# ICLR_sweep1 — six single-GPU pretraining trials (decided with the user 2026-08-25)

Data: `/scratch/colliderml/ICLR_retraining_geom/single_muon_uniform` (geometry hit
order, event_id-joined; permanent copy `/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_geom`).
All trials: 4 L / dim 128 / **d_state 64** except C (10 L / dim 192 / d_state 128),
ICLR loss ranges (d0 ±7.1, z0 ±270, q/p ±1.0), packed batches, fixed 1 M-track val
subset, `TRK_MATMUL_PRECISION=highest`. `base.yaml` here is a verbatim copy of
`ssm_cls/ICLR/base.yaml` (train.py loads the base next to the leaf).

| trial | file | machine / GPU | network | optimizer + schedule | batch | epochs (steps) | axis tested |
|---|---|---|---|---|---|---|---|
| A | `A_4L_ds64_lion_cosine_bs2048.yaml` | other, 0 | 4L ds64 | Lion + OneCycle (legacy recipe) | 2 048 | 20 (1.87 M) | small-batch reference |
| B | `B_4L_ds64_lion_cosine_bs36k.yaml` | sess3, 0 | 4L ds64 | Lion + OneCycle, LR ×√19.5 | 36 000 | 20 (106 k) | batch size (vs A) |
| C | `C_10L_ds128_lion_cosine_bs2048.yaml` | other, 1 | 10L ds128 | Lion + OneCycle | 2 048 | 20 (1.87 M) | network size (vs A) |
| D | `D_4L_ds64_lion_wsd_bs36k.yaml` | sess3, 1 | 4L ds64 | Lion + WSD (5 % / 70 % / 25 %) | 36 000 | 20 (106 k) | schedule at large batch (vs B) |
| E | `E_4L_ds64_lion_cosine_bs36k_data25pct.yaml` | sess3, 2 | 4L ds64 | as B, **25 % of the training tracks** | 36 000 | 80 (106 k) | low-pT data sufficiency (vs B, per pT bin) |
| F | `F_4L_ds64_muon_wsd_bs36k.yaml` | sess3, 3 | 4L ds64 | Muon-hybrid + WSD (09c54481 recipe) | 36 000 | 20 (106 k) | fine-tune optimizer from scratch (vs A, B, D) |

Equal *epochs* (equal track presentations: 3.83 B) is the comparison unit; A and C
therefore run ~18× more optimizer steps than the large-batch trials — that is the
point of the test, not a confound. Read-out: `bash scripts/04_eval_ckpt_iclr.sh
<run_dir> <ckpt> <out_dir>` → `rms_summary.txt` (all five eval sets) and
`rms_by_pt.txt` (per-pT-bin iter-3σ RMSE, SSM vs truth-KF/CKF) — the latter is
the E-vs-B read-out. Comet: project `ssm-track-regression-iclr`, names `SW1-*`;
`val/<p>/ssm_rms3s` is the unbinned iter-3σ RMSE per epoch.

Launch (each script checks that the store is on /scratch and copies it from /eos
if not, then starts its trials with nohup; logs in `launch_logs/sweep1/`):

    bash src/track_regression/config/ssm_cls/ICLR_sweep1/launch_sess3_gpus0-3.sh     # B D E F
    bash src/track_regression/config/ssm_cls/ICLR_sweep1/launch_other_gpus0-1.sh     # A C  (stop baeedc59 first)

Memory / speed (dry run 2026-08-25, 60 steps each, GPUs 1–3): 40 000 tracks/step
of the 4 L / d_state 64 network peaked at **90.6 GB of 95.8 GB** — hence 36 000
(~2.2 MiB/track + 3 GB fixed ≈ 82 GB). Throughput in the real runs
2.1–2.4 steps/s at 36 k (the dry run's 1.1 was compile warm-up) → ~40 min/epoch
→ ~14 h per large-batch trial. Trials A/C at 2 048 fit anywhere
(incl. L40S); A ≈ 1.5–2 h/epoch, C ≈ 3 h/epoch on an H100 (6.2 h on an L40S).
The dry runs left three `dryrun-*` experiments in the Comet project — delete them.
