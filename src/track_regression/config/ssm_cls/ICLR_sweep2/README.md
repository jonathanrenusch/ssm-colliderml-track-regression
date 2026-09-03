# ICLR_sweep2 — attacking the precision floor (defined 2026-08-26)

Sweep 1 showed a pT-independent floor in every trial (z0 ≈ 117 µm, θ ≈ 0.5 mrad above
10 GeV while the truth-KF reaches 19 µm / 0.04 mrad) and the ACTS three-pixel-point seed
alone (`track_regression.seed`) already gives z0 ≈ 26 µm, θ ≈ 0.05 mrad, d0 ≈ 33 µm,
φ ≈ 0.87 mrad at those momenta. Sweep 2 tests whether the floor is (a) head-side
representation — predict the correction to the seed (`delta_anchor: seed_<p>`, residual
norm ranges), (b) encoder-side representation — finer input Fourier scales, or (c) fp32
arithmetic — an fp64 probe. Baseline for all comparisons: sweep-1 trial B
(4 L / dim 128 / d_state 64, Lion + OneCycle, batch 36 000, 20 epochs, geometry store).

| trial | file | GPU | change vs B | reads against |
|---|---|---|---|---|
| G | `G_4L_ds64_lion_cosine_bs36k_seedanchor.yaml` | 0 | all five heads predict target − seed; ranges d0 ±0.4 mm, z0 ±3.5 mm, φ wrapped, θ ±0.01 rad (plain quantile, not η), q/p ±0.5 | B |
| H | `H_4L_ds64_lion_cosine_bs36k_seedanchor_fourier10.yaml` | 1 | G + Fourier scales 2⁻¹⁰…2⁵ (finest period ~37 mm in z, ~13 mm in x/y; default finest ~1.2 m) | G, J |
| I | `I_4L_ds64_lion_cosine_bs18k_fp64.yaml` | 2 | float64 end-to-end (64-true, kernel v3c train+eval), batch 18 000, LR ×2.96 | B |
| J | `J_4L_ds64_lion_cosine_bs36k_fourier10.yaml` | 3 | Fourier scales 2⁻¹⁰…2⁵ only (absolute heads) | B, H |

Seed residual ranges were measured on `ICLR_retraining_geom/single_muon_uniform/test/part_0000`
(300 k tracks): |truth − seed| 99.99 % quantiles d0 0.40 mm, z0 3.4 mm, φ 14 mrad,
θ 10 mrad, q/p 0.48 e/GeV. The collate (`flat_data._pack`) computes `seed_<p>` for every
track (~0.5 M tracks/s, numpy); `predict_physical` adds the anchor back, so logged metrics,
`val/<p>/ssm_rms3s` and the prediction h5 are absolute parameters as before.

Launch (one process per GPU, nohup): `bash src/track_regression/config/ssm_cls/ICLR_sweep2/launch_sess3_gpus0-3.sh`
Read-out: `scripts/04_eval_ckpt_iclr.sh` → `rms_summary.txt` + `rms_by_pt.txt`; compare
G/H/J to B (`eval_plots/sweep1/B`) per pT bin, I to B.
