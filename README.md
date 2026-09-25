# track-regression

Learned charged-particle trajectory regression: seed-guided bidirectional
sequence models that fit the five perigee track parameters from the hits of a
track. The repository contains data preprocessing, training, the fused GPU
inference kernels, the inference-speed benchmarks and the scripts that produce
the resolution tables and figures, for four encoders: **minGRU** (the main
model), **Mamba-2**, **Transformer** and the **non-selective diagonal SSM**.

## Install

Needs Linux, an NVIDIA GPU with CUDA 12.8 drivers, and either
[pixi](https://pixi.sh) (recommended: `pixi.lock` pins the tested environment,
including the C compiler that `torch.compile`/Triton need) or pip with Python 3.10–3.13 and `gcc`.

```bash
pixi install && pixi shell        # or: pip install -e .
pytest                            # unit tests (a few minutes, needs a GPU)
```

## 1. Data

The ColliderML `drift_beamspot` samples are downloaded from the public portal and
turned into flat track stores (~0.75 TB raw download, ~0.3 TB of stores).

```bash
for ds in single_muon_uniform single_muon_loguniform single_muon_2GeV \
          single_muon_10GeV single_muon_50GeV ttbar; do
  bash scripts/fetch_data.sh $ds data/raw
done
bash scripts/build_stores.sh data/raw data/stores data/eval
```

This writes the training store `data/stores/mix3` (uniform-pT and log-uniform-pT
muons plus ttbar tracks with 1 ≤ pT ≤ 110 GeV), the four muon test samples
`data/eval/single_muon_{2GeV,10GeV,50GeV,uniform}` (with the truth-seeded Kalman
filter fits shipped with the data as the reference), and `data/eval/ttbar_bench`
(the sample the throughput is measured on).

## 2. Training

Stage 1 runs on one GPU (~30 h on an H100), stage 2 on two GPUs (~15 h).
Training is strict fp32.

```bash
python -m track_regression.train fit --config configs/minGRU_stage1.yaml
python -m track_regression.train fit --config configs/minGRU_stage2_finetune.yaml
```

The encoder ablation uses stage 1 only:
`configs/mamba2_stage1.yaml`, `configs/transformer_stage1.yaml`,
`configs/diagssm_stage1.yaml`.
Checkpoints land in `runs/<config name>/version_0/checkpoints/last.ckpt` (the paper
uses `last.ckpt`).

## 3. Resolution tables and figures

```bash
M=runs/minGRU_stage2_finetune/version_0/checkpoints/last.ckpt
bash scripts/eval_checkpoint.sh configs/minGRU_stage2_finetune.yaml $M data/eval results/minGRU
bash scripts/make_paper_figures.sh results/minGRU/preds data/eval figures/minGRU
```

`eval_checkpoint.sh` predicts the four test samples at the deployment settings
(fused kernels, fp16 encoder, float64 GPU seed) and writes a summary
(`results/minGRU/plots/rms_summary.txt`). `make_paper_figures.sh` writes the
resolution-vs-η, resolution-vs-pT and residual figures and prints the ratio table.

Encoder ablation (evaluate each stage-1 model with `PREDICT_ARGS="--encoder-dtype float32"`,
as in the paper, and run `make_paper_figures.sh` on each, then):

```bash
python scripts/table_encoder_ablation.py --mingru figures/abl_mingru --mamba2 figures/abl_mamba2 \
    --transformer figures/abl_transformer --diagssm figures/abl_diagssm
python scripts/plot_encoder_ablation.py --mingru results/abl_mingru/plots --mamba2 results/abl_mamba2/plots \
    --transformer results/abl_transformer/plots --diagssm results/abl_diagssm/plots
```

Gradient-cosine probe (appendix):
`python scripts/grad_cos_probe.py --config C --ckpt K --data-dir data/eval/ttbar_bench --out-dir figures/grad_cos`,
then `python scripts/plot_grad_cos.py figures/grad_cos/grad_cosines.npz figures/grad_cos`.

## 4. Inference speed

```bash
C=configs/minGRU_stage1.yaml; K=runs/minGRU_stage1/version_0/checkpoints/last.ckpt
python scripts/bench_infer.py --config $C --ckpt $K --data-dir data/eval/ttbar_bench --mode deployed
python scripts/bench_infer.py --config $C --ckpt $K --data-dir data/eval/ttbar_bench --mode reference
bash scripts/bench_batch_sweep.sh $C $K data/eval/ttbar_bench results/bench
python scripts/plot_throughput.py results/bench none figures
python scripts/bench_stage_share.py --config $C --ckpt $K --data-dir data/eval/ttbar_bench
```

`deployed` is the optimized path (packed layout, fused Triton kernels, compiled
Fourier front end, fp16 encoder GEMMs); `reference` is the exact code path each
encoder trains with (padded layout, strict fp32). The same switch is available
everywhere as `TRK_REFERENCE_KERNELS=1`. Run benchmarks on an idle GPU.

## Code

| path | content |
|---|---|
| `src/track_regression/seed.py`, `seed_torch.py` | analytic three-hit helix seed and per-hit residual features (numpy / GPU) |
| `src/track_regression/model.py` | regressor (normalisation, Fourier features, encoder, quantile heads) and Lightning module |
| `src/track_regression/losses.py` | seed-anchored quantile losses |
| `src/track_regression/mingru.py`, `mamba.py`, `transformer.py` | the four encoders |
| `src/track_regression/ops/`, `txf_packed.py` | fused Triton inference kernels |
| `src/track_regression/data.py`, `flat_data.py` | flat-store data loading |
| `scripts/` | data preparation, evaluation, benchmarks, figures |
| `configs/` | the five training configurations of the paper |

License: GPL-3.0-or-later.
