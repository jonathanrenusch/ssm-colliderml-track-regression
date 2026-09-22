# Inference precision study (2026-09-21, evening) -- Mamba-2 at fp16, and the headroom left in the fastest model

Everything below was measured on this node (sess5, 2x H100 NVL) in this session:
throughput on the idle GPU 0 (`scripts/bench_infer_flat.py --gpu-seed --iters 100`, fp64
GPU seed inside the timed loop, `ttbar_new_pt1` store, kernel switches on, TF32 matmuls,
`os._exit` wrapper), physics on GPU 1 (`scripts/04b_eval_ckpt_deploy.sh` on
`ICLR_eval_v2_new`, six test sets, `TRK_ABS_ETA_MAX=2`, truth-KF reference), compared cell by
cell with the new `scripts/compare_rms_summary.py` (60 cells = 6 sets x 5 parameters x
post/pre-clip; N must be identical).  Raw logs: scratchpad `precision/bench_logs/summary.log`,
`precision/stages_queue_gpu0.out`, `precision/map_*.txt`; eval bundles
`eval_plots/ablations_2026-09/v2_evals/<tag>` with `<tag>.log`.

Precision rules kept throughout: the seed stays float64 (`TRK_SEED_DTYPE`, the rc-R
cancellation argument of CLAUDE.md 4.32); nothing below fp16 is deployed.

## 1. Precision map of the deployed forward, as it is today (measured, not read off the code)

Probe: scratchpad `precision/precision_map.py` -- forward hooks on every Linear / norm
module plus explicit probes on the seed, the Fourier tensor, the scan ops and the
quantile ladder, on one real `ttbar_new_pt1` batch (8 192 tracks, 108 500 hits), eager
front end so every stage is visible, `TRK_SSD_BUCKET16=1`, TF32 matmuls, encoder autocast
fp16, seed float64.  Output: `precision/map_mingru192_deployed.txt`, `map_mamba2_fp16.txt`,
`map_mingru192_all16.txt`.

### 1.1 minGRU h=192 (the paper's model), deployed path

| stage | tensor | dtype (measured) | arithmetic |
|---|---|---|---|
| store -> batch | `hit_features` (1, n_hits, 12), `cu_seqlens` | fp32, int32 | -- |
| `gpu_seed_features` | seed (B, 5), residual features (n_hits, 3) | **float64 internally** (`TRK_SEED_DTYPE`), both **cast to fp32** on return | 292 tiny kernels; the fp32 cast of the anchors costs <= 1 ulp of |z0| <= 270 mm = 0.03 um |
| `torch.cat([hits, res])` | (1, n_hits, 15) | fp32 | -- |
| `_normalise` | (1, n_hits, 15) | fp32 | elementwise |
| `fourier_encode` | (1, n_hits, **480**) | **fp32 -- 198.7 MiB for 108 500 hits, i.e. ~3.3 GB written + read at 131 k tracks** | sin/cos, compiled with the normalisation |
| `input_net` = Linear(480->128) -> SiLU -> Linear(128->128) | (1, n_hits, 128) | fp32 in / fp32 out; GEMMs run as **TF32** (`TRK_MATMUL_PRECISION=high`) | cuBLAS TF32 (10-bit mantissa operands, fp32 accumulate) |
| encoder autocast (fp16) `layers.0.in_proj` Linear(128->768) | (n_hits, 768) | fp32 in -> **fp16** out | fp16 GEMM, fp32 accumulate |
| `mingru_bidi_packed` (layer 0) | (n_hits, 384) | fp16 in -> fp16 out | loads fp16, converts in registers, **recurrence accumulates in fp32**, stores fp16 |
| `layers.1.in_proj` Linear(384->768) | (n_hits, 768) | fp16 -> fp16 | fp16 GEMM |
| `mingru_bidi_packed` (layer 1) | (n_hits, 384) | fp16 -> fp16 | as above |
| terminal-state gather | (B, 384) | fp16 | index |
| `pool_norm` = nn.RMSNorm(384) | (B, 384) | **fp16 in -> fp16 out** -- autocast does NOT promote `rms_norm` to fp32 (it does promote LayerNorm; verified) | torch's non-fused rms_norm (weight fp32, input fp16) |
| `pool_proj` Linear(384->256) | (B, 256) | fp16 -> fp16 | fp16 GEMM |
| `pooled.to(heads dtype)` | (B, 256) | **fp32** | cast |
| `pool_head` = Linear(256->128) -> SiLU -> Linear(128->128) | (B, 128) | fp32 (TF32 GEMMs) | |
| `output_head` = Linear(128->128) -> SiLU -> Linear(128->35) | (B, 35) | fp32 (TF32 GEMMs) | |
| `predict_physical`: per parameter `raw[:, k:k+7]` -> softplus gaps -> `cumsum` -> median -> denormalise -> + anchor (`qop`: `* (|a|+0.02) + a`; `phi` wrapped) | 5 x (B,) | **fp32 throughout** (ladder, median, anchors) | five `torch.cumsum` launches over 6-wide rows |

Two facts the brief's sketch did not have: (i) the encoder's own `pool_norm` and
`pool_proj` are fp16 (the RMSNorm is not promoted), so the pooled vector reaching the heads
is an fp16 quantity cast up -- the heads' fp32 buys nothing on their *input*; (ii)
`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` is `True` by default
but toggling it to `False` changes **no prediction at all** on this batch (max |dpred| = 0
on all five parameters): cuBLAS accumulates these fp16 GEMMs in fp32 either way.

### 1.2 Mamba-2 conv-free 2L on the v5pc path with the fp16 encoder (after the change of section 2)

| stage | dtype (measured) |
|---|---|
| layer `norm` (RMSNorm on the fp32 residual stream) | fp32 -> fp32 |
| `forward_mamba.in_proj` / `backward_mamba.in_proj` Linear(128->648) | fp32 -> **fp16** (T_aug, 648) |
| `ssd_short_fwd_packed` (both directions) | fp16 -> fp16 (T_aug, 256); conv, softplus, decay, both IEEE `tl.dot` in fp32 in-kernel |
| `gated_rmsnorm` | fp16 -> fp16; gate, mean-square, rsqrt in fp32 in-kernel |
| `out_proj` Linear(256->128) | fp16 -> fp16 |
| `gate` Linear(128->128) | fp32 in -> fp16 out; `gate*x_fwd + (1-gate)*x_bwd` fp16 |
| `skip + combined` | fp32 + fp16 -> **fp32 residual stream** (as the padded autocast path) |
| `final_norm`, `cls_norm`, encoder output (B, 256) | fp32 |

So the Mamba-2 block keeps its residual stream and the class-token readout in fp32; only
the projections and the scan's activations are half precision -- the same split the minGRU
and the packed transformer use.

## 2. Task A -- Mamba-2 deployment path at fp16

### 2.1 What failed and what changed

Under `encoder_autocast_dtype: float16` the projections emit fp16 rows.  The scan kernel
compiled by implicit promotion (fp16 loads times fp32 conv weights), but
`_gated_rmsnorm_kernel` called `tl.sigmoid(z)` on an fp16 `z`; Triton 3.5's `tl.sigmoid` is
`1/(1+math.exp(-x))` and `math.exp` carries `_check_dtype(["fp32","fp64"])` -- hence
`ValueError: Expected dtype ['fp32', 'fp64'] but got fp16` at `16:17` of that kernel (the
`backbones_bs131000_float16.log` row).

Change (`src/track_regression/ops/ssd_short_triton.py`, additive, the minGRU kernel's
convention): in `_ssd_short_fwd_kernel2p` every load from the projection rows (conv'd B, C,
x and `dt_raw`) gets `.to(tl.float32)`; the conv, SiLU, softplus, decay matrix and both IEEE
`tl.dot` calls are unchanged fp32; the store casts to the output's storage dtype (the output
tensor follows the input dtype).  `_gated_rmsnorm_kernel` converts `y`, `z` and the weight
on load and computes gate, mean square and rsqrt in fp32.  `ssd_short_fwd_packed` and
`gated_rmsnorm` accept fp32/fp16/bf16; the opt-in merged-bidi op asserts fp32 with a
message (it is not the deployment path).  Glue (`mamba_short.fused_bidi_scan_packed`,
`mamba_cls._forward_packed`) needed no change: `in_proj` under autocast produces fp16
rows, the ops return fp16, `out_proj` runs fp16, the residual add promotes to fp32.  No
training code touched; the padded kernels are untouched.

### 2.2 Parity

* **fp32 path bit-identical to before** (scratchpad `precision/save_ref.py` before the edit,
  `check_kernel.py` after): 12 tensors -- packed scan fwd/rev x bucket16 on/off, gated norm
  on each, and a full conv-free layer (`d_conv=1`, 2 048 tracks) fwd+bwd with bucket16
  on/off -- all `torch.equal`.  (`.to(tl.float32)` on an fp32 load is elided by Triton.)
* **fp16 vs the fp32 torch reference** (`_packed_scan_torch_ref`, 3 000 tracks, max |delta| /
  max |ref|): scan 5.9-6.1e-4 (of which 3.7-4.0e-4 remains against the fp32 kernel fed the
  same fp16-rounded input, i.e. the in-kernel part is the output rounding), gated norm
  4.2-4.3e-4, full layer under fp16 autocast 6.8-7.0e-4 max / 5.9e-4 rms.  bf16: 3.0-3.9e-3 /
  3.4-3.8e-3 / 5.1-5.9e-3.
* Tests added to `tests/test_ssd_variants.py` (all on the GPU): fp16 packed scan vs
  reference for reverse x bucket16 (tolerance 2e-3), bf16 control (2e-2), fp32 output dtype
  unchanged, gated norm fp16 vs fp32 and vs a torch oracle, full conv-free layer under
  fp16/bf16 autocast.  `tests/test_ssd_variants.py` + `tests/test_precision_flags.py`: 23
  passed; `tests/test_mingru.py`: 28 passed (this session, GPU 1).

### 2.3 Physics gate (Task A.1)

`SSM_baseline_25ep_fp16` (encoder fp16 on v5pc + BUCKET16 + compiled front end, TF32
elsewhere, fp64 GPU seed) vs the reference bundle `SSM_baseline_25ep` (same path, fp32
encoder): **worst |delta ratio| = 0.00056 over 60 cells, 1 cell above 0.0005
(single_muon_2GeV post-clip z0 0.9896 -> 0.9902), N identical on all six sets.**  That is the
same class as the paths already accepted for the paper (minGRU fp16: 3 of 30 cells 0.001
apart; transformer packed fp16: worst 0.0008) -- **gate passed**.  Rounded to the three
decimals the paper prints, 58 of the 60 cells are identical.

### 2.4 Throughput (Task A.2; GPU 0 idle, `--gpu-seed --iters 100`, `ttbar_new_pt1`, seed fp64, switches on)

| Mamba-2 conv-free 2L, stage-1 ckpt | 131 000 tracks/batch | peak VRAM | 32 768 tracks/batch | peak VRAM |
|---|---:|---:|---:|---:|
| as trained (v3c, strict fp32, padded, no switches) | **578,568** tracks/s | 26.48 GiB | -- | -- |
| deployed, TF32 (v5pc + BUCKET16 + compiled front end), same session | 1,769,196 | 14.63 GiB | 1,637,887 | 3.69 GiB |
| (same, measured earlier today, `TRANSFORMER_PACKED` doc) | 1,737,255 | | | |
| **deployed, fp16 encoder (this work)** | **2,062,647** (repeat 2,057,571) | **9.38 GiB** | **1,861,753** | 2.48 GiB |

fp16 buys **+16.6 %** over TF32 measured in the same session (+18.7 % against the 1,737,255
quoted earlier today) and **-36 % peak VRAM** at 131 k; at 32 k +13.7 %.  Against the
as-trained path the deployed fp16 Mamba-2 is **3.57x** (0.579 M -> 2.06 M).  The gain is
smaller than the minGRU's fp16 step because the Mamba-2 encoder's time is in the scan kernel
(fp32 IEEE `tl.dot`, unchanged) rather than in the GEMMs: the fp16 win is the two `in_proj`
/ `out_proj` GEMMs per direction and half the traffic of the (T_aug, 648) projection rows
that the scan and the gated norm read.

## 3. Task B -- headroom on the fastest model (minGRU h=192, deployed fp16 encoder)

### 3.1 The flags (all opt-in, default behaviour unchanged, inference only)

| flag | where | what it changes |
|---|---|---|
| `TRK_FRONTEND_DTYPE=float16` (bench `--frontend-dtype`) | `model._frontend_eager` | the Fourier tensor is materialised in fp16 (`x.to(fd)` right after `fourier_encode`) and the input net runs under autocast(fd): both GEMMs fp16, SiLU fp16, embedding leaves in fp16.  Normalisation and sin/cos stay fp32. |
| `TRK_HEADS_DTYPE=float16` (bench `--heads-dtype`) | `model.forward` | `pool_head` + `output_head` under autocast(fd); the raw (B, 35) output is cast back to fp32 before the quantile decoding and the anchors. |
| `TRK_QUANTILE_LADDER=matmul` (bench `--quantile-ladder`) | `losses._ladder_prefix_sum` (used by `QuantileLoss` / `EtaQuantileLoss._ordered_from_raw`) | the ladder's prefix sum `base + cumsum(softplus gaps)` as one `(N,6) @ (6,6)` upper-triangular product **in float64**, rounded once to fp32.  CPU tensors and the training path (env unset) keep `torch.cumsum`. |
| `--encoder-dtype bfloat16` / `TRK_EVAL_ENCODER_DTYPE=bfloat16` | existing | control: 7 mantissa bits instead of 10 |

Why float64 for the ladder: `TRK_MATMUL_PRECISION=high` makes every fp32 GEMM TF32, which
would round the gaps to 10 bits -- an fp32 `deltas @ tri` (the measurement-only patch of
the transformer doc) is therefore *not* exact under the deployment setting.  In float64
the product of six fp32 numbers with 0/1 weights is exact, and its single rounding to fp32
is the correctly rounded sum; `torch.cumsum`'s sequential fp32 additions round up to five
times.  Measured (`tests/test_precision_flags.py::test_ladder_matmul_is_the_exact_sum`,
200 000 random ladders): the fp64 product equals the exact prefix sum to 4 fp64 ulp, its
fp32 result is within 0.5 ulp of the exact value on every element, the cumsum path is never
closer to the exact value than it, both keep the ladder strictly increasing, and the median
column is unaffected only when the median is the base -- here `tau=0.5` is the 4th of 7
quantiles, so the median IS a prefix sum and both paths can differ by 1 fp32 ulp of the
normalised value (1e-7 of a +-1 range, i.e. 4e-8 of |z0|'s 3.5 mm window = 0.14 nm).

### 3.2 Where the deployed minGRU forward spends its time (stage profile, 131 k tracks, uncontended GPU 0, torch.profiler kernel time, `precision/kernel_stages_prec.py`)

Reference row (deployed fp16 encoder, fp32/TF32 front end and heads, cumsum ladder):
seed 292 kernels / 2.14 ms | front end 6 / 6.44 ms | encoder 27 / 8.99 ms | heads + predict
67 / 4.82 ms | total 392 kernels, 22.38 ms (= 5.85 M tracks/s of pure kernel time; the
wall-clock bench gives 5.28 M).  Inside the two shared stages:

* front end: the compiled `normalise + sin/cos` kernel **3.71 ms** (it writes the 480-wide
  fp32 Fourier tensor), the three TF32 GEMMs of the input net **2.01 ms** (`sm90_xmma_gemm
  f32f32_tf32f32`), the fused `addmm + SiLU` 0.50 ms, the `cat([hits, residuals])` 0.23 ms;
* heads + predict: the **five `tensor_kernel_scan_innermost_dim` (cumsum) launches 4.37 ms
  = 91 % of the stage**, the three TF32 head GEMMs 0.16 ms, everything else < 0.1 ms.

So of the 22.4 ms, 4.4 ms are the quantile-ladder scans and 3.7 ms the Fourier write; the
head GEMMs (0.16 ms) and the input-net GEMMs (2.0 ms) are what the fp16 flags can touch.

### 3.3 The bf16 control, in detail

bf16 encoder (7 mantissa bits) against the fp16 reference: **worst |delta ratio| = 0.0101,
16 of 60 cells above 0.0005, only 42/60 identical at three decimals**, all in the direction
of *worse* SSM/truth-KF ratios and concentrated on q/p and phi at high pT -- 100 GeV pre-clip
q/p 0.9651 -> 0.9752, 50 GeV post-clip q/p 1.0261 -> 1.0310, uniform post-clip q/p
1.0181 -> 1.0249, 100 GeV pre-clip phi 0.9141 -> 0.9162.  Throughput is the same as fp16
(5.38 M vs 5.28 M at 131 k, within the run-to-run spread; the GEMMs are identical
tensor-core shapes).  So bf16 costs ~1 % of the q/p resolution where the network is
measurement-limited and buys nothing: it stays out, as the standing precision policy says
(fp16 == TF32 mantissa; bf16 is 8x coarser).

### 3.4 The quantile-ladder scans (Task B.3): 4.4 ms -> 0.09 ms, exact

Stage profile of the heads + `predict_physical` stage at 131 k tracks: with `torch.cumsum`
**4.82 ms, of which 4.37 ms are the five `tensor_kernel_scan_innermost_dim` launches**
(one per parameter, each over a (131 000, 6) fp32 tensor = 3 MB; that is 0.87 ms per
launch for 3 MB in / 3 MB out, ~7 GB/s -- torch's innermost-dim scan is a per-row block
scan built for long rows and is latency bound at width 6).  With
`TRK_QUANTILE_LADDER=matmul` the same stage is **0.59 ms**: the five fp64 GEMMs
(`sm90_xmma_gemm_f64f64`) take **0.085 ms together**, the rest is the unchanged softplus /
cat / denormalise / anchor elementwise work (~0.5 ms, 15 small `add` kernels among them).
Whole forward (kernel time): 22.38 -> 17.88 ms; wall-clock bench: **5.28 M -> 6.35 M
tracks/s (+20.3 %) at 131 k, 4.30 M -> 4.91 M (+14.4 %) at 32 k**, no VRAM change.
The transformer doc's measurement-only patch had predicted +19 % with an fp32/TF32 product;
the exact fp64 product delivers the same speed.

Physics gate of the ladder variant vs the reference: **worst |delta ratio| = 0.00000, 60/60
cells identical at three decimals** (and at four), N identical -- as the numerics promise
(correctly rounded vs sequentially rounded fp32 sums differ by <= 1 ulp of a normalised
value, i.e. 0.14 nm on z0).

Status: **implemented as an opt-in flag, NOT switched on by default** -- the training path
and every existing eval keep `torch.cumsum` unless `TRK_QUANTILE_LADDER=matmul` is set.
Making it the inference default is a one-line decision (the deployment script / bench flag)
and needs nothing else: `tests/test_precision_flags.py` covers exactness, monotonicity and the
CPU/default path.  What is left in that stage after the change (0.5 ms of ~15 elementwise
launches over (B,) and (B,6) tensors) is launch-bound and would need `torch.compile` of
`predict_physical` or a single fused kernel; at 131 k it is 3 % of the forward.

### 3.5 The fp16 front end: two formulations, one of which works

**a. cast after the concat** (first implementation: `fourier_encode` in fp32, then
`x.to(fp16)` and the input net under autocast).  Stage profile: the compiled sin/cos kernel
still writes the 480-wide tensor in fp32 (3.57 ms), then Inductor emits a *separate*
`triton_poi_fused__to_copy` kernel (**1.37 ms**) that reads it back and writes the fp16 copy
-- it does not fuse the cast into the concat kernel -- and the three input-net GEMMs go
from 2.01 ms (TF32 `xmma`) to 1.02 ms (fp16 `nvjet`).  Net front end 6.44 -> 6.47 ms: the
GEMM saving is eaten by the extra pass.  Bench: 5.28 -> 5.35 M tracks/s (+1.2 %), VRAM
6.03 -> 5.62 GiB.

**a'. round on store** (`fourier_encode(..., out_dtype=fp16)`: every sin/cos component is cast
before the concat, so the fused kernel computes sin/cos in fp32 and writes the 480-wide
tensor once, in fp16).  The Fourier tensor is bit-identical to formulation a's
(`torch.equal` on a real 108 500-hit batch); the embedding is not bit-identical (compiled
fp16 GEMM + fused bias/SiLU vs the eager autocast GEMM -- the same class of difference as
compiled-vs-eager fp32, max |d| 0.011 vs 0.016 in embedding units), and is 5.0e-4 relative
to the fp32 front end (the fp16 mantissa).  Bench: **5.28 -> 5.79 M tracks/s (+9.6 %) at
131 k, 4.30 -> 4.55 M (+6.0 %) at 32 k**, VRAM 5.62 GiB.  Stage profile and physics gate:
section 3.7.

**b. heads in fp16**: the three head GEMMs are 0.16 ms of the 22.4 ms forward; fp16 makes
them 0.07 ms + 0.07 ms of dtype copies (9 `float16_copy_kernel` launches) -- kernel count
67 -> 77, stage 4.82 -> 4.79 ms, bench +0.1 % (5.28 -> 5.28 M): **nothing to gain**, as
the precision map predicted (the pooled vector is already an fp16 quantity; the heads' cost
is the ladder, not the GEMMs).  Physics: worst |delta| 0.0008 (two q/p cells at 50 and
100 GeV), i.e. fine but pointless.

### 3.6 Mamba-2 stage profile for completeness (131 k, kernel time)

TF32: seed 2.14 | front end 6.34 | **encoder 93 kernels, 58.75 ms** | heads 4.71 -> 71.9 ms;
fp16: encoder **103 kernels, 47.15 ms** (-19.7 %; the ten extra kernels are autocast's
weight/activation casts) -> 60.4 ms total.  The scan kernel's IEEE fp32 `tl.dot`s are
untouched by the change, so the encoder gain is the GEMMs and the halved traffic of the
(T_aug, 648) projection rows; the shared seed / front end / heads are the same 13.2 ms as
for the minGRU, i.e. 22 % of the fp16 Mamba-2 forward vs 59 % of the minGRU's.

### 3.7 Round-on-store front end: stage profile

`minGRU h192 + frontend fp16 (round on store)`, 131 k: front end **9 kernels, 5.01 ms** (was
6 / 6.44): the fused `to_copy + clamp/cos/div/sin/sub` kernel 3.47 ms (the cast is now inside
it, no separate pass), the three fp16 `nvjet` GEMMs 1.03 ms (were 2.01 ms TF32), the fused
`addmm + SiLU` 0.27 ms (was 0.50), the `cat` 0.23 ms.  Forward kernel time 22.38 -> 20.33 ms;
with the fp16 heads and the fp64 ladder as well **16.25 ms (8.06 M tracks/s kernel time)**,
bench **6.97 M / 6.92 M (repeat) tracks/s at 131 k, 5.22 M at 32 k**, VRAM 5.62 GiB.

A finding worth recording: the sin/cos kernel takes 3.47 ms whether it writes fp32 (3.71 ms)
or fp16 -- it is **bound by the transcendental evaluations** (131 000 tracks x 13.3 hits x
15 features x 16 scales x {sin, cos} = 0.84 G sin/cos in 3.5 ms), not by the 480-wide write.
Half the remaining front end is therefore arithmetic that no dtype change touches; the
levers there are cheaper sines (half-angle recursion across the 16 dyadic scales, or the
hardware `__sinf` on the fine scales) or generating the features inside the GEMM -- kernel
work, outside this study.

### 3.8 Results table (Task B.2) -- one variable at a time on the deployed h=192 minGRU

Reference = `V2_mingru_h192_25ep` (fp16 encoder, TF32 front end and heads, cumsum ladder,
|eta| <= 2).  Physics: 60 cells (6 sets x 5 parameters x post/pre-clip), all N identical.
Throughput: GPU 0 idle, 131 000 and 32 768 tracks/batch, fp64 GPU seed in the loop.

| variant | worst \|delta ratio\| (cell) | cells > 0.0005 | identical at 3 decimals | N identical |
|---|---|---:|---:|---|
| Mamba-2 fp16 encoder (Task A) | 0.0006 (single_muon_2GeV post z0 0.9896->0.9902) | 1 | 58/60 | True |
| minGRU h192: + front end fp16 | 0.0007 (single_muon_100GeV post qop 0.9580->0.9573) | 2 | 53/60 | True |
| minGRU h192: + heads fp16 | 0.0008 (single_muon_50GeV post qop 1.0261->1.0253) | 2 | 53/60 | True |
| minGRU h192: + ladder matmul (fp64) | 0.0000 (single_muon_uniform pre phi 0.9735->0.9735) | 0 | 60/60 | True |
| minGRU h192: bf16 encoder (control) | 0.0101 (single_muon_100GeV pre qop 0.9651->0.9752) | 16 | 42/60 | True |
| minGRU h192: all three | 0.0009 (single_muon_50GeV pre qop 1.0284->1.0292) | 2 | 52/60 | True |
| minGRU h192: reference path re-run | 0.0000 (None) | 0 | 60/60 | True |

Mamba-2 fp16 encoder (Task A): single_muon_2GeV post z0 0.9896->0.9902
minGRU h192: + front end fp16: single_muon_100GeV post qop 0.9580->0.9573; single_muon_50GeV pre qop 1.0284->1.0290
minGRU h192: + heads fp16: single_muon_100GeV post qop 0.9580->0.9586; single_muon_50GeV post qop 1.0261->1.0253
minGRU h192: bf16 encoder (control): single_muon_100GeV post d0 0.9626->0.9638; single_muon_100GeV post phi 0.9102->0.9120; single_muon_100GeV post qop 0.9580->0.9618; single_muon_100GeV pre d0 0.9589->0.9598; single_muon_100GeV pre phi 0.9141->0.9162; single_muon_100GeV pre qop 0.9651->0.9752; single_muon_10GeV post qop 1.0103->1.0116; single_muon_10GeV pre qop 0.9959->0.9966; single_muon_2GeV post qop 1.0200->1.0214; single_muon_50GeV post qop 1.0261->1.0310; single_muon_50GeV pre phi 0.9809->0.9815; single_muon_50GeV pre qop 1.0284->1.0354; single_muon_uniform post d0 0.9930->0.9935; single_muon_uniform post phi 0.9739->0.9753; single_muon_uniform post qop 1.0181->1.0249; single_muon_uniform pre qop 0.9743->0.9754
minGRU h192: all three: single_muon_100GeV pre qop 0.9651->0.9657; single_muon_50GeV pre qop 1.0284->1.0292
| variant (on the h=192 minGRU) | physics: worst \|delta ratio\| vs reference | cells > 0.0005 | identical @3 dec. | tracks/s @131k (vs ref) | peak VRAM @131k | tracks/s @32k |
|---|---:|---:|---:|---:|---:|---:|
| deployed reference: fp16 encoder, fp32/TF32 front end + heads, cumsum ladder | 0.0000 | 0 | 60/60 | 5,280,814 (+0.0 %) | 6.03 GiB | 4,296,995 |
| a. + front end fp16 (`TRK_FRONTEND_DTYPE`) | 0.0007 | 2 | 53/60 | 5,345,122 (+1.2 %) | 5.62 GiB | 4,310,206 |
| b. + heads fp16 (`TRK_HEADS_DTYPE`) | 0.0008 | 2 | 53/60 | 5,284,511 (+0.1 %) | 6.03 GiB | 4,284,491 |
| 3. + quantile ladder as fp64 triangular GEMM (`TRK_QUANTILE_LADDER=matmul`) | 0.0000 | 0 | 60/60 | 6,350,421 (+20.3 %) | 6.03 GiB | 4,914,065 |
| c. bf16 encoder instead of fp16 (control) | 0.0101 | 16 | 42/60 | 5,376,772 (+1.8 %) | 6.03 GiB | 4,340,801 |
| a + b + 3 together (cast-after-concat front end) | 0.0009 | 2 | 52/60 | 6,368,011 (+20.6 %) | 5.62 GiB | 4,900,981 |
| a'. front end fp16, rounded on store (fused into the sin/cos kernel) | 0.0007 | 2 | 53/60 | 5,788,113 (+9.6 %) | 5.62 GiB | 4,554,717 |
| a' + b + 3 together -- the candidate deployment path | 0.0009 | 2 | 52/60 | 6,965,590 (+31.9 %) | 5.62 GiB | 5,215,734 |

repeat at 131k, all three (cast-after-concat): 6,403,143 vs 6,368,011 tracks/s (+0.6 %)

repeat at 131k, all three (round on store): 6,915,845 vs 6,965,590 tracks/s (-0.7 %)

repeat at 131k, reference: 5,329,923 vs 5,280,814 tracks/s (+0.9 %)

Reading the table:

* The reproducibility floor is **0.0000 / 60 of 60 cells** (reference path re-run, and the
  ladder variant): the eval is deterministic to the fourth decimal, so every non-zero
  delta above is a real dtype effect.
* fp16 in the front end or the heads moves at most two q/p cells at 50-100 GeV by
  0.0006-0.0009 -- the same size as the accepted Mamba-2 (0.0006) and transformer (0.0008)
  fp16 paths, one class below bf16 (0.0101, 16 cells).  Both round-on-store and
  cast-after-concat front ends give **identical physics to four decimals** (60/60 cells),
  as they must: the same fp16 Fourier tensor feeds the same GEMMs.
* The only variant that costs nothing in physics is the ladder (exact), and it is also the
  largest single gain (+20 %).  The front end adds +10 % (round on store) for a
  0.0007 deviation; the heads add nothing.
* Run-to-run spread of the bench at 131 k: +0.9 % / +0.6 % / -0.7 % on three repeated rows.

## 4. Recommendation

1. **Mamba-2 at fp16 goes into the three-network table**: as trained 0.579 M -> deployed
   fp16 **2.06 M tracks/s (3.6x)**, physics identical to the TF32 path within the bar the
   other two encoders already met (worst 0.0006, 58/60 cells identical at three decimals,
   N identical).  Row: "Mamba-2, 2L conv-free | 0.58 M | 2.06 M | 3.6x | v5pc fused SSD
   kernels (fp32 IEEE maths in-kernel, fp16 rows in/out) + BUCKET16 + compiled front end,
   fp16 projections".  The gap to the minGRU narrows from 3.0x to 2.6x (5.31 / 2.06) at the
   deployment batch, and the earlier caption note "TF32 for Mamba-2, whose kernel is
   fp32-typed" is no longer needed.
2. **Switch the quantile ladder to the fp64 triangular product at inference**
   (`TRK_QUANTILE_LADDER=matmul`; today opt-in, recommended as the deployment default in
   `04b_eval_ckpt_deploy.sh` and `bench_infer_flat.py`).  It is exact -- correctly rounded
   where cumsum is sequentially rounded -- with 60/60 cells identical, and it is worth
   **+20 % on the minGRU (5.28 -> 6.35 M)**; the same 4.3 ms is in every encoder's forward
   (Mamba-2, transformer), so they gain the same absolute time.  The training path keeps
   `torch.cumsum`; nothing changes for the loss.
3. **Adopt the round-on-store fp16 front end** (`TRK_FRONTEND_DTYPE=float16`) if a 0.0007
   worst-cell deviation is acceptable for +10 % (5.28 -> 5.79 M alone; 6.35 -> 6.97 M on top
   of the ladder, **+32 % in total, 5.62 GiB**).  It is the same precision class as the fp16
   encoder itself (fp16 and TF32 share the 10-bit mantissa; the encoder's first GEMM already
   rounds the embedding to fp16), and the two flagged cells are q/p at 50/100 GeV where the
   reference itself is the known miscalibrated region.  If the paper is to keep the "fp16
   agrees to three decimals on every cell" sentence literally, the honest phrasing becomes
   "52 of 60 cells identical, worst 0.0009" -- the user decides; both paths are one env var.
4. **Do not bother with fp16 heads** (+0.1 %, adds nine cast kernels) and **keep bf16 out**
   (no speed, 1 % worse q/p at high pT).  Nothing below fp16 is proposed; fp8 for the two
   in_proj GEMMs would need per-tensor scaling of the Fourier features and a physics gate
   nobody has run -- idea only.
5. The seed stays float64 (2.15 ms, 10 % of the forward; not a precision lever, a kernel
   count lever: 292 launches).  After 1-3 the h=192 minGRU forward at 131 k is 16.3 ms of
   kernel time: seed 2.15 (13 %), front end 5.0 (31 %, of which 3.5 ms is the sin/cos
   arithmetic itself), encoder 8.5 (52 %), heads 0.6 (4 %).  The next lever is no longer a
   dtype: it is the transcendental cost of the 16-scale Fourier features and the seed's
   launch count.

## 5. Code, tests, tooling (all additive; defaults unchanged)

| file | change |
|---|---|
| `src/track_regression/ops/ssd_short_triton.py` | `_ssd_short_fwd_kernel2p`: `.to(tl.float32)` on the four projection-row loads (conv'd B, C, x, `dt_raw`); `_gated_rmsnorm_kernel`: fp32 conversion on load of y, z, w; `ssd_short_fwd_packed` / `gated_rmsnorm` accept fp16/bf16 (output follows input); merged-bidi op asserts fp32 with a message; module docstring. fp32 path bit-identical (12 tensors, `torch.equal`). |
| `src/track_regression/model.py` | `_env_dtype()`; `frontend_dtype` / `heads_dtype` attributes read once from `TRK_FRONTEND_DTYPE` / `TRK_HEADS_DTYPE` in `__init__`; `fourier_encode(..., out_dtype=None)`; `_frontend_eager` rounds the Fourier components on store and runs the input net under autocast when set; the head block runs under an optional autocast and casts `pred` back. Unset = identical code path. |
| `src/track_regression/losses.py` | `_ladder_prefix_sum(deltas, owner)` (fp64 triangular product, opt-in `TRK_QUANTILE_LADDER=matmul`, CPU and default keep `torch.cumsum`); `QuantileLoss` / `EtaQuantileLoss._ordered_from_raw` call it. |
| `scripts/bench_infer_flat.py` | `--frontend-dtype`, `--heads-dtype`, `--quantile-ladder` (set the env vars before the model is built); the flags are printed in the report. |
| `scripts/04b_eval_ckpt_deploy.sh` | prints the three flags in its header line (they pass through the environment). |
| `scripts/compare_rms_summary.py` (new) | cell-by-cell gate of two `rms_summary.json` bundles: SSM/reference ratios post- and pre-clip, worst |delta|, cells above a tolerance, cells identical at three decimals, N identity. |
| `tests/test_ssd_variants.py` | +10 tests: fp16 packed scan vs torch reference (reverse x bucket16), bf16, fp32 output dtype, gated norm fp16 + torch oracle, full conv-free layer under fp16/bf16 autocast. |
| `tests/test_precision_flags.py` (new) | env parsing; ladder matmul exactness (fp64 = exact sum to 4 ulp64, fp32 result within 0.5 ulp of the exact value, never farther than cumsum, monotone, base column identical); CPU/default untouched. |

`pixi run -e default python -m pytest tests/test_ssd_variants.py tests/test_mingru.py tests/test_precision_flags.py -q`
-> **50 passed** (19 + 28 + 3; GPU 1, this session).

Not changed: training code, the padded Mamba-2 kernels, the merged-bidi kernel (fp32-only,
now with a clear assertion), CLAUDE.md (the main session is editing it), anything under
`/shared/tracking/NeurIPS_2026_SSM_Tracking`.

Measurement scripts (scratchpad `precision/`): `save_ref.py` / `check_kernel.py` (bit-identity
and fp16/bf16 deviation of the kernels), `precision_map.py` (dtype hooks), `probe_dtypes.py`
(autocast promotion rules), `kernel_stages_prec.py` (stage profile with the flags),
`fe16_parity.py`, `bench.sh` / `bench_nohang.py` / `tp_queue*_gpu0.sh` (throughput),
`eval_queue*_gpu1.sh` (physics), `gate_table.py` / `assemble_b_table.py` / `diff3.py` (tables).
