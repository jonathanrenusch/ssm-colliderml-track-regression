# v2 architecture ablations, 2-GPU node — round of 2026-09-19

Data: `/scratch/colliderml/ICLR_retraining_v2_mix3` (the pre-digitization-fix v2
mix3 the paper used). Recipe: the established stage-1 pretrain — Lion, OneCycle
1e-5 -> 5e-5 -> 1e-6, batch 2048, **25 epochs**, strict fp32
(`TRK_MATMUL_PRECISION=highest`), unchanged in every arm. Evaluation:
`04b_eval_ckpt_deploy.sh` on `ICLR_eval_v2_new`, `TRK_ABS_ETA_MAX=2`, reference =
the truth-seeded KF shipped with the data. Table: `scripts/abl_v2_table.py`.

## Reference arm (done)

`SSM_baseline_25ep` = the R2Lnoconv stage-1 checkpoint
(`logs/comet_offline/8f7e4ac9...`), i.e. exactly this recipe with the paper's
Mamba-2 encoder. Post-clip GM5 vs truth-KF: **0.994 / 0.994 / 0.997 / 0.960 /
0.991 / 0.996** on mu 2 / 10 / 50 / 100 GeV / uniform / ttbar_new_pt1; pre-clip
0.984 / 0.972 / 0.988 / 0.953 / 0.972 / 0.902. Every v2 arm is read against this
row, not against the v3 numbers.

## Arms in flight on this node

| arm | encoder | what it isolates | ETA |
|---|---|---|---|
| `V2_clru_25ep` | `ComplexLRUCLSEncoder` h270 | a **complex** (oscillatory) decay instead of a real one | Mon ~01:00 |
| `V2_mingru_fp16_25ep` | `MinGRUCLSEncoder` h194, fp16 encoder | reduced-precision arithmetic, nothing else | Sun ~10:00 |
| `V2_diagrnn_narrow_25ep` (queued) | `DiagRNNCLSEncoder` h194 | delete the gate GEMM and **keep the saving** | Mon ~06:00 |
| `V2_minlstm_25ep` (queued) | `MinLSTMCLSEncoder` h157 | minGRU's sibling | — |

Running on the other node: minGRU h194, one-directional Mamba, non-selective
diagonal SSM (h270, **parameter-matched**), Transformer.

### Why a complex decay

A track is a helix, so its azimuth advances almost linearly along the hit
sequence: the signal is periodic in phi, not merely decaying. A real per-channel
decay (minGRU, DiagRNN) can only forget; a complex eigenvalue `r e^{i w}` can
rotate. Parameterised LRU-style (`r = exp(-exp(nu))`, stable by construction).
Because `a` is a learned constant rather than an input-dependent gate, the scan
never materialises an `a` tensor — the composed multiplier after round `k` is
just `a^(2^k)` in closed form, so each Hillis-Steele round is one complex
multiply-add on `b` alone. That rewrite took the layer from 20.6 to 12.8 ms
fwd+bwd (1.6x) and the run from 26.4 to 33.5 it/s.

### The two non-selective arms are a pair, and only the pair is interpretable

Removing the input-dependent gate halves the in-projection (`D -> 2H` instead of
`D -> 4H`). That can be spent two ways, and they answer different questions:

| arm | hidden | encoder params | encoder FLOPs/track | question |
|---|---|---|---|---|
| minGRU (selective) | 194 | 401,968 | 10.69 M | baseline |
| diagRNN **param-matched** (other node) | 270 | 362,880 | 9.65 M (1.11x) | does selectivity matter at equal capacity? |
| diagRNN **width-matched** (queued here) | 194 | 201,760 | **5.37 M (1.99x)** | can we delete the gate and pocket 2x? |

The width-matched arm deliberately confounds gate-removal with capacity — the
parameter drop is a *consequence* of deleting the mechanism, not a free knob.
Read it only next to the param-matched twin. Literature support for expecting
selectivity not to matter at L = 20: Block-Biased Mamba (arXiv:2505.09022) and
arXiv:2609.16540 both argue input-dependent decay is a long-context/linguistic
bias; **no published ablation exists below L ~ 32**, so this is new ground.

## The precision question (`scripts/precision_feature_audit.py`)

Measured on 200 k real hits, not assumed:

* **10 mantissa bits cannot hold a 2 m detector to 43 um** — that needs 15.5
  bits. Min-max normalisation does *not* help: near 1.0 fp16 steps by 2^-11,
  which is 0.5 mm of x and 1.5 mm of z. Absolute coordinates must never be
  carried or differenced in fp16.
* **The network does not take its precision from them.** The fine Fourier
  components resolve x to **0.69 um** in fp16 at scale 2^-10, and the seed
  residuals `asinh(du/0.1mm)` are O(1) by construction. Both are pointwise, so
  autocast leaves them in fp32; only the GEMMs that consume them run reduced.
* **fp16 and TF32 have the SAME 10 mantissa bits.** The campaign already measured
  TF32 harmless (<= 0.3 %, 4.12/4.32), so fp16's forward precision inherits a
  measured result; what is new is only the 5-bit exponent, and nothing in this
  model approaches 65504.
* **bf16 is the wrong trade here**: 7 mantissa bits, 8x coarser, buying range we
  do not need. arXiv:2510.26788 is the citable evidence that bf16 rounding breaks
  value-level agreement where fp16 does not.
* The seed stays **fp64** regardless (4.32: the rc-R cancellation makes even fp32
  unusable at high pT).

Design first tried: `precision: 32-true` + `encoder_autocast_dtype: float16`,
i.e. reduced precision **exactly where the FLOPs are**, with the input
normalisation, the Fourier encoding, the heads, the loss and the seed all left in
fp32/fp64 and no GradScaler. **That diverged** (`V2_mingru_fp16_noscaler_DIVERGED.log`):
healthy to step 23,500 (val d0 32.7 um at 20 k), then `train/total` 0.074 ->
0.403 -> NaN at ~24,000. A loss spike overflowed an fp16 gradient and, with no
scaler, there is no skip-the-step mechanism, so the weights were poisoned in one
update. Loss scaling is **part of the fp16 recipe, not a confound** -- omitting
it tests "fp16 without loss scaling", which is known-broken. Relaunched as
`precision: 16-mixed` (Lightning's GradScaler, unscale-then-clip at the existing
`gradient_clip_val: 1.0`), which also casts the heads; the encoder is still the
only place the arithmetic matters. The
scan is forced to fp32 inside the layer (a product of 20 gates in fp16 would lose
mantissa and can underflow the 6e-8 subnormal floor); the packed Triton kernel
casts at its boundary (`tests/test_mingru.py::test_packed_kernel_runs_under_fp16_autocast...`).

**Measured: fp16 trains at 55.3 it/s against the fp32 control's 55.6.**
No training speedup at all — training is bound by the eager padded path, the
fp32 scan and the loader, not by the GEMMs. fp16 is an *inference* lever in this
model, and the training run's only job is to show the physics survives.

## Throughput does NOT scale with projection FLOPs at this size (`scripts/mingru_flop_scaling.py`)

Every "cheaper backbone" argument rests on the claim that ~all the cost is the
dense in-projections, so a FLOP cut should buy proportional throughput. Measured
on the packed inference kernel (random packed batches; only shapes matter),
interleaved repeats, at 32 k **and** 131 k tracks/batch -- the two agree to 2 %:

| hidden | encoder FLOPs/track | FLOP ratio vs 192 | measured speed ratio |
|---|---|---|---|
| 112 | 4.22 M | 2.49x | **1.60x** |
| 128 | 5.26 M | 2.00x | **1.52x** |
| 144 | 6.40 M | 1.64x | 1.24x |
| 176 | 9.03 M | 1.16x | 1.04x |
| 192 | 10.50 M | 1.00 | 1.00 |
| 224 | 13.78 M | 0.76x | 0.81x |
| 256 | 17.49 M | 0.60x | 0.72x |

**Below ~192 a FLOP cut returns only 60-75 % of itself**; above it the penalty is
sub-linear (cheap to widen). This is not launch overhead -- it is identical at
131 k tracks, where launches are fully amortised.

Consequences, and a correction to what I expected:

* the width-matched non-selective arm is a **1.99x FLOP** cut but should be
  expected to deliver only ~**1.3-1.5x** throughput, not 2x;
* structured / block-diagonal (Monarch, BLAST) projections would underdeliver
  for the same reason at this model size -- not worth a GPU-day yet;
* the levers that remain are **fewer tokens**, **cheaper arithmetic** (fp16) and
  **kernel work**, not fewer projection FLOPs.

### Width alignment is worth more than any of it

Raw times at 131 k tracks, interleaved:

| hidden | 4H | ms |
|---|---|---|
| 188 | 752 | 17.73 |
| 190 | 760 | 18.73 |
| **192** | **768 = 12 x 64** | **17.05** |
| 193 | 772 | **32.16** |
| 194 | 776 | 21.70 |
| 196 | 784 | 20.62 |

**h = 192 is 27 % faster than h = 194 for 2 % fewer parameters, and h = 193 is
88 % slower than either.** The projection width lands on or off a tensor-core
tile boundary, and nothing else about the model changes.

This is a methodological problem for the whole comparison: parameter-matching
the arms forced odd widths (minGRU 194, diagRNN 270/274, minLSTM 157, cLRU 270),
which penalises their throughput for reasons that have nothing to do with the
architecture. **Physics numbers are unaffected** (they are what the matched
widths were for), but every deployment throughput number must be re-measured at
each architecture's nearest 64-aligned width before it is quoted.

Caveat: these were measured with the other GPU training, so the absolute times
are contended. The 32 k / 131 k agreement and the interleaving make the
*ratios* trustworthy; they still want an uncontended re-run before publication.
