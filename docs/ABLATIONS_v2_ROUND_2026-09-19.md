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

### Width alignment: a real effect, but narrower than it first looked

Raw times at 131 k tracks, interleaved, packed Triton kernel:

| hidden | channel blocks at BD=64 | ms |
|---|---|---|
| 188 | 3 | 17.73 |
| **192** | **3** | **17.05** |
| 193 | 4 | 37.84 (suspect, see below) |
| 194 | 4 | 21.70-25.25 |
| 196 | 4 | 20.62 |

**h = 192 is ~25 % faster than h = 194 for 1 % fewer parameters.** The
mechanism is not mysterious and does not need to be inferred from timings --
it is the launch grid, `mingru_short_triton.py:122`:

    grid = lambda meta: (B, triton.cdiv(H, meta["BD"]), 2)

At `BD = 64`, `H = 192` is exactly 3 channel blocks and `H = 194` is 4: **+33 %
programs for +1 % channels**, with the last block 97 % idle.

Control (`TRK_MINGRU_KERNEL=off`, same widths, same batch): on the eager padded
path the spread collapses to **5.7 %** (192 / 193 / 194 = 354 / 394 / 374 ms)
against 21-25 % with the kernel. So the effect really is the kernel's channel
blocking, not the GEMM shapes -- an independent check of the grid argument.

Three corrections to the first reading of this:

1. **It is a property of my minGRU kernel, not of the hardware or of the
   model.** The dense GEMMs are nearly insensitive: an isolated cuBLAS
   `(1.7 M x 128) @ (128 x N)` costs only 7 % more at `N = 648` (8 past a
   boundary) than at `N = 640`, and padding it to 704 wins 1 %.
2. **The paper's deployment model is therefore fine.** Its four Mamba-2
   `in_proj` matrices are `648 x 128` -- 8 past a 64-boundary, exactly the
   suspicious shape -- which is why this was worth checking; at 7 % on one GEMM
   inside a well-tuned v5pc path, there is no free headline throughput here.
3. **The 88 % outlier at h = 193 should not be quoted.** It has the same block
   count as 194, so the extra cost is almost certainly a bad autotune pick made
   while the GPU was contended, not a real cliff.

So the earlier claim that "every deployment throughput number must be
re-measured at 64-aligned widths" was too broad. Only the minGRU arm uses this
kernel (`DiagRNNCLSEncoder`, `MinLSTMCLSEncoder` and `ComplexLRUCLSEncoder` all
set `_use_packed_kernel = False`; Mamba-2 and the transformer have their own
paths). The actionable version: **deploy minGRU at h = 192, not 194** -- 1 % of
the parameters for ~25 % of the throughput -- and re-measure that arm's
tracks/s at the aligned width before quoting it.

Caveat: measured with the other GPU training. The 32 k / 131 k agreement and the
interleaving make the ratios trustworthy, and the grid arithmetic explains them
independently, but an uncontended re-run (with a cleared autotune cache) is owed
before publication.

## Early signal, ~80 k of 4.73 M steps (do not over-read)

Both runs happen to be step-matched at four validation points (every 20 k
steps), on the pooled **mixed** val set (muons + ttbar, so not comparable to the
muon-only test tables):

| | d0 [um] | z0 [um] | phi [mrad] | theta [mrad] | q/p [1/GeV] |
|---|---|---|---|---|---|
| complex-LRU | 25.9 | 43.7 | 0.574 | 0.229 | **1.71e-3** |
| minGRU fp16 | 26.0 | 44.1 | 0.572 | 0.232 | **3.25e-3** |

Geometry is indistinguishable; **complex-LRU is ~1.9x better on q/p** at equal
steps. That is the parameter the oscillatory prior was aimed at -- curvature
*is* phase advance per unit path length, which a complex eigenvalue can express
and a real decay cannot.

Caveats that matter more than the number: this is 1.7 % of training; the two are
different architectures whose optimal LR may differ; and this campaign has
repeatedly seen early orderings reverse, because the precision arrives in the
OneCycle anneal (4.1 reading 0, 4.11). Nothing here is a result yet.

## Throughput comparisons between these arms are NOT yet apples-to-apples

Only `MinGRUCLSEncoder` has a fused packed Triton kernel. `DiagRNNCLSEncoder`,
`MinLSTMCLSEncoder` and `ComplexLRUCLSEncoder` all set
`_use_packed_kernel = False` and run the eager padded path, which the width
sweep measured at **~17x slower** than the kernel (354 ms vs 20 ms at 131 k
tracks, h = 192). Mamba-2 has its own v5pc kernel; the transformer has
flash-attention.

So a raw tracks/s table across the arms would mostly rank *how much kernel work
each one has had*, not the architectures. Two defensible options, and the second
is what the physics ablation actually needs:

1. compare every arm on the **eager padded path** (apples to apples, but ~17x
   below what any of them can do), and quote the fused number separately for the
   arms that have a kernel;
2. quote physics parity from the ablation and throughput **only for encoders
   with a tuned kernel**, stating plainly that the others were not kernel-tuned.

Writing packed kernels for the non-selective and complex variants is not hard --
they are strictly simpler than minGRU's (no gate to read; the complex one needs
two accumulators instead of one) -- but it is not needed to answer the physics
question, which is what these 25-epoch runs are for.

Practical consequence noticed during a smoke test: a physics eval of an
eager-path arm takes considerably longer than a kernel-path one. That is
slowness, not a defect.

## Deployment throughput of the arms (2026-09-20, H100 NVL, same store/flags)

All parameter-matched at ~0.65 M; `bench_infer_flat.py --gpu-seed
--matmul-precision high`, GPU seed (fp64) inside the timed loop. Measured with
the minGRU fine-tune running on two other GPUs, so absolute values carry a few
per cent of contention; the ordering does not.

| encoder | 32 k | 131 k | VRAM @131 k |
|---|---|---|---|
| minGRU, fp16 encoder (native kernel) | 3.23 M | **4.09 M** | **6.09 GiB** |
| minGRU, fp32 encoder (TF32 matmuls) | 3.17 M | 3.93 M | 11.10 GiB |
| Mamba-2 deployment model (4.32) | 1.85 M | 1.91 M | 14.6 GiB |
| **Transformer** | **0.690 M** | **0.693 M** | 17.40 GiB |

**Four backbones reach the truth-KF's precision and differ by ~6x in
throughput** — so the architecture choice is a pure deployment decision, which
is a cleaner claim than "our architecture is better".

The transformer is the instructive case: physics within +-0.005 GM5 of every
other arm, but 4.7x (32 k) to 5.9x (131 k) slower and 2.9x the memory, and its
curve is FLAT from 32 k to 131 k — already saturated where the recurrent arms
still have headroom.

**CAVEAT — this number handicaps the transformer and must not be published as
is.** The ablation's transformer subclass routes packed batches through a
PADDED attention path (`_PaddedRoutedTransformerCLS`), because the efficient
packed route is flash-varlen, which is fp16/bf16 only and therefore excluded by
this study's strict-fp32 training rule. That is the right call for TRAINING
(all arms must share a precision), but it penalises the INFERENCE number twice:
20 padded tokens instead of the true ~13.3 (1.5x of the projection work wasted)
and no flash-attention at all. Packing is NOT impossible for transformers —
`flash_attn_varlen_func` (installed here, 2.8.3) and FlexAttention block masks
both do exactly this block-diagonal attention over a `cu_seqlens` stream. Only
*naive* packed attention is O((B*L)^2).

The fair comparison is the transformer on its own fast path (packed,
flash-varlen, fp16) against the minGRU on its own. **That number is still
UNMEASURED.** A quick monkeypatch attempt (set `enc.encoder.attn_type =
'flash-varlen'`, bind the parent `forward`) produced 14.9 k tracks/s at 32 k,
which is not a property of flash-attention but a broken patch: a batch scan
gives 0.041 s at 4 k and 2.202 s at 32 k, i.e. **8x the tokens for 53.7x the
time, N^1.9** — the signature of the dense O((B*L)^2) block mask. The patch
never reached flash-varlen; it fell through to the packed `torch` route.
Discard 14.9 k. Doing it properly needs a flag on
`_PaddedRoutedTransformerCLS` to skip the padded routing plus constructing the
encoder with `attn_type: flash-varlen` (fp16 only), with a parity check against
the padded path (~10 lines, ~30 min).

**Bounds in the meantime**: 653-690 k is a LOWER bound; the padding waste is
bounded at 20/13.2 = 1.5x, so a correct packed number lands near ~1 M — still
~4x below the minGRU's 4.09 M, so no conclusion in this study depends on it.
Quote the transformer as ">= 4x slower at equal physics, measured on the padded
path" with the caveat, which understates rather than overstates our margin.
What remains structurally true is the kernel-count argument: per layer the
transformer launches LayerNorm + QKV + SDPA + out-proj + two FFN GEMMs, and at
L = 20 the attention is nearly free, so the cost is launches and small GEMMs —
exactly where a single fused scan wins.

### fp16 only pays with a kernel that takes fp16 natively

| matmul | encoder | 131 k |
|---|---|---|
| strict fp32 | fp32 | 1.69 M |
| strict fp32 | fp16 | 2.67 M (+58 %) |
| TF32 | fp32 | 3.93 M |
| TF32 | fp16, cast at the scan boundary | 2.90 M (**slower**) |
| TF32 | fp16, kernel converts on load | **4.09 M**, 6.09 GiB |

fp16 GEMMs are worth +58 % over strict IEEE fp32, but TF32 already captures
that, so fp16-vs-TF32 is nearly a wash on compute — and materialising an fp32
copy of the (T, 4H) projection per layer turned it into a 26 % LOSS. With the
kernel converting on load (a register convert on a load that happens anyway),
fp16 edges ahead **and halves the memory**. The speed gain is small because
both the H100 and the Ada run this model at only ~8 % of their TF32
tensor-core peak — neither is arithmetic-bound — so the prize is the VRAM,
which is what decides the batch that fits on a 32 GB card.

## Is FlashAttention the right tool at L ~ 20?  NO — it is 40-55 % slower (2026-09-20)

Referee-facing study, `scripts/attn_strategy_study.py`: the **attention step
alone**, at the trained transformer's exact shapes (4 heads, head dim 32,
Lmax = 22 = 20 hits + 2 CLS), microseconds per 1e6 tokens, H100:

| strategy | fp16 B=8k / 32k / 131k | fp32 B=8k / 32k / 131k |
|---|---|---|
| **dense padded, torch SDPA** | **3.7 / 3.6 / 3.8** | 8.9 / 8.9 / 9.2 |
| dense padded, explicit QK^T/softmax/AV | 4.5 / 4.4 / 4.4 | **5.8 / 6.1 / 6.5** |
| torch SDPA forced onto its flash backend | 5.6 / 5.6 / **fails** | **no kernel at all** |
| `flash_attn_varlen_func` (packed) | 5.2 / 5.2 / **fails** | n/a (fp16/bf16 only) |

* FlashAttention is **40-55 % slower** than plain dense attention here. It
  exists to avoid materialising the L x L score matrix for LONG sequences; at
  L = 22 that matrix is 22 x 22 per head = 484 numbers, register-resident, so
  the tiling, online softmax and varlen index plumbing are pure overhead.
* Both flash paths **fail to launch at B = 131 k** (`CUDA error: invalid
  configuration argument` — a grid-dimension limit), i.e. they are not merely
  slower but unusable at our deployment batch sizes.
* In strict fp32 there is **no flash kernel at all**, and the naive explicit
  implementation beats SDPA by 30-40 %.
* Note the dense rows pay padding (22 vs a 15.2 mean) and still win — which
  makes the conclusion stronger, not weaker.

**So our transformer already uses the best attention strategy**, and the
earlier plan to "fix" it with flash-varlen would have made it slower. The
inefficiency is somewhere else entirely — per layer at L = 22, d = 128:

| | kFLOP/token | share |
|---|---|---|
| FFN | 262.1 | 64.8 % |
| QKV projection | 98.3 | 24.3 % |
| out projection | 32.8 | 8.1 % |
| **attention** | **11.3** | **2.8 %** |

Attention is 2.8 % of the work, so padding it is nearly free; padding the
**projections** is the whole waste — 45 % overhead on the 97 % that is dense
GEMM. The right fix for a transformer in this domain is therefore **packed
projections with padded (dense) attention**: run LN/QKV/out/FFN on the packed
(T, D) stream, scatter to (B, 22) only around the attention step, gather back.
That recovers ~1.44x on 97 % of the cost, which would take the measured
653-690 k tracks/s to roughly **0.95-1.0 M** — still ~4x below the minGRU, so
no conclusion here changes, but it is the honest number and it is the one a
referee would ask for.

## minGRU deployment config: width x precision, END-TO-END (2026-09-20)

Earlier width numbers were encoder-only; these are the full deployment path
(GPU seed fp64 in-forward, Fourier front-end, heads, H2D), `bench_infer_flat
--gpu-seed --matmul-precision high`, H100, 131 k tracks/batch. Untrained
weights (throughput depends on shapes, not values); fine-tune running on two
other GPUs, so absolute values carry a few per cent but each pair was measured
back to back.

| config | 32 k | 131 k | VRAM @131 k |
|---|---|---|---|
| h=194, fp32 (**what we deploy today**) | 3.35 M | 3.86 M | 11.1 GiB |
| h=194, fp16 | — | 4.09 M | 6.09 GiB |
| h=192, fp32 | 3.61 M | 4.48 M | 11.1 GiB |
| **h=192, fp16** | — | **5.04 M** | **6.04 GiB** |

1.31x over the current deployed config and 2.6x the paper's Mamba-2 model
(1.91 M). h=192 gains more in fp16 (+23 %) than in fp32 (+16 %): with the GEMMs
cheaper the scan's share rises, and the scan is what the width quantisation
hits.

**The width gain CANNOT be recovered by a smarter kernel** (correcting an
earlier suggestion in this file): Triton's `tl.arange` needs a power-of-two
length, so the channel block BD is in {32, 64, 128, 256}, and 194 = 2 x 97 with
97 prime — no legal BD tiles it cleanly, at any size. 192 = 3 x 64 exactly. The
autotuner is already choosing the best available option for both. So the 16-23 %
requires the width change and therefore a retrain (25 ep stage 1 + 50 ep
fine-tune, ~45 h).

Status of the two levers:
* **fp16 — available now**, no retrain, physics unchanged (<= 0.2 % GM5,
  measured over 25 epochs), already in the `/eos` share bundle. Worth as much
  for the halved VRAM as for the speed: it roughly doubles the batch that fits
  on a 32 GB Ada.
* **h=192 — next round.** Physics is *assumed* identical (1 % fewer parameters,
  well inside the +-0.005 GM5 spread between arms) but has not been trained.

## Precision policy for this branch

**PRECISION RULE (user decision, 2026-09-20): train in strict fp32, switch to
fp16 only at inference.** Standing policy for this R&D branch. What the
measurements say in support:
* fp16 buys **nothing** in training here — 55.3 vs 55.6 it/s. The step is bound
  by the eager padded path, the fp32 scan and the loader, not by the GEMMs, so
  there is no throughput to trade for coarser arithmetic.
* fp16 training **needs loss scaling** and diverges without it: a loss spike at
  ~24 k steps (0.074 -> 0.403) overflowed a gradient and NaN'd the weights in
  one update. That is an extra failure mode for no gain.
* strict fp32 keeps every ablation arm on the same arithmetic, which is what
  makes the cross-architecture comparison valid at all.
* at inference fp16 IS worth it, now that the packed kernel consumes it
  natively: +4 % (h=194) to +23 % (h=192) and **-45 % VRAM**, with physics
  unchanged.
Note on the mechanism, for the record: fp16 does not make the model see less
data — the epoch count and sample count are identical. What it lowers is the
precision of each accumulated update. The conclusion is the same either way;
with zero speed-up on offer there is simply nothing to buy with that precision.
