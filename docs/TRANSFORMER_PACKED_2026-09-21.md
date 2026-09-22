# The transformer, given the same kernel treatment as the minGRU (2026-09-21)

Question from the user: the ICLR_v2 draft says the encoders tie on physics but
"only linear recurrences benefit" from kernel work, "the transformer gains
nothing", and it attributes that to token mixing needing far more kernel
launches.  Task: give the transformer the treatment the minGRU got, gate the
physics, see where it maxes out, audit every kernel sentence in the paper
against the code, judge the minGRU kernel, and recommend paper changes.

Everything below is measured on this node (sess5, 2x H100 NVL), one idle GPU
for throughput, `scripts/bench_infer_flat.py --gpu-seed --iters 100`, the
`ttbar_new_pt1` store, GPU fp64 seed inside the timed loop, i.e. the exact
quantity the paper quotes.  Physics: `04b_eval_ckpt_deploy.sh` on
`ICLR_eval_v2_new`, `TRK_ABS_ETA_MAX=2`, truth-KF reference, compared to the
transformer's own reference bundle `v2_evals/V2_txf_25ep`.  Raw logs:
scratchpad `txf/bench_logs/summary.log`, `txf/kernel_stages_all.out`,
`txf/queue5_stages131k.out`, `txf/queue8_finalprofile.out`.

## 1. Result in one table -- the three networks, "as trained" vs deployed

"As trained" = the exact code path and precision each network trains with:
padded layout, strict IEEE fp32, its training kernel (Mamba-2: `v3c` compiled
pure-torch quadratic dual; minGRU: compiled Hillis-Steele scan,
`TRK_MINGRU_KERNEL=off`; transformer: padded SDPA), no inference switches.
"Deployed" = the fastest physics-gated path of each (fp16 encoder where the
kernel takes it).  131 000 tracks/batch, tracks/s.

| encoder (trunk-matched, ~0.63 M) | as trained | deployed | gain | deployed path |
|---|---:|---:|---:|---|
| minGRU h=194 (earlier draft's physics) | 0.793 M | 4.07 M | **5.1x** | packed Triton scan, fp16 in-kernel, TF32 input net, compiled front end |
| **minGRU h=192 (the paper's model)** | 0.880 M | **5.31 M** | **6.0x** | same |
| Mamba-2, 2L conv-free (previous paper model) | 0.573 M (0.579 M re-measured) | 1.74 M TF32; **2.06 M fp16** (later session, `docs/PRECISION_STUDY_2026-09-21.md`) | **3.0x** TF32, **3.6x** fp16 | v5pc fused SSD kernels + BUCKET16 + compiled front end; fp16 after the kernel learned to read fp16 (physics worst delta 0.0006) |
| Transformer, 3L (padded, what the draft measured) | 0.480 M | 0.726 M | 1.5x | fp16 + compiled front end, still padded |
| **Transformer, 3L, packed (this work)** | 0.480 M | **3.18 M** | **6.6x** | packed stream, per-track attention kernel, fused-epilogue GEMMs, fp16 |

So with comparable effort (one afternoon: a packed layout, one attention
kernel, one fused GEMM/norm kernel) the transformer gains as much as the
minGRU did, and ends up **1.67x slower than the h=192 minGRU and 1.8x
faster than the Mamba-2 model the paper used until last week**.  The draft's
"transformer gains nothing" measured that nobody had packed it, not attention.

Run-to-run spread of the headline numbers (repeat measurements): minGRU h192 fp16 5.31 M / 5.29 M; transformer final 3.18 M / 3.18 M.

## 2. What was built (all additive, opt-in, default off)

* `src/track_regression/ops/attn_short_triton.py`
  * `attn_packed_tracks`: one Triton program per track reads its q/k/v rows
    straight from the packed `(T, 3D)` projection via `cu_seqlens`, applies the
    model's `RMSNorm(D)` on q, k and v in-kernel (needs the whole 128-wide
    row, so it cannot be split per head), runs the four 22x22 heads in
    registers (`tl.dot`, fp16 tensor cores with fp32 softmax, or IEEE fp32),
    writes `(T, D)` in place.  No padding, no `(B, L, L)` mask, no
    scatter/gather around attention.
  * `add_rmsnorm_packed`: single-pass residual-add + RMSNorm (Inductor's
    generated 128-wide reduction ran at ~0.4 TB/s).
  * `gemm_epilogue_fp16`: fp16 Triton GEMM with the layer's epilogues fused --
    bias; bias+SiLU (FFN up-projection); LayerScale-residual + RMSNorm of the
    new residual row (out-projection and FFN down-projection: `BLOCK_N = N =
    128`, so one program owns whole rows and the row statistic is in-tile).
    A layer is then **five kernels**: QKV GEMM, attention, out-proj(+res+norm),
    FFN1(+SiLU), FFN2(+res+norm).
* `src/track_regression/txf_packed.py`: the packed forward of
  `_PaddedRoutedTransformerCLS` -- one augmented stream of `T + 2B` rows (two
  class tokens interleaved per track with two index writes, not an argsort),
  the index positional encoding as a 20-row lookup table (the key only takes
  the values 0..19), every projection on real rows only, class-token readout
  gathered from known positions.  Residual stream fp32 (mirrors the padded
  autocast path), GEMMs in the encoder autocast dtype.
* Hook: `ablation_encoders._PaddedRoutedTransformerCLS.forward`, guarded by
  `TRK_TXF_PACKED=1`, inference only, CUDA only.  Flags:
  `TRK_TXF_PACKED_FUSED_NORM`, `TRK_TXF_PACKED_FUSED_GEMM` (fp16 only),
  `TRK_TXF_PACKED_RESID16` (fp16 residual stream), `TRK_TXF_PACKED_COMPILE[_MODE]`.
* `tests/test_txf_packed.py` (18 tests): kernels vs torch oracles (fp32
  1e-5, fp16 2e-3), packed vs padded encoder in fp32 (1e-4) and under fp16
  autocast, segment independence, fused norm / fused GEMM paths vs padded.

## 3. The optimisation ladder (131k tracks/batch, tracks/s)

| step | strict fp32 | TF32 | fp16 | what it removed |
|---|---:|---:|---:|---|
| padded, as trained (no switches) | 0.480 M | 0.635 M | -- | -- |
| padded + compiled front end (the draft's "fused" row) | -- | 0.675 M | 0.726 M | 5 kernels in the Fourier front end |
| **packed** (Inductor glue, torch attention kernel) | 0.988 M | 1.60 M | 2.47 M | 1.65x padding on 97 % of the FLOPs; the mask; scatter/gather |
| + fused add+RMSNorm kernel | 1.03 M | -- | 2.72 M | Inductor's slow 128-wide reductions |
| + fused-epilogue GEMMs (5 kernels/layer) | -- | -- | 3.05 M | bias/SiLU/residual/norm passes over the activations |
| + positional-encoding lookup table | -- | -- | **3.18 M** | 22 sin/cos kernels + a 22-way cat + a GEMM per forward |
| (+ fp16 residual stream, `RESID16`) | -- | -- | 3.28 M | half the residual traffic -- physics not separately gated, see 5 |
| (+ fast quantile ladder in the shared head, measurement only) | -- | -- | 3.53 M | five `cumsum` scan kernels over 6-wide rows, see 6 |

Without `torch.compile` on the glue the packed fp16 path is 1.73 M; Inductor
`max-autotune` on top of the plain packed path gave +7 % (2.65 M with fp16
residual) but is superseded by the hand-fused epilogues.

Batch dependence (packed fp16, first round): 256: 61 k | 1 k: 242 k | 4 k:
944 k | 16 k: 2.00 M | 32 k: 2.38 M | 65 k: 2.49 M | 131 k: 2.47 M; final path
16 k: 2.37 M | 32 k: 2.87 M | 65 k: 3.17 M | 131 k: 3.18 M.  Saturation from ~32 k, like the recurrent encoders.

CUDA graphs (small-batch / HLT regime; `--cuda-graph`, dummy-track padding,
graph-vs-eager `max |dpred| = 0`): packed fp16 at 2 048: 0.48 M eager -> 1.20 M
(+150 %); 4 096: 0.96 M -> 1.66 M (+73 %); with the fused GEMMs 1.42 M / 1.95 M.
The padded path at 2 048 is 0.36 M.  So at small batch the packed transformer
under a graph is 4x the padded eager path.

## 4. Where the time goes -- stage-wise kernel profile (uncontended, torch.profiler, real batch)

Kernel counts and GPU kernel time per forward, split into the on-GPU fp64 seed,
the front end (min-max -> Fourier -> input net), the encoder, and the heads +
`predict_physical`.  32 768 tracks/batch:

| path | seed | front end | **encoder** | heads+predict | total -> kernel-time tracks/s |
|---|---|---|---|---|---|
| txf padded, as trained fp32 | 292 k, 0.89 ms | 73 k, 6.07 ms | **150 k, 56.2 ms** | 67 k, 1.37 ms | 582 k, 64.6 ms -> 0.51 M |
| txf padded, deploy fp16 | 292, 0.88 | 5, 1.56 | **246, 39.1** | 67, 1.26 | 610, 42.8 -> 0.77 M |
| txf packed fp16 (round 1) | 292, 0.89 | 5, 1.57 | **101, 9.28** | 67, 1.31 | 465, 13.1 -> 2.51 M |
| txf packed, fused GEMM (final) | 292, 0.90 | 5, 1.57 | **45, 6.34** | 67, 1.33 | 409, 10.14 -> 3.23 M |
| minGRU h194, as trained fp32 | 292, 0.88 | 73, 6.09 | **40, 30.5** | 67, 1.48 | 472, 39.0 -> 0.84 M |
| minGRU h194, deploy fp16 | 292, 0.89 | 5, 1.57 | **27, 4.06** | 67, 1.29 | 391, 7.81 -> 4.20 M |
| minGRU h192, deploy fp16 | 292, 0.90 | 5, 1.57 | **27, 2.32** | 67, 1.41 | 391, 6.19 -> 5.29 M |
| Mamba-2, as trained fp32 (v3c) | 292, 0.90 | 73, 6.09 | **94, 45.2** | 67, 1.42 | 526, 53.6 -> 0.61 M |
| Mamba-2, deploy (v5pc, TF32) | 292, 0.89 | 5, 1.56 | **83, 14.1** | 67, 1.30 | 447, 17.8 -> 1.84 M |

131 000 tracks/batch (the deployment batch):

| path | seed | front end | encoder | heads+predict | total |
|---|---|---|---|---|---|
| minGRU h192 deploy fp16 | 2.15 ms (10 %) | 6.41 ms (**29 %**) | 8.94 ms (40 %) | 4.77 ms (**21 %**) | 22.3 ms |
| txf packed + fused norm fp16 | 2.16 | 6.46 | 31.6 | 4.72 | 44.9 |
| txf packed, final | 292, 2.15 | 6, 6.37 | 45, 24.75 | 67, 4.74 | 410, 38.01 -> 3.45 M |
| Mamba-2 v5pc TF32 | 2.15 | 6.44 | 59.5 | 4.73 | 72.8 |

Readings:

1. **The "12 kernel launches per forward" in the paper is not what the code
   does.**  The minGRU *encoder* issues 27 kernels (2 GEMMs + 2 scan launches
   carry >95 % of its time; the rest are casts, gathers, the pool norm).  The
   full deployed forward issues ~390: the fp64 seed alone is 292 tiny kernels,
   `predict_physical` + heads 67.  The transformer encoder was 150 (fp32) /
   246 (fp16 autocast: extra casts) padded and is 45 packed.
2. **For the fastest model the encoder is only 40 % of the deployed forward.**
   At 131 k the h=192 minGRU spends 6.4 ms in the Fourier front end (a 480-wide
   fp32 tensor written and read back), 4.8 ms in the heads (of which ~1 ms per
   32 k, i.e. most of it, is five `torch.cumsum` scan kernels over 6-element
   rows in the quantile ladder) and 2.2 ms in the seed.  A measurement-only
   replacement of the ladder's cumsum by a 6x6 triangular matmul lifts the
   h=192 minGRU from 5.31 M to **6.34 M tracks/s (+19 %)** and the transformer
   by +15 %.  The paper's "seed costs 2.9 % of the forward" was measured against
   the slower Mamba-2 forward; for the shipped minGRU it is **~9-10 %**.
3. **Launch overhead is not the transformer's problem at deployment batch.**
   At 131 k each of its ~100 encoder kernels does 0.1-2 ms of work; the gap to
   the minGRU is memory traffic and GEMM count: per layer the transformer moves
   the activations through four GEMMs plus attention, the minGRU through one
   GEMM plus one scan, and the transformer needs three layers for the same
   parameter count.  At 2-4 k tracks launch gaps do dominate -- for both
   families -- and CUDA graphs fix both.
4. **"Attention needs shared memory and barriers, 73x working set" is not the
   mechanism.**  The per-track attention kernel keeps a track's 22x128 q, k, v
   in registers (Triton stages `tl.dot` operands through shared memory
   internally, as every tensor-core kernel does) and costs 13-16 % of the
   packed encoder; the padded path spent more time in Inductor's norm
   reductions than in attention.

## 5. Physics gate

Reference: `V2_txf_25ep/plots/rms_summary.json` (padded path, TF32, fp32
encoder, |eta| <= 2).  Ratios SSM / truth-KF, post- and pre-clip, all six test
sets, identical N on every set:

* packed fp16, round 1 (Inductor glue): **worst |delta ratio| = 0.0005** over
  all 60 cells (`V2_txf_25ep_packed_fp16`).
* padded fp16 control (the draft's own fp16 path, `V2_txf_25ep_padded_fp16`): worst |delta ratio| = **0.0010**, 3 of 60 cells above 0.0005 -- i.e. the packed path is *closer* to the fp32 reference than the paper's existing fp16 path.
* packed fp16 + fused-epilogue GEMMs + posenc table (the final path, `V2_txf_25ep_packed_fusedgemm_fp16`): worst |delta ratio| = **0.0008**, 3 of 60 cells above 0.0005, N identical on all six sets -- inside the third decimal the paper quotes and inside the deviation of the paper's own padded fp16 path (0.0010). **Gate passed.**
* real-batch check of every variant against strict fp32 (8 192 tracks): the
  packed strict-fp32 path reproduces the padded one to 2e-6 mm in d0 (1e-7
  relative, bit-level up to summation order); packed TF32 / fp16 / fused-norm /
  fused-GEMM / fp16-residual all deviate from strict fp32 by the same amount
  the padded TF32 and fp16 paths do (d0 RMS 1.1-1.6e-4 mm, q/p 1.1-1.5e-4).
  The fp16 residual stream has the largest theta max deviation (3.2e-4 vs
  1.8e-4 mrad) and was not put through the full eval; it is a +6 % option, not
  the recommended path.

## 6. Verdict on the four hypotheses of `docs/TRANSFORMER_KERNEL_PROMPT.md`

* **H1 (padding + glue, not attention) -- confirmed.**  Packing alone took the
  fp16 path 0.73 -> 2.47 M (3.4x); attention was 2.8 % of FLOPs and is 13-16 %
  of the packed encoder's time.
* **H2 (a packed transformer could match the minGRU) -- refuted, but by less
  than a factor two.**  3.18 M vs 5.31 M (h=192) / 4.07 M (h=194): the
  residual gap is the extra GEMMs and layers at matched parameters, and it is
  arithmetic/traffic, not fusability.
* **H3 (the residual advantage is fusability) -- partly.**  What the linear
  recurrence has is one elementwise pass for the whole bidirectional token
  mixing, so a layer is two kernels; attention needs a per-track reduction and
  three GEMM-shaped ops around it, so a layer is five even with everything
  fused.  The working-set / shared-memory / barrier argument does not bite.
* **H4 (131 k tiny sequences is the obstacle) -- refuted.**  It is a limit of
  FlashAttention's grid, not of attention; a one-program-per-track kernel has
  no such limit and saturates from 32 k like the recurrences.

## 7. Was the minGRU kernel done properly?

The kernel itself: yes.  Packed layout, both directions in one launch (grid
axis 2), in-kernel reverse via segment bounds, loop bounded by the track's own
length, fp16/bf16 consumed on load with fp32 accumulation, autotuned channel
block, zero explicit shared memory, parity tests, AOT sm_89 compile check.
At 131 k the two scan launches take 3.4 ms and the two `in_proj` GEMMs 3.8 ms
of a 8.9 ms encoder; the documented fused GEMM+scan attempt was correctly
abandoned.  Three things around it are not done:

1. the h=194 -> h=192 width alignment (measured again today: 4.07 -> 5.31 M);
2. the shared pipeline is 60 % of the forward (front end 29 %, heads 21 %,
   seed 10 %): the quantile-ladder cumsums alone are worth +19 %, the Fourier
   front end could be generated inside the input-net GEMM instead of being
   materialised as a 480-wide fp32 tensor;
3. the doc's "0.98 M padded strict-fp32 minGRU" could not be reproduced under
   the as-trained definition (compiled Hillis-Steele: 0.79 M h194 / 0.88 M
   h192); with the padded Triton kernel instead 0.98 M (h194) -- that is the doc's 0.98 M: it already is a custom kernel, not the training path.  The paper
   should state which path a "reference" number is.

## 8. Paper audit -- every kernel/architecture sentence vs the code (ICLR_v2 @ d9af0b4)

Status legend: WRONG = contradicts the code or a measurement; STALE = was true
for the Mamba-2 model, not for the shipped minGRU; UNSUPPORTED = no measurement
in the campaign backs it; OK.

| where | claim | status | what the code/measurement says |
|---|---|---|---|
| `method.tex:76-82`, `:125-126`; `introduction.tex:47-48`; `fig:architecture` ("CLS pool") | two learned class tokens bracket the sequence; each minGRU layer applies RMSNorm, a forward and reverse scan, merges directions through a learned sigmoid gate before the residual connection; the class-token states are concatenated to 256-d | **WRONG** (describes the Mamba-2 block) | `mingru.py`: a layer is `in_proj (D -> 4H)` + the bidirectional scan, output `[h_fwd \| h_bwd]` (2H) feeds the next layer; **no class tokens, no per-layer norm, no gate, no residual**.  Readout = terminal states `[h_fwd(last hit) \| h_bwd(first hit)]` (384-d) -> RMSNorm -> Linear(384 -> 256) -> heads.  Parameters: h=192 total 0.642 M (trunk 0.621 M); h=194 0.650 M.  The `eq:mingru-update` itself is right. |
| `method.tex:185-190` | minGRU kernel: "inference runs on unpadded, length-sorted tracks, so roughly three-quarters of tracks with 16 hits or fewer are routed through a smaller compute tile" | **WRONG kernel** | That is the Mamba-2 `TRK_SSD_BUCKET16` mechanism (no sorting: an index list; the split is at 16 *augmented tokens* = 14 hits, ~60 % of tracks).  The packed minGRU kernel has nothing to bucket. |
| `method.tex:190-192` | Fourier front end "fused into a single compiled kernel"; "entire bidirectional token mixing in a single kernel launch" | half OK | The compiled front end is 5-6 kernels (1.6 ms at 32 k, 6.4 ms at 131 k -- the largest single cost of the minGRU forward).  One launch per *layer* for both directions: correct. |
| `method.tex:194-197`; `results.tex:222-224` | "an identical kernel effort brings 5.0x for the minGRU while the transformer gains nothing" / "5.1x ... gains nothing" | **WRONG** | The effort was not identical (the transformer had no packed path).  With it: minGRU 5.1x (h194) / 6.0x (h192), transformer 6.6x, Mamba-2 3.0x, all against the as-trained path.  5.0/5.1x is reproducible only as padded-Triton-fp32 -> h192-fp16 (0.98 -> 5.04 M); define the baseline in the table caption. |
| `method.tex:197-204`; `appendix.tex:220-227` | attention needs shared memory and barriers; 73x larger on-chip working set; "12 kernel launches per forward pass" | **UNSUPPORTED / WRONG** | Per-track attention runs in registers and is 13-16 % of the packed encoder.  minGRU encoder = 27 kernels, full forward ~390; transformer packed encoder 45. |
| `method.tex:148-149` | training in strict fp32 "since TF32 does not achieve the required precision during optimization" | **UNSUPPORTED** | No TF32-training run exists in the campaign log.  What was measured: fp16 training buys no step rate (55.3 vs 55.6 it/s) and diverges without loss scaling.  Say that, or drop the causal claim. |
| `method.tex:150-156` | "the selective-scan kernel's own matrix products stay IEEE fp32"; "0.01 % median deviation over five parameters and four samples" | STALE / provenance unknown | The minGRU scan has no matrix products; it accumulates the recurrence in fp32 and reads fp16.  The measured statement in `ICLR_v2_CHANGELOG.md` is: fp16 vs fp32 inference identical to three decimals in 27 of 30 cells, 0.001 apart in 3. |
| `method.tex:182`; `appendix.tex:213` | "measures 15 hits on average" | WRONG number | 13.3 hits (mix3 13.28, test stores 13.2-13.3); 22 vs 13.3 real tokens is 1.65x padding, not 1.47x. |
| `results.tex:233-235`; `method.tex:51` | seed "0.53 us per track ... 0.015 us ... 2.9 % of the forward" | STALE | Measured on the Mamba-2 forward.  For the h=192 minGRU at 131 k the whole forward is 0.19 us/track and the seed 0.016 us: ~9 %. |
| `appendix.tex:99-125` (`tab:kernel-bench`) | Mamba-2 stock 589 k -> 1 784 k at 32 k | OK as history, but | stale 589 k (remeasured 608 k on 2026-09-06); replace by the three-network table of section 1 as the user asked. |
| `appendix.tex:209-232` | "Transformer optimization issues": padded path, 0.94x arithmetic, 12 launches, 73x, flash 40-55 % slower, attention 2.8 % | rewrite | Keep 0.94x, 2.8 % and the FlashAttention measurement; replace the rest with sections 3-4 of this document. |
| `data.tex:26`, `introduction.tex:47`, `method.tex:11` | hits "ordered by the time at which they were measured/recorded" | reviewer trap | Strip hits carry no time; the stores are ordered by the *simulated* production time, and at inference the truth-free detector-geometry order reproduces it on 100 % (muons) / 99.7 % (ttbar) of tracks (CLAUDE.md 4.17, 5.2). |
| `results.tex:217`, `main.tex \deployThroughputAligned` | h=192 fp16 5.21 M | OK | 5.31 M today, uncontended (+2 %). |
| `method.tex:108-112`, `tab:arch-ablation` | only the non-selective diagonal SSM fails, on q/p | OK | consistent with the tables. |

## 9. Recommendation for the storyline (my opinion, for the user to decide)

Do not sell "the transformer is harder to optimise because token mixing needs
more kernel launches".  It is false at deployment batch and the measured
number that would have supported it (0.69 M) was an un-packed path.  What the
measurements support, and what survives a reviewer who reads our own tables:

1. Precision is a property of the formulation (seed, residual features,
   anchored quantile heads, small-batch recipe): four encoders tie.
2. Throughput is a property of the kernel path, and *every* encoder responds to
   the same treatment: 3.0x (Mamba-2), 5.1-6.0x (minGRU), 6.6x
   (transformer) over its training path.
3. After that treatment the ranking is set by how many passes over the
   activations a layer needs at this sequence length: the minGRU's token
   mixing is one elementwise scan (a layer = 1 GEMM + 1 scan, two layers),
   attention needs a per-track reduction with three GEMM-shaped ops around it
   (a layer = 5 kernels even fully fused, three layers at equal parameters).
   That is why the minGRU stays 1.67x ahead -- an arithmetic and
   memory-traffic argument, not a launch-count or shared-memory argument.
4. The minGRU is the deployment choice; the transformer is a legitimate
   fallback where a recurrent kernel is unavailable.
5. Say where the rest of the time is: at the deployment batch the shared
   seed + Fourier front end + quantile heads are 60 % of the minGRU forward.
   This is honest and it is the natural "future work" sentence.

Concrete edits are listed per line in section 8; proposed replacement text for
`sec:kernels`, the results sentence and the appendix subsection is in the
chat report.  No tex was edited.

## 10. Proposed replacement text (for the user to adapt; nothing applied)

### `sec:kernels` (method.tex:176-212), replacing the three paragraphs

> Standard GPU kernels for sequence models are built for long sequences,
> typically $10^{3}$--$10^{5}$ tokens.  Our domain demands the opposite: a
> trajectory is at most $22$ tokens including the two class tokens and carries
> $13.3$ hits on average, and inference processes $10^{5}$ such trajectories
> at once.  Run as-is, every encoder pads each trajectory to the longest one,
> spends about $1.65\times$ its arithmetic on padding, and issues hundreds of
> small kernels per batch.
>
> We therefore give every encoder the same treatment: a packed layout with one
> row per recorded hit and the trajectory boundaries in a prefix-sum, a fused
> Triton kernel for its token mixing, the projections in fp16 or TF32 where the
> physics allows it, and the per-hit Fourier front end compiled into a handful
> of kernels.  For the minGRU the token-mixing kernel scans both directions of
> every trajectory in one launch, keeps each channel's state in a register and
> reads fp16 directly; a layer is one GEMM and one scan.  For the transformer
> one program per trajectory reads its queries, keys and values from the packed
> projection, applies the q/k/v normalization and the $22\times22$ attention of
> all heads in registers, and the remaining GEMMs carry their bias, activation,
> residual and normalization as epilogues; a layer is five kernels.  For the
> Mamba-2 block the scan is a single small matrix product per trajectory
> (\cref{app:kernels}).
>
> \cref{tab:backbones} gives the result.  Measured against each encoder's own
> training path (padded, strict fp32), the minGRU gains $6.0\times$, the
> transformer $6.6\times$ and the Mamba-2 block $3.0\times$; all three respond
> to the same engineering.  After it, the ranking follows how many passes over
> the activations a layer needs at this sequence length: the minGRU's token
> mixing is one elementwise scan, attention needs a reduction across the
> trajectory with three further projections around it, and three layers for
> the same parameter count, so the minGRU remains $1.7\times$ faster at equal
> precision and is the encoder we deploy.  At the deployment batch size the
> remaining time of the minGRU forward is dominated by the parts all encoders
> share: the analytic seed, the Fourier front end and the quantile heads
> (\cref{sec:results-throughput}).

### The table (replacing `tab:kernel-bench`, appendix, or in the main text)

| encoder | as trained | deployed | gain |
|---|---:|---:|---:|
| minGRU (h=192, the paper's model) | 0.88 M | 5.31 M | 6.0x |
| Transformer | 0.48 M | 3.18 M | 6.6x |
| Mamba-2, bidirectional | 0.57 M | 1.74 M | 3.0x |

Caption: "Inference throughput on one H100 NVL, $10^{6}$ tracks/s at
$131$k tracks per batch, on-GPU seed included.  *As trained*: the padded
layout, strict IEEE fp32 and the kernel each encoder is trained with, no
inference switches.  *Deployed*: the packed layout, the encoder's fused
token-mixing kernel, compiled front end, fp16 projections (TF32 for Mamba-2,
whose kernel is fp32-typed).  Physics identical to three decimals on every
ratio of \cref{tab:ratios}."

### `results.tex:222-224`, replacing "no longer tie ... gains nothing"

> The same four encoders no longer tie once each is run on a packed layout with
> a fused kernel for its token mixing (\cref{tab:backbones}): every one of them
> gains between $3\times$ and $6.6\times$ over its training path, and the
> minGRU ends $1.7\times$ ahead of the transformer and $3\times$ ahead of
> Mamba-2 at identical precision.

### `results.tex:233-235`, the seed sentence

> At the saturating batch size the deployed minGRU forward costs
> \SI{0.19}{\micro\second} per track on the H100, of which the on-GPU seed
> (float64, \cref{app:seed}) contributes $\sim$\SI{0.016}{\micro\second}, about
> $9\%$; the Fourier front end and the quantile heads take a further $50\%$,
> so the encoder itself is $40\%$ of the forward.

### `method.tex:76-82`, the encoder paragraph (must change)

> The $15$ per-hit features are min--max normalized, expanded by a $16$-scale
> Fourier featurization to $480$ dimensions and projected to $d=128$.  The
> backbone stacks two bidirectional minGRU layers~\citep{Feng2024minGRU} of
> hidden width $192$: each layer is a single linear projection of its input to
> the gates and candidates of both directions, followed by the forward and the
> reverse scan, whose states are concatenated ($384$-d) and fed to the next
> layer.  The readout is the terminal state of each direction, the forward
> state at the outermost hit and the reverse state at the innermost, normalized
> and projected to $256$ dimensions for the heads; the encoder has $0.64$\,M
> parameters in total.

(and `introduction.tex:47-48`: drop "with two learned class tokens that
bracket the hit sequence"; `fig:architecture`: the block label "CLS pool"
becomes "terminal-state pool", or the caption notes that the class-token
readout is the Mamba-2 variant.)

### `method.tex:148-156`, the precision paragraph

> Training runs end-to-end in strict fp32.  Reduced-precision training buys
> nothing here -- the step is bound by the eager padded path and the loader,
> not by the GEMMs ($55.3$ vs $55.6$ steps/s) -- and fp16 adds a divergence
> mode without loss scaling.  At inference the projections run in fp16 and the
> recurrence accumulates in fp32 inside the kernel; on the shipped checkpoint
> fp16 and fp32 inference agree to three decimals on every ratio of
> \cref{tab:ratios} (27 of 30 cells identical), so the fp16 path is used for
> every number in this paper.

## 11. Have all Mamba-2 kernel strategies been tried on the minGRU?  (checked in code, 2026-09-21)

| Mamba-2 deployment strategy (where) | minGRU path | status |
|---|---|---|
| fused Triton scan kernel, one program per track (`_ssd_short_fwd_kernel2p`, v5pc) | `_mingru_bidi_packed_kernel` | done |
| packed stream, no pad rows, `cu_seqlens` (v5p) | `mingru_bidi_packed` | done |
| both directions from ONE in-projection GEMM, stacked weights (`_register_fused_weights`) | one `in_proj` emits `[z_fwd\|n_fwd\|z_bwd\|n_bwd]` | done |
| direction by channel offset; in-kernel reverse via segment bounds (`REVERSE`) | `z_off = pid_dir*2H`, `phys = end-1-t` | done |
| merged bidirectional launch (`TRK_SSD_MERGED_BIDI`, opt-in, a wash for Mamba-2) | grid axis 2 = direction, always | done (built in) |
| `torch.library.custom_op` opaque to Inductor; autotuned `num_warps`/tile | yes; `BD` 64-256 x warps | done |
| strict IEEE fp32 accumulation, no TF32 inside the scan (`TRK_SSD_DOT_PRECISION` measured slower) | fp32 accumulation; no dots at all | done / n.a. |
| compiled front end (`TRK_COMPILE_FRONTEND`, shared in `model.forward`) | same code | done |
| TF32 projections (`TRK_MATMUL_PRECISION=high`) | yes, and fp16 GEMMs with fp16 read in-kernel (Mamba-2 cannot: kernel typed fp32/fp64) | done (minGRU ahead) |
| CUDA graphs with dummy-track padding, host syncs removed | measured +140 % at 2 048, `max \|dpred\| = 0`; kernel has no device sync | done |
| length bucketing `TRK_SSD_BUCKET16` (two BL launches via an index list, one device sync) | **not applicable**: the packed scan loops `while t < Lr` per track, no static tile, nothing to bucket | n.a. |
| fused gated RMSNorm kernel (`_gated_rmsnorm_kernel`) | **not applicable**: the minGRU block has no gate/norm; one `pool_norm` per track | n.a. |
| width alignment to the channel block | h=194 -> h=192 (+30 %, 4.07 -> 5.31 M) | done (minGRU-specific) |
| AOT sm_89 compile test against Hopper-only PTX | `test_kernels_compile_for_ada_sm89` | done (minGRU only; Mamba-2 has no such test) |
| fusing the in-projection into the scan kernel | tried, 5-6x slower, documented negative result (`mingru_short_triton.py`) | tried, rejected |

Not tried on **either** encoder, in order of expected value at 131 k: (i) the
quantile-ladder `cumsum` -> triangular matmul in `losses._ordered_from_raw`
(measured +19 % for the minGRU with a measurement-only patch); (ii) generating
the 480-wide Fourier features inside the input-net GEMM instead of
materialising them (front end = 29 % of the minGRU forward); (iii) the seed's
292 tiny kernels (10 %); (iv) wrapping the minGRU packed glue (fp16 weight
casts, terminal-state gathers, pool norm: ~1.5 ms of the 8.9 ms encoder) in
`torch.compile`; (v) two tracks per 16-row tile for Mamba-2 (its own open
item).  None of these is a kernel-family question; they are shared pipeline
costs.

## 12. Missing passage: why these four encoders (draft by a Sonnet agent, two facts corrected by me; for the user to place)

Placement: Results, immediately before `\cref{tab:arch-ablation}`, as the
rationale for the comparison; the appendix "Protocol" subsection then stays
about *how* the arms were matched, not *why* they were chosen.

> \cref{tab:arch-ablation} compares four encoders chosen along two axes:
> recurrence versus attention for mixing tokens, and an input-dependent
> (selective) versus a fixed state update. Mamba-2, the original encoder, is
> selective: an update rule motivated by analogy to the Kalman gain of the
> classical Kalman-filter track fit. minGRU \citep{Feng2024minGRU} keeps that
> input-dependent gate but drops Mamba-2's state expansion, gated output
> normalization and per-layer residual structure, so its gates depend only on
> the current hit and the recurrence is linear and elementwise. A classical GRU
> or LSTM gates on the previous hidden state, so each step is a dependent
> matrix product with no parallel or fused kernel; we studied one in an
> earlier round and do not carry it forward. The Transformer, the architecture
> of the closest prior work on this detector \citep{Couthures2025CTD}, mixes
> tokens by attention and is the control. The non-selective diagonal
> state-space model keeps the linear recurrence but drops the input-dependent
> gate, isolating whether selectivity matters. All four run bidirectionally,
> matching the filter-plus-smoother structure of the classical fit. Three reach
> the same precision; only the non-selective model falls behind, in $\qop$.
