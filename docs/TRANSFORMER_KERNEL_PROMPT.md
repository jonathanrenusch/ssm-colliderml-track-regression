# Task: is the transformer really slower here, or did we just not try?

You are working in `/shared/tracking/ssm-colliderml-track-regression` on a
charged-particle track-fitting model. Take as long as you need; correctness of
the conclusion matters far more than speed.

## The question

We compare four sequence encoders at matched parameters on the same task. They
tie on physics. They do not tie on throughput, and our paper currently
attributes that to the architecture. **We may be wrong, and the purpose of this
task is to find out.** We wrote a purpose-built fused kernel for one family and
not for the other, so the measured gap confounds "architecture" with
"engineering effort we chose to spend".

Your job: give the transformer the same treatment the minGRU got, and report
what actually happens. **If the transformer ends up faster, that is a fine and
publishable outcome** — we would rather correct the claim before a reviewer
does. Do not try to confirm the existing result.

## The regime (this is unusual and drives everything)

* A "sequence" is one particle trajectory: **6 to 20 hits, mean ~13**, padded
  to 22 tokens including two class tokens. This is two to four orders of
  magnitude shorter than the language-model regime every stock kernel targets.
* A "batch" is **131,072 tracks** — about 1.8 M real tokens spread over 131 k
  independent, very short sequences. Inverted from the usual shape: enormous
  batch, tiny sequence.
* The readout is a pooled pair of class-token states per track, not per-token
  outputs.
* Deployment targets are an H100 NVL and, more importantly for the cost
  argument, an RTX 5000 Ada (99 KiB shared memory per SM).
* Inference runs at fp16 for the recurrent path; physics must be unchanged.

## What is measured today

One idle H100 NVL, 131 k tracks/batch, identical tool and input sample,
$10^6$ tracks/s. "Reference" is each encoder's stock PyTorch implementation;
"fused" adds our kernel work.

| encoder | shape | reference | fused | fused fp16 |
|---|---|---:|---:|---:|
| minGRU | 2 layers, hidden 194 | 1.49 | **3.90** | **4.00** |
| Mamba-2, bidirectional | 2 layers, d_state 64, expand 2, headdim 32 | 0.99 | 1.78 | — (kernel is fp32/fp64-typed) |
| Transformer | 3 layers, dim 128, 4 heads, FFN x3 | 0.69 | 0.69 | 0.74 |
| diagonal SSM, non-selective | 2 layers, hidden 270 | 0.38 | 0.38 | 0.37 |

Trunk parameters are matched to 0.3 % (~0.63 M). Reproduce with
`scripts/abl_v2_backbone_throughput.sh <idle-gpu> 131000 float32`.

Supporting measurements, all on the padded transformer path:

* Kernel count and GPU time per forward, same batch: minGRU **12** launches /
  1,559 us; transformer **153** / 6,793 us. The 4.36x GPU-time ratio matches
  the 4.59x throughput ratio, so the gap is not a measurement artefact.
* FLOP split of a transformer layer at L=22, d=128: **FFN 64.8 %, QKV 24.3 %,
  out-proj 8.1 %, attention itself 2.8 %.** The cost is the projections.
* Padding waste: 22 tokens against a mean of ~13 real ones, so roughly
  **1.6x** of the projection work is spent on pad rows.
* At matched parameters and *packed*, the transformer performs **0.94x** the
  arithmetic of the minGRU (12.30 vs 13.13 MFLOP/track). **It is the cheaper
  model on paper.**
* Long-sequence attention kernels are counter-productive at this length. Our
  exact shapes (4 heads, head dim 32), us per 1e6 tokens, batch 8k/32k/131k:
  dense padded SDPA 3.7/3.6/3.8; explicit QK^T+softmax+AV 4.5/4.4/4.4; SDPA on
  its flash backend 5.6/5.6/FAILS; `flash_attn_varlen` 5.2/5.2/FAILS. Both
  flash paths hit a grid-dimension limit at 131 k sequences. In strict fp32
  there is no flash kernel at all.
* On-chip working set per track per layer if you tried to fuse a whole layer:
  minGRU 194 floats = 0.76 KiB; transformer peak (FFN stage) 22x128 + 22x512 =
  14,080 floats = 55.0 KiB.

## What we did NOT do for the transformer — the honest gap

1. **We never packed it.** The minGRU runs on a packed, unpadded layout
   (`cu_seqlens`, one row per real hit). The transformer arm still pads every
   track to 22. Packing alone removes ~1.6x of projection work, and the
   projections are ~97 % of the FLOPs.
2. **We never wrote it a kernel.** The minGRU has a hand-written Triton scan
   (`src/track_regression/ops/mingru_short_triton.py`). The transformer runs
   stock `scaled_dot_product_attention` plus `nn.Linear`s. Its "fused" column
   is the same path as its "reference" column plus a compiled front end.
3. **We did not try one-program-per-track attention.** A 15x15 attention over
   4 heads of dim 32 is a tiny problem that fits comfortably in registers and
   shared memory. This is the obvious design for this regime and nobody wrote
   it. The flash kernels we tested are the opposite design — built for few long
   sequences, which is why they fail at 131 k of them.

## My hypotheses, and the evidence against each

State these as hypotheses to falsify, not as conclusions.

**H1 — most of the gap is padding and launch overhead, not attention.**
For: attention is 2.8 % of FLOPs; 153 vs 12 launches; 1.6x padding waste.
Against: nothing measured. This is the hypothesis I consider most likely, and
if it holds, a packed transformer with a modest custom kernel should land
somewhere between 1.5 and 3 M tracks/s.

**H2 — a packed transformer could match or beat the minGRU.**
For: it performs 0.94x the arithmetic; its projections are dense GEMMs, which
is what GPUs are best at; and unlike a scan it has **no sequential dependency
at all** — the minGRU's advantage from short sequences is smaller than it looks
because its scan is already only ~15 steps (log-depth under Hillis-Steele).
Against: 3 layers instead of 2, so more GEMMs and more launches per track.
**Take this hypothesis seriously.** If it holds, our paper's framing changes.

**H3 — the residual advantage, if any, is fusability rather than arithmetic.**
The recurrent update is elementwise per channel, so one thread owns a channel
and keeps its state in a register: no shared memory, no barriers. Attention
needs a softmax reduction across positions, hence shared memory and barriers,
and a fully fused layer has a 73x larger working set, which bites hardest on
the smaller card. For: the working-set arithmetic. Against: **you do not have
to fuse the whole layer.** Fusing only the attention block and leaving the
projections to cuBLAS on packed tokens is much easier and may capture most of
the win. Test that before accepting H3.

**H4 — 131 k tiny sequences is the real obstacle.** Both flash paths fail to
launch at that batch. For: measured. Against: a custom kernel with one program
per track has no such limit; this may be a property of those libraries, not of
attention.

## What to build (suggested, not prescriptive)

1. A packed transformer encoder: `cu_seqlens` layout, projections as dense
   GEMMs over real tokens only, attention block-diagonal per track. Measure
   before writing any kernel — packing alone may be most of the win.
2. If that is not enough, a Triton attention kernel with **one program per
   track**: load up to 22x32 per head into registers/shared, compute the small
   QK^T, softmax and AV in-kernel, never materialize a padded L x L matrix.
3. Consider fusing QKV + attention + out-proj per track; report whether the
   working-set argument actually bites at 4 heads x 32.
4. CUDA graphs are worth trying for the small-batch regime; they gave the
   minGRU +93 % to +140 % at 2-4 k tracks.

## Rules

* **Physics must not change.** Re-evaluate with
  `scripts/04b_eval_ckpt_deploy.sh <run_dir> last.ckpt <out> /scratch/colliderml/ICLR_eval_v2 <gpu>`
  (set `TRK_ABS_ETA_MAX=2`) and compare against
  `eval_plots/ablations_2026-09/v2_evals/V2_txf_25ep/plots/rms_summary.json`.
  Anything beyond the third decimal of a ratio is a bug, not a speed-up. Add a
  parity test next to `tests/test_mingru.py` / `tests/test_ssd_variants.py`.
* **Measure on an idle GPU.** Throughput under contention is noise;
  `abl_v2_backbone_throughput.sh` refuses a busy GPU by design. Physics evals
  tolerate contention, throughput does not.
* Report the **same** quantity: `scripts/bench_infer_flat.py`, `--gpu-seed`,
  `--iters 100`, batch 131000, the `ttbar_new_pt1` store, on-GPU fp64 seed
  inside the timed loop.
* Keep the parameter count matched to ~0.63 M trunk. If you change the shape,
  say so and re-match.

## Tooling gotchas that will cost you hours

* `bench_infer_flat.py` prints its full report and then **hangs on exit** (its
  dataloader threads are never joined). Run it detached, wait for the
  "peak VRAM" line, then kill it — see `scripts/abl_v2_batch_sweep.sh`.
  Otherwise every measurement costs an extra ~6 minutes.
* **Select the GPU with `CUDA_VISIBLE_DEVICES`, not `--device cuda:N`.** Triton
  launches on the *current* device, so a custom kernel on a non-zero index dies
  with "Pointer argument cannot be accessed"; and `max_memory_allocated()`
  reads the current device, so peak VRAM comes back as 0.00 GiB.
* `pkill -f <pattern>` matches your own shell whenever the pattern appears in
  its command line. Use a bracket class and keep the literal name out of the
  rest of the command, or kill by PID.
* `pixi run python -c "...'...'..."` strips the inner single quotes. Use the
  system `python3` for inline `-c`, or write a script file.
* The pod's RAM cap is `memory.max` in `/sys/fs/cgroup`, ~335 GB; `free -g`
  shows the whole node and is misleading.

## Where things live

* Encoders: `src/track_regression/ablation_encoders.py`
  (`IndexPosEncTransformerCLS` and its base `_PaddedRoutedTransformerCLS`),
  `src/track_regression/mingru.py`.
* The minGRU kernel, as the worked example of what a good kernel looks like
  here: `src/track_regression/ops/mingru_short_triton.py`. Read its header —
  the conventions (in-kernel reverse via per-segment bounds, direction by
  channel offset, `@torch.library.custom_op`, autotuned `BD`, strict fp32
  accumulation, device pinning) all apply to whatever you write.
  It also documents a **measured negative result**: a fused
  in-projection + scan kernel that was 5-6x *slower*, because one track per
  program starves the GEMM. Do not repeat it.
* Trained transformer checkpoint and config:
  `eval_plots/ablations_2026-09/v2_evals/V2_txf_25ep/`.
* Attention micro-benchmark: `scripts/attn_strategy_study.py`.
* Campaign log with the full history: `CLAUDE.md`, sections 4.39 and 4.40.

## What to report

1. Throughput of the packed transformer, before and after any kernel, at
   131 k batch on an idle H100, with the physics re-check attached.
2. A decomposition: how much came from packing, how much from the kernel, how
   much from fp16, how much from CUDA graphs.
3. Which of H1-H4 survived and which did not, with the measurement that
   decided each.
4. A one-paragraph verdict we can put in a paper, in either direction. If the
   honest answer is "the transformer is competitive once packed and the earlier
   gap was our engineering", write that.
