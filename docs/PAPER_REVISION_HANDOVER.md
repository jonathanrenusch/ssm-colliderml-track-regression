# Handover — ICLR paper revision, branch `ICLR_v2`

Rewritten from scratch 2026-09-21 (was `PAPER_REVISION_HANDOVER_2026-09-20.md`).
The previous version is superseded completely: it described Mamba-2 as the
shipped model and an ablation that still had to be argued into the paper.
Neither is true any more.

You are helping revise an ICLR submission at
`/shared/tracking/NeurIPS_2026_SSM_Tracking` (LaTeX: `main.tex` +
`sections/*.tex` + `references.bib`). The deadline is days away.

---

## WHERE THINGS STAND — read this before anything else

**The figures and tables are done.** Branch `ICLR_v2` (12 commits on top of
`ICLR` @ `1b10bad`, head `f047dc4`) already carries every regenerated figure,
every recomputed table and every measured number. The build is clean: 0 errors,
0 overfull boxes, 18 pages, main body ending on page 9.

**What remains is the prose, and the user writes it.** Your job is to propose
text, one item at a time, and to apply only what the user signs off. The
complete account of what changed on the branch, and why, is
`ICLR_v2_CHANGELOG.md` in the paper repo — read it once before you start.

**The user also edits prose directly**, on `ICLR` and by cherry-pick onto
`ICLR_v2` (see commit `77e490b`). Check `git log` before assuming a passage is
still as you last saw it.

## HOW TO WORK

**Do not apply any change until the user has signed off on that specific
change.** Work through the numbered items below ONE AT A TIME:

1. Quote the current text verbatim, with `file:line`.
2. Say what is wrong with it and why it must change.
3. Show the exact proposed replacement.
4. **Stop and wait for approval.** Do not batch items.
5. On approval, apply that one edit, rebuild, confirm the build is still clean
   and the page count has not regressed, then move on.

If the user rewrites your prose, use their version verbatim.

Build (the system texlive lacks `fancyhdr`):

    PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2025/bin/x86_64-linux:$PATH bash build.sh

After adding a citation, `bibtex main` runs inside `build.sh`, but the citation
only resolves on the following pass — check the final pass, not the first.

## GUARDRAILS

* **The shipped model is the fine-tuned bidirectional minGRU**, not Mamba-2.
  Every physics number in the draft comes from it.
* **Do not hand-edit tables or figures.** Every table body sits between
  generator markers (`% ARCH_ROWS_BEGIN`, `% BACKBONE_ROWS_BEGIN`, …) and every
  figure is produced by a script; `ICLR_v2_CHANGELOG.md` lists the commands. If
  a number looks wrong, regenerate it, do not retype it.
* Main body must stay at 9 pages (References on page 10). The appendix may grow.
* Unit macros `\gibi` / `\kibi` are ONLY valid inside `\SI{}{}`
  (`\SI{14.6}{\gibi\byte}`). Bare use is a fatal LaTeX error.
* `sections/{background,related_work,non_gaussian,metrics,interpretability,discussion}.tex`
  are commented out of `main.tex` and are NOT compiled. Ignore them.
* `\tbd` (red, two characters) marks material still to be replaced. It appears
  twice, both on the RTX 5000 Ada, whose numbers are a collaborator's and are
  still the Mamba-2 model. **A `\tbd` always replaces the stale number rather
  than sitting next to it** — keep it that way.

---

## THE STORYLINE

A seed-guided bidirectional sequence model reaches the precision of the
truth-seeded Kalman filter on every perigee parameter, and fits tracks about
**30x** faster than the classical fit saturating a 64-thread CPU
(5.21 M vs 172.7 k tracks/s). Keep that factor honest: per device dollar it is
1.5-3x, not an order of magnitude (see P12).

Two results were added since the last draft, and they change the framing:

1. **The encoder barely matters for precision.** Four backbones — minGRU,
   bidirectional Mamba-2, a transformer, and a non-selective diagonal SSM —
   trained with an identical recipe, dataset, schedule, parameter budget and
   seed, land within 0.004 of each other on the geometric mean, except the
   non-selective one, which fails on $q/p$ alone.
2. **The encoder matters a great deal for speed, and the reason is the
   kernel.** The two linear-recurrence families we wrote fused kernels for
   gained 2.6x and 1.8x; the transformer gained nothing.

So the paper's recommendation is not "this architecture is more accurate". It
is: *the precision comes from the formulation — the analytic seed, the
residual features, the anchored quantile heads, the small-batch recipe — and
once that is fixed, pick the encoder whose kernel you can actually make fast.*
That is the state-space / linear-recurrence family.

This is a stronger and more honest paper than "Mamba-2 is the right
architecture", and it costs the abstract nothing (see item P6).

## WHY THE minGRU — the argument to make, step by step

The paper has to justify a model choice it cannot justify on accuracy. Build it
in this order; every step is backed by a measurement in this document.

1. **The precision comes from the formulation, not the encoder.** The analytic
   three-hit seed, the per-hit residual features in the seed frame, the
   seed-anchored quantile heads (scale-free for $q/p$) and the small-batch
   recipe are what put the fit at the Kalman filter. Four different encoders
   dropped into that scaffold land within 0.004 of each other. Say this
   plainly: it is the paper's most transferable result.

2. **Among encoders, one property does matter: input-dependent gating.**
   The non-selective diagonal SSM is the only arm that separates, and it
   separates on $q/p$ alone — 1.237(1) against 1.01-1.02 — while its geometric
   parameters stay within 2 %. That is the Kalman-gain analogy earning its
   keep, and it is why `eq:mamba-update` should stay in the paper.

3. **Given that, choose on engineering.** Precision is a tie, so the tiebreak
   is how fast each family can be made to run. That is measured, not asserted:
   2.6x for the minGRU, 1.8x for Mamba-2, 1.0x for the transformer, from the
   same kernel effort applied to each.

4. **Why the linear recurrence fuses and attention does not.** The recurrent
   update is elementwise per channel, so a thread keeps its state in a register
   — no shared memory, no barriers, 12 kernels per forward. Attention needs a
   softmax reduction across positions, so it needs shared memory, and its
   on-chip working set is 73x larger; at L~22 FlashAttention is 40-55 % *slower*
   than dense, and attention is only 2.8 % of the layer's FLOPs anyway — the
   cost is in padded projections.

5. **Why the minGRU among the linear recurrences.** It is the fastest before
   any kernel work (1.49 M vs 0.99 M for Mamba-2 in stock PyTorch) and after
   (3.90 M vs 1.78 M); its kernel takes fp16 natively where Mamba-2's fused
   gating kernel is typed fp32/fp64; and it uses no `tl.dot`, no TMA and zero
   shared memory, so it ports to smaller cards. Its physics is a tie with
   Mamba-2 to within one standard deviation after fine-tuning.

**One thing this argument must NOT claim.** We have no evidence in the paper
that bidirectionality is necessary — the arm that would show it is held back
(see WHAT NOT TO SAY). Justify the bidirectional block as the more general
operator, free at this sequence length, and leave it there.

---

---

## THE EVIDENCE

All measured. Cite these numbers; do not round them differently.

### Precision of the shipped model (`tab:ratios`, in the paper)

Ratio to the truth-seeded KF, $|\eta|\le2$ (uniform row also $p_{\mathrm T}\le70$\,GeV),
fine-tuned minGRU at fp16 inference. Brackets are 1-sigma paired-bootstrap
errors on the last digit, 400 replicas per sample.

| sample | d0 | z0 | phi | theta | q/p |
|---|---|---|---|---|---|
| mu 2 GeV | 0.992(2) | 0.991(3) | 0.989(2) | 0.993(2) | 1.006(4) |
| mu 10 GeV | 0.991(1) | 0.994(3) | 0.990(2) | 0.991(2) | 1.000(3) |
| mu 50 GeV | 0.997(1) | 1.000(1) | 0.976(2) | 0.994(2) | 1.011(2) |
| mu 1-70 GeV | 0.9959(3) | 0.9985(4) | 0.9816(6) | 0.9965(5) | 1.0115(8) |

Note that $q/p$ at 50 GeV and on the uniform sample is above unity by about ten
standard deviations. `sections/results.tex` currently says the SSM is "at or
below the reference in every entry of both halves of the table within 1 %",
which no longer reads quite right for those two cells (item P7).

### Encoder ablation (`tab:arch-ablation`, uniform sample)

**All four are first-stage models** — 25 epochs, identical training, no
fine-tuning — so every row is slightly worse than the shipped model above. That
identity is exactly what makes the comparison controlled, and the caption says
so. There is no fine-tuned row: only the minGRU has ever been fine-tuned, and
mixing it in confounded backbone with training length.

| encoder | d0 | z0 | phi | theta | q/p | geom. mean |
|---|---|---|---|---|---|---|
| minGRU | 0.9972(3) | 0.9994(4) | 0.9837(6) | 0.9972(5) | 1.0223(8) | 0.9999(2) |
| Mamba-2, bidirectional | 0.9957(3) | 0.9984(3) | 0.9816(6) | 0.9965(5) | 1.0113(8) | 0.9967(2) |
| Transformer | 0.9962(3) | 0.9980(3) | 0.9820(5) | 0.9954(5) | 1.0126(8) | 0.9968(2) |
| diagonal SSM, non-selective | 1.0180(4) | 1.0179(4) | 1.0221(7) | 1.0083(6) | 1.237(1) | 1.0572(4) |

With errors of 0.0002 on the geometric mean, "indistinguishable" is a
measurement, not an impression: Mamba-2 and the transformer are inside one
standard deviation of each other. The non-selective arm is 300 standard
deviations away, and $q/p$ at 1.237(1) is the whole of it.

Every test sample is in `tab:arch-ablation-all` (appendix), and
`fig:encoder-ablation` (appendix) shows the same numbers with the un-clipped
ratios the tables omit.

### Throughput by backbone (`tab:backbones`)

One idle H100 NVL, 131k tracks/batch, identical tool, sample and settings, in
$10^6$ tracks/s.

| encoder | reference PyTorch | our fused path | gain | fused, fp16 |
|---|---:|---:|---:|---:|
| minGRU | 1.49 | 3.90 | **2.6x** | 4.00 |
| Mamba-2, bidirectional | 0.99 | 1.78 | **1.8x** | — |
| Transformer | 0.69 | 0.69 | 1.0x | 0.74 |
| diagonal SSM, non-selective | 0.38 | 0.38 | 1.0x | 0.37 |

*Reference* is each encoder's stock PyTorch implementation — no custom kernel,
no inference switches. The minGRU leads there already, before any kernel work.

The empty Mamba-2 fp16 cell is not a missing measurement: its fused gating
kernel is typed fp32/fp64 (`Expected dtype ['fp32','fp64'] but got fp16`), so
fp16 would need an explicit cast at the kernel boundary. The minGRU kernel
converts on load inside the scan, which is why fp16 is free there.

**Careful:** the diagonal SSM also shows 1.0x, but only because nobody wrote it
a fused kernel (`_use_packed_kernel = False`). That is not the transformer's
fact and must not be presented as if it were.

### Device throughput (`tab:throughput`, `fig:throughput`)

| device | configuration | tracks/s | cost [$] | tracks/s/$ |
|---|---|---:|---:|---:|
| 32-core CPU | ACTS Kalman filter fit (reference) | ~170 k | 4,000 | 42.5 |
| RTX 5000 Ada | this work, deployment path | **TBD** | 4,000 | **TBD** |
| H100 NVL | this work, deployment path (fp16, h=194) | 4.06 M | 30,000 | 135 |
| H100 NVL | this work, deployment path (fp16, h=192) | 5.21 M | 30,000 | 174 |

The figure shows one GPU curve, the minGRU at 5.21 M, against the CPU thread
scan. The Ada series was removed rather than left stale.

### Precision policy

Train in strict fp32, infer in fp16. On the shipped checkpoint, fp16 and fp32
inference agree to **three decimals on every parameter of every test sample**
(27 of 30 cells identical, 3 differing by 0.001). fp16 buys nothing in training
here — 55.3 vs 55.6 it/s, the step is bound by the eager padded path, the fp32
scan and the loader — and it adds a divergence mode: without a GradScaler a
loss spike at ~24k steps overflowed a gradient and NaN'd the weights in one
update. The on-GPU analytic seed stays **float64** regardless; fp32 there
injects a catastrophic cancellation at high momentum.

### Kernel / fusion argument (the intellectually load-bearing part)

* Profiled, same batch: minGRU **12** CUDA kernels per forward (1,559 us GPU
  time); transformer **153** (6,793 us). The 4.36x GPU-time ratio matches the
  4.59x throughput ratio. Decomposition: 1.32x more arithmetic (1.65x tokens
  from padding x 0.80x parameters) times 3.47x overhead.
* minGRU's update `h_t = h_{t-1} + z_t*(n_t - h_{t-1})` is elementwise per
  hidden channel: one thread owns a channel and keeps its state in a
  **register** — no shared memory, no barriers. Attention needs every query to
  meet every key plus a softmax reduction across positions, which *requires*
  shared memory and barriers.
* **On-chip working set per track per layer**: minGRU 194 floats = 0.76 KiB;
  transformer peak (FFN stage) 22x128 + 22x512 = 14,080 floats = 55.0 KiB —
  **73x more**. An RTX 5000 Ada has 99 KiB of shared memory per SM, so one
  track in fp32, three in fp16; an H100 (227 KiB) four or eight. The minGRU
  uses registers, so that ceiling does not bind it. A fused transformer layer
  is occupancy-starved, and it gets *worse* on smaller cards — the actual
  deployment target.
* **FlashAttention is the wrong tool at L=22.** Attention step alone, our exact
  shapes (4 heads, head dim 32), microseconds per 1e6 tokens, batches
  8k/32k/131k: dense padded SDPA 3.7/3.6/3.8; explicit QK^T+softmax+AV
  4.5/4.4/4.4; SDPA on its flash backend 5.6/5.6/FAILS; `flash_attn_varlen`
  5.2/5.2/FAILS. **40-55 % slower**, and both flash paths fail to launch at
  131k (grid-dimension limit). In strict fp32 there is no flash kernel at all.
* Attention is only **2.8 %** of a layer's FLOPs at L=22, d=128 (FFN 64.8 %,
  QKV 24.3 %, out-proj 8.1 %). The transformer's cost is the projections.
* Portability: the minGRU kernel uses no `tl.dot`, no TMA, no warp
  specialisation, no fp8 and **zero** shared memory; every autotune config is
  AOT-compiled and checked for Hopper-only PTX against sm_89.

### Training protocol — the first thing a reviewer will ask

Every architecture arm uses an identical protocol. Verified from the configs:

| setting | value (all arms) |
|---|---|
| optimiser | Lion |
| schedule | OneCycle, 1e-5 -> 5e-5 -> 1e-6 |
| weight decay | 1e-3 |
| batch size | 2048 |
| precision | `32-true` (strict fp32) |
| seed | 42 |
| epochs | 25 |
| data | `ICLR_retraining_v2_mix3` |

No per-architecture tuning of any kind.

**The shared learning rate is nobody's optimum.** An eleven-point sweep per
architecture exists (`eval_plots/ablations_2026-09/lr_sweep_table.txt`,
protocol in `INTERIM_REPORT.md` section 4): stage 1 at 46,000 steps over
x1/8 to x8 of the production 5e-5, then the top three re-run at 138,000 steps
to re-rank. 33 runs, none diverged, every optimum bracketed inside the grid.
Selection statistic: pooled-val GM5 against the truth-KF on the identical 1 M
validation subset — a selection statistic only, not comparable to the
per-sample test ratios.

| architecture | stage-1 minimum | selected after re-rank | x production |
|---|---|---|---|
| Bi-Mamba-2 | 0.996 at 2e-4 | **1e-4** | x2 |
| Transformer | 1.089 at 7.07e-5 | **2.5e-5** | x0.5 |
| Bi-GRU (minGRU class) | 1.183 at 7.07e-5 | **7.07e-5** | x1.41 |

The three span a factor of four and **none of them is the production value**,
though 5e-5 sits inside the 3 % tie band of the recurrent minimum. The
re-ranking also measures a short-horizon bias directly: the SSM prefers 2e-4
at 46 k steps and 1e-4 at 138 k, drifting towards production as the budget
grows.

**Do not repeat the earlier "the shared LR handicaps the transformer most by
8 %" line — it was wrong.** It compared 5e-5 against each architecture's
*selected* LR rather than its stage-1 minimum, which hid that the largest
stage-1 gap is Mamba-2's. The defensible statement is the one now in the
paper's appendix: the optima span 4x, the production value is nobody's
optimum, the ablation runs 25 epochs (~100x the sweep budget) where the
sensitivity is much smaller, and the single shared recipe is stated as a
limitation rather than spun either way.

**Second caveat that must travel with those numbers**: the sweep ran on v3
data at a short step budget, not on the v2 25-epoch runs. Only the direction
transfers.

### Which checkpoint

The physics comes from the fine-tune's **`last.ckpt`, epoch 49 of 50** — not
from the validation-loss `best`, which for this run is epoch 17. The parameters
that matter keep improving long after the pooled validation loss stops moving.

---

## THE PROSE CHANGES — present these one at a time

Line numbers are as of `f047dc4`; re-check them, the user edits prose in
parallel.

**P1 (MUST) `sections/method.tex:78-79`** — "stacks two bidirectional Mamba-2
layers ... with $d_{\mathrm{state}} = 64$, expansion factor 2 and head
dimension 32". **This is the single most important factual error in the
draft**: it specifies an encoder the paper no longer ships. Replace with the
minGRU (2 layers, hidden 194, 0.650 M parameters) and its update rule, and move
the Mamba-2 description into the ablation discussion.

**P2 (MUST) `sections/method.tex:8`** — section title "A seed-guided
bidirectional Mamba-2 trajectory parameter estimator". Name the family or the
minGRU. `\cref{sec:method}` resolves to a number, so nothing downstream breaks.

**P3 (SHOULD) `sections/method.tex:87-110`** — the selective-scan / Kalman-gain
analogy around `eq:mamba-update`. **Keep it.** The ablation *supports* it: the
one arm that drops input-dependent gating is the one that fails, and only on
$q/p$. What needs adding is that result, inline: dropping the
input-dependent gate takes $q/p$ from 1.0113(8)-1.0223(8) to **1.237(1)** and
leaves the geometric parameters within 2 % — selectivity buys the momentum
estimate and nothing else.

**P4 (SHOULD) `sections/method.tex:172-193`** — "The reference Mamba-2 GPU
kernel is built for language-model..." plus `\kernelSpeedup`. This is where the
recommendation lives. Currently a Mamba-2 anecdote; the ablation makes it a
family-level claim. Use the `tab:backbones` gains (2.6x / 1.8x / 1.0x) and the
73x working-set argument.

**P5 (SHOULD) `sections/introduction.tex:41-46`** — "motivates a
Mamba-2-inspired architecture ... We propose a bidirectional Mamba-2
state-space encoder". One clause: alternatives were tested, they tie on
precision, the family is chosen for its kernel.

**P6 (INFORM, no edit) `sections/abstract.tex:25,31`** — "bidirectional
State-Space Models (SSM)" and "a small bidirectional State-Space encoder" are
**already family-level and survive the swap unchanged**. Tell the user this:
it is the strongest practical argument for framing the recommendation at family
level rather than naming an architecture.

**P7 (SHOULD) `sections/results.tex`**, the sentence "The SSM is at or below
the reference in every entry of both halves of the table within 1 %". With
uncertainties now visible, $q/p$ at 50 GeV (1.011(2)) and on the uniform sample
(1.0115(8)) are above unity by ~10 sigma. Soften to match the table.

**P8 (SHOULD) `sections/results.tex:33`** — "the seed-guided bidirectional
Mamba-2 trajectory estimator matches ... the model fits `\adaThroughput` tracks
per second on an NVIDIA RTX 5000 Ada GPU ... about a factor 3 higher". Names
the old encoder, and quotes an Ada number that is `TBD` in the table because it
is still the Mamba-2 measurement. Needs both fixed.

**P9 (SHOULD) `sections/results.tex`** — two new sentences, one after
`tab:arch-ablation` ("the encoders tie on precision") and one after
`tab:backbones` ("and this is where they stop tying"). These carry the
storyline and currently do not exist.

**P10 (SHOULD) `sections/results.tex:256`** — "throughput of the seed-guided
bidirectional Mamba-2 trajectory parameter estimator". Same rename as P2.

**P11 (SHOULD) `sections/conclusion.tex:8-9,13`** — "a seed-guided
bidirectional Mamba-2 state-space model matches..." needs the rename; "State-Space
Models like Mamba-2 resemble classical Kalman filters" is already family-level
and survives.

**P12 (RAISE, do not act) `sections/results.tex:170`** — claims the throughput
advantage is "more than one order of magnitude ... even if normalized to the
approximate cost of the compute device". The CPU baseline was corrected on
2026-09-17 to 172.7 k tracks/s, which puts the per-dollar advantage at 1.5-3x.
Pre-existing, unrelated to the ablation, already logged in the training repo's
`CLAUDE.md` 4.38.

**P13 (INFORM) `sections/appendix.tex:104-121`** (`tab:kernel-bench`) — still
the Mamba-2 kernel ablation. Correct as a statement about that campaign, but no
longer about the shipped model; `tab:backbones` is the family-level version.
The user already trimmed this table by hand (`77e490b`), so ask before touching it.

---

## WHAT NOT TO SAY

* **Do not mention a one-directional Mamba-2 arm.** It exists, it ties the
  bidirectional reference, and it is deliberately held back (user decision,
  2026-09-20): it would force rewriting the bidirectionality motivation, and
  the abstract sells "bidirectional" in its first line. It is in no table and
  no figure. If it is ever reinstated, the abstract has to move with it.
* **Do not claim transformers are fundamentally slower.** Our 0.69 M number is
  a padded attention path — 22 tokens per track against a true mean of 13.3 —
  with no fused kernel. Packing *is* possible for transformers (flash-attn
  varlen, FlexAttention block masks); only naive packed attention is
  O((B*L)^2). Corrected, it lands near ~1 M. At matched parameters and packed,
  the transformer does **0.94x** the arithmetic of the minGRU. The defensible
  claim is that *a bidirectional linear recurrence is far easier to make fast
  at L~20, and the resulting kernel is simpler and more portable.*
* **Do not present the diagonal SSM's 1.0x kernel gain as evidence** — no
  kernel was written for it.
* **Do not quote the 8 % LR handicap as if it applied to the ablation table.**
* **Do not describe the ablation rows as final-model numbers.** They are
  stage-1, pre-fine-tuning, and slightly worse than `tab:ratios` by design.

---

## STILL OPEN

* **RTX 5000 Ada (`\tbd` x2).** Every Ada number is still the Mamba-2 model.
  The benchmarking bundle for the minGRU is staged at
  `/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_v2/rtx_share`.
* **minGRU h=192.** The kernel-aligned width is the throughput headline
  (5.21 M vs 4.06 M: at BD=64, hidden 192 is three channel blocks where 194 is
  four — +28 % for 1 % fewer parameters). Its 25-epoch run was at epoch 11/25
  on 2026-09-21 and its curve tracks the h=194 run, so it is expected to land
  at the same physics. The physics in the draft is still h=194.
* **`fig:architecture`** is fixed — its encoder block now reads "Bidirectional
  state-space encoder". It is TikZ
  (`material/main_9_pages/architecture/architecture_iclr.tex`), so further
  relabelling is a source edit plus one `pdflatex`, not a redraw.
* **Other arms still training** (complex-decay LRU, width-matched non-selective,
  inward minGRU). None is in the paper and none is needed for the storyline.
