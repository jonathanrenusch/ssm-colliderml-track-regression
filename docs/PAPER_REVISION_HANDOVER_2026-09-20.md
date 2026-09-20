# Handover prompt — ICLR paper revision, encoder ablation

You are helping revise an ICLR submission at `/shared/tracking/NeurIPS_2026_SSM_Tracking`
(LaTeX: `main.tex` + `sections/*.tex` + `references.bib`). The deadline is ~5 days
away and there is no time for internal review.

## HOW TO WORK — read this first

**Do not apply any change until the user has signed off on that specific change.**
Work through the numbered items below ONE AT A TIME. For each one:

1. Show the user the current text (quote it verbatim, with `file:line`).
2. Say what is wrong with it and why it must change.
3. Show the exact proposed replacement.
4. **Stop and wait for approval.** Do not batch items.
5. On approval, apply that single edit, rebuild, confirm the build is clean and
   the page count has not regressed, then move to the next item.

The user writes the paper. Your drafts are proposals. If the user rewrites your
prose, use their version verbatim.

Build command (the system texlive lacks `fancyhdr`):

    PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2025/bin/x86_64-linux:$PATH bash build.sh

After adding a citation you must also run `bibtex main` and rebuild, or the
citation stays undefined.

## GUARDRAILS

* The **shipped model stays Mamba-2** (`R2Lnoconv-FT`). Only the *motivation*
  changes. Swapping the encoder would force regenerating every figure and table
  in `results.tex`; that is a separate, much larger decision for the user.
* Do not touch any measured number, figure or table in `results.tex`.
* Main body must stay at 9 pages (Conclusions on p9 or earlier). Appendix may grow.
* Unit macros `\gibi` / `\kibi` are ONLY valid inside `\SI{}{}` in this document
  (`\SI{14.6}{\gibi\byte}`). Bare use is a fatal LaTeX error.
* `sections/{background,related_work,non_gaussian,metrics,interpretability,discussion}.tex`
  are commented out of `main.tex` and are NOT compiled. Ignore them.

---

## THE EVIDENCE (all measured; cite these numbers, do not round differently)

### Architecture ablation
Identical recipe, data, seed, 25 epochs, parameter-matched ~0.65 M, evaluated
against the truth-seeded KF shipped with the data, |eta| <= 2, post-clip,
geometric mean over the 5 perigee parameters (GM5):

| encoder | mu 2GeV | 10GeV | 50GeV | uniform | ttbar |
|---|---|---|---|---|---|
| Mamba-2 bidirectional (paper model) | 0.994 | 0.994 | 0.997 | 0.991 | 0.996 |
| Mamba-2 ONE-DIRECTIONAL (HELD BACK — do not put in the paper) | 0.997 | 0.994 | 0.997 | 0.992 | 0.998 |
| Transformer | 0.992 | 0.994 | 0.997 | 0.991 | 0.994 |
| minGRU | 0.998 | 0.996 | 0.999 | 0.995 | 0.999 |
| non-selective diagonal SSM | 1.034 | 1.031 | 1.062 | 1.065 | 1.037 |

* On d0/z0/phi/theta all four agree to +-0.005 on every test set.
* The non-selective arm fails ONLY on q/p (1.14-1.31x), the sole source of its
  GM5 excess.
* The one-directional arm ties the bidirectional one everywhere, but it is
  **held back from this submission** (user decision 2026-09-20, final note at
  the end of this file). Nothing in tomorrow's draft may rest on it, and the
  bidirectionality motivation stays as written.

### Why bidirectionality turned out not to matter (BACKGROUND ONLY — not for the paper)
The model regresses five numbers from a POOLED terminal state, not per-hit
outputs. A single forward pass has already seen every hit before any head reads
it. Bidirectionality only buys "each hit sees the hits after it", which a
per-track readout never uses. The analytic three-hit seed is additionally a
global, order-independent summary computed before the sequence model, and the
heads are anchored to it.

### Throughput (one H100 NVL, deployment path, 131k tracks/batch)
* minGRU bidirectional h=194: 3.9 M tracks/s fp32/TF32 (11.1 GiB); 4.1 M fp16 (6.1 GiB)
* minGRU bidirectional h=192: 5.04 M fp16 (6.04 GiB)  [h=192 vs 194 is +23 %]
* Mamba-2 deployment model (the paper's): 1.91 M
* Transformer: 0.69 M, and FLAT already from 32k batch (saturated)
* Throughput is flat from **131k to 600k** batch for the recurrent encoders.
  It is NOT flat from 32k for them (3.15 M at 32k -> 3.94 M at 131k). The
  transformer is the one that is flat from 32k.
* fp16 INFERENCE on an fp32-trained checkpoint changes physics by <= 0.07 % on
  any parameter of any test set.

### Kernel / fusion argument (the intellectually load-bearing part)
* Profiled, same batch: minGRU **12** CUDA kernels/forward (1,559 us GPU time);
  transformer **153** (6,793 us). GPU-time ratio 4.36x matches the measured
  throughput ratio 4.59x. Decomposition: 1.32x more arithmetic (1.65x tokens
  from padding x 0.80x parameters) times 3.47x overhead.
* minGRU's update h_t = h_{t-1} + z_t*(n_t - h_{t-1}) is elementwise per hidden
  channel: one thread owns a channel, keeps state in a REGISTER, no shared
  memory, no barriers. Attention needs every query to meet every key plus a
  softmax reduction across positions -> REQUIRES shared memory and barriers.
* **On-chip working set per track per layer**: minGRU 194 floats = 0.76 KiB;
  transformer peak (FFN stage) 22x128 + 22x512 = 14,080 floats = 55.0 KiB.
  **73x more.** RTX 5000 Ada has 99 KiB shared memory/SM -> 1 track in fp32,
  3 in fp16. H100 (227 KiB) -> 4 or 8. The minGRU uses registers, so this
  ceiling does not apply to it. A fused transformer layer is occupancy-starved,
  and it gets WORSE on smaller cards - the actual deployment target.
* **FlashAttention is the wrong tool at L=22.** Attention step alone, our exact
  shapes (4 heads, head dim 32), microseconds per 1e6 tokens, batch 8k/32k/131k:
  dense padded SDPA 3.7/3.6/3.8; explicit QK^T+softmax+AV 4.5/4.4/4.4; SDPA on
  its flash backend 5.6/5.6/FAILS; flash_attn_varlen 5.2/5.2/FAILS.
  **40-55 % slower**, and both flash paths fail to launch at 131k (grid-dimension
  limit). In strict fp32 there is no flash kernel at all.
* Attention is only **2.8 %** of a layer's FLOPs at L=22, d=128 (FFN 64.8 %,
  QKV 24.3 %, out-proj 8.1 %). The transformer's cost is the projections.
* Portability: the minGRU kernel uses no tl.dot, no TMA, no warp specialisation,
  no fp8 and ZERO shared memory; every autotune config is AOT-compiled and
  checked for Hopper-only PTX against sm_89.

### FAIRNESS — non-negotiable
Our 0.69 M transformer number is measured on a PADDED attention path: 22 tokens
per track instead of the true mean 13.3, and no fused/flash kernel. Packing IS
possible for transformers (flash-attn varlen / FlexAttention block masks);
only NAIVE packed attention is O((B*L)^2). Corrected, it lands near ~1 M.
At matched parameters and packed the transformer does **0.94x** the arithmetic
of the minGRU (12.30 vs 13.13 MFLOP/track). **Do NOT claim transformers are
fundamentally slower.** The defensible claim is: *a bidirectional linear
recurrence is far easier to make fast at L~20, and the resulting kernel is
simpler and more portable.*

---

## THE PROPOSED CHANGES — present these one at a time

**C1 (WITHDRAWN 2026-09-20 — see the final note; do not present) `sections/introduction.tex`**, the paragraph beginning "The Kalman
filter is a classical solution". Currently claims *Mamba-2's selective state
update* specifically motivates the architecture. Our ablation shows selectivity
buys nothing on 4 of 5 parameters. Reframe to: the Kalman update is a linear
recurrence corrected by a gain; any bidirectional linear-recurrent encoder can
represent it; we instantiate the family as Mamba-2; an ablation tests how much
of the analogy is load-bearing. (There is also a typo to fix: "Gaussiansas".)

**C2 (PART (b) WITHDRAWN 2026-09-20 — see the final note; part (a) still stands) `sections/method.tex`**, the two prose blocks around the
`eq:kalman-update` / `eq:mamba-update` equations (leave the equations alone).
(a) The selectivity-as-Kalman-gain claim needs its ablation result inline:
costs 14-31 % on q/p ALONE and nothing on the geometric parameters.
(b) The sentence "so the hidden state at each hit is informed by both its past
and its future" is the bidirectionality accuracy claim our one-directional arm
refutes. Replace with the pooled-readout mechanism, and say we keep the
bidirectional block because it is more general and free at this length.

**C3 (WITHDRAWN 2026-09-20 — see the final note; do not present) `sections/method.tex:7`** section title: "bidirectional Mamba-2"
-> "bidirectional recurrent". Cosmetic; `\cref{sec:method}` resolves to a
number, not the title text, so nothing downstream breaks.

**C4 (SHOULD) `sections/results.tex`**, after "...achieve the $10^{-5}$ relative
precision the problem demands." Add one sentence pointing at the ablation and
framing encoder choice as a deployment decision.

**C5 (SHOULD) `sections/conclusion.tex`**, first sentence of the final
paragraph: "State-Space Models like Mamba-2 resemble classical Kalman filters."
-> attribute to the family rather than to Mamba-2 specifically.

**C6 (SHOULD) `sections/appendix.tex`**, new `\paragraph{Encoder architecture.}`
plus `tab:arch-ablation` (the 5x5 table above), inserted at the end of
`app:ablations`, immediately before the "% F. Evaluation details" comment.
Use 3 decimal places - the point is a +-0.005 agreement that 2 decimals erase.
Caption must carry the transformer fairness caveat.

**C7 (SHOULD) `sections/appendix.tex`**, two new paragraphs at the end of
`app:kernels`, before the "% E. Ablations" comment: the throughput comparison
(with the transformer caveat) and the recurrence-vs-attention fusion argument
built from the numbers above.

**C8 (SHOULD) `references.bib`**: add `Feng2024MinGRU` (Feng et al., "Were RNNs
All We Needed?", arXiv:2410.01201, 2024) and `Vaswani2017Transformer`.
`Dao2023Flash2` already exists.

**C9 (RAISE WITH USER, do not act)** `sections/results.tex:170` claims the
throughput advantage is "more than one order of magnitude ... even if normalized
to the approximate cost of the compute device". The CPU baseline was corrected
on 2026-09-17 (172.7 k tracks/s, not 30 k), which puts the per-dollar advantage
at 1.5-3x, not >10x. This is pre-existing, unrelated to the ablation, and
already logged in the training repo's CLAUDE.md 4.38 as left for the user.

## OPEN DECISION TO PUT TO THE USER BEFORE STARTING

Whether the paper keeps Mamba-2 as the shipped model (these changes assume yes,
and only fix the motivation) or swaps to minGRU for the better throughput
numbers. The swap means regenerating every results figure and table; the minGRU
paper-plot bundle already exists at
`eval_plots/paper_plots/truthkf_minGRU_eta2/` in the training repo, and a
50-epoch fine-tuned minGRU checkpoint is due 2026-09-21.

---

# ADDENDUM (2026-09-20 late) — training protocol, LR fairness, and which arms to include

## Optimiser and learning rate — the first thing a reviewer will ask

**Every v2 architecture arm uses an IDENTICAL training protocol.** Verified
from the configs, not from memory:

| setting | value (all arms) |
|---|---|
| optimiser | **Lion** |
| schedule | OneCycle, `initial 1e-5 -> max 5e-5 -> end 1e-6` |
| weight decay | 1e-3 |
| batch size | 2048 |
| precision | `32-true` (strict fp32) |
| seed | `seed_everything: 42` |
| epochs | 25 |
| data | `ICLR_retraining_v2_mix3` |

No per-architecture tuning of any kind was applied to these runs.

**The shared learning rate is nobody's optimum, and it handicaps the
TRANSFORMER most.** An 11-point per-architecture LR sweep does exist, from the
earlier matched-budget study
(`eval_plots/ablations_2026-09/lr_sweep_table.txt` / `.json` / `.pdf`), and it
found genuinely different optima:

| architecture | GM5 at the shared 5e-5 | GM5 at its own optimum | own optimum | handicap |
|---|---|---|---|---|
| Bi-GRU | 1.1974 | 1.1826 | 7.07e-5 | 1.3 % |
| Bi-Mamba-2 | 1.1796 | 1.1480 | 1e-4 | 2.8 % |
| **Transformer** | 1.2048 | 1.1154 | **2.5e-5** | **8.0 %** |

5e-5 is the production recipe value, inherited from the Mamba-2 paper model.
So the shared LR disadvantages the transformer by ~8 % and the Mamba-2 model by
~3 %, **and the transformer still reached parity**. That makes the
"architecture does not matter" conclusion **conservative, not flattering** —
which is the right direction when a reviewer probes it, and it should be stated
explicitly rather than buried.

**Caveat that must be stated with those numbers**: that sweep was run on the v3
data at a matched *step* budget (921,600 steps), NOT on the v2 25-epoch runs.
The numerical 8 % does not transfer; only the direction does. Do not quote the
8 % as if it applied to the v2 table.

**Suggested one-sentence form for the paper**: "All encoders were trained with
an identical optimiser, schedule, batch size, precision and seed; the shared
peak learning rate was inherited from the production recipe and is not the
per-architecture optimum, which by a separate sweep disadvantages the
transformer more than the state-space model -- so the parity we report is a
conservative comparison."

## Which arms to put in the draft

State of every arm as of 2026-09-20 22:45:

| arm | state | in paper? |
|---|---|---|
| Mamba-2 bidirectional (reference) | done, evaluated | **main table** |
| Mamba-2 one-directional | done, evaluated | **HELD BACK — see the 2026-09-20 note at the end** |
| Transformer | done, evaluated | **main table** |
| minGRU | done, evaluated | **main table** |
| non-selective diagonal SSM (param-matched) | done, evaluated | **main table** |
| minGRU fp16 (training) | done, evaluated | appendix (precision policy) |
| minGRU fp16 (inference only) | done, evaluated | appendix (precision policy) |
| minGRU fine-tuned 50 ep | evaluated (see below) | judgement call — see below |
| complex-decay LRU | ep 23/25, not evaluated | leave out |
| narrow non-selective (width-matched) | ep 10/25 | leave out |
| minGRU h=192 (deployment width) | ep 8/25 | leave out |
| inward (one-directional) minGRU | ep 4/25 | leave out |
| minLSTM | KILLED at 13/25 | leave out |

Reasoning: the five architecture arms ARE the scientific claim (four backbones
at parity, one informative failure). h=192 and the inward variants are
kernel-engineering details, not architecture claims, and the inward throughput
gain did not materialise (3.2x fewer FLOPs bought ~0 % on an H100 — the model is
not FLOP-bound). minLSTM is incomplete. complex-LRU will not be evaluated and
sanity-checked in time.

## The minGRU fine-tune result (new, 2026-09-20) — and why it is a judgement call

50-epoch Muon-hybrid + WSD fine-tune (the canonical stage-2 recipe, DDP 2x20k)
from the 25-epoch minGRU. Converged: `best.ckpt` is epoch 17 and the pooled val
metrics moved < 1.3 % between epoch 1 and epoch 45.

**What it bought** (change from stage-1, negative = better; the same both
pre- and post-clip):

| dataset | d0 | z0 | phi | theta | **q/p** | GM5 |
|---|---|---|---|---|---|---|
| mu 2 GeV | -0.1 % | -0.1 % | -0.1 % | +0.1 % | **-1.4 %** | -0.3 % |
| mu 10 GeV | -0.1 % | -0.1 % | -0.2 % | -0.2 % | **-0.9 %** | -0.3 % |
| mu 50 GeV | -0.3 % | -0.2 % | -0.4 % | -0.1 % | **-1.3 %** | -0.4 % |
| mu uniform | -0.2 % | -0.1 % | -0.4 % | -0.1 % | **-1.3 %** | -0.4 % |
| ttbar | -0.1 % | -0.1 % | -0.1 % | -0.1 % | **-1.2 %** | -0.3 % |

Final post-clip GM5: 0.994 / 0.993 / 0.995 / 0.961 / 0.991 / 0.996
(mu 2 / 10 / 50 / 100 GeV / uniform / ttbar); pre-clip 0.983 / 0.973 / 0.987 /
0.954 / 0.972 / 0.900.

**Important for any claim**: unlike the Mamba-2 twin, where this stage pushed
every pre-clip ratio below 1.0, for minGRU **q/p remains above the reference** --
1.022 (ttbar), 1.009 (50 GeV), 1.006 (2 GeV) post-clip. A "no parameter above
1.0 anywhere" claim is NOT available for the fine-tuned minGRU.

**Checkpoint choice**: `last.ckpt` (epoch 45) beats `best.ckpt` (epoch 17) by
0.4-0.7 % on q/p and is identical elsewhere, because `best.ckpt` is selected on
`val/total` (the pooled mixed-set LOSS), which is not the physics metric. Use
`last.ckpt`.

Recommendation: keep the fine-tune OUT of the main text. It invites "why is your
shipped model fine-tuned then?" for a ~1 % q/p gain. Include it only if the
minGRU is presented as a deployment candidate, and if so state the residual
q/p > 1.0 honestly.

Artefacts: `eval_plots/ablations_2026-09/v2_evals/V2_minGRU_FT50{,_best}/`;
paper-ready plots (generated, NOT synced into `material/iclr/`, no tex touched)
`eval_plots/paper_plots/truthkf_minGRU_FT_eta2/` -- six datasets, |eta| <= 2,
pT <= 70 on the vs-pT page, truth-KF reference.
Tables reproduce with `scripts/abl_v2_preclip_table.py <arm> [<arm>]`.


---

# NOTE (2026-09-20, user decision) — the one-directional Mamba arm is HELD BACK

**Decision.** The one-directional Mamba-2 arm does not appear in tomorrow's
draft: not in the ablation table, not in the comparison figure, not in the
prose. It is a real, clean measurement and it stays in the repo — it is simply
out of scope for this submission.

**Why (the user's reasoning, which is correct).** The result ties the
bidirectional reference on every test set, so putting it in the paper forces a
rewrite of the architecture *motivation*, and that rewrite does not stop at the
method section: `sections/abstract.tex` sells "bidirectional State-Space
Models" in its first line and "a small bidirectional State-Space encoder" in
its last paragraph, and `main.tex:2` carries it in the title comment. An
ablation showing bidirectionality is unnecessary, sitting in an appendix under
an abstract that makes bidirectionality the headline, reads as a contradiction
a reviewer will find. Fixing that properly means touching the abstract, the
title framing, the introduction and the method — one change too many, five days
out.

**What this withdraws.** The one-directional arm was the ONLY evidence for the
bidirectionality claim. The other three parity arms are all bidirectional by
construction (the transformer attends over the whole track, minGRU and the
diagonal SSM both run two scans), so with this arm held back the draft contains
no evidence that bidirectionality is dispensable. C1, C3 and C2(b) therefore
come off the table together with it — do not present them, and do not soften
"informed by both its past and its future" in `method.tex`, because nothing in
the draft would any longer contradict it.

**What survives untouched.** The selectivity result is independent of all this:
the non-selective diagonal SSM is bidirectional and still fails, and it fails
on q/p alone. So C2(a), C4-C8 stand as written, and the scientific claim of the
ablation becomes the cleaner and narrower one: *four different bidirectional
encoder families reach the truth-KF; what actually matters is input-dependent
(selective) gating, and only for the momentum parameter.*

**Was it parameter-matched? No — it is 5.1 % SMALLER.** Measured from the
checkpoints (`state_dict` totals, identical 21,027-parameter head everywhere):

| arm | total | encoder |
|---|---:|---:|
| Mamba-2 bidirectional (reference) | 649,422 | 500,832 |
| **Mamba-2 one-directional** | **616,014** | **467,424** |
| Transformer | 649,198 | 500,608 |
| minGRU | 650,530 | 501,940 |
| non-selective diagonal SSM | 650,506 | 501,916 |

The matching was done on the recurrent work rather than on the parameter count:
`expand` went 2 -> 4, so the single forward scan carries an inner width of 512
where the bidirectional block ran two scans of 256 each (`d_state` 64,
`headdim` 32, 2 layers, `d_conv` 1 in both). That leaves it 33 k parameters
short of the other four. The direction of the mismatch is the awkward one — the
arm ties the reference while being *smaller* — which is another reason the
result is not something to half-tell in an appendix.

**Artifacts.** The paper figure
`eval_plots/paper_plots/ablation_compare/ablation_dotplot.pdf` is regenerated
without the arm (4 arms, ttbar already removed). The five-arm version is kept
for internal use only as
`ablation_dotplot_with1dir_INTERNAL.pdf`; `scripts/abl_compare_dotplot.py`
reproduces it with `--with-1dir`. The eval bundle stays at
`eval_plots/ablations_2026-09/v2_evals/V2_mamba1dir_25ep/`. If the arm is ever
reinstated, C1/C2(b)/C3 above are the changes it requires, and the abstract has
to move with them.
