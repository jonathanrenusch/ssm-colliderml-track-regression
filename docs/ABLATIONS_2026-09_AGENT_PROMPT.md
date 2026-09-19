# Prompt for the compute-node agent — architecture ablations for the ICLR 2027 track-fitting paper

Hand everything below the line to the agent. Fill the two `<<...>>` placeholders first.
Everything else is already resolved against the live repo and the live data (checked 2026-09-14).

---

You are running on **sess3**: 4 × NVIDIA H100 NVL (95 GB each, currently idle), 190 CPU cores,
~720 GB RAM (cgroup `memory.max`), and a **local `/scratch` that is empty** with 1.9 TB free.
The repository is `/shared/tracking/ssm-colliderml-track-regression` (NFS, shared between nodes —
another job of mine may be using GPUs on a *different* node through the same checkout; leave its
`logs/`, `launch_logs/` and `eval_plots/` entries alone). Python env: `pixi run -e default python`
from `<repo>/src/track_regression`.

Your job is a **controlled cross-architecture comparison** for an ICLR 2027 submission whose
full-paper deadline is **Fri 25 Sep 2026**. I need complete results by **Sun 20 Sep, 18:00 CEST**,
and an interim report at the end of Phase A (LR sweeps). I am the first author; you report to me,
`<<YOUR_NAME>>`. Work autonomously, log everything, and **when the protocol below cannot be met,
stop and report rather than improvise** — a badly controlled run is worse than a missing one and
will be discarded.

Before anything else, read `CLAUDE.md` in the repo root: the section "THE FINAL RECIPE" plus
§4.17 (data v2), §4.21/4.22 (parity), §4.27–4.32 (final model), §4.37 (the digitization fix and
the v3 data), and §8 ("Repo facts that bite"). It is the campaign log and it is accurate.

---

## 0. Decisions already taken (do not re-open)

| Question | Answer |
|---|---|
| Which data | **v3 only** (regenerated 2026-09-12 after the digitization bug fix). v2 is *wrong data* — a collaborator found a digitization bug, every v2 store and every published number sits on it. |
| Reference fit | the **production truth-seeded KF shipped with the data** (`truth_tracks` → `truth_kf_reco.npy` side-cars). **Never** the in-pipeline ad-hoc ACTS KF refit (miscalibrated 1.3–3.2×, `docs/BUGREPORT_acts_pipeline_kf.md`). |
| Fiducial | |η| ≤ 2 (truth η) for every headline number: `TRK_ABS_ETA_MAX=2`. |
| Precision | strict IEEE fp32 in training: `TRK_MATMUL_PRECISION=highest` **and** `encoder_autocast_dtype: float32`. No TF32, no bf16, no autocast anywhere in training. |
| Recipe stage | **stage 1 only** (Lion + OneCycle + bs 2048). The paper model additionally gets a 50-epoch Muon-hybrid fine-tune; nobody in this study gets it, so absolute ratios here will be worse than the paper's headline. Say so in the report. |
| Optimiser | **Lion for every architecture**, with a per-architecture LR sweep (§3). One AdamW spot-check for the transformer only (run 13). |
| Hit order | true_time (baked into the stores; the packed encoder never re-sorts). |

---

## 1. What the paper claims and what this experiment is for

The paper shows that a **0.649 M-parameter bidirectional Mamba-2 encoder** (2 layers, dim 128,
d_state 64, headdim 32, expand 2, `d_conv: 1`), trained with a specific recipe, matches a
truth-seeded Kalman filter to within ~1 % on all five perigee parameters (d₀, z₀, φ, θ, q/p) at
1.9 M tracks/s on one H100. The recipe has four ingredients, none of which touches the encoder:

1. **Seed anchoring** — an analytic three-pixel-hit ACTS helix seed (Bz = 3 T, transported to the
   perigee); every head predicts a *correction* to it, and every hit additionally carries three
   *residual-to-seed-helix* input features.
2. **Scale-free q/p target** — `(q/p − seed_qop)/(|seed_qop| + 0.02)`.
3. **7-quantile pinball heads**, all loss weights 1.0, median = point estimate.
4. **Strict fp32 at batch 2048** with Lion and a one-cycle schedule (the small batch is
   load-bearing — the precision arrives in the anneal).

The thesis is that **the recipe, not the encoder, is what reaches KF precision.** This experiment
tests (a) whether the recipe transfers to other encoders and (b) whether the recipe is necessary
for *each* encoder. Both halves are needed — see the 3 × 2 factorial in §5a.

---

## 2. Data — v3, and where it is

**Source of truth (verified complete, 2026-09-14):**
`/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_v3/`

| store | role | size / files |
|---|---|---|
| `single_muon_uniform` | training (+ its test split = the uniform test sample) | 148 GB / 2,630 |
| `single_muon_loguniform` | training | 147 GB / 2,604 |
| `ttbar_new_pt1_tr` | training (hadrons, 1–110 GeV, runs 46–784) | 13 GB / 2,396 |
| `single_muon_2GeV`, `single_muon_10GeV`, `single_muon_50GeV`, `single_muon_100GeV` | test guns | 76 MB each |
| `ttbar_new_pt1` | test (hadrons, runs 6–45) | 745 MB / 524 |

All were preprocessed with the exact v2 recipe on the *regenerated* raw data:
`--sort-key true_time --bz 3.0 --apply-d0z0-windows --d0-window 7.1 --z0-window 270`
(ttbar additionally `--pt-min 1 --pt-max 110`). Check `manifest.json` → `hit_sort_key == "true_time"`
on every store before using it.

**Staging (do this first, ~10–15 min total; EOS reads at ~1.2 GB/s with 16 streams):**

```bash
cd /shared/tracking/ssm-colliderml-track-regression
EOS=/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_v3
OUT=/scratch/colliderml/ICLR_retraining_v3
for ds in single_muon_uniform single_muon_loguniform ttbar_new_pt1_tr ttbar_new_pt1 \
          single_muon_2GeV single_muon_10GeV single_muon_50GeV single_muon_100GeV; do
  bash scripts/copy_dataset.sh $EOS/$ds $OUT/$ds 16
done
```

**Build the three-way training mix (symlinks, seconds, no disk):**

```bash
P="pixi run -e default python"
$P scripts/07_build_mixed_store.py --base $OUT/single_muon_uniform \
   --extra $OUT/single_muon_loguniform --extra-max-tracks 1000000000 --extra-val \
   --out /scratch/colliderml/ICLR_retraining_v3_mixLU
$P scripts/07_build_mixed_store.py --base /scratch/colliderml/ICLR_retraining_v3_mixLU \
   --extra $OUT/ttbar_new_pt1_tr --extra-max-tracks 1000000000 --extra-val \
   --out /scratch/colliderml/ICLR_retraining_v3_mix3
```

**Expected, and a hard gate:** `train` = **377,295,364** tracks, `val` = **20,872,359**
(the loader uses `max_val_tracks: 1000000` of them, spread over all parts), `test` = uniform test
only. If the train count is not 377,295,364 ± a few thousand, **stop and report** — something is
missing and nothing downstream is comparable.

**Eval farm:**

```bash
$P scripts/build_eval_farm.py --store-root $OUT --eval-root /scratch/colliderml/ICLR_eval_v3 \
   --union single_muon_2GeV --union single_muon_10GeV --union single_muon_50GeV \
   --union single_muon_100GeV --union single_muon_uniform --union ttbar_new_pt1
```
(the fixed-pT guns were produced as a single shard, so the farm unions train+val+test for them —
the model never saw those events; the uniform entry uses its test split.)

**ttbar test set — built 2026-09-14, use it.** `ICLR_retraining_v3/ttbar_new_pt1` (also on /eos,
524 files / 745 MB, verified): runs 6–45, pT 1–110 GeV, true_time order, truth-KF reference,
**951,923 tracks** (train 857,039 / val / test — the farm unions them, the model never saw these
runs; training uses runs 46–784, disjoint). It is the only hadron / genuinely-low-pT test sample
and the only test of whether the recipe transfers off the muon gun, so include it as a test row
(add `--union ttbar_new_pt1` to the eval-farm command).
Note the ttbar rows are a *diagnostic*: ttbar was removed from the paper's physics results
(CLAUDE.md §4.32) and stays training data there.

**Held-out test samples (used once, at the end):** `single_muon_{2,10,50}GeV`,
`single_muon_uniform` and `ttbar_new_pt1`. `single_muon_100GeV` is a **diagnostic only** — on v2 the *reference* fit
itself was shown to lose calibration above ~95 GeV (CLAUDE.md §4.33); whether the digitization fix
removed that is unknown, so 100 GeV does not enter any headline claim.

**Never select on the test samples.** Selection (LR, anything else) uses the mix3 **val** split only.

---

## 3. Protocol — non-negotiable

**Held fixed across every run** (identical code paths, not re-implementations):
the 15 per-hit input features (12 absolute + 3 seed residuals); the min–max normalisation and the
16-scale Fourier featurisation; seed anchoring and the scale-free q/p target; the 7-quantile
pinball heads with equal weights; **the entire head stack** (see §4, "identical readout");
batch size 2048; strict fp32; Lion; the one-cycle schedule *shape*; `gradient_clip_val: 1.0`;
no dropout; weight decay 1e-3; the training mixture, its ordering and `seed_everything: 42`;
the validation split; the evaluation code; **trunk parameter count 632,928 ± 10 %**
(= the paper model's encoder+input-net trunk; its total incl. heads is 0.649 M — print and report
both for every model); and the training budget in **optimiser steps** (§5).

**Varied:** the encoder module only.

**Learning rate — symmetric sweep for every architecture, including the paper's own SSM.**
Production stage-1 schedule (read from
`src/track_regression/config/ssm_cls/ICLR_sweep7/R2Lnoconv_qrel_2L_dconv1_mix3_bs2048_onecycle25.yaml`):
Lion, OneCycle, `initial 1.0e-5 → max 5.0e-5 → end 1.0e-6`, `pct_start 0.10`, `weight_decay 1.0e-3`.

- **Stage 1 — coarse grid, 11 points, half-octave spacing with octave guards at the edges**
  (×⅛, ×¼, ×0.354, ×½, ×0.707, ×1, ×1.41, ×2, ×2.83, ×4, ×8 of the production peak):
  peak LR ∈ **{6.25e-6, 1.25e-5, 1.77e-5, 2.5e-5, 3.54e-5, 5.0e-5, 7.07e-5, 1.0e-4, 1.41e-4,
  2.0e-4, 4.0e-4}**, at the sweep budget (46 k steps).
- **Stage 2 — refine.** Re-run the **best three** points at **3× the sweep budget (138 k steps)**
  and select on those. This is the part that matters: a short sweep is biased toward larger LRs
  than a long run wants, and re-ranking the finalists at a longer horizon attacks that bias
  directly — far more useful than adding yet more points at the short horizon. Report both stages
  in `lr_sweep.pdf` (stage-1 curve + stage-2 points overlaid).
- **Scale `initial` and `end` with the peak** (keep `initial = max/5`, `end = max/50`), so the
  schedule *shape* is identical at every grid point. Changing only `max` would turn the low-LR
  points into near-constant-LR runs and the sweep would measure the wrong thing.
- Keep `weight_decay = 1e-3` fixed at every point. Note in the report that Lion's decoupled decay
  is applied as `lr × λ`, so the *effective* decay varies across the grid by the same 16× as the
  LR; at these magnitudes the cumulative shrink over the sweep budget is ≲ 1 % at the low end and
  a few % at the high end. This is a known, stated confound, not a silent one.
- Selection criterion: **GM₅** = geometric mean over the five parameters of the iterative-3σ-clipped
  RMS on the val split (the `[val epoch N] iter-3σ RMSE` line the trainer prints once per
  validation, and the Comet `val/<p>/ssm_rms3s` metrics). Use ratios to the truth-KF if the val
  split gives them; otherwise absolute clipped RMS — for *selection within one val set* they
  differ by a constant.
- **Tie-break: if two points are within 3 % on GM₅, take the smaller LR.**
- The ×⅛ and ×8 guards exist so the optimum is bracketed. If the stage-1 best is still one of
  them, extend by one octave in that direction, once, and say so. If the **SSM's** selected LR is not strictly inside the grid, **stop and report**.
- A run that diverges (NaN/inf, or val GM₅ worse than the seed baseline) is a **sweep result** —
  record it, kill that run only, continue the others.
- Ablation runs (no-seed variants) inherit the **selected LR of their own architecture**.

**Inductive bias is not equalised.** Each encoder uses its natural form; fairness comes from
everything else being identical, from the per-architecture LR sweep, and from matched parameters
and matched optimiser steps. Do not add tricks to help or hinder any encoder.

**Parameter-matched, not FLOP-matched.** State this explicitly and report, for every model:
trunk params, total params, measured training steps/s, and inference tracks/s (§7). Do not
silently equalise anything else.

---

## 4. Encoders

The repo already contains two of the four. **Use what exists** — a re-implementation is a
confound, not a control.

**Identical readout (this is what makes the head stack fair).** The regressor builds the same
head stack for `pool: ssm_cls` and `pool: register_token` whenever the encoder's pooled output is
`2 × dim = 256`: `pool_head = Dense(256 → 128, hidden [128])` → `output_head(128 → 35)`. So every
encoder must expose **`pool_dim = 256`** and return `(sequence_output, pooled)` with
`pooled.shape == (B, 256)`. Verify by diffing the printed parameter counts of the heads between
runs — they must be identical to the parameter.

**Packed batches are mandatory.** `data.seed_residual_features: true` requires
`packed_batches: true` (`flat_data.py:286`), so every encoder must accept the packed layout:
`forward(x, x_sort_value=..., seq_idx=..., cu_seqlens=...)` with `x` of shape `(1, ΣLᵢ, dim)` and
`cu_seqlens` the cumulative segment boundaries. Sequences are ≤ 20 hits (+ CLS tokens).

1. **Bidirectional Mamba-2 (paper model)** — `track_regression.mamba_cls.BidirectionalMambaCLSEncoder`,
   unchanged, `pool: ssm_cls`. 2 layers, dim 128, d_state 64, d_conv 1, expand 2, headdim 32,
   ngroups 1, RMSNorm. Config to copy verbatim (change only `data.preprocessed_dir`, `max_epochs`,
   and the LR block): `config/ssm_cls/ICLR_sweep7/R2Lnoconv_qrel_2L_dconv1_mix3_bs2048_onecycle25.yaml`.
2. **Transformer** — `track_regression.transformer_encoder.EncoderWithCLS` **already exists** and
   already supports packed input, 2 CLS tokens (`pool_dim = 2·dim`), RMSNorm, pre-norm, LayerScale
   and a content-derived Fourier-of-sort-key positional encoding. Start from
   `config/transformer/pretrain_transformer_2cls.yaml` for the encoder block **only** — that file
   is a campaign-1 config (dim 192, 12 layers, legacy losses, legacy data): take the encoder
   `init_args`, drop everything else, and graft it into the sweep-7 recipe.
   Required settings: `pool: register_token`, `dim: 128`, `num_cls_tokens: 2`,
   **`attn_type: torch`** (SDPA — flash-attn-2 is bf16/fp16 only and the encoder runs in fp32),
   `norm: RMSNorm`, `qkv_norm: true`, `layer_scale: 1.0e-5`, `attn_kwargs: {num_heads: 4}`,
   `dense_kwargs: {hidden_dim_scale: 4}`, posenc left at its default ladder.
   Set `num_layers` (start at 3) and, if needed, `hidden_dim_scale` to land the trunk at
   632,928 ± 10 %. Report the exact count.
3. **Bidirectional GRU** — new module, the only one you must write. `nn.GRU(bidirectional=True,
   num_layers=2)`, hidden size tuned to the trunk target (≈ 160), input `dim` 128. Convert packed
   → padded internally (`cu_seqlens` → lengths → `pad_sequence` → `pack_padded_sequence`,
   `enforce_sorted=False`), pool by concatenating the final forward and final backward hidden
   states of the last layer, project to 256 if hidden ≠ 128, expose `pool_dim = 256`, return
   `(sequence_output, pooled)`. Include the DDP unused-parameter tie the other encoders use
   (`pooled + 0.0 * sequence_output.sum()`) — see `mamba_cls.py` for the pattern, and note the
   §4.27 bug: apply it **in training mode only** (a global sum poisons the batch at inference).
   No dropout. cuDNN fused GRU is fine and is the GRU's "best available kernel".
4. **Forward-only Mamba-2** — the paper's block with the reverse scan and the direction-merge gate
   removed; width raised to the trunk target; pool from the forward CLS state and project to 256.
   No LR sweep: reuse the bidirectional SSM's selected LR (same block, same optimiser).
5. **MLP (optional, run 11)** — padded, flattened 20×15 input; only if everything else is done.

**Per-encoder acceptance tests before any long run** (all five, logged):
(a) shape test on one real batch from the mix3 loader;
(b) **packed vs padded equivalence** to ≤ 1e-6 max abs on the same tracks (skip for the GRU only
if the padded path is genuinely unavailable — then say so);
(c) permutation/segment-independence: shuffling the *track order* within a packed batch must not
change any track's output (catches cu_seqlens bugs — this is the single most common way to get a
silently wrong packed encoder);
(d) overfit 512 tracks to near-zero loss in a few hundred steps;
(e) parameter counts (trunk, heads, total) printed and inside the budget;
(f) `torch.backends.cuda.matmul.allow_tf32 == False`, no autocast, `precision: 32-true`,
`encoder_autocast_dtype: float32`.

---

## 5. Budgets and the run list

**Measure first.** Do a 200-step timed dry run of each architecture at bs 2048 on mix3 and report
steps/s before launching Phase B. One epoch of mix3 at bs 2048 = **184,226 steps**; the paper
model runs it in ≈ 1.2 h (≈ 42 steps/s) on one H100. Other encoders will differ — **budgets are in
steps, not epochs, and are identical for every architecture**; wall-clock is a reported quantity,
not a control.

- **LR-sweep budget: 46,000 steps** (stage 1) and **138,000 steps** (stage-2 refine of the top 3),
  the *complete* one-cycle compressed into each. ≈ 20 min / 60 min for the SSM.
- **Matched budget: 921,600 steps** (5 epoch-equivalents), one-cycle compressed into it.
  ≈ 6.1 h for the SSM.
- **Ladder budget (§5b): 307,200 steps** (⅓ of matched), identical across every size and
  architecture in the ladder.

Set these with `--trainer.max_steps` and `--trainer.max_epochs -1`; make sure the OneCycle total is
derived from `max_steps` (check the printed LR curve in the first/last 100 steps of a dry run).

### 5a. The main run list — a 3 × 2 factorial

The core of the study is **{Mamba-2, Transformer, GRU} × {with seed, without seed}** at the matched
budget. That design answers both halves of the paper's claim at once: the *rows* say whether the
recipe transfers across encoders, the *columns* say whether it is necessary for each of them, and
the interaction says whether any encoder is unusually dependent on it.

| # | Run | LR | Seeds | Priority |
|---|-----|----|-------|----------|
| 1 | SSM LR sweep (11 × 46 k, then 3 × 138 k) | grid | — | A |
| 2 | Transformer LR sweep (11 × 46 k, then 3 × 138 k) | grid | — | A |
| 3 | GRU LR sweep (11 × 46 k, then 3 × 138 k) | grid | — | A |
| 4 | **SSM + seed** matched | selected | **3** (42/43/44) | B |
| 5 | **Transformer + seed** matched | selected | **2** (42/43) | B |
| 6 | **GRU + seed** matched | selected | **2** | B |
| 7 | **SSM − seed** matched | SSM's selected | **2** | B |
| 8 | **Transformer − seed** matched | Transformer's selected | **2** | B |
| 9 | **GRU − seed** matched | GRU's selected | **2** | B |
| 10 | Forward-only Mamba-2 + seed matched (bidirectionality ablation) | SSM's selected | 1 | C |
| 11 | Saturation ladder (§5b) | per-size | 1 each | C |
| 12 | **Store control**: SSM + seed on the uniform-muon store only, matched budget (§5c) | SSM's selected | 1 | C |
| 13 | Transformer + seed, **AdamW** (3-point mini-sweep, then matched) | own | 1 | D |
| 14 | MLP on flattened 20×15 input (own sweep + matched) | grid | 1 | E |

**Runs 4–9 are the study.** Nothing may be dropped from them. Everything else is expendable, in
reverse priority order (14, then 13, then 11, then 12, then 10).

**Two seeds everywhere is deliberate.** Every cross-architecture and every with/without-seed
difference must be quotable against a measured spread, not an assumed one. If two numbers differ
by less than the larger of the two seed spreads, report them as **indistinguishable** — do not
rank them. Seeds differ in `seed_everything` only.

**"Minus seed" means the seed is gone entirely**, not half of it — the seed enters the recipe
twice, as the head anchor and as 3 input features, and removing only one of them measures nothing
interpretable. Configure it as:

- `data.seed_residual_features: false`; `input_dim: 12` with the first 12 entries of
  `input_fields` / `norm_min` / `norm_max` from the base config;
- absolute heads, **keeping the rest of the recipe** (7-quantile pinball, all weights 1.0), with
  **no `delta_anchor` and no `scale_anchor_eps`** anywhere:

```yaml
d0:    {type: quantile, weight: 1.0, norm_min: -7.1,   norm_max: 7.1,    quantiles: [0.05,0.1,0.25,0.5,0.75,0.9,0.95]}
z0:    {type: quantile, weight: 1.0, norm_min: -270.0, norm_max: 270.0,  quantiles: [...]}
phi:   {type: circular, weight: 1.0, beta: 0.01}
theta: {type: quantile, weight: 1.0, norm_min: 0.0,    norm_max: 3.1416, quantiles: [...]}
qop:   {type: quantile, weight: 1.0, norm_min: -1.0,   norm_max: 1.0,    quantiles: [...]}
```

**One forced confound, and you must name it in the report:** φ cannot keep the pinball head when
the anchor goes — the recipe's φ head is a *wrapped delta* to the seed, and absolute φ lives on a
circle where a quantile ladder is not representable. So the no-seed φ head is the `circular`
(Huber on sin/cos) head, at weight 1.0. That is one head-family change riding along with the seed
removal, in the one parameter where the recipe's own log (`CLAUDE.md` §4.14) says the head family
matters most. Do not hide it, and do not "fix" it by inventing a third φ head.
For orientation, `config/ssm_cls/ICLR_sweep1/B_4L_ds64_lion_cosine_bs36k.yaml` is the campaign's
absolute-head loss block — but it uses the *legacy* weights (0.1/1/0.1/1/1), which would add a
second confound. Use the block above, not that file verbatim.
Print the resolved config and eyeball the loss block before launching.

**Seeds.** Only the SSM gets 3 seeds (42, 43, 44 — differing in `seed_everything` only). Use the
SSM's seed spread as the reference uncertainty for the single-seed runs. **Rule:** if two
architectures' GM₅ differ by less than that spread, run 2 extra seeds of the contender before
claiming any difference; if you cannot, report the comparison as "indistinguishable at 1 seed".

### 5b. The saturation ladder — the answer to "you can only compare architectures under scaling"

That objection is right in general and it has a specific, cheap answer in *this* setting, which
you must produce: a **saturation curve**, not a scaling-law fit.

Run each of the three encoders **with the recipe** at **three trunk sizes** — nominally
**≈ 0.2 M, 0.649 M (the paper point), ≈ 2.0 M** trainable trunk parameters, scaled by width first
(and depth only if width alone cannot reach the target) — all at the **ladder budget** (307,200
steps), and plot **GM₅ vs trunk parameters, one curve per encoder**, with the truth-KF line at 1.0.

- **Per-size LR is mandatory.** Do *not* reuse the 0.649 M LR at another size: a dim-96 variant of
  this exact model **diverged** at the standard peak LR in an earlier round (CLAUDE.md §4.22).
  Run a 3-point mini-sweep (×½, ×1, ×2 of that architecture's selected LR) at each new size,
  46 k steps each, and select as in §3.
- Re-run the middle (0.649 M) size at the ladder budget too, so the curve is internally consistent;
  the matched-budget runs of §5a stay the headline numbers.
- **What the figure has to show, and what it would mean.** If all three curves flatten at the
  truth-KF line at or below 0.649 M parameters, the task is *saturated* at that scale, and a
  single-size comparison is then justified **by data** rather than asserted — which is exactly
  what the paper needs, because the claim is "the recipe reaches the information floor at
  0.65 M parameters", not "this encoder scales better". If instead a curve is still falling at
  2 M, say so plainly: the single-point comparison then understates that encoder and the paper
  must scope its claim to the tested scale.
- State the limitation either way: this is a saturation ladder at a fixed budget, not a
  compute-optimal scaling study, and the conclusion is scoped to the regime where this task
  saturates against its reference fit.

### 5c. Which training store — and why it is mix3

**Every run in §5a and §5b trains on the full three-way v3 mix (`ICLR_retraining_v3_mix3`,
377,295,364 tracks).** This was a deliberate choice; the reasoning matters, because the obvious
alternative — train the ablations on the uniform-muon store alone (181.5 M tracks) since only
*relative* performance matters — looks cheaper and is not.

- **It is not cheaper.** The budget is fixed in **optimiser steps**, so a smaller store costs
  exactly the same wall-clock; it only changes how many times each track is seen
  (921,600 × 2048 = 1.89 B presentations = 5.0 passes over mix3, or 10.4 over uniform-only).
  There is no compute to be bought back here.
- **It keeps every test row in-domain.** The uniform gun is flat in pT over 1–110 GeV, so only
  ~0.9 % of its tracks sit in 1–2 GeV; the log-pT gun is what supplies low-pT statistics and
  ttbar supplies hadrons. Train uniform-only and the 2 GeV row (and the ttbar row) stop measuring
  "how close is this encoder to the KF" and start measuring "how well does it extrapolate off its
  training distribution" — a noisier quantity that can *amplify* architecture differences which
  would not survive in the shipped setting.
- **It matches the model the paper ships**, which is trained on mix3. A conclusion drawn on a
  different mixture invites exactly one reviewer question we would rather not answer.

The one genuine advantage of uniform-only — a homogeneous validation metric instead of one pooled
over samples with 2.5–8× different intrinsic resolution — does not affect *selection*, because the
same pooled metric is applied identically to every architecture and the offset cancels. Say in the
report that pooled-val GM₅ is a selection statistic only and is not comparable to the per-dataset
test numbers.

**Run 12 settles it empirically rather than by argument:** the SSM + seed, matched budget,
uniform-muon store only. One run, ≈ 6 GPU-h. If its per-dataset ratios land on top of the mix3
SSM run, the store choice is irrelevant and we can say so in one sentence; if they do not, we have
measured the size of the effect instead of assuming it.

**These ablations are a pre-training-stage comparison only** (§0): stage-1 recipe, Lion, OneCycle,
bs 2048, no Muon-hybrid fine-tune for anybody. The fine-tune is a tail-cleaning polish that moved
q/p by 1–2 % in the campaign (CLAUDE.md §4.21) and it would cost a second run per arm; leaving it
out is the right call, and the report must state that the absolute ratios are therefore not the
paper's headline numbers.

### 5d. Budget, schedule and the drop order — the node goes away after 20 Sep

Hard constraint: **the 4-GPU node is available until Sun 20 Sep only.** Plan to be finished
**Sun 20 Sep, 12:00 CEST**, with the report written by 18:00. There is no extension and no second
attempt, so build the schedule backwards from that and keep a real margin.

Rough GPU-hour cost at the estimated step rates (SSM 42, transformer ≈20, GRU ≈30 steps/s —
replace with your measured numbers in the interim report):

| block | GPU-h |
|---|---|
| LR sweeps, two-stage (runs 1–3, 42 jobs) | ≈ 27 |
| 3 × 2 factorial with seeds (runs 4–9, 13 runs) | ≈ 125 |
| forward-only (run 10) + store control (run 12) | ≈ 12 |
| saturation ladder incl. per-size mini-sweeps (run 11) | ≈ 36 |
| evaluation + throughput | ≈ 8 |
| **total** | **≈ 208** |

Against 4 GPUs × ~5.5 days ≈ **530 GPU-h**, so the plan uses under 40 % of the machine and has
room for two full re-runs. Use that margin for seeds and for re-running anything that looks
marginal — not for adding new architectures.

**Checkpoints I expect:**
- **Tue 15 Sep, ~09:00** — interim report: staging done, eval-pipeline regression test passed,
  measured steps/s per architecture, LR sweeps finished with the selected LRs and `lr_sweep.pdf`.
- **Thu 17 Sep** — runs 4–9 finished, the 3 × 2 table with seed spreads.
- **Sat 19 Sep** — ladder + forward-only + throughput done.
- **Sun 20 Sep, 18:00** — final report.

**Scheduling: treat the 4 GPUs as one job queue, not as four pinned lanes.**
Do **not** assign one architecture per GPU — the transformer is the slowest per step, so a pinned
layout leaves three GPUs idle waiting for it. Put every run on a single work list and have four
workers pull from it, longest-estimated-first:

```bash
# jobs.txt: one full launch command per line (no trailing &), longest first.
# Four workers, one per GPU, each pulling the next line under a lock.
run_worker () {                      # $1 = gpu id
  while :; do
    job=$(flock /tmp/abl.lock -c 'head -1 jobs.txt; sed -i 1d jobs.txt')
    [ -z "$job" ] && break
    echo "[gpu $1] $(date +%H:%M) $job" >> queue.log
    CUDA_VISIBLE_DEVICES=$1 bash -c "$job" >> queue.log 2>&1
  done
}
for g in 0 1 2 3; do run_worker $g & done; wait
```

This matters most in Phase A: 33 stage-1 sweep points + 9 refine points is 42 short jobs, and as a
queue they finish in ≈ 9 h of wall time instead of being serialised behind the slowest
architecture. Same for Phase B's 14 matched runs. Order the list longest-first (transformer runs
before SSM runs) so the tail of the queue is short jobs, not a 13-hour one.

Phase A (≈ 9 h): all LR-sweep jobs in the queue; in parallel, on whichever GPU frees first, the
staging + mix3 + eval-farm build and the eval-pipeline regression test. **Interim report at the
end of Phase A.**
Phase B (≈ 1.5 days): the 13 factorial runs + the store control, queued longest-first.
Phase C: ladder mini-sweeps → ladder runs → forward-only → throughput; evaluation folded in.

**If run 1 selects a peak LR different from the production 5.0e-5, report immediately and do not
start Phase B until I answer** — the whole comparison then has to use the sweep-selected SSM LR.

## 6. Evaluation

**Regression-test the eval pipeline first.** Evaluate the released paper checkpoint
`eval_plots/sweep7/R2LnoconvFT/ckpts/model.ckpt` (md5 `cabbf59091c12e8981e7ea5fc2bb2315`) on the
v3 farm and reproduce this table (it is the *v3* evaluation of that checkpoint — **not** the
paper's Table 1, which is on the buggy v2 data and must not be used as the target here):

```bash
bash scripts/04b_eval_ckpt_deploy.sh eval_plots/sweep7/R2LnoconvFT model.ckpt \
     <out_dir> /scratch/colliderml/ICLR_eval_v3 <gpu>
```

|η| ≤ 2, iter-3σ-clipped RMSE, **SSM / truth-KF** (the header in `rms_summary.txt` says "CKF" —
that label is a known cosmetic bug; the reference column is the truth-KF):

| dataset | N | d0 µm | z0 µm | φ mrad | θ mrad | q/p GeV⁻¹ |
|---|---|---|---|---|---|---|
| µ 2 GeV | 63,869 | 33.0 / 33.4 | 51.3 / 51.1 | 0.943 / 0.958 | 0.537 / 0.535 | 2.28e-3 / 2.27e-3 |
| µ 10 GeV | 65,169 | 13.9 / 14.0 | 16.7 / 16.1 | 0.263 / 0.271 | 0.138 / 0.135 | 4.90e-4 / 4.84e-4 |
| µ 50 GeV | 64,580 | 8.47 / 8.49 | 8.46 / 8.56 | 0.090 / 0.096 | 0.042 / 0.044 | 1.45e-4 / 1.36e-4 |
| µ 100 GeV | 64,936 | 7.63 / 7.66 | 7.67 / 7.63 | 0.068 / 0.074 | 0.031 / 0.033 | 1.06e-4 / 1.02e-4 |
| µ uniform | 6,287,826 | 9.15 / 9.19 | 9.43 / 9.49 | 0.103 / 0.109 | 0.052 / 0.054 | 1.68e-4 / 1.61e-4 |

Reproduce to ≲ 1 % or **stop and report** — the pipeline is wrong on this node and nothing else
matters. (This checkpoint was trained on v2; it is a pipeline test, not a performance target.
A v3-retrained checkpoint of the same architecture lands in `eval_plots/` around 2026-09-15; if it
exists when you get here, quote it too but keep the regression test above as the gate.)

**For every matched-budget run**, on `single_muon_{2,10,50}GeV` + `single_muon_uniform` +
`ttbar_new_pt1` (100 GeV as a diagnostic):
- per-parameter **iterative-3σ-clipped RMS ratio to the truth-KF** and its GM₅;
- the **un-clipped (pre-clip) ratios** and the clipped fraction — the campaign's experience is
  that the clipped core saturates and the *tails* are what separate models;
- `TRK_ABS_ETA_MAX=2`; matching rule = the repo's double-matched subset (unchanged);
- for the uniform sample, report both the full-range row and the `TRK_PT_MAX=70` row (the paper's
  "µ, 1–70 GeV" row) — headline comparison uses the full range, identically for all architectures.

Train with the standard path; evaluate with `04b_eval_ckpt_deploy.sh` (deployment settings) so
every architecture is scored by the same script. Note in the report that the deployment eval path
applies TF32 + fused kernels + the GPU seed **to the SSM only** (they do not exist for the other
encoders) and that it was measured to be equal to strict fp32 to displayed precision.

---

## 7. Throughput

Measure inference throughput for every architecture with `scripts/bench_infer_flat.py`
(`--gpu-seed --seed-residuals`, batch sizes 2¹¹ … 2¹⁷, uncontended GPU, `--iters 100`), and report:

- transformer and GRU, **each on its best available kernel** (SDPA at fp32 and, separately,
  flash-attn-2/bf16 for the transformer if you can run it — label it, it is not the fp32 number;
  cuDNN fused GRU);
- the paper model on its **stock PyTorch/mamba-ssm path** (`--no-kernel-switches`) **and** on its
  fused Triton deployment path (TF32 + `TRK_SSD_BUCKET16=1` + `TRK_COMPILE_FRONTEND=1` + GPU seed,
  seed dtype float64).

Reference numbers already measured on an H100 NVL for the paper model: stock fp32 607.6 k
tracks/s at 32 k batch, deployed 1.849 M at 32 k and 1.913 M at the 131 k plateau. Reproducing
these is a second, cheap sanity check on the node.

The point of reporting all of them: the paper must be able to say which part of the SSM's
throughput advantage is **architecture** and which is **kernel engineering**. Do not conflate them.

---

## 8. Deliverables

Write everything to `<<RESULTS_DIR>>/ablations_2026-09/`:

- **`report.md`** — (i) one table of all matched-budget runs: encoder, trunk params, total params,
  selected LR, steps, wall-clock, steps/s, GM₅ ± (± = s.d. over the 3 SSM seeds; single runs quote
  that spread as the reference uncertainty), the five clipped ratios, the five un-clipped ratios;
  (ii) a throughput table; (iii) the full sweep table (all 15+ points, including divergences);
  (iv) the **3 × 2 factorial** (runs 4–9) called out separately as its own table — encoders down,
  with/without seed across, GM₅ ± seed spread in each cell, plus the per-encoder ratio
  (without/with) that is the actual measure of how much of the precision the recipe supplies;
  (v) the saturation ladder table; (v) a **Deviations**
  section listing every departure from this protocol, however small; (vii) a **Not done** section.
- **`saturation_ladder.pdf`** — GM₅ vs trunk parameters, one curve per encoder, truth-KF line
  at 1.0, log-x, the 0.649 M paper point marked.
- **`lr_sweep.pdf`** — val GM₅ vs peak LR, one curve per architecture, selected point marked, grid
  edges visible, divergent points marked as such.
- **`runs.csv`** — one row per run and per sweep point with every number above.
- **`configs/`** (resolved config of every run), **`logs/`** (full training logs),
  **`checkpoints/`** (final checkpoint of every matched run), plus the git commit hash,
  `pixi list` / `pip freeze`, `nvidia-smi`, and the mix3 manifest track counts.

Report format for the interim and final messages: what ran, what finished, the numbers, the
deviations, the time used, what is still running.

---

## 9. Repo facts that bite (from CLAUDE.md §8 — all of these have cost days before)

- `train.py` auto-loads `base.yaml` **from the config file's own directory**. A leaf `callbacks:`
  list **replaces** the base list — if you write one, re-add `KernelSwapCallback` (SSM runs) and
  `Checkpoint`. Put new configs in a directory that contains the right `base.yaml`, or copy it.
- Launch pattern (one process per GPU, nohup, logs to disk):
  ```bash
  cd /shared/tracking/ssm-colliderml-track-regression/src/track_regression
  TRK_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=0 nohup pixi run -e default python train.py fit \
    --config config/<...>.yaml --trainer.devices 1 --trainer.max_steps 921600 \
    > ../../launch_logs/ablations/<run>_$(date +%Y%m%d_%H%M).log 2>&1 &
  ```
  `TRK_MATMUL_PRECISION=highest` is **required**: `train.py` defaults to `high` (TF32). The log's
  first line must read `float32_matmul_precision = 'highest' (full IEEE fp32)`.
- Flat stores under DDP need `trainer.use_distributed_sampler: false` (the block sampler shards
  itself). Not relevant for single-GPU bs-2048 runs, but keep it.
- `RegressionPredictionWriter` writes `<ckpt-stem>__test_predictions.h5` **next to the
  checkpoint** — copy the checkpoint elsewhere before evaluating several datasets, or the h5 is
  overwritten. `04b_eval_ckpt_deploy.sh` already does this; hand-rolled eval loops do not.
- Do **not** pass `--data.max_test_tracks` with `fast_rms_eval` (misaligns the reference column).
- Never widen loss norm ranges on a kept head. (Not relevant here — all runs are from scratch.)
- `pixi run python3 -c "…'…'…"` **strips inner single quotes** → SyntaxError. Use the system
  `python3` for inline `-c` guards, or write a script file.
- Comet runs offline into `src/track_regression/logs/comet_offline/<id>/ckpts/`.
- The validation-skip bug (block sampler `__len__`) is fixed; still, confirm you get one
  `[val epoch N] iter-3σ RMSE …` line per validation in every log.

## 10. Rules of engagement

- Do not modify data splits, the evaluation code, the loss, the heads, the features, the schedule
  shape, or the paper model's hyperparameters. New code goes in new files/classes behind config
  switches. **Add, don't edit.**
- Do not delete anything. Do not overwrite existing checkpoints, `eval_plots/` bundles or
  `launch_logs/` entries. Another job of mine is running from this same checkout on another node.
- Run long jobs under `nohup` (or `tmux`); check GPU memory and utilisation after launch; check
  for NaN/inf in the first 500 steps of every run.
- Fix and record every random seed.
- If any encoder cannot complete its LR sweep by the deadline, drop that encoder entirely and say
  so; do not run it at a borrowed LR.
- If you are unsure whether a step is a protocol change, treat it as one: stop and ask.
