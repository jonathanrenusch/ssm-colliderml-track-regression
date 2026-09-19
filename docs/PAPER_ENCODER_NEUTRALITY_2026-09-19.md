# If the encoder changes, what in the paper has to change

Triage of all 44 `Mamba` occurrences in `/shared/tracking/NeurIPS_2026_SSM_Tracking`.
**No tex was edited** — this is a list for you to work from. Grouped by whether
the occurrence is a claim about *the model we ship*, a claim about *why the
architecture suits the task*, or a *baseline* we compare against.

## A. Claims about the shipped model — must change if the encoder changes

| file:line | current | note |
|---|---|---|
| `abstract.tex:32` | "an optimized SSM inspired by a bidirectional Mamba-2 encoder" | the load-bearing one |
| `method.tex:7` | section title "A seed-guided bidirectional Mamba-2 trajectory parameter estimator" | |
| `method.tex:16` | figure caption "Architecture of the seed-guided bidirectional Mamba-2 model" | |
| `method.tex:72-73` | "stacks two bidirectional Mamba-2 layers ... $d_{\mathrm{state}} = 64$" | the concrete spec; rewrite to whichever block ships |
| `introduction.tex:44-45` | "we propose a bidirectional Mamba-2 state-space encoder ... inspired by Vision Mamba" | |
| `discussion.tex:10` | "a $0.65$M-parameter bidirectional Mamba-2" | |
| `conclusion.tex:7-8`, `conclusion.tex:12` | "seed-guided bidirectional Mamba-2 state-space model" (x3) | |
| `results.tex:32`, `results.tex:170`, `results.tex:181` | "the seed-guided bidirectional Mamba-2 trajectory estimator" | incl. the throughput figure caption |

A neutral phrase that survives any of the arms: **"seed-guided bidirectional
recurrent encoder"** (or "...linear-recurrent encoder"). Every arm we have
tested — Mamba-2, minGRU, minLSTM, a non-selective diagonal SSM, a complex-decay
LRU — is a bidirectional linear recurrence; the transformer is the only one that
is not, and it is not a candidate to ship.

Your earlier title idea fits this exactly:
*"Recurrent models for high-throughput, high-precision charged-particle
trajectory reconstruction at the Large Hadron Collider."*
It is also more honest about what the study now shows: the recipe carries the
result, and the backbone is a swappable component.

## B. Baselines and kernel work — stay as they are

`method.tex:176-197` and `appendix.tex:299-481` compare our kernel against the
**stock `mamba_ssm` implementation**. That is a baseline, not a claim about our
model, and it stays correct whatever we ship — though if the shipped encoder is
no longer Mamba-2, `\kernelSpeedup` needs re-deriving against that encoder's own
reference implementation, and the comparison should be relabelled as what it is
(a reference point from the SSM literature, not our predecessor).
`appendix.tex:598` (removing the depthwise causal conv) is a fact about the
Mamba-2 twin and stays if that twin stays in the ablation table.

## C. The motivation — conditional on the non-selective result, and this is the scientific one

`introduction.tex:40-42` and `method.tex:81-104` build the paper's central
analogy: **Mamba-2's *selective* (input-dependent) state update has structurally
the same form as the Kalman gain**, with $\Delta_t$, $B_t$, $C_t$ learned rather
than derived. `eq:mamba-update` is that argument in symbols.

If the non-selective diagonal SSM (running, parameter-matched) reaches the same
GM5, **that motivation is overstated as written** — selectivity would then be
demonstrably unnecessary for this task, and a claim that it is what makes the
architecture Kalman-like would not survive a referee who reads our own ablation
table.

The honest replacement is *stronger*, not weaker: the Kalman filter's state
update is a **linear recurrence with a gain**, and any bidirectional linear
recurrence can represent it — forward pass plus backward pass is exactly the
filter-plus-smoother structure (the depth argument already in 3.6). Selectivity
is then an orthogonal question that our L = 20 ablation answers empirically, and
the literature supports: input-dependent decay is a long-context mechanism
(Block-Biased Mamba, arXiv:2505.09022; arXiv:2609.16540), and **no published
ablation exists below L ~ 32**, so this is a contribution rather than a caveat.

Decision point: wait for `V2_diagssm_25ep` (parameter-matched, other node) before
touching §C. If it *lags*, the current selectivity argument stands as written and
only §A needs neutral wording.

## What is NOT affected

Physics numbers, the truth-KF reference rule, the $|\eta| \le 2$ and $p_T \le 70$
cuts, the seed (fp64, three pixel hits, 3 T), the quantile heads, the data
sections, and every figure in the eta2 bundle. The ablation changes which
encoder sits inside the recipe, not the recipe or its evaluation.
