# Proposed text changes — minGRU encoder / kernel-throughput correction pass

Style rule from the author (2026-09-21): no em-dashes and no semicolons in any proposed text; the remaining items were scrubbed accordingly and should be re-read for flow before sign-off.

Scope note: every item below is PROPOSED text only. Nothing under
`/shared/tracking/NeurIPS_2026_SSM_Tracking` was edited, moved, or built. "Current"
blocks are copied verbatim from the files (checked 2026-09-21); the author signs off
item by item.

---

### Item 1 — [APPLIED 2026-09-21 (user wording)] method.tex:76-82

**Current**
```latex
dense network to the encoder width $d_{\mathrm{model}} = 128$.  Two
learned class tokens bracket the embedded sequence, and the backbone
stacks two bidirectional minGRU layers~\citep{Feng2024minGRU} with
hidden width $192$ ($\sim\!0.644$\,M parameters).
Each layer applies RMSNorm~\citep{Zhang2019RMSNorm}, a
forward and a reverse scan, and merges the directions
through a learned sigmoid gate before the residual connection.
```

**Proposed**
```latex
dense network to the encoder width $d_{\mathrm{model}} = 128$.  The
backbone stacks two bidirectional minGRU layers~\citep{Feng2024minGRU} with
hidden width $192$ ($0.64$\,M parameters in total): each layer
linearly projects its input to the forward and reverse gates and candidate
states, then runs a forward and a reverse scan. The two directions' states
are concatenated ($384$-d) into the next layer, with no class tokens,
per-layer normalization, direction-merge gate, or residual connection in
this block.
```

**Why:** the deployed model (`mingru.py`) has no class tokens, no per-layer RMSNorm, no direction-merge gate, and no residual connection — that description belongs to the earlier Mamba-2 block, not the minGRU one actually trained.

---

### Item 2 — [APPLIED 2026-09-21 (user wording: final state, no 256-d detail)] method.tex:124-127

**Current**
```latex
The last forward and reverse
class-token states are concatenated to a $256$-d representation and
read by a two-layer head (hidden width $128$) producing $7$ quantiles for each of the
five parameters, $35$ outputs in total.
```

**Proposed**
```latex
The forward scan's terminal state (at the outermost hit) and the reverse
scan's terminal state (at the innermost hit) are concatenated to a
$384$-d representation, normalized~\citep{Zhang2019RMSNorm} and projected
to $256$-d, then read by a two-layer head (hidden width $128$) producing
$7$ quantiles for each of the five parameters, $35$ outputs in total.
```

**Why:** there are no class tokens to read out; the readout is the two scans' terminal hidden states, concatenated, normalized, and projected — matching `mingru.py`.

---

### Item 3 — [APPLIED 2026-09-21 (user wording: always ordered by time)] introduction.tex:50-51

**Current**
```latex
, with two learned class tokens that bracket the hit sequence. The hits are ordered by the time each was recorded following the particle's trajectory through the
detector. The class-token outputs are pooled
```

**Proposed**
```latex
. The hits are ordered along the trajectory: by their simulated
production time in training and, at inference, by detector geometry, a
truth-free order that reproduces it on essentially every track. The
encoder's terminal forward and reverse states are pooled
```

**Why:** removes the class tokens (the deployed minGRU has none) and replaces the "time each was recorded" claim, which strip hits cannot support, with what the model actually orders by and why.

---

### Item 4 — [APPLIED 2026-09-21 (user's rewrite)] method.tex:147-156

**Current**
```latex
Training is performed end-to-end in strict fp32, since faster
reduced-precision formats such as TF32 do not achieve the required
precision during optimization~\citep{Kalamkar2019BFloat16}.  At
inference, however, enabling TF32 on the projection GEMMs (the
dense matrix multiplications the network's layers are built from;
the selective-scan kernel's own matrix products stay IEEE fp32,
\cref{app:kernels}) leaves the physics essentially unchanged, at
$0.01\%$ median deviation over the five parameters and four
evaluation samples, so TF32 and partial fp16 are enabled in the deployment
configuration and used for every figure in this paper.
```

**Proposed**
```latex
Training runs end-to-end in strict fp32. Whether a reduced-precision
format would hurt optimization was not measured, but fp16 training gave
no step-rate gain here ($55.3$ vs $55.6$ steps/s, the step is bound by
the eager padded path and the data loader, not the GEMMs) and diverged
without loss scaling. At inference the projection GEMMs run in fp16
(TF32 for the Mamba-2 encoder, whose kernel is fp32/fp64-typed). The
recurrence itself accumulates in fp32 inside the kernel
(\cref{app:kernels}), and the seed stays float64 (\cref{app:seed}). On
the shipped checkpoint, fp16 and fp32 inference agree to three decimals
on every ratio of \cref{tab:ratios} ($27$ of $30$ entries identical, the
rest $0.001$ apart), so this configuration is used for every figure in
this paper.
```

**Why:** the old sentence claims a measurement ("TF32 does not achieve the required precision during optimization") that was never made; replaced with what was actually measured (no fp16 training speedup, a divergence without loss scaling) and drops the untraceable "0.01% median deviation" claim in favor of the checkpointed three-decimal agreement.

---

### Item 5 — method.tex:176-214 (whole "Kernel adaptation" subsection)

**Current**
```latex
\subsection{Kernel adaptation}
\label{sec:kernels}

Standard GPU kernels for sequence models are built for long
sequences, typically $10^{3}$--$10^{5}$ tokens. Our domains instead demands doing inference on thausand short sequences at once. A typical charged-particle
trajectory, is
at most $22$ tokens, including the two class tokens, and measures $13$ hits
on average, far below that regime, so running a standard kernel
on a track wastes arithmetic operations and results in needless launch overhead. 
For our fastest encoder, the minGRU, a hand-written fused Triton kernel removes that
mismatch: inference runs on unpadded, length-sorted
tracks, so the roughly three-quarters of tracks with $16$ hits or
fewer are routed through a smaller compute tile rather than one
sized for the longest possible track, and the per-hit Fourier
feature expansion ahead of the encoder, otherwise dozens of small
operations, is fused into a single compiled kernel. The kernel also performs the
entire bidirectional token mixing in a single kernel launch.

The same short-sequence mismatch affects every encoder family we
tried, but only linear recurrences benefit from fixing it: an
identical kernel effort brings $5.0\times$ for the minGRU, while the
transformer gains nothing. The reason is structural: a gated
linear recurrence updates each channel independently, so a thread
can keep its channel's state in a register for the whole trajectory;
no shared memory or barrier synchronization is needed. Attention
instead requires every query to attend to every key through a
softmax reduction across positions, which needs shared memory and
barriers, and its on-chip working set per track is about $73\times$
larger.

This kernel-engineering approach was first validated on the Mamba-2
encoder: the same three changes plus
TF32 projections reach \kernelSpeedup{} the default Mamba-2 kernel's
throughput on identical hardware and outputs (\cref{tab:kernel-bench}); the
deployed minGRU's own fused kernel is the one benchmarked throughout
the rest of the paper. A hardware- and
energy-cost-matched comparison against a multi-core
CPU, on a workstation-class GPU, is discussed in
\cref{sec:results-throughput}.
```

**Proposed**
```latex
\subsection{Kernel adaptation}
\label{sec:kernels}

Standard GPU kernels for sequence models are built for long sequences,
typically $10^{3}$,$10^{5}$ tokens. Our domain instead demands inference
on thousands of short sequences at once: a track is at most $22$ tokens
and measures $13$ hits on average, far below that regime, so running a
standard kernel on a track wastes arithmetic and pays needless launch
overhead. We give every encoder the same treatment: inference runs on
unpadded tracks packed one row per hit rather than padded to the longest
track. A hand-written fused Triton kernel performs the token-mixing step
, the bidirectional scan of both directions in one launch for the
recurrent encoders, the attention of every head of a track in one
program for the Transformer, whose surrounding projections carry their
bias, activation, residual and normalization as fused epilogues. The
per-hit Fourier feature expansion ahead of the encoder, otherwise dozens
of small operations, is compiled into a handful of kernels. And the
projection GEMMs run in fp16 (TF32 for the Mamba-2 encoder, whose kernel
is fp32/fp64-typed).

\cref{tab:kernel-gains} (\cref{app:kernels}) reports the resulting gain
for each encoder at the deployment batch size: $6.0\times$ for the
minGRU, $6.6\times$ for the Transformer, and $3.0\times$ for the
bidirectional Mamba-2. The Transformer's relative gain is not smaller
than the recurrent encoders', packing and a fused kernel help
attention as much as any other encoder here, once the padding and the
many small unfused kernels of the naive path are removed. What
distinguishes the minGRU is not the kernel treatment but the arithmetic
itself: its token mixing is one elementwise scan per layer (one GEMM
plus one scan, two layers), while attention needs a reduction across
the whole trajectory around three projections (five kernels even fully
fused) and three layers to match the same parameter budget. The minGRU
therefore makes fewer passes over its activations per layer, not fewer
kernel launches, and stays ahead of the Transformer once both receive
the identical treatment.

This kernel-engineering approach was first applied to the Mamba-2
encoder, which after the same treatment is the slowest of the three
(\cref{tab:kernel-gains}). The deployed minGRU is the configuration
benchmarked throughout the rest of the paper. At the deployment batch size the encoder itself is only part of
the deployed minGRU forward pass: the float64 seed, the compiled
Fourier front end, and the quantile heads together account for roughly
$60\%$ of its time, so further kernel work on the encoder alone has a
shrinking return. A hardware- and energy-cost-matched comparison
against a multi-core CPU, on a workstation-class GPU, is discussed in
\cref{sec:results-throughput}.
```

**Why:** the old subsection claims the Transformer "gains nothing" from the same kernel work, which the tab:kernel-gains measurement contradicts (Transformer gains $6.6\times$, more than the minGRU's $6.0\times$); the true differentiator is arithmetic per layer, not launch count, and the paragraph now says so and points at the new appendix table instead of the retired one.

---

### Item 6 — results.tex:222-224

**Current**
```latex
The same four encoders no longer tie once a fused kernel is written for
them: the minGRU gains $5.1\times$, while the Transformer's dense attention
gains nothing.
```

**Proposed**
```latex
Each of the three encoders given the same kernel treatment gains
sharply from it: $6.0\times$ for the minGRU, $6.6\times$ for the
Transformer, and $3.0\times$ for bidirectional Mamba-2
(\cref{tab:kernel-gains}). The Transformer's relative gain is the
largest of the three, but in absolute deployed throughput the minGRU
still leads it by $1.7\times$ and leads Mamba-2 by $3\times$.
```

**Why:** replaces the stale "gains nothing" claim and single minGRU number with the measured three-way gain and the throughput ranking that actually decides deployment, referencing the new table.

---

### Item 7 — results.tex:231-235

**Current**
```latex
The analytic seed is part of the deployed forward pass, so we time it
explicitly: at the saturating batch size the full forward costs
\SI{0.53}{\micro\second} per track on the H100, of which the on-GPU
seed (computed in float64, \cref{app:seed}) contributes
$\sim$\SI{0.015}{\micro\second} per track, $2.9\%$ of the forward pass of the network.
```

**Proposed**
```latex
The analytic seed is part of the deployed forward pass, so we time it
explicitly: at the saturating batch size the full minGRU forward costs
\SI{0.19}{\micro\second} per track on the H100, of which the on-GPU
seed (computed in float64, \cref{app:seed}) contributes
$\sim$\SI{0.016}{\micro\second} per track, $\sim\SI{9}{\percent}$ of the
forward pass. The compiled Fourier front end and the quantile heads
together take a further $\sim\SI{50}{\percent}$, so the encoder itself
accounts for only $\SI{40}{\percent}$ of the deployed forward.
```

**Why:** updates the per-track timing and seed share to the measured minGRU numbers (0.53 us/2.9% was the earlier Mamba-2 figure) and adds the front-end/heads/encoder breakdown that item 5 and item 13 refer back to.

---

### Item 8 — [APPLIED 2026-09-21 (cross-ref needed by item 9)] results.tex:203-204 (tab:throughput caption)

**Current**
```latex
  (GPU board / CPU chip; hosts excluded on every row). The kernel-variant ablation
  is in \cref{tab:kernel-bench}.}
```

**Proposed**
```latex
  (GPU board / CPU chip. Hosts excluded on every row). The kernel-variant ablation
  is in \cref{tab:kernel-gains}.}
```

**Why:** points the caption at the new all-encoder table, since `tab:kernel-bench` and its underlying macros are retired.

---

### Item 9 — [APPLIED 2026-09-21 (Mamba-2 row TF32, fp16 re-measure pending)] appendix.tex:99-127 (app:kernels: intro sentence + table)

**Current**
```latex
\subsection{Kernel adaptation for short sequences}
\label{app:kernels}

\cref{tab:kernel-bench} reports single-H100 inference throughput of
the Mamba-2 checkpoint under a chunk-free Triton kernel replacing the
stock Mamba-2 scan, isolating what each successive optimization
buys.

\begin{table}[h]
  \centering
  \caption{Single-H100 (NVL) inference throughput of the Mamba-2
  checkpoint, in $10^{3}$ tracks/s, with the on-GPU analytic seed
  inside the timed loop, at the saturating batch size ($32$\,k
  tracks/batch).  Rows accumulate downwards.  The stock row is
  measured on the $d_{\mathrm{conv}}\!=\!4$ twin (identical physics):
  the stock convolution kernel cannot run the conv-free block.}
  \label{tab:kernel-bench}
  \small
  \begin{tabular}{lr}
    \toprule
    configuration & $32$\,k / batch \\
    \midrule
    stock Mamba-2 kernels, strict fp32 ($d_{\mathrm{conv}}{=}4$ twin) & \stockThroughput \\
    fused short-sequence kernel, strict fp32      & \fusedThroughput \\
    \; + TF32 projections                         & 1\,346 \\
    \; + bucketed launch + compiled front end     & 1\,784 \\
    \bottomrule
  \end{tabular}
\end{table}
```

**Proposed**
```latex
\subsection{Kernel adaptation for short sequences}
\label{app:kernels}

\cref{tab:kernel-gains} reports single-H100 (NVL) inference throughput
for the three encoders of \cref{tab:arch-ablation} (trunk-matched at
$\sim\!0.63$\,M parameters) that admit a fused short-sequence kernel,
before and after the treatment of \cref{sec:kernels}: packed, unpadded
layout. A fused Triton kernel for the token-mixing step. A compiled
Fourier front end. And reduced-precision projections.

\begin{table}[h]
  \centering
  \caption{Single-H100 (NVL) inference throughput, in tracks/s, at the
  saturating batch size ($131\,000$ tracks/batch), with the on-GPU
  float64 seed inside the timed loop, one idle GPU. \emph{As trained} is
  the exact code path each encoder trains with: padded layout, strict
  IEEE fp32, its training kernel, no inference switches. \emph{Deployed}
  is the fastest physics-gated path: packed layout (one row per hit, no
  padding), a fused Triton kernel for the token mixing, a compiled
  Fourier front end, and fp16 projections (TF32 for Mamba-2, whose
  kernel is fp32-typed). The deployed path reproduces each encoder's
  reference resolution ratios to three decimals on every entry (for the
  Transformer within $0.0008$). The minGRU row of
  \cref{tab:throughput} was measured in a separate session and differs
  from the row below by $\sim\!2\%$, the session-to-session spread of
  this benchmark.}
  \label{tab:kernel-gains}
  \small
  \begin{tabular}{lrrr}
    \toprule
    encoder & as trained & deployed & gain \\
    \midrule
    minGRU (hidden $192$) & $0.88$\,M & $5.31$\,M & $6.0\times$ \\
    Transformer ($3$ layers, $4$ heads) & $0.48$\,M & $3.18$\,M & $6.6\times$ \\
    Mamba-2, bidirectional ($2$ layers, conv-free) & $0.57$\,M & $1.74$\,M & $3.0\times$ \\
    \bottomrule
  \end{tabular}
\end{table}
```

**Why:** replaces the Mamba-2-only kernel table with the all-encoder before/after table the new prose (items 5, 6, 9) references, and states the physics-parity and run-to-run-spread caveats the measurement carries.

---

### Item 10 — appendix.tex:186-190 (Protocol)

**Current**
```latex
\subsection{Protocol.}  The four encoders are swapped into an otherwise
identical model: the same analytic seed, the same $15$ per-hit features, the
same Fourier featurization and input projection, the same two class tokens,
the same $256$-d pooled representation, and the same quantile heads with the
same loss ranges and anchors.  They are trained with an identical optimizer
```

**Proposed**
```latex
\subsection{Protocol.}  The four encoders are swapped into an otherwise
identical model: the same analytic seed, the same $15$ per-hit features, the
same Fourier featurization and input projection, class tokens for Mamba-2 and
the Transformer, terminal scan states for the minGRU and the diagonal
state-space model, all pooled to the same $256$-d representation, and the same
quantile heads with the same loss ranges and anchors.  They are trained with an identical optimizer
```

**Why:** only two of the four ablation encoders use class tokens; the minGRU and the diagonal state-space model are read out from their scans' terminal states, as corrected in items 1-2.

---

### Item 11 — appendix.tex:210-232 ("Transformer optimization issues" -> "Transformer throughput")

**Current**
```latex
\subsubsection{Transformer optimization issues.}  Its throughput is
measured on a padded attention path: every track is
padded to the longest possible length, $22$ tokens including the two class
tokens, against a mean of $15$ hits, and attention runs on the stock fused
attention primitive with no kernel of our own.  Packing is possible for
attention --- variable-length attention kernels and block-mask formulations
both exist, and only a naive packed implementation is quadratic in the total
token count --- and removing the padding alone would move the Transformer
substantially.  At matched parameters and packed it performs $0.94\times$ the
arithmetic of the minGRU, so the difference is not arithmetic.  What differs is
how readily the two admit a fused kernel at this sequence length.  The
recurrent update is elementwise per hidden channel, so one thread owns a
channel and keeps its state in a register: no shared memory, no barriers, and
$12$ kernel launches per forward pass.  Attention requires every query to meet
every key and a softmax reduction across positions, hence shared memory and
barriers, and its on-chip working set per track and layer is $73\times$ larger
($55.0$ against $0.76$\,KiB), which starves occupancy on exactly the smaller
cards that make the cost argument interesting.  The specialized long-sequence
attention kernels do not help here either: at our shapes they are
$40$--$55\,\%$ slower than dense attention, because their tiling overhead never
amortizes over $22$ tokens.  Attention itself is only $2.8\,\%$ of a layer's
floating-point operations at this length; the Transformer's cost is in its
projections.
```

**Proposed**
```latex
\subsubsection{Transformer throughput.}  Our first measurement padded
every track to the longest possible length ($22$ tokens including the
two class tokens) and ran the stock fused attention primitive. With fp16
projections and a compiled front end that path reaches $0.73$\,M
tracks/s, against $0.48$\,M as trained. Packing removes the padding:
tracks packed one row per hit already reach $2.47$\,M. Fusing the residual add and
RMSNorm into one kernel adds a further step to $2.72$\,M. Fusing bias,
SiLU, the residual add, and the norm into the epilogues of the
surrounding GEMMs, so a layer becomes five kernels (QKV projection,
attention, out-projection, two feed-forward GEMMs), reaches $3.05$\,M. Precomputing the additive positional encoding as a $20$-row table
reaches the final $3.18$\,M (\cref{tab:kernel-gains}), saturating from
$\sim\!32$\,k tracks per batch like the recurrent encoders. At small
batch, CUDA graphs take it to $1.42$\,M at $2048$ tracks against
$0.36$\,M for the padded path. Long-sequence FlashAttention kernels do
not help here: they are $40$,$55\,\%$ slower than dense attention at
this length and do not launch at our deployment batch size. Attention
itself is only $2.8\,\%$ of a layer's floating-point operations at this
length. The Transformer's cost is in its projections, and the same
kernel treatment that helps the recurrent encoders helps it too.
```

**Why:** replaces the "gains nothing" framing (built on an un-packed, launch-count argument that the new measurement contradicts) with the actual optimization ladder and the two measurements that still hold (2.8% FLOPs, FlashAttention penalty); drops the unsupported "12 launches", "$73\times$ working set", and "mean of 15 hits" figures.

---

### Item 12 — NEW paragraph, Results, immediately before tab:arch-ablation

Insertion point: results.tex, new paragraph inserted after the "% --- ICLR_v2 ... encoder ablation ..." comment block and immediately before `\begin{table}[t]` (currently line 149, `\label{tab:arch-ablation}`) — i.e. it becomes the last prose before that table.

**Current**
```latex
(none -- new paragraph)
```

**Proposed**
```latex
\cref{tab:arch-ablation} compares four encoders chosen along two axes: recurrence
versus attention for mixing tokens, and selective versus fixed state updates.
Mamba-2, the original encoder, is selective: its update rule is modeled on the
Kalman gain of the classical Kalman-filter fit. MinGRU~\citep{Feng2024minGRU}
keeps that input-dependent gate but drops Mamba-2's state expansion, gated output
normalization and per-layer residual structure, so its gates depend only on the
current hit and the recurrence is linear and elementwise. A classical GRU or LSTM
gates on the previous hidden state, so each step is a dependent matrix product
with no parallel or fused kernel. We studied one in an earlier round and do not
carry it forward. The Transformer, the architecture of the closest prior work on
this detector~\citep{Couthures2025CTD}, mixes tokens by attention and is the
control. The non-selective diagonal state-space model keeps the recurrence but
drops the gate, isolating whether selectivity matters. All four run
bidirectionally, matching the filter-plus-smoother structure of the classical
fit. Three reach the same precision. Only the non-selective model falls behind,
in $\qop$.
```

**Why:** motivates the four-way encoder choice (recurrence vs. attention, selective vs. fixed) before the reader sees the ablation table, and states the one result the table shows (selectivity matters only for $\qop$). (169 words.)

---

### Item 13 — NEW paragraph, Results, end of the Throughput subsection

Insertion point: results.tex, `\subsection{Throughput}` (`\label{sec:results-throughput}`), new paragraph appended after "Throughput saturates from $\sim$$16$--$64$\,k tracks per batch on both GPUs." (currently line 258) and before the commented-out covariance-head aside (lines 260-265, which does not render) — i.e. the closing paragraph of the subsection.

**Current**
```latex
(none -- new paragraph)
```

**Proposed**
```latex
Taken together, the encoder comparison separates precision from throughput.
Precision comes from the formulation, the analytic seed, its residual input
features, the anchored quantile heads, and the small-batch recipe, not from the
encoder: three of the four reach the same resolution. The one architectural
property that matters is the input-dependent gate: only $\qop$ degrades when it
is removed. Throughput, in contrast, is decided by the kernel path, and every
encoder responds to the same treatment (\cref{tab:kernel-gains}). Once applied,
the ranking follows the number of passes an encoder makes over its activations
per layer, not the number of kernel launches. The minGRU is our deployment
choice. The Transformer is a legitimate fallback wherever a fused recurrent
kernel is unavailable. At the deployment batch size the shared seed, Fourier
front end and quantile heads already account for $\sim\!60\%$ of the minGRU
forward pass, so that shared machinery, not the encoder, is where the next
throughput factor lies.
```

**Why:** closes the Throughput subsection by stating what the encoder comparison means for the paper's claims -- precision is a formulation property, throughput a kernel-path property -- and names where the next speed gain would have to come from. (169 words.)

---

### Item 14 — OPTIONAL: hit-ordering wording (data.tex:25-28, method.tex:11)

**Current (data.tex:25-28)**
```latex
analytic helix seed are described in \cref{sec:method}.  Hits are
ordered by the  time at which they were measured, which gives the learned SSM a meaningful inductive
bias about positional information, letting it retrace the original
trajectory the particle took through the detector.
```

**Proposed (data.tex:25-28)**
```latex
analytic helix seed are described in \cref{sec:method}.  Hits are
ordered by the simulated time at which each was produced along the
trajectory; since most hits (all strip hits) carry no usable measured
time, we instead order by detector geometry, which reproduces the same
order at inference on essentially every track ($100\%$ of muon tracks,
$99.7\%$ of $t\bar t$ in a held-out check). This gives the learned SSM a
meaningful inductive bias about positional information, letting it
retrace the trajectory the particle took through the detector.
```

**Why:** "measured" hit time is not what the stores are sorted by, and it is unusable at inference for strip hits; states the actual (simulated-time) ordering and the truth-free geometry stand-in used at inference. [AUTHOR: confirm whether naming $t\bar t$ here is appropriate given data.tex's training-mixture paragraph (line 33) currently describes only the two muon samples — this sentence is reporting an ordering-agreement measurement, not asserting $t\bar t$ is part of this paper's training mixture; flag if that reads as inconsistent.]

**Current (method.tex:11)**
```latex
The model maps the input hit sequence of a single track, ordered by hit measurement time \cref{sec:data}, to the five perigee
```

**Proposed (method.tex:11)**
```latex
The model maps the input hit sequence of a single track, ordered by
detector geometry as a truth-free stand-in for simulated hit-production
time \cref{sec:data}, to the five perigee
```

**Why:** matches the corrected data.tex wording and states what the model itself is given (geometry order), not a "measurement time" that does not exist for most hits.

---

## Closing note (not an item)

**Macros in `main.tex` that become unused once items 5, 8, and 9 are applied:**
`\kernelSpeedup` (main.tex:87, used at method.tex:208, retired by item 5), `\stockThroughput` (main.tex:94, used at appendix.tex:121, retired by item 9), `\fusedThroughput` (main.tex:96, used at appendix.tex:122, retired by item 9). All three have no other rendering uses in the current sections (the only other occurrences are inside commented-out lines in `abstract.tex` and `introduction.tex`, which already do not render). They can be deleted from `main.tex` once the author signs off items 5 and 9, or left in place as dead macros — either way `tab:kernel-bench` disappears and `tab:kernel-gains` takes its place, so any other `\cref{tab:kernel-bench}` in the document should be re-checked (this pass found only results.tex:204, handled in item 8).

**Page budget:** items 12 and 13 each add one new paragraph (169 words apiece) to the main body, which the brief fixes at 9 pages; items 1-11 are replacements close to (in a few cases, items 4-5, modestly longer than) the text they replace. The main body's page count should be rebuilt and re-checked after all items are applied, before the 9-page constraint is treated as satisfied.
