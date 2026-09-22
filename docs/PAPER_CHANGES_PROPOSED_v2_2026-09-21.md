# Proposed text changes, round 2: kernel-gains rewrite and the encoder comparison

STATUS 2026-09-22 (after merging the senior author's "first wave"): item 5 APPLIED (on the merged text, "every encoder we tested" instead of the table reference); item 7 APPLIED (9 %, relative to the H100); items 10 and 11 MOOT (the protocol and transformer subsections were dropped at the senior author's request; only the dot plot fig:encoder-ablation stays, in a slim "Encoder comparison" subsection); items 12+13 WITHDRAWN (replaced by a one-sentence rationale, pending sign-off); item 15 MOOT except the new `ChipCost` entry (added). Item 6 still open (optional main-text sentence pointing at tab:kernel-gains).

Scope note: every item below is PROPOSED text only. Nothing under
`/shared/tracking/NeurIPS_2026_SSM_Tracking` was edited, moved, or built. "Current"
blocks are copied verbatim from the files as they stand today (checked 2026-09-21,
after items 1, 2, 3, 4, 8 and 9 of the previous round were applied). The author
signs off item by item. Numbering continues from the previous round. Items 1-4, 8
and 9 are done and not repeated here.

---

### Item 5 — method.tex:173-211 (whole "Kernel adaptation" subsection)

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
throughput on identical hardware and outputs (\cref{tab:kernel-gains}); the
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

Standard GPU kernels for sequence models target sequences of
$10^{3}$ to $10^{5}$ tokens. A charged-particle trajectory has at most
$20$ hits and $13$ on average. A kernel built for the longer regime
wastes arithmetic on a track and pays launch overhead the track never
needs.

We give every encoder in \cref{tab:arch-ablation} the same
treatment. Tracks run in a packed, unpadded layout with one row per
hit. A fused Triton kernel handles the token mixing for each
encoder. The recurrent encoders get one launch that covers both scan
directions. The Transformer gets one program per track, with bias,
activation, residual addition and normalization fused into the
surrounding GEMM epilogues. A compiled Fourier front end replaces
the per-hit feature expansion ahead of the encoder. Projections run
in fp16 for all three. This treatment gains $6.0\times$ for the
minGRU, $6.6\times$ for the Transformer, and $3.6\times$ for Mamba-2
(\cref{tab:kernel-gains}, \cref{app:kernels}).

The minGRU still leads the Transformer by $1.7\times$ after this
treatment. Its layer is one GEMM and one scan, and it has two
layers. The Transformer's layer is attention plus three projections,
and it has three layers at the same parameter count. The minGRU's
lead comes from fewer passes over the activations per track. The
number of kernel launches is similar for the two. At the deployment batch size the
shared seed, Fourier front end and quantile heads account for
$60\,\%$ of the deployed minGRU forward. \cref{sec:results-throughput}
reports the resulting throughput on real hardware.
```

**Why:** the old subsection describes a minGRU-only kernel and a Transformer that "gains nothing" (both wrong, see brief.md). The new text states the shared kernel treatment, the measured gains for all three encoders, and the mechanical reason the minGRU still leads, without `\kernelSpeedup`.

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
The four encoders no longer tie once three of them get a fused kernel.
The minGRU gains $6.0\times$, the Transformer $6.6\times$, and
Mamba-2 $3.6\times$ (\cref{tab:kernel-gains}). The minGRU still
leads, at $1.7\times$ the Transformer's throughput and $2.6\times$
Mamba-2's.
```

**Why:** the Transformer and Mamba-2 gains were never measured before this round. "Gains nothing" was wrong. Mamba-2 is now measured at fp16 like the other two (docs/PRECISION_STUDY_2026-09-21.md).

---

### Item 7 — results.tex:231-235 (seed-timing sentence)

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
The analytic seed is part of the deployed forward pass, so we time
it explicitly. At the saturating batch size the full minGRU forward
costs \SI{0.19}{\micro\second} per track on the H100. The on-GPU
seed, computed in float64 (\cref{app:seed}), takes
\SI{0.016}{\micro\second} per track, $9\,\%$ of the total. The
Fourier front end and quantile heads take a further $50\,\%$, and the
encoder itself $40\,\%$.
```

**Why:** the old $0.53$\,\micro\second figure and $2.9\,\%$ seed share are from an earlier, slower kernel path. The current deployed minGRU forward is $0.19$\,\micro\second per track, and the seed's share of it is $9\,\%$, with the rest split between the front end plus heads and the encoder.

---

### Item 10 — appendix.tex:201-205 (Protocol)

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
identical model, with the same analytic seed, the same $15$ per-hit features,
the same Fourier featurization and input projection, class tokens for
Mamba-2 and the Transformer, final scan states for the minGRU and the
diagonal state-space model, the same $256$-d pooled representation for all,
and the same quantile heads with the same loss ranges and anchors.  They are
trained with an identical optimizer
```

**Why:** only Mamba-2 and the Transformer use class tokens. The minGRU and the diagonal state-space model are read out from the final states of their scans, as corrected in items 1-2 of the previous round.

---

### Item 11 — appendix.tex:225-247 ("Transformer optimization issues" to "Transformer throughput")

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
\subsubsection{Transformer throughput.}  The Transformer trains on a
padded attention path, with every track padded to $22$ tokens and
attention run through the stock fused attention primitive. Adding
fp16 and a compiled front end to that padded path reaches $0.73$\,M
tracks/s, against $0.48$\,M as trained. Packing the layout reaches
$2.47$\,M. Fusing the residual addition and the RMSNorm into one
kernel reaches $2.72$\,M. Fusing bias, SiLU, the residual and the
norm into the surrounding GEMM epilogues, five kernels per layer,
reaches $3.05$\,M. Precomputing the positional encoding as a
$20$-row lookup table reaches $3.18$\,M. This path saturates from
about $32$k tracks per batch, as the recurrent encoders do. CUDA
graphs reach $1.42$\,M tracks/s at $2048$ tracks per batch, against
$0.36$\,M for the padded path at the same batch size. Attention
itself is $2.8\,\%$ of a layer's floating-point operations at this
length. The specialized long-sequence attention kernels are
$40$ to $55\,\%$ slower than dense attention here, and they do not
launch at all at $131$k tracks per batch.
```

**Why:** the padded path the old text measured (0.48 M as trained, "gains nothing") was never the deployed Transformer. The new text gives the actual packing and fusion ladder that reaches $3.18$\,M tracks/s, keeps the two measurements that still hold (attention's FLOP share, the long-sequence kernels), and drops the launch-count and working-set claims that came from the abandoned padded-only measurement.

---

### Item 12+13 — NEW subsection "Choice of encoder", Results, before `tab:arch-ablation`

Insertion point: after `fig:rms_pt` and before the `ICLR_v2` comment that
introduces the `tab:arch-ablation` table.

**Current**
```latex
  estimators.  Bands are the analytic \rms{} standard error.  The
  fourth parameter is the polar angle $\theta$.}
  \label{fig:rms_pt}
\end{figure}

% --- ICLR_v2 (2026-09-21): encoder ablation, table only, no prose. ----------
% Rows generated by scripts/abl_v2_arch_table.py main 400 (training repo).
% The figure that used to sit here is now fig:encoder-ablation in the
% appendix, and this table points at it.
\begin{table}[t]
```

**Proposed**
```latex
  estimators.  Bands are the analytic \rms{} standard error.  The
  fourth parameter is the polar angle $\theta$.}
  \label{fig:rms_pt}
\end{figure}

\subsection{Choice of encoder}

Sequence encoders for this task fall into two families. Attention
\citep{Vaswani2017Attention} is the general-purpose choice. The prior
learned track fit on this detector uses it \citep{Couthures2025CTD},
as do related transformer trackers \citep{VanStroud2025Tracker,
Caron2025TrackFormers}. Linear recurrences returned as structured
state-space models with a fixed transition \citep{Gu2022S4D,
Smith2023S5, Orvieto2023LRU}, gained an input-dependent transition in
Mamba and Mamba-2 \citep{Gu2023Mamba, Dao2024Mamba2}, and developed in
parallel as gated linear recurrences down to the minGRU
\citep{Katharopoulos2020LinearAttention, De2024Griffin, Yang2024GLA,
Beck2024xLSTM, Feng2024minGRU}. The LSTM and GRU
\citep{Hochreiter1997LSTM, Cho2014GRU} gate on the previous hidden
state, so each step is a dependent matrix product with no parallel
kernel. We trained one earlier and do not carry it forward.

We pick one encoder per corner that matters at $20$ tokens. The
Transformer is the general encoder and the prior art. Mamba-2 is the
selective state-space model we started from, its update motivated by
the Kalman gain (\cref{sec:method}) and run bidirectionally after
Vision Mamba \citep{Zhu2024VisionMamba}. The minGRU keeps the
input-dependent gate and drops the rest of the Mamba-2 block,
leaving one elementwise scan. The non-selective diagonal state-space
model keeps the recurrence and drops the gate. All four run forward
and reverse over the track, the filter-plus-smoother structure of
the classical fit.

Three of the four reach the same precision. Only the non-selective
model falls behind, and only in $\qop$. Precision tracks the
formulation, the seed, the residual features, the anchored quantile
heads and the small-batch training. The encoder is a swappable part,
and the input-dependent gate is the one property that matters for
precision. Throughput tracks the kernel path, and every encoder
responds to the same treatment (\cref{tab:kernel-gains}).

Four encoders are a sample of the space. We did not test linear
attention, hybrid or convolutional mixers, or an order-blind model
such as an MLP over the fixed-length track. A simpler, faster encoder
than the minGRU may exist. The one simpler arm we tested lost only
on $\qop$, and the Transformer gained $6.6\times$ from kernel work,
so both rankings depend on engineering effort as much as on the
architecture. We deploy the minGRU because it is the fastest of the
four at equal precision. We do not claim it is the optimal encoder
for this task. At the deployment batch size the shared seed, Fourier
front end and quantile heads make up $60\,\%$ of its forward pass.

% --- ICLR_v2 (2026-09-21): encoder ablation, table only, no prose. ----------
% Rows generated by scripts/abl_v2_arch_table.py main 400 (training repo).
% The figure that used to sit here is now fig:encoder-ablation in the
% appendix, and this table points at it.
\begin{table}[t]
```

**Why:** the reader reaches four unexplained architectures right before `tab:arch-ablation`. This subsection says which families they come from, why these four and not others, and what the comparison did and did not show, and it states the limits plainly (word count about $360$, over the $220$-$300$ target, see the closing note).

---

### Item 15 — new bibliography entries

**Current:** none of these ten keys exist in `references.bib` (checked by grep).

**Proposed**
```bibtex
@inproceedings{Vaswani2017Attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {30},
  year      = {2017},
  note      = {Transformer architecture, the attention encoder of the ablation.}
}

@inproceedings{Gu2022S4D,
  title     = {On the Parameterization and Initialization of Diagonal State Space Models},
  author    = {Gu, Albert and Gupta, Ankit and Goel, Karan and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  year      = {2022},
  note      = {S4D diagonal state-space model, fixed non-selective transition.}
}

@inproceedings{Smith2023S5,
  title     = {Simplified State Space Layers for Sequence Modeling},
  author    = {Smith, Jimmy T. H. and Warrington, Andrew and Linderman, Scott W.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  note      = {S5 state-space layer.}
}

@inproceedings{Orvieto2023LRU,
  title     = {Resurrecting Recurrent Neural Networks for Long Sequences},
  author    = {Orvieto, Antonio and Smith, Samuel L. and Gu, Albert and Fernando, Anushan and Gulcehre, Caglar and Pascanu, Razvan and De, Soham},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  note      = {Linear recurrent unit (LRU).}
}

@inproceedings{Katharopoulos2020LinearAttention,
  title     = {Transformers Are {RNN}s: Fast Autoregressive Transformers with Linear Attention},
  author    = {Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c c}ois},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020},
  note      = {Linear attention as a gated linear recurrence.}
}

@article{De2024Griffin,
  title   = {Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models},
  author  = {De, Soham and Smith, Samuel L. and Fernando, Anushan and Botev, Aleksandar and Cristian-Muraru, George and Gu, Albert and Haroun, Ruba and Berrada, Leonard and Chen, Yutian and Srinivasan, Srivatsan and Desjardins, Guillaume and Doucet, Arnaud and Budden, David and Teh, Yee Whye and Pascanu, Razvan and De Freitas, Nando and Gulcehre, Caglar},
  journal = {arXiv preprint arXiv:2402.19427},
  year    = {2024},
  url     = {https://arxiv.org/abs/2402.19427},
  note    = {Griffin gated linear recurrence.}
}

@inproceedings{Yang2024GLA,
  title     = {Gated Linear Attention Transformers with Hardware-Efficient Training},
  author    = {Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  note      = {Gated linear attention (GLA).}
}

@inproceedings{Beck2024xLSTM,
  title     = {{xLSTM}: Extended Long Short-Term Memory},
  author    = {Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  note      = {xLSTM gated linear recurrence.}
}

@article{Hochreiter1997LSTM,
  title   = {Long Short-Term Memory},
  author  = {Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997},
  note    = {LSTM, gates on the previous hidden state.}
}

@inproceedings{Cho2014GRU,
  title     = {Learning Phrase Representations using {RNN} Encoder-Decoder for Statistical Machine Translation},
  author    = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2014},
  note      = {GRU, gates on the previous hidden state.}
}
```

**Why:** these ten keys are cited by the new "Choice of encoder" subsection (item 12+13) and are not yet in `references.bib`. The author pastes them in alongside the existing `Feng2024minGRU`, `Gu2023Mamba`, `Dao2024Mamba2`, `Zhu2024VisionMamba`, `Couthures2025CTD`, `VanStroud2025Tracker`, `Caron2025TrackFormers` and `Zhang2019RMSNorm` entries, which already exist and are unchanged.

---

## Closing note (not an item)

**Word count.** The "Choice of encoder" subsection (item 12+13) runs to
about $360$ words in this draft, above the $220$-$300$ word target. The four
required points (the two encoder families and their citations, why these
four, what the comparison showed, the honest limits) name eleven works and
did not compress further without dropping one of them. [AUTHOR: say which of
the eleven citations, if any, can go, or accept the subsection at its
current length.]

**Page budget.** The new subsection is pure addition, since nothing in the
main body is removed to make room for it. Added to a main body already fixed
at 9 pages, it is likely to push the document over. Two candidates to
cut elsewhere, both already flagged for the author's own attention.
One is the ATLAS farm electricity paragraph in results.tex (the paragraph
after the throughput comparison, starting "To put this into perspective").
The other is the second half of the Analytic seed subsection in method.tex
(the paragraph on float64 precision and the commented-out paragraph above
it). The main body should be rebuilt and the page count re-checked once
items 5-13 are applied.

**Unused macros.** `\kernelSpeedup` (main.tex:87) loses its only remaining
use once item 5 is applied. Its two siblings, `\stockThroughput` and
`\fusedThroughput`, were already retired when item 9 of the previous round
replaced `tab:kernel-bench` with `tab:kernel-gains`. All three can be
deleted from main.tex, or left as dead macros, once the author signs off.
