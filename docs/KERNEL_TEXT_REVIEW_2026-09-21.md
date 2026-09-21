# Review of `sec:kernels` on `ICLR_v2` — what the minGRU kernel actually does

Checked against `src/track_regression/ops/mingru_short_triton.py`
(`mingru_bidi_packed`, the deployed path) and `tab:backbones`.

Short answer to the question: **yes, the kernel does more than the text says,
and one sentence currently describes the wrong kernel.**

## What the deployed minGRU kernel actually does

1. **The whole token mixing is ONE kernel launch.** `mingru_bidi_packed`
   launches on a grid `(tracks, channel-blocks, 2)` — the third axis *is* the
   direction. Forward and reverse scans for every track and every channel
   happen in a single launch, not two.
2. **The reverse pass is flipped inside the kernel**, via a `REVERSE`
   constexpr and per-segment bounds, and the store un-flips. No flip kernel,
   no gather, no extra memory traffic.
3. **The layout is fully packed — there is no padding at all.** One row per
   real hit, track boundaries in `cu_seqlens`.
4. **The kernel consumes fp16/bf16 natively**, converting on load in registers
   and accumulating the recurrence in fp32. This is what makes fp16 inference
   free here: the alternative is a full $(T, 4H)$ cast kernel at the boundary,
   which costs more than the faster GEMMs save.
5. **Zero shared memory, no barriers, no `tl.dot`** — the state lives in
   registers, which is also what makes it portable to the Ada deployment
   target (an AOT compile test guards against Hopper-only PTX).
6. Net effect: **12 CUDA kernel launches per forward pass** for the whole
   model, against 153 for the transformer.

## Problems in the current text

**P-A (wrong kernel described).** The paragraph says of the minGRU kernel:
"inference runs on unpadded, length-sorted tracks, so the roughly
three-quarters of tracks with 16 hits or fewer are routed through a smaller
compute tile rather than one sized for the longest possible track." That is
the **Mamba-2 bucketed launch** (`TRK_SSD_BUCKET16`), which exists because the
Mamba path *is* padded. The minGRU kernel has nothing to bucket — it is packed,
so there is no "longest possible track" tile and no length sorting. Carried
over from the `ICLR` text, where it was correct.

**P-B (the headline is missing).** Nothing says the bidirectional token mixing
collapses to a single launch, or that the reverse pass costs no separate
kernel. That is the most concrete thing the kernel does and it is exactly the
claim the throughput number rests on.

**P-C (the fp16 link is missing).** The text does not say the kernel takes
fp16 natively. Without it, the empty Mamba-2 cell in `tab:backbones` — whose
caption explains that its fused gating kernel is fp32/fp64-typed — reads as an
unexplained gap rather than as the same design decision seen from the other
side.

**P-D (a number that contradicts our own table).** "an identical kernel effort
brings $5.0\times$ for the minGRU". `tab:backbones` gives reference $1.49$ →
fused $3.90$, i.e. **$2.6\times$**, or $2.7\times$ counting fp16 ($4.00$). I
could not reproduce $5.0\times$ from any measurement in the campaign. Recommend
$2.6\times$, and quote it from the table.

**P-E (mean hit count).** "measures $15$ hits on average" — the stores give
**13.3** (mix3 13.28, uniform test 13.28, ttbar test 13.21). The padding-waste
argument uses this number, so it is worth being right: 22 tokens against 13.3
real hits is $1.65\times$, not $1.47\times$.

**P-F (typos).** "Our domains instead demands doing inference on thausand
short sequences at once" → *domain instead demands ... on thousands of short
sequences*.

## Suggested replacement for the first two paragraphs

> Standard GPU kernels for sequence models are built for long sequences,
> typically $10^{3}$--$10^{5}$ tokens. Our domain demands the opposite: a
> single charged-particle trajectory is at most $22$ tokens including the two
> class tokens and carries $13.3$ hits on average, and inference processes
> $10^{5}$ such trajectories at once. A standard kernel therefore spends most
> of its arithmetic on padding and most of its time on launch overhead.
>
> For the deployed minGRU a hand-written fused Triton kernel removes the
> mismatch. It runs on a fully packed layout, one row per recorded hit with no
> padding anywhere, and it performs the entire bidirectional token mixing in a
> **single kernel launch**: the launch grid carries the scan direction as one
> of its axes, and the reverse pass is flipped inside the kernel rather than by
> materializing a reversed copy. The recurrence is elementwise, so each thread
> keeps its channel's state in a register and the kernel allocates no shared
> memory and needs no barrier. It also reads fp16 inputs directly, converting
> on load and accumulating in fp32, which is what makes reduced-precision
> inference free here rather than costing a separate cast over the whole
> projected sequence. Together with the per-hit Fourier feature expansion,
> otherwise dozens of small operations, fused into one compiled kernel, a
> forward pass over a batch of trajectories issues $12$ CUDA kernels in total.

Then keep the existing "only linear recurrences benefit" paragraph, with
$5.0\times \to 2.6\times$, and the existing Mamba-2 validation paragraph
(where the bucketed launch belongs and is correctly described).
