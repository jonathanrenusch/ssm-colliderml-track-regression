# Why the RTX 5000 Ada beats the H100 at small batch — answered

Question: the minGRU curve showed the Ada ahead of the H100 below ~4k tracks
per batch, and that survived aligning the batch ladder, the input sample and
the host load.  Is the Ada genuinely the faster device in that regime?

**No.  The small-batch regime is host/dispatch bound, not device bound.**

CUDA-graph replay removes per-launch dispatch gaps and changes nothing else.
H100 NVL, h=192 minGRU, fp16, quiet host, 200 timed iterations:

| tracks/batch | eager | CUDA graph | graph / eager | dispatch share of wall clock |
|---|---:|---:|---:|---:|
| 256  |    77,068 |   298,786 | 3.88x | 74 % |
| 1024 |   307,126 |   977,793 | 3.18x | 69 % |
| 2048 |   606,344 | 1,715,450 | 2.83x | 65 % |
| 4096 | 1,198,393 | 2,571,578 | 2.15x | 53 % |

Against the collaborator's eager Ada numbers (91,081 / 365,725 / 737,211 /
1,045,558 at the same batch sizes), the graphed H100 is **3.3x / 2.7x / 2.3x /
2.5x** faster.  So the H100 is ahead at every batch size once dispatch is taken
out; the Ada only leads while both sides are paying that overhead, and its lead
there (~17-25 %) is small next to the overhead itself (53-74 % of wall clock).

Reading: below ~4k tracks the measurement compares two hosts' launch latency,
not two GPUs.  The plateau comparison (H100 5.41 M vs Ada 1.51 M, 3.6x) is the
device result.

Practical consequence for deployment in the small-batch / trigger regime: use
CUDA graphs.  The minGRU's packed kernel composes with graph capture because it
needs no host synchronization -- unlike a bucketed launch, which needs a
device sync to split the tracks and therefore cannot be captured.

Reproduce: scratchpad/smallbatch.sh (eager vs --cuda-graph, TRK_SSD_BUCKET16=0,
every co-located trainer SIGSTOPped with a resume trap).
