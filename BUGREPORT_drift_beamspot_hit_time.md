# drift_beamspot: `tracker_hits.time` — two regressions vs the previous production

Checked directly in the **published raw parquet** (nothing of ours applied).
Both findings are identical across all five datasets.

## Finding 1 — the strip volumes no longer measure time

`subspace` (bitmask: 1 = loc0, 2 = loc1, 4 = time) splits the hits exactly:

| subspace | bits | hits      | volumes      | time == 0 |
|---------:|-----:|----------:|--------------|----------:|
|        7 |  111 | 6,762,570 | 16, 17, 18   |   0.00 %  |
|        3 |  011 | 5,637,620 | 23, 24, 25   | 100.00 %  |
|        1 |  001 | 3,355,562 | 28, 29, 30   | 100.00 %  |

So the ~57 % of hits at exactly 0.0 are **not corrupt** — the strip volumes
simply have no time in their measurement subspace, and `var_time` is 0 there
too. `subspace` reports it honestly.

That said, it is a change: in the previous production **every** hit carried a
valid time (0.00 % zeros, see the table below). Was dropping time from the
strips intentional? If not, look at the per-volume `time` entries in
`odd-digi-smearing-config.json` (the `digi_config` of the digitization stage).

## Finding 2 — where time IS written, it is not referenced to the bunch crossing

Pixel volumes 16/17/18 only, against the expected flight time `s/c`
with `s = sqrt(x^2+y^2+z^2)`:

|                        | 1 %    | 50 %   | 99 %    |
|------------------------|-------:|-------:|--------:|
| `time` [ns]            | -39.23 | 271.49 | 1530.23 |
| expected `s/c` [ns]    |   0.11 |   0.92 |    5.09 |

`corr(time, s/c) = 0.275`, fitted `slope = 298`, and the maximum across
datasets runs to 1.0 ms (`ttbar`: 4.8 ms). `tracker_simhits.true_time` shows
the same thing (max 4.8 ms), so it is not purely digitisation — the truth time
looks like a raw Geant4 timestamp, including late secondaries (nuclear
interactions, neutron capture), rather than a time relative to the bunch
crossing.

For comparison, the **previous** production, same container
(`ghcr.io/opendatadetector/sw:pr-8`):

| dataset (legacy)     | time == 0 | corr(t, s/c) | slope     | median t | median s/c | max t   |
|----------------------|----------:|-------------:|----------:|---------:|-----------:|--------:|
| p200_core_kf_matched |    0.00 % |    **0.995** | **1.008** |  2.55 ns |    2.46 ns | 12.3 ns |
| p0_core_kf_hits      |    0.00 % |      0.485   | **1.012** |  2.25 ns |    2.24 ns |  3.2 us |

Slope ~1 and median t ~ median s/c is what a correct per-hit time looks like.
Neither holds in the new campaign.

## Per-dataset summary (raw parquet, one shard each)

| dataset             | hits       | time == 0 | corr(t, s/c) | slope | median t | median s/c | max t    |
|---------------------|-----------:|----------:|-------------:|------:|---------:|-----------:|---------:|
| single_muon_2GeV    |     15,368 |  56.79 %  |       -0.19  |   -22 | 0.000 ns |   2.69 ns  |  3.5 us  |
| single_muon_10GeV   |     15,653 |  56.44 %  |       -0.19  |   -22 | 0.000 ns |   2.48 ns  |  3.8 us  |
| single_muon_100GeV  |     15,732 |  57.02 %  |       -0.03  |  -111 | 0.000 ns |   2.55 ns  |  432 us  |
| single_muon_uniform | 15,755,752 |  57.08 %  |       -0.07  |   -23 | 0.000 ns |   2.66 ns  |  1.0 ms  |
| ttbar               |  3,524,552 |  63.63 %  |       -0.06  | -1702 | 0.000 ns |   3.10 ns  |  4.8 ms  |

(The negative correlations here mix the zero-filled strips with the pixels;
Finding 2 above is the clean pixel-only measurement.)

## Why it matters to us

We order each track's hits by time to build the input sequence for a sequence
model. Finding 2 alone makes that ordering wrong even on pixel-only tracks.
Ordering by `s` is a workaround but it is a proxy, not a time.

Everything else we have checked in the campaign is fine.
