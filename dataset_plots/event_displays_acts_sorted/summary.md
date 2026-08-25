# Hit ordering vs the ACTS reference (sort by `tracker_simhits.true_time`)

Exact agreement of the whole per-track permutation, flat `test` stores, one part per dataset.

| method | single_muon_2GeV | single_muon_10GeV | single_muon_100GeV | single_muon_uniform | ttbar |
|---|---|---|---|---|---|
| `stored`  | 0.000 % (4806) | 0.000 % (4840) | 0.000 % (4821) | 0.005 % (19140) | 0.618 % (19939) |
| `s_origin`  | 87.162 % (617) | 86.777 % (640) | 87.492 % (603) | 87.247 % (2441) | 84.922 % (3025) |
| `r`  | 70.537 % (1416) | 75.744 % (1174) | 76.893 % (1114) | 76.590 % (4481) | 73.578 % (5301) |
| `geometry`  | 100.000 % (0) | 100.000 % (0) | 100.000 % (0) | 99.995 % (1) | 99.731 % (54) |
| `s_perigee` (truth) | 100.000 % (0) | 99.959 % (2) | 100.000 % (0) | 99.995 % (1) | 99.950 % (10) |
| `helix_T` (truth) | 71.036 % (1392) | 75.785 % (1172) | 76.934 % (1112) | 76.621 % (4475) | 75.472 % (4921) |
| `helix` (truth) | 99.958 % (2) | 99.979 % (1) | 99.917 % (4) | 99.916 % (16) | 99.905 % (19) |

Tracks used / with a truth time for every hit / whose hits belong to another event (row-order mis-join in preprocess_flat.select_shard; excluded above):

| dataset | part | used | matched | mislabelled | partially unmatched |
|---|---|---|---|---|---|
| single_muon_2GeV | part_0000 | 4,998 | 4,806 | 192 (3.84 %) | 0 |
| single_muon_10GeV | part_0000 | 5,000 | 4,840 | 160 (3.20 %) | 0 |
| single_muon_100GeV | part_0000 | 4,998 | 4,821 | 177 (3.54 %) | 0 |
| single_muon_uniform | part_0002 | 20,010 | 19,141 | 869 (4.34 %) | 0 |
| ttbar | part_0000 | 20,063 | 20,063 | 0 (0.00 %) | 0 |

Share of discordant hit pairs that sit in the same (volume, layer) — i.e. swaps of the two staggered sensors of one layer rather than cross-layer errors:

| method | single_muon_2GeV | single_muon_10GeV | single_muon_100GeV | single_muon_uniform | ttbar |
|---|---|---|---|---|---|
| `stored` | 3.8 % | 3.9 % | 3.8 % | 3.8 % | 4.2 % |
| `s_origin` | 23.4 % | 23.9 % | 24.0 % | 23.6 % | 22.8 % |
| `r` | 76.8 % | 82.7 % | 87.4 % | 85.7 % | 66.0 % |
| `geometry` | 0.0 % | 0.0 % | 0.0 % | 100.0 % | 5.1 % |
| `s_perigee` | 0.0 % | 100.0 % | 0.0 % | 100.0 % | 69.6 % |
| `helix_T` | 93.5 % | 84.6 % | 87.8 % | 86.8 % | 92.7 % |
| `helix` | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 11.8 % |
