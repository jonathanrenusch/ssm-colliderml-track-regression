# Hit ordering without hit time: what ACTS does, and what reproduces it on drift_beamspot

*2026-08-25. Side-by-side study; nothing in the data pipeline or the model was changed.*
*Code: `src/track_regression/hit_sorting.py` (pure numpy, tested in `tests/test_hit_sorting.py`),
`src/track_regression/scripts/hit_sorting_study.py` (the numbers and figures below).
Figures: `dataset_plots/event_displays_acts_sorted/`.*

## TL;DR

1. **ACTS orders a truth track's measurements by simulated hit time** — `std::sort` on
   `SimHit::time()` in both `TruthTrackFinder` and `TruthSeedingAlgorithm`. It never sorts by
   path length, radius or geometry id. The Kalman filter and the CKF do not need an order at
   all: measurements are keyed by surface and visited in propagation order.
2. That key exists in the new data: `tracker_simhits.true_time`, reached through
   `tracker_hits.simhit_ids`. It is Geant4 global time in **ACTS native units (mm/c, 1 ns =
   299.792458 mm)** — which is also what the bug report's "slope 298" was. It is monotonic along
   every track (median dt/ds = 1.0003 over whole shards) and is the reference used below.
3. On the flat `test` stores, the legacy key `s = |X|` from the origin reproduces the ACTS order
   on only **85–87 % of tracks** (legacy campaign: 98 %). The failures are **not a low-pT
   effect** — the rate is flat from 0.5 GeV to 300 GeV — they are the **wide beamspot**: 0 % of
   tracks with |z0| < 40 mm are affected, 31 % of those with |z0| = 160–200 mm.
4. A **truth-free geometry order** (pixel → short strip → long strip; within a group barrel
   before endcap; barrel layers by radius, discs by z along the track's direction of flight)
   reproduces the ACTS order on **100.000 % of the three fixed-pT muon sets, 99.995 % of
   `single_muon_uniform` and 99.73 % of ttbar**. The ttbar misses are all pT < 1 GeV tracks whose
   hits genuinely curl back inward (energy-loss kinks / loopers), where no geometric key can
   recover a time order. It beats every truth-based key we tried except `|X − P|` from the truth
   perigee (which is unusable at inference).
5. **Recommendation: sort by the geometry order, at preprocessing and at inference** (same key
   on both sides, no truth needed). Do **not** stay on `s`: the `ICLR_retraining_ssort` rebuild
   still scrambles 13–15 % of tracks, and precisely the large-|z0| ones this campaign was made
   to study.
6. **Two things found on the way that are not about ordering** (details in §7): (a) the
   `particles` and `tracker_hits` parquet tables do not list events in the same row order, and
   `preprocess_flat.select_shard` joins them by row index, so **3.2–4.3 % of muon-gun tracks in
   every flat store carry the hits of a different event than their targets** (ttbar is clean);
   (b) the on-disk hit order of the `ICLR_retraining` stores (by digitised `time`) agrees with the
   ACTS order on 0.0 % of tracks — already known, see `CLAUDE.md` §0.1.

---

## 1. What ACTS does (checked in the repository, commit `a267dbb2`, 2026-08-25)

Permalinks are to `https://github.com/acts-project/acts/blob/a267dbb28be7fa297d7b8c5d18bc220f296b0dc2/`.

### 1.1 Truth track building: sort by sim-hit time

`Examples/Algorithms/TruthTracking/ActsExamples/TruthTracking/TruthTrackFinder.cpp` L106–116
([permalink](https://github.com/acts-project/acts/blob/a267dbb28be7fa297d7b8c5d18bc220f296b0dc2/Examples/Algorithms/TruthTracking/ActsExamples/TruthTracking/TruthTrackFinder.cpp#L106-L116)):

```cpp
    std::vector<std::size_t> indices;
    indices.resize(hits.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::ranges::sort(indices, [&hits](std::size_t a, std::size_t b) {
      return hits[a]->time() < hits[b]->time();
    });
    ProtoTrack sortedTrack;
    for (const auto& idx : indices) {
      sortedTrack.emplace_back(track[idx]);
    }
```

where `hits[i]` is the `SimHit` reached from each measurement through the
`measurementSimHitsMap` (L82–100). The class doc (`TruthTrackFinder.hpp` L26–31): *"Convert true
particle tracks into 'reconstructed' proto tracks … This algorithm should be able to replace any
other real track finder in the reconstruction chain."*

`TruthSeedingAlgorithm.cpp` L131–137
([permalink](https://github.com/acts-project/acts/blob/a267dbb28be7fa297d7b8c5d18bc220f296b0dc2/Examples/Algorithms/TruthTracking/ActsExamples/TruthTracking/TruthSeedingAlgorithm.cpp#L131-L137)) does the same:

```cpp
    std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
      return a.first->time() < b.first->time();
    });

    for (const auto& [hit, index] : hits) {
      track.push_back(index);
    }
```

and then *assumes* the time order is an r order when it picks the seed triplet (L178–185:
`ACTS_WARNING("Space points are not sorted in r. …")` whenever `m.r() - b.r() < 0`).

This is the path the ColliderML `truth_tracks` table was produced with:
`Examples/Scripts/Python/colliderml_truth_tracking.py` L104–131 calls
`addSeeding(..., seedingAlgorithm=SeedingAlgorithm.TruthEstimated)` followed by `addKalmanTracks`,
and `TruthEstimated` instantiates `acts.examples.TruthSeedingAlgorithm`
(`Python/Examples/python/reconstruction.py` L757–767).

### 1.2 `SimHit::index()` — the per-particle sequence number, and why it is useless here

`Fatras/include/ActsFatras/EventData/Hit.hpp` L38, L69–72, L120–121
([permalink](https://github.com/acts-project/acts/blob/a267dbb28be7fa297d7b8c5d18bc220f296b0dc2/Fatras/include/ActsFatras/EventData/Hit.hpp#L69-L72)):

```cpp
  /// Hit index along the particle trajectory.
  ///
  /// @retval negative if the hit index is undefined.
  constexpr std::int32_t index() const { return m_index; }
```

It is assigned as a running counter per particle: Geant4 path
`Examples/Algorithms/Geant4/src/SensitiveSteppingAction.cpp` L286–292 (`eventStore().particleHitCount.at(particleId) - 1`),
Fatras path `Fatras/include/ActsFatras/Kernel/detail/SimulationActor.hpp` L178–184 (`result.hits.size()`).
It is written by `RootSimHitWriter.cpp` L123–124 and `CsvSimHitWriter.cpp` L78–79
(*"// TODO write hit index along the particle trajectory"*). **In the ColliderML parquet the
`tracker_simhits.hit_index` column is 65535 (= −1 as uint16, "undefined") for every hit of every
dataset checked** (2 GeV, 100 GeV, ttbar shards; 3.6 M sim hits), so ACTS never sorts on it and
neither can we.

### 1.3 Geometry order is a *container* order, not a track order

`Examples/Framework/include/ActsExamples/EventData/SimHit.hpp` L19–20:
`/// Store hits ordered by geometry identifier.  using SimHitContainer = GeometryIdMultiset<SimHit>;`.
`GeometryContainers.hpp` L83–106 (`CompareGeometryId`) orders by geometry id and, for equal ids,
by time. `Core/include/Acts/Geometry/GeometryIdentifier.hpp` packs
volume (8 bit) | boundary | layer (12 bit) | approach | sensitive (20 bit) | extra into a 64-bit
value (L186–197) and compares the encoded value (L240–243). That is volume → layer → sensor
order — **not** trajectory order: in the ODD volume 16 (−z pixel discs) sorts before 17 (pixel
barrel), and on the −z side the disc layer id *decreases* outward (layer 16 at |z| = 619 mm,
layer 4 at 1522 mm, measured from the data). Nothing in `Examples/Algorithms/{TruthTracking,
TrackFinding,TrackFitting,Digitization}` sorts measurements by geometry id, radius or path length.

### 1.4 Fitting does not need an order

`Core/include/Acts/TrackFitting/KalmanFitter.hpp` L295 and L748–753
([permalink](https://github.com/acts-project/acts/blob/a267dbb28be7fa297d7b8c5d18bc220f296b0dc2/Core/include/Acts/TrackFitting/KalmanFitter.hpp#L748-L753)):

```cpp
    // To be able to find measurements later, we put them into a map
    // We need to copy input SourceLinks anyway, so the map can own them.
    ACTS_VERBOSE("Preparing " << nMeasurements << " input measurements");
    std::unordered_map<const Surface*, SourceLink> inputMeasurements;
```

The propagator visits surfaces in navigation order and looks each one up; the proto-track order
only matters for the seed estimate. The CKF (`Core/include/Acts/TrackFinding/MeasurementSelector.hpp`
L51–54) selects candidates per surface by χ², again with no notion of a sequence. The digitisation
(`DigitizationAlgorithm.cpp` L304–309) only records measurement → particle and
measurement → sim-hit maps.

**Distinction asked for in the task:** ACTS does *neither* "sort by path length from the
production vertex" nor "sort by geometry". It sorts by **truth time**, which for one particle is
path length / (βc) and therefore *equivalent* to a path-length sort, but it needs truth.

## 2. Where the ACTS key is in our data, and the time-unit finding

* `tracker_hits.simhit_ids` (list per hit, aligned one-to-one with `particle_ids`) is the
  **positional index into the event's `tracker_simhits` list**. Verified on every dataset: 100 %
  of (hit, particle) pairs resolve to a sim hit with the same `particle_id` (17.96 M pairs in the
  ttbar test shard, 0 mismatches).
* `tracker_simhits.true_time` is `SimHit::time()`. Median Δt/Δs between consecutive sim hits of
  the same particle is **1.0003** (2 GeV), **1.0000** (100 GeV), 1.014 (ttbar, curved tracks) —
  i.e. the number is a *length*: ACTS stores time with c = 1, `Units.hpp` L120–126
  (`constexpr double s = 299792458000.0; … ns = 1e-9 * s` → **1 ns = 299.792458 mm**). Divided
  by that, the ttbar shard spans −0.22 ns (vertex smearing, ±0.32 ns) to 1.46 µs (late
  secondaries). This re-reads Finding 2 of `BUGREPORT_drift_beamspot_hit_time.md`: the pixel
  `time` is referenced to the event and the "slope 298" *is* the unit. Finding 1 (strips have no
  time) stands, and the pixel digitised time is smeared by tens of mm, so the digitised column is
  still not a usable sort key — the **truth** one is.
* Availability: `tracker_simhits` is on `/scratch` for the three fixed-pT sets and ttbar. For
  `single_muon_uniform` only events 0–2 M were local (train shards); the shard for the test part
  used by the existing displays (`part_0002`, events 80 000 000–80 999 999, 684 MB) was fetched
  from the NERSC portal into `…/single_muon_uniform/v1/parquet/truth/tracker_simhits/`. The full
  set is 202 × 0.68 GB ≈ 138 GB.

## 3. Candidate orderings implemented (`track_regression/hit_sorting.py`)

All functions take one track's `xyz (L, 3)` in mm; parameters may be scalars or per-hit arrays, so
a whole CSR store can be processed in one call. ACTS conventions: `P = (−d0 sin φ, d0 cos φ, z0)`,
`q = sign(qop)`, `B = (0, 0, +2 T)`; validated against `perigee.py` (`test_centre_convention_matches_perigee_module`).

| key | needs truth | definition |
|---|---|---|
| `stored` | – | on-disk order of the `ICLR_retraining` stores (digitised `tracker_hits.time`) |
| `s_origin` | – | `s = sqrt(x² + y² + z²)` — the legacy key, hit feature 6 |
| `r` | – | `sqrt(x² + y²)` |
| `geometry` | – | `primary = (2·group + endcap)·4096 + c`, `c = r` (quantised to 0.1 mm) on barrel layers, `c = dir·z` on discs; `secondary = dir·z` (barrel) / `r` (disc). `group` = pixel 0 / short strip 1 / long strip 2 from `volume_id`; `dir = sign(z_outer − z_inner)` from the track's own hits (`z_direction`) |
| `s_perigee` | yes | `|X − P|` |
| `helix_T` | yes | `l_T = R·Δα / sin θ`, `R = pT/(0.29979·B)`, `C = P + qR(sin φ, −cos φ)`, `Δα = wrap[0,2π)(−q(atan2(X−C) − atan2(P−C)))`, unwrapped with z for loopers |
| `helix` | yes | `l_T` on barrel volumes, `l_L = (z − z0)/cos θ` on endcap volumes |

Why the geometry key is built the way it is: barrel sensors sit at a fixed radius and disc sensors at
a fixed z, so `r` (barrel) and `z` (disc) are the coordinates the module *fixes* regardless of the
strip's unmeasured direction (barrel long strips have z errors up to ±40 mm, endcap strips have r
errors of the same size). The ODD groups are nested shells (pixel r < 175 mm, short strip
240–705 mm, long strip 810–1035 mm) and every disc sits beyond its own barrel's half-length at a
radius inside the barrel envelope, so an outgoing track always crosses pixel → sstrip → lstrip
and, inside a group, the barrel before the discs. `r` is monotonic along a helix from the beam
line until `2R + |d0|`, i.e. for pT > 0.31 GeV inside the ODD; `z` is monotonic along any track
in a solenoid. The ODD `layer_id` orders identically inside a volume (grows with r in barrels,
grows with |z| on +z discs, shrinks with |z| on −z discs), so this *is* the layer order without a
lookup table. Unknown volume ids raise.

## 4. Quantitative comparison

Reference: ACTS order = sort by `true_time`. Metric: the whole per-track permutation is identical.
Tracks: one part of each flat `test` store — all tracks of the fixed-pT parts, 20 000 random
tracks of `single_muon_uniform/part_0002` and of `ttbar/part_0000` (plus the events of the
existing displays). Tracks whose hits are not in their own raw event (§7a) are excluded.

| method | mu 2 GeV | mu 10 GeV | mu 100 GeV | mu uniform | ttbar |
|---|---|---|---|---|---|
| tracks compared | 4 806 | 4 840 | 4 821 | 19 141 | 20 063 |
| `stored` (on-disk, digitised time) | 0.000 % | 0.000 % | 0.000 % | 0.005 % | 0.618 % |
| `s_origin` (legacy `s`) | 87.16 % | 86.78 % | 87.49 % | 87.25 % | 84.92 % |
| `r` | 70.54 % | 75.74 % | 76.89 % | 76.59 % | 73.58 % |
| **`geometry` (truth-free)** | **100.000 %** | **100.000 %** | **100.000 %** | **99.995 %** (1) | **99.731 %** (54) |
| `s_perigee` (truth) | 100.000 % | 99.959 % (2) | 100.000 % | 99.995 % (1) | 99.950 % (10) |
| `helix_T` (truth) | 71.04 % | 75.79 % | 76.93 % | 76.62 % | 75.47 % |
| `helix` mixed (truth) | 99.958 % (2) | 99.979 % (1) | 99.917 % (4) | 99.916 % (16) | 99.905 % (19) |

(in brackets: number of disagreeing tracks). Full binned curves in `summary.json`, table in
`summary.md`, figure `summary_disagreement.pdf` (rows: vs pT, |η|, |d0|, |z0|; hollow markers =
no disagreement in that bin).

### 4.1 Where the legacy `s` fails — the beamspot, not low pT

Disagreement rate of `s_origin` (uniform / ttbar, the two sets with a pT spectrum):

| pT [GeV] | 0.5–1 | 1–2 | 2–3 | 3–5 | 5–10 | 10–20 | 20–50 | 50–100 | >100 |
|---|---|---|---|---|---|---|---|---|---|
| uniform | – | 13.9 % | 15.0 % | 10.7 % | 13.0 % | 12.8 % | 12.3 % | 13.0 % | 12.6 % |
| ttbar | 14.2 % | 14.9 % | 16.0 % | 15.0 % | 15.6 % | 17.9 % | 15.1 % | 14.3 % | 18.2 % |

| \|z0\| [mm] | 0–40 | 40–80 | 80–120 | 120–160 | 160–200 | 200–240 |
|---|---|---|---|---|---|---|
| uniform | 0.0 % | 1.2 % | 10.1 % | 22.4 % | 30.9 % | 17.2 % |
| ttbar | 0.1 % | 1.4 % | 12.1 % | 27.2 % | 31.1 % | 22.3 % |

Versus |η| it peaks at 25–29 % for 0.5 < |η| < 1.5 and vanishes above |η| ≈ 2.5 (all datasets
alike). The mechanism: with `|z0|` up to 200 mm, a track whose first hits move *towards* z = 0
has `z²` shrinking faster than `r²` grows, so `s` decreases over the first pixel layers and the
innermost hits come out reversed (visible as the hook at the vertex end in
`lowpt_track_00_single_muon_uniform.pdf`, top row). The 2 % figure from the legacy campaign was
the same effect with a σ_z ≈ 50 mm beamspot. `s` anchored at the *truth* perigee (`s_perigee`)
removes it completely — which is the proof that the origin, not the shape of `s`, is the problem.

### 4.2 Why `r` and the transverse helix arc fail (71–77 %)

77–93 % of their discordant pairs are two hits in the *same* (volume, layer): the endcap strip
sensors measure φ but not r (the ODD places the cluster at the strip centre, up to ±50 mm off),
so any key built from transverse coordinates alone swaps the two staggered sensors of a disc. The
mixed helix key (`z` on discs) fixes exactly that: 99.9 %.

### 4.3 Residuals of `geometry`, `helix`, `s_perigee`

* Muon guns: the first version of the geometry key (plain `r`, `|z|` tie-break) missed 0.06–0.1 %
  of tracks, all of them a pair of hits at the same radius to 0.01 mm, 0.6–2 mm apart in z — a
  track crossing the boundary between two modules of one stave, where digitisation noise on `r`
  decided. Quantising `r` to 0.1 mm and breaking the tie by z *along the flight direction* took
  that to 0 on all three fixed-pT sets and to 1/19 141 on uniform. The mixed helix key misses the
  same pairs (the truth helix is off by ~1 mm after multiple scattering) and cannot be fixed
  without measured positions.
* ttbar: all 54 geometry misses are pT < 1 GeV (0.9 % of the 0.5–1 GeV bin, 0 % above). Their
  hits genuinely turn back inward — e.g. long-strip barrel at r = 1030 mm followed by hits at
  r = 1017 and 818 mm at larger |z|, or pixel L2 crossed three times — energy-loss kinks,
  interactions, decays in flight, and hit-association oddities. The truth helix (`helix`: 19
  misses) and `|X − P|` (10) fail on the same tracks. Only the truth time itself orders them.

## 5. Event displays (`dataset_plots/event_displays_acts_sorted/`)

Same events, same RNG draw, same axes (ODD envelope |r| < 1100 mm, |z| < 3100 mm, equal aspect)
as the existing `event_displays_s_sorted/` and `event_displays_stored_time_order/` figures, so
they are directly comparable:

* `<dataset>/overlay_10events.pdf` (muon sets) and `ttbar/event_NN_id<eid>.pdf` — hits connected
  in **ACTS truth-time order**.
* `<dataset>/side_by_side_*.pdf` — three rows, one per ordering: legacy `s` / ACTS truth time /
  geometry order. On the muon sets the ACTS and geometry rows are pixel-identical; the `s` row
  shows short hooks at the inner end of tracks with large |z0|.
* `lowpt_disagreements.pdf`, `lowpt_disagreements_zoom.pdf` — up to 8 pT < 3 GeV tracks whose
  `s` order differs from ACTS (uniform, ttbar, 2 GeV), one colour each, same three rows.
* `lowpt_track_NN_<dataset>.pdf` — one such track per figure, axes zoomed to its hits, where
  the mis-ordering is actually visible at print size.

## 6. Recommendation

**Switch the encoder sort key to the geometry order** (`hit_sorting.geometry_order(H[:, :3], H[:, 7])`
per track, or the vectorised `geometry_keys` with a per-track `direction`) — at preprocessing
*and* at inference, since it uses only hit positions and volume ids that the model receives
anyway. It reproduces ACTS's own truth-time order on 100 % of the muon-gun tracks and 99.7 % of
ttbar, with the remaining 0.3 % being sub-GeV tracks that no geometric rule can order; the model
sees identical sequences in training and deployment, and the ordering no longer depends on where
the beamspot is. Concretely, in `scripts/preprocess_flat.py` this is a third `--sort-key`
alongside `time` and `s` (a `lexsort` over `(secondary, primary, particle)` instead of
`(s, particle)`); the packed model path then needs no change.

What not to do:

* **Do not keep `s` from the origin.** On this campaign it scrambles 13–15 % of tracks, and the
  affected ones are exactly the large-|z0| tracks the drift beamspot was introduced to cover; the
  `ICLR_retraining_ssort` rebuild is much better than the time-sorted stores but still carries
  this.
* **Do not train on the ACTS truth-time order itself.** It is the right *reference* — and would
  be the right training order if inference had it — but at inference there is no truth time, so
  the sequences would differ systematically between training and deployment. The same holds for
  `s_perigee` and the helix arc length: they need `d0, z0, φ, θ, q/p`, i.e. the answer. Use them
  as diagnostics only (`s_perigee` is the sharpest available truth-based check on the geometry
  order). Fetching the 138 GB of uniform `tracker_simhits` is therefore not needed.

## 7. Caveats and side findings

**(a) Mislabelled tracks in every flat store — row-order mis-join.** `preprocess_flat.select_shard`
pairs hits with particles through the parquet *row index* (`p_ev`, `h_ev` are row numbers; the
composite keys are `(row << 32) | particle_rank`). The `particles` and `tracker_hits` tables hold
the same events but **not in the same row order**: 10–94 of 1 000 rows differ per fixed-pT shard,
41 047 / 42 845 of 1 000 000 in the two uniform shards checked (events 0–1 M and 80–81 M), 0–2 per
ttbar shard. Every differing row gives one track whose **targets come from event A and whose hits
come from event B** — e.g. flat track (event 26957, 2 GeV test) has θ = 0.274 (η = +2.0) and eleven
hits at z < −380 mm. Counted directly (hits absent from their own raw event): **3.84 % / 3.20 % /
3.54 % of the 2 / 10 / 100 GeV test tracks, 4.34 % of the uniform test tracks, 0 % of ttbar**;
the per-shard counts match the row-order differences exactly (192 = 90 + 10 + 30 + 29 + 33 for
2 GeV). Extrapolated: ≈ 8 M of the 191.5 M uniform training tracks. The `ICLR_retraining_ssort`
rebuild uses the same `select_shard` and inherits it. The fix is to join on `event_id` *values*
(map each particles row to the hits row with the same `event_id`), not on row position; the
`tracks` table is already joined by value (`ev_to_local`) and is unaffected. Not changed here per
the task's constraint; the study excludes these tracks (they have no sim hit in their own event).

**(b) The `stored` order is 0 % correct** (strip hits, `time == 0`, first) — the packed path reads
it verbatim. Known since this morning (`CLAUDE.md` §0.1); quantified here on all five sets.

**(c) What the reference can and cannot be.** The truth time is exact for a single Geant4 track.
A digitised cluster merging two sim hits of the same particle takes the earlier one. Clusters
merging *different* particles are attached to each of them with their own sim hit, as the flat
stores do.

**(d) Scope of the geometry key.** It hard-codes the ODD silicon volume ids 16–30 and the
nested-shell layout; it is the layer order, not a physical arc length, so hits on the same
module are ordered by digitisation noise (irrelevant for a sequence model). Loopers with
pT < 0.31 GeV (below every selection) would need the truth time.

**(e) Sampling.** One part per dataset (all tracks of the fixed-pT parts; 20 k of 1 M for uniform;
20 k of 26 k for ttbar). The uniform truth check needed one 684 MB `tracker_simhits` shard, now at
`/scratch/colliderml/drift_beamspot/single_muon_uniform/v1/parquet/truth/tracker_simhits/…events80000000-80999999.parquet`.

## 8. Reproduce

```bash
cd /shared/tracking/ssm-colliderml-track-regression
pixi run -e default python -m pytest tests/test_hit_sorting.py -q          # 23 tests, <1 s
pixi run -e default python src/track_regression/scripts/hit_sorting_study.py --n-sample 20000
# ~6 min on CPU; writes dataset_plots/event_displays_acts_sorted/{summary.json,summary.md,*.pdf}
```
