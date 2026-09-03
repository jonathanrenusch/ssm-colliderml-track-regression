"""SSM track fitting as an ACTS ``IAlgorithm`` -- pyacts-based physics validation (TASK.md item c).

Runs the trained SSM inside a real ACTS event loop (ColliderML -> gen3 ODD + JSON material map),
reading only reconstructed (digitized) information -- never truth -- so it can be evaluated by
ACTS's own performance writers exactly like the ACTS Kalman-filter baseline. This is a
reconstruction algorithm, not a truth-comparison tool: physics comparison happens downstream via
``TrackTruthMatcher`` + the performance writers, identically for SSM and KF.

Requires pyacts>=47.5.0 (2026-08-24): ``Cluster.globalPosition`` gives the exact digitized
position directly (no reconstruction from local params), and ``IndexSourceLink(geometryId,
index)`` has a real Python constructor (no more ``SpacePointMaker`` detour, which previously
caused a 98.4% prototrack-drop bug via a mismatched ``geometrySelection``).

A valid ``uncalibratedSourceLink`` on every kept track state is required: ``TrackTruthMatcher``
silently drops any track state without one.

Covariance is left unset for every parameter: ``BoundMatrix`` has no Python-side mutation API in
47.5.0 (no pull plots possible), matching the ``pypi_finding_fitting_demo.py`` reference.

Geometry/field setup follows ``~/odd-json/colliderml_full_chain.py``: gen3 ODD from JSON topology
+ JSON material map, 3.0 T field (ColliderML's ttbar_pu200 was generated at 3 T, not ODD's default
2 T). Particle-level selection mirrors ``core_kf_hits`` (pt >= 0.5 GeV, |eta| <= 3, 6-20
measurements, charged, primary) via ``ParticleSelectorConfig``, which has no d0/z0 or
hard-scatter fields -- those training-selection cuts are a known, documented gap (see
PROGRESS.md).

Usage::

    uv run --project /shared/jonathan-ssm-colliderml-track-regression \\
        python -m track_regression.acts_integration \\
        --particles-dir <dir> --hits-dir <dir> \\
        --geo-json ~/odd-json/odd.json \\
        --material-json ~/odd-json/gen3_material_map_map.json \\
        --geoid-map-csv ~/odd-json/geoid_map.csv \\
        --seeding-config ~/odd-json/odd-seeding-config-gen3.json \\
        --ckpt <ckpt> --config <model config yaml> \\
        --variant v0 --events 100 --output <outdir>
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import acts
import acts.examples
import acts.examples.json  # noqa: F401 -- registers acts.examples.json.readJsonGeometryList
import acts.examples.scipy as acts_scipy
import numpy as np
import torch
from acts.examples.arrow import ColliderMLRelease1InputConverter, ParquetReader
from acts.examples.reconstruction import SeedingAlgorithm, addKalmanTracks, addSeeding
from acts.examples.simulation import ParticleSelectorConfig, addDigiParticleSelection
from acts.json import JsonMaterialDecorator, MaterialMapJsonConverter, TrackingGeometryJsonConverter

# This copy lives at <repo>/scripts/acts_integration.py (upstream: /eos/user/b/bhuth/
# jonathan_ssm/acts_integration.py, synced 2026-09-01); upstream sat two levels deeper.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "perf"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# geoid: gen3 ACTS geoId <-> ColliderML (volume_id, layer_id, surface_id) mapping
# ---------------------------------------------------------------------------
#
# ``ColliderMLRelease1InputConverter`` remaps each hit's geometry ID from the ColliderML
# (volume, layer, sensitive) triplet to a gen3 geoId via a CSV lookup. ``load_geoid_inverse``
# inverts that CSV so a gen3 geoId can be mapped back to the ColliderML triplet -- three of the
# model's twelve input features.
#
# ``volume_id -> detector`` has no ACTS EDM representation; derived once from the raw ColliderML
# parquet (100-event shard, 24.8M hits) as an exhaustively-confirmed bijection.
VOLUME_TO_DETECTOR: dict[int, int] = {
    16: 0, 17: 1, 18: 2,
    23: 3, 24: 4, 25: 5,
    28: 6, 29: 7, 30: 8,
}

N_HIT_FEATURES = 12
# The selection yaml lives in src/track_regression/ in this repo (upstream kept a copy next
# to the script).
DEFAULT_SELECTION_PATH = REPO_ROOT / "src" / "track_regression" / "selection_p200_datasets.yaml"
DEFAULT_VARIANT = "core_kf_hits"


def load_geoid_inverse(csv_path: str | Path) -> dict[int, tuple[int, int, int]]:
    """Invert ``geoid_map.csv`` on ``gen3_packed``.

    Returns ``{gen3_packed_geoid_value: (volume_id, layer_id, surface_id)}``. Raises ``ValueError``
    if ``gen3_packed`` is not unique (the inversion would be ambiguous).
    """
    inverse: dict[int, tuple[int, int, int]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            gen3_packed = int(row["gen3_packed"])
            triplet = (int(row["gen1_volume"]), int(row["gen1_layer"]), int(row["gen1_sensitive"]))
            if gen3_packed in inverse and inverse[gen3_packed] != triplet:
                raise ValueError(
                    f"geoid_map.csv: gen3_packed={gen3_packed} maps to both "
                    f"{inverse[gen3_packed]} and {triplet} -- inversion is ambiguous"
                )
            inverse[gen3_packed] = triplet
    if not inverse:
        raise ValueError(f"geoid_map.csv at {csv_path} produced an empty inverse map")
    return inverse


def volume_to_detector(volume_id: int) -> int:
    """Look up the ColliderML ``detector`` feature value for a volume_id."""
    try:
        return VOLUME_TO_DETECTOR[volume_id]
    except KeyError as e:
        raise KeyError(
            f"volume_id={volume_id} is not one of the 9 known ColliderML tracker volumes "
            f"{sorted(VOLUME_TO_DETECTOR)}"
        ) from e


# ---------------------------------------------------------------------------
# features: the model's 12 hit input features from digitized ACTS EDM objects
# ---------------------------------------------------------------------------
#
# Reproduces ``preprocess_colliderml_compact.py:247-269`` exactly (same formulas, column order),
# from ``Cluster.globalPosition`` (exact digitized position, no reconstruction needed).


def build_hit_features(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    volume_id: np.ndarray, layer_id: np.ndarray, surface_id: np.ndarray,
) -> np.ndarray:
    """Vectorized (L, 12) feature array from global positions + geoIds.

    Column order matches ``preprocess_colliderml_compact.py:257-269`` exactly: x, y, z, r,
    phi_hit, theta_hit, s, volume_id, layer_id, surface_id, detector, eta_hit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = x.shape[0]

    r = np.sqrt(x**2 + y**2)
    phi_hit = np.arctan2(y, x)
    theta_hit = np.arccos(np.clip(z / (np.sqrt(x**2 + y**2 + z**2) + 1e-12), -1.0, 1.0))
    s = np.sqrt(x**2 + y**2 + z**2)
    eta_hit = np.clip(-np.log(np.tan(np.clip(theta_hit, 1e-8, np.pi - 1e-8) / 2.0)), -10.0, 10.0)
    detector = np.fromiter((volume_to_detector(int(v)) for v in volume_id), dtype=np.float32, count=n)

    return np.stack(
        [
            x, y, z, r, phi_hit, theta_hit, s,
            np.asarray(volume_id, dtype=np.float64), np.asarray(layer_id, dtype=np.float64),
            np.asarray(surface_id, dtype=np.float64), detector, eta_hit,
        ],
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# selection: the core_kf_hits hit-count cut for SSMTrackFitter
# ---------------------------------------------------------------------------
#
# Only the hit-count cut is reconstructed-quantity-based and applies inside the algorithm; the
# particle-level cuts (pt, eta, primary, charged) are mirrored upstream via the sequencer's
# ``ParticleSelectorConfig``, same as the KF path, without touching particle truth here.


def load_hit_count_bounds(
    path: str | Path = DEFAULT_SELECTION_PATH, variant: str = DEFAULT_VARIANT,
) -> tuple[int, int]:
    """Return ``(min_hits, max_hits)`` for the given ``selection_p200_datasets.yaml`` variant."""
    from track_regression.selection_utils import load_selection_variant

    sel = load_selection_variant(path, variant)
    return int(sel["min_hits"]), int(sel["max_hits"])


def hit_count_ok(n_hits: int, min_hits: int, max_hits: int) -> bool:
    return min_hits <= n_hits <= max_hits


# ---------------------------------------------------------------------------
# SSMTrackFitter: the ACTS IAlgorithm
# ---------------------------------------------------------------------------

PARAM_NAMES = ("d0", "z0", "phi", "theta", "qop")


class SSMTrackFitter:
    """Construction parameters + per-event feature/selection logic, as plain methods rather than
    living inside the ``IAlgorithm`` subclass itself -- keeps :class:`SSMTrackFitterAlgorithm` a
    thin ACTS wrapper around this.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        geoid_csv: str,
        *,
        min_hits: int | None = None,
        max_hits: int | None = None,
        device: str = "cuda",
        sort_by_s: bool = True,
        sort_key: str | None = None,
        seed_residual_features: bool = False,
    ) -> None:
        self.model = model
        self.geoid_inverse = load_geoid_inverse(geoid_csv)
        # v2 checkpoints (campaign 2, CLAUDE.md §0.1/D.2): hits in detector-geometry order,
        # optionally with 3 extra per-hit seed-residual features (P', §4.10) and seed-anchored
        # heads whose anchors predict_physical() adds back.  ``sort_key`` overrides the legacy
        # ``sort_by_s`` bool: "s" | "geometry" | "none".
        if sort_key is None:
            sort_key = "s" if sort_by_s else "none"
        if sort_key not in ("s", "geometry", "none"):
            raise ValueError(f"sort_key must be s|geometry|none, got {sort_key!r}")
        self.sort_key = sort_key
        self.seed_residual_features = bool(seed_residual_features)
        # Anchored heads need the seed values at predict time (losses.predict_physical adds
        # them back); read the anchor spec straight off the checkpoint's loss module.
        self._delta_anchors = dict(getattr(model.loss_module, "_delta_anchors", {}) or {})
        if min_hits is None or max_hits is None:
            min_hits, max_hits = load_hit_count_bounds(DEFAULT_SELECTION_PATH, DEFAULT_VARIANT)
        self.min_hits = min_hits
        self.max_hits = max_hits
        self.device = device
        # Default True (2026-08-28, Benjamin): the training checkpoints were trained on
        # s = sqrt(x^2+y^2+z^2)-sorted hits, not TruthTrackFinder's truth-time order -- re-sort
        # to match. Set False only to reproduce the (incorrect-for-these-checkpoints) truth-time
        # ordering, e.g. for an A/B comparison.
        self.sort_by_s = sort_by_s
        self.n_tracks_seen = 0
        self.n_tracks_kept = 0
        self.n_tracks_dropped_hitcount = 0
        self.n_tracks_dropped_geoid = 0

    def _extract_track_hits(self, meas_indices, measurements, clusters, surface_map):
        """Build (features dict, source-link list) for one prototrack, or None to drop it.

        Hits start in ``meas_indices`` order (truth-time order from ``TruthTrackFinder``), then get
        re-sorted by s if ``self.sort_by_s`` (default True, matches training order -- see
        ``__init__``). ``clusters`` and ``measurements`` share the same index space (1:1).
        """
        xs, ys, zs, vols, lays, surfs, sls = [], [], [], [], [], [], []
        for midx in meas_indices:
            gid = measurements[midx].geometryId
            triplet = self.geoid_inverse.get(gid.value)
            if triplet is None or gid not in surface_map:
                self.n_tracks_dropped_geoid += 1
                return None

            pos = clusters[midx].globalPosition
            xs.append(float(pos[0])); ys.append(float(pos[1])); zs.append(float(pos[2]))
            vol, lay, surf = triplet
            vols.append(vol); lays.append(lay); surfs.append(surf)
            sls.append(acts.examples.IndexSourceLink(gid, midx).toSourceLink())

        n = len(meas_indices)
        feats = build_hit_features(xs, ys, zs, vols, lays, surfs)
        if self.sort_key != "none":
            if self.sort_key == "s":
                order = np.argsort(feats[:, 6], kind="stable")
            else:  # "geometry": detector order, exactly what the v2 stores use (hit_sorting)
                from track_regression.hit_sorting import geometry_order
                order = geometry_order(feats[:, :3].astype(np.float64), feats[:, 7])
            feats = feats[order]
            sls = [sls[i] for i in order]
            meas_indices[:] = [meas_indices[i] for i in order]
        item = {
            "hit_features": feats,
            "hit_s": feats[:, 6].copy(),
            # Placeholder: packed encoder ignores hit_time's value (mamba_cls.py:361,419-420);
            # ordering comes from input order + cu_seqlens/seq_idx.
            "hit_time": np.arange(n, dtype=np.float32),
            "targets": np.zeros(5, dtype=np.float32),  # placeholder: predict_physical(targets=None)
            "length": n,
        }
        return item, sls

    def process_event(self, prototracks, measurements, clusters, surface_map):
        """Returns a list of ``(meas_indices, source_links, d0, z0, phi, theta, qop)``."""
        from track_regression.data import collate_tracks_packed

        kept_items, kept_meas = [], []
        for proto in prototracks:
            meas_indices = list(proto)
            self.n_tracks_seen += 1
            if not hit_count_ok(len(meas_indices), self.min_hits, self.max_hits):
                self.n_tracks_dropped_hitcount += 1
                continue
            extracted = self._extract_track_hits(meas_indices, measurements, clusters, surface_map)
            if extracted is None:
                continue
            item, sls = extracted
            kept_items.append(item)
            kept_meas.append((meas_indices, sls))

        if not kept_items:
            return []

        # v2 checkpoints: ACTS three-pixel-point seed per track from the digitized hits only
        # (track_regression.seed, Bz = 3 T) -- mirrors flat_data._pack exactly.  Feeds (a) the
        # 3 extra per-hit residual features (seed_residual_features, input_dim 15) and (b) the
        # seed_<p> anchors that predict_physical adds back onto the anchored heads.
        seed_targets = None
        if self._delta_anchors or self.seed_residual_features:
            from track_regression.seed import compress_residuals, seed_from_csr, seed_residuals

            lens = np.array([it["length"] for it in kept_items], dtype=np.int64)
            flat = np.concatenate([it["hit_features"] for it in kept_items], axis=0)
            seed64 = seed_from_csr(flat, lens)
            if self.seed_residual_features:
                row = np.repeat(np.arange(len(kept_items)), lens)
                res = compress_residuals(seed_residuals(flat[:, :3], seed64, row)).astype(np.float32)
                off = 0
                for it in kept_items:
                    n_it = it["length"]
                    it["hit_features"] = np.concatenate(
                        [it["hit_features"], res[off:off + n_it]], axis=1)
                    off += n_it
            seed32 = seed64.astype(np.float32)
            seed_targets = {
                f"seed_{name}": torch.from_numpy(np.ascontiguousarray(seed32[:, i])).to(self.device)
                for i, name in enumerate(PARAM_NAMES)
            }

        inputs, _ = collate_tracks_packed(kept_items)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(inputs)
        preds = self.model.loss_module.predict_physical(out["pred"], targets=seed_targets)
        params = {name: preds[name].cpu().numpy() for name in PARAM_NAMES}

        self.n_tracks_kept += len(kept_items)
        return [
            (meas_indices, sls, *(float(params[name][i]) for name in PARAM_NAMES))
            for i, (meas_indices, sls) in enumerate(kept_meas)
        ]


class SSMTrackFitterAlgorithm(acts.examples.IAlgorithm):
    """ACTS ``IAlgorithm`` wrapper: reads prototracks/measurements/clusters, delegates to an
    ``SSMTrackFitter``, writes a ``ConstTrackContainer`` of SSM-fitted tracks."""

    def __init__(
        self, name: str, level, *, fitter: SSMTrackFitter, surface_map,
        prototracks_key: str = "prototracks", measurements_key: str = "measurements",
        clusters_key: str = "clusters", tracks_key: str = "ssm_tracks",
    ) -> None:
        acts.examples.IAlgorithm.__init__(self, name, level)
        self.fitter = fitter
        self.surface_map = surface_map

        def read(Type, label, key):
            h = acts.examples.ReadDataHandle(self, Type, label)
            h.initialize(key)
            return h

        self.prototracks = read(acts.examples.ProtoTrackContainer, "Prototracks", prototracks_key)
        self.measurements = read(acts.examples.MeasurementContainer, "Measurements", measurements_key)
        self.clusters = read(acts.examples.ClusterContainer, "Clusters", clusters_key)

        self.tracks = acts.examples.WriteDataHandle(self, acts.examples.ConstTrackContainer, "Tracks")
        self.tracks.initialize(tracks_key)

        self._perigee = acts.Surface.createPerigee(acts.Vector3(0, 0, 0))

    def execute(self, context):
        prototracks = self.prototracks(context.eventStore)
        measurements = self.measurements(context.eventStore)
        clusters = self.clusters(context.eventStore)

        results = self.fitter.process_event(prototracks, measurements, clusters, self.surface_map)

        container = acts.examples.TrackContainer()
        for meas_indices, sls, d0, z0, phi, theta, qop in results:
            track = container.makeTrack()
            track.referenceSurface = self._perigee
            track.parameters = acts.BoundVector(d0, z0, phi, theta, qop, 0.0)
            track.particleHypothesis = acts.ParticleHypothesis.muon

            for midx, sl in zip(meas_indices, sls):
                surface = self.surface_map[measurements[midx].geometryId]
                ts = track.appendTrackState()
                ts.typeFlags.isMeasurement = True
                ts.uncalibratedSourceLink = sl
                ts.referenceSurface = surface
            track.nMeasurements = len(meas_indices)

        self.tracks(context, container.makeConst())
        return acts.examples.ProcessCode.SUCCESS


def add_ssm_algorithm(
    s, *, model: torch.nn.Module, geoid_csv: str, tracking_geometry,
    min_hits: int | None = None, max_hits: int | None = None,
    device: str = "cuda", sort_by_s: bool = True,
    sort_key: str | None = None, seed_residual_features: bool = False,
    prototracks_key: str = "prototracks", measurements_key: str = "measurements",
    clusters_key: str = "clusters", tracks_key: str = "ssm_tracks",
) -> SSMTrackFitter:
    """Build an ``SSMTrackFitter`` + its ``SSMTrackFitterAlgorithm`` and add it to sequencer ``s``.

    Returns the fitter (callers read its ``n_tracks_*`` counters after ``s.run()``).
    """
    fitter = SSMTrackFitter(
        model, geoid_csv, min_hits=min_hits, max_hits=max_hits, device=device, sort_by_s=sort_by_s,
        sort_key=sort_key, seed_residual_features=seed_residual_features,
    )
    alg = SSMTrackFitterAlgorithm(
        "SSMTrackFitter", acts.logging.INFO, fitter=fitter, surface_map=tracking_geometry.geoIdSurfaceMap(),
        prototracks_key=prototracks_key, measurements_key=measurements_key,
        clusters_key=clusters_key, tracks_key=tracks_key,
    )
    s.addAlgorithm(alg)
    return fitter


# ---------------------------------------------------------------------------
# runner: sequencer assembly + CLI
# ---------------------------------------------------------------------------


# Plot grid: d0, z0, phi, theta, qop, relative-qop/pT resolutions (3x2). Each maps to a
# `reswidth_<param>_vs_pT` Histogram1 -- a fit-sigma Profile from `fitFunction`
# (`acts.examples.scipy.makeScipyHistogramFitFunction`, ROOT-free scipy Gaussian fit) not raw RMS.
RESOLUTION_PARAMS = {
    "d0": ("d0 resolution", r"$\sigma(d_0)$ [mm]"),
    "z0": ("z0 resolution", r"$\sigma(z_0)$ [mm]"),
    "phi": ("phi resolution", r"$\sigma(\phi)$ [rad]"),
    "theta": ("theta resolution", r"$\sigma(\theta)$ [rad]"),
    "qop": ("q/p resolution", r"$\sigma(q/p)$ [c/GeV]"),
    "qopt_rel": ("relative q/pT resolution", r"$\sigma(q/p_T)/|q/p_T|$ [%]"),
}

# (nBins, half_width) per param's `Residual_<param>` histogram axis in `resPlotToolConfig`. Only
# z0/phi are widened past ResPlotTool.hpp's stock defaults (100 bins, +-0.5mm/+-0.01rad) -- SSM's
# coarser resolution overflowed the stock window there; d0/theta/qop/qopt_rel fit fine as-is.
#
# nBins is NOT scaled proportionally with the wider window (bins get coarser, not denser): the
# scipy fit weights bins by 1/count, so scaling nBins to preserve density populated sparse tail
# bins that dragged the fit around (this destabilized d0 the first time it was tried). Coarser
# bins pool tail entries instead.
RESOLUTION_RESIDUAL_WINDOWS = {
    "d0": (100, 0.5),        # mm,    ACTS stock -- unchanged, already adequate
    "z0": (150, 2.0),        # mm,    was (100, 0.5) -- the one that was completely empty
    "phi": (150, 0.04),      # rad,   was (100, 0.01) -- the other one that was empty/erratic
    "theta": (100, 0.01),    # rad,   ACTS stock -- unchanged, already adequate
    "qop": (100, 0.1),       # 1/GeV, ACTS stock -- unchanged, already adequate
    "qopt_rel": (100, 0.1),  # %,     ACTS stock -- unchanged, already adequate
}

# A Gaussian sigma the scipy fit reports beyond its own residual histogram's half-width is
# definitionally not constrained by the data (degenerated on a near-empty bin) -- drop it in the
# plot rather than let it blow out the axis.
RESOLUTION_SANITY_MAX = {param: half_width for param, (_, half_width) in RESOLUTION_RESIDUAL_WINDOWS.items()}


def override_residual_windows(specs: list[str]) -> None:
    """Apply ``param:nbins:half_width`` overrides (repeatable --residual-window).

    The stock qop window (+-0.1, 100 bins) is far coarser than the fixed-pT muon
    residuals (e.g. sigma(q/p) ~ 5e-4 at 10 GeV vs a 2e-3 bin) -- the Gaussian
    fit then degenerates to sigma = 0 or spikes; per-dataset windows fix it.
    """
    for spec in specs or []:
        param, nbins, half = spec.split(":")
        assert param in RESOLUTION_RESIDUAL_WINDOWS, param
        RESOLUTION_RESIDUAL_WINDOWS[param] = (int(nbins), float(half))
        RESOLUTION_SANITY_MAX[param] = float(half)


# Independent variables to plot each resolution against -- each is one PDF page. Maps a
# display name to the `reswidth_<param>_vs_<key>` histogram-name suffix.
RESOLUTION_VARIABLES = {"pT": "pT", "eta": "eta"}

u = acts.UnitConstants

# Post-fit TrackSelectorAlgorithm cuts, identical for every track system (SSM, KF, KF-beamspot):
# approximates core_kf_hits's training window (|d0|<=2.5mm, |z0|<=200mm) plus the min-hits floor
# already enforced upstream, restated here explicitly so none of the systems' resolution is
# dominated by outliers. Plain floats also mirrored below (not just read off the Config) so
# main() can quote them in the summary page.
TRACK_SELECTOR_MIN_MEASUREMENTS = 6
TRACK_SELECTOR_D0_MAX_MM = 2.0
TRACK_SELECTOR_Z0_MAX_MM = 150.0


def _make_track_selector_cuts() -> "acts.TrackSelector.Config":
    return acts.TrackSelector.Config(
        minMeasurements=TRACK_SELECTOR_MIN_MEASUREMENTS,
        loc0=(-TRACK_SELECTOR_D0_MAX_MM * u.mm, TRACK_SELECTOR_D0_MAX_MM * u.mm),
        loc1=(-TRACK_SELECTOR_Z0_MAX_MM * u.mm, TRACK_SELECTOR_Z0_MAX_MM * u.mm),
    )


# v2 datasets carry |d0| up to 7.1 mm / |z0| up to 270 mm by construction (CLAUDE.md §4.17);
# main() rebuilds this Config when --d0-max/--z0-max override the legacy window above.
TRACK_SELECTOR_CUTS = _make_track_selector_cuts()

# ColliderML ttbar_pu200 beamspot smearing (d0, z0 stddevs) -- KF-beamspot-only refit pass, see
# build_sequencer's RefittingAlgorithm block. Not applied to SSM: RefittingAlgorithm needs each
# input track's own covariance from a prior real fit, which SSM tracks don't have (module
# docstring: no Python-side BoundMatrix mutation API to set one).
BEAMSPOT_D0_STDDEV_MM = 12.5
BEAMSPOT_Z0_STDDEV_MM = 55.0


def extract_resolution_data(hists: dict) -> dict:
    """Pull plain-numpy resolution curves out of a writer's ``.histograms()``.

    Picklable/ACTS-independent (unlike the live bound-C++ objects in ``hists``): for
    ``RESOLUTION_PARAMS`` x ``RESOLUTION_VARIABLES``, extracts bin centers/values/errors/xlabel
    from the ``reswidth_<param>_vs_<var>`` fit-sigma profile, so the result can be pickled and
    ``plot_resolutions`` re-run later without repeating ACTS inference.
    """
    data = {}
    for param in RESOLUTION_PARAMS:
        data[param] = {}
        for var_key, hist_suffix in RESOLUTION_VARIABLES.items():
            h = hists[f"reswidth_{param}_vs_{hist_suffix}"]
            bh = h.histogram
            edges = np.asarray(bh.axis(0).edges)
            data[param][var_key] = {
                "centers": 0.5 * (edges[:-1] + edges[1:]),
                "values": np.asarray(bh.values()),
                "errors": np.asarray(bh.errors()),
                "xlabel": bh.axis(0).label,
            }
    return data


def extract_raw_residual_data(hists: dict) -> dict:
    """Pull the raw (unfit) 1D residual histograms directly out of a writer's ``.histograms()``.

    ``res_<param>`` is the actual bin-count histogram of ``predicted - truth`` integrated over
    eta and pT, using the same ``RESOLUTION_RESIDUAL_WINDOWS`` binning as ``Residual_<param>``
    elsewhere in this module. Only reliably distinct from the pull/profile histograms of the same
    param name since pyacts ``999.999.999.dev20260827071648``, which fixed these being silently
    overwritten by same-named histograms in the ``.histograms()`` dict export (confirmed via a
    before/after key-listing probe: the pre-fix dict had no ``res_<param>``/``resVsEta_<param>``/
    etc. keys at all -- only the unrelated ``pullVsEtaPt_<param>`` survived under a bare
    ``<param>`` key). Picklable.
    """
    data = {}
    for param in RESOLUTION_PARAMS:
        h = hists[f"res_{param}"]
        bh = h.histogram
        edges = np.asarray(bh.axis(0).edges)
        data[param] = {
            "centers": 0.5 * (edges[:-1] + edges[1:]),
            "edges": edges,
            "counts": np.asarray(bh.values()),
            "xlabel": bh.axis(0).label,
        }
    return data


def plot_resolutions(
    data_by_label: dict, output_pdf: str | Path,
    *, integrated_by_label: dict | None = None, raw_residuals_by_label: dict | None = None,
    summary: dict | None = None,
) -> None:
    """Multi-page PDF: one 3x2 page per ``RESOLUTION_VARIABLES`` entry (pT, then eta) with
    d0/z0/phi/theta/qop/qopt_rel fit-sigma resolution curves; optional eta/pT-integrated
    residual-shape page (raw histogram + Gaussian fit overlay); optional run-summary page.

    ``data_by_label`` maps a legend label (e.g. ``"SSM"``, ``"ACTS KF"``) to an
    ``extract_resolution_data()`` dict. Bins with zero entries or a degenerated fit (sigma wider
    than the residual histogram, see ``RESOLUTION_SANITY_MAX``) are dropped rather than plotted.
    When ``"ACTS KF"`` is one of the labels, each pT/eta panel gets a thin HEP-style ratio panel
    below it (every other label's fit sigma divided by ACTS KF's, bin-by-bin) -- ACTS KF itself is
    not drawn there, just the y=1 reference line.

    ``integrated_by_label`` (optional): same labels -> ``extract_integrated_resolution()`` dict,
    Gaussian-fit mean/sigma integrated over eta and pT.

    ``raw_residuals_by_label`` (optional): same labels -> ``extract_raw_residual_data()`` dict,
    the actual residual bin counts backing that fit. Rendered as a filled step histogram
    (density-normalized) with the Gaussian fit overlaid in the same color, so the fit quality is
    visible directly. If omitted (e.g. re-plotting from an older pickle without this data), only
    the Gaussian curve is drawn, peak-normalized to 1 as before.

    ``summary`` (optional): ``{section_title: [line, ...]}`` text page.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_labels = list(data_by_label.keys())
    label_colors = {label: color_cycle[i % len(color_cycle)] for i, label in enumerate(all_labels)}
    ratio_ref_label = "ACTS KF" if "ACTS KF" in data_by_label else None

    with PdfPages(output_pdf) as pdf:
        for var_key in RESOLUTION_VARIABLES:
            has_ratio = ratio_ref_label is not None
            fig = plt.figure(figsize=(11, 15) if has_ratio else (11, 12))
            outer = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)
            for i, param in enumerate(RESOLUTION_PARAMS):
                title, ylabel = RESOLUTION_PARAMS[param]
                row, col = divmod(i, 2)
                if has_ratio:
                    inner = GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=outer[row, col], height_ratios=[3, 1], hspace=0.08,
                    )
                    ax = fig.add_subplot(inner[0])
                    ax_ratio = fig.add_subplot(inner[1], sharex=ax)
                else:
                    ax = fig.add_subplot(outer[row, col])
                    ax_ratio = None

                all_values = []
                xlabel = var_key
                curves = {}
                for label in all_labels:
                    curve = data_by_label[label][param][var_key]
                    centers, values, errors = curve["centers"], curve["values"], curve["errors"]
                    xlabel = curve["xlabel"]
                    mask = (values > 0) & (values <= RESOLUTION_SANITY_MAX[param])
                    curves[label] = (centers, values, mask)
                    ax.errorbar(
                        centers[mask], values[mask], yerr=errors[mask], color=label_colors[label],
                        marker="o", markersize=3, linestyle="-", label=label,
                    )
                    all_values.append(values[mask])
                # y-limits from the fitted sigmas only, not the error bars: a poorly-constrained
                # (low-stats) bin's sigma error can be orders of magnitude larger than every
                # sigma value, which would otherwise blow out the axis and hide the rest of the
                # curve.
                all_values = np.concatenate(all_values) if all_values else np.array([])
                if all_values.size:
                    lo, hi = float(all_values.min()), float(all_values.max())
                    pad = 0.1 * (hi - lo) if hi > lo else 0.1 * max(abs(hi), 1e-12)
                    ax.set_ylim(min(0.0, lo - pad), hi + pad)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)

                if has_ratio:
                    ax.tick_params(labelbottom=False)
                    ref_centers, ref_values, ref_mask = curves[ratio_ref_label]
                    ax_ratio.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
                    for label in all_labels:
                        if label == ratio_ref_label:
                            continue
                        centers, values, mask = curves[label]
                        common = mask & ref_mask
                        if not common.any():
                            continue
                        ax_ratio.plot(
                            centers[common], values[common] / ref_values[common],
                            color=label_colors[label], marker="o", markersize=3, linestyle="-",
                        )
                    ax_ratio.set_xlabel(xlabel)
                    ax_ratio.set_ylabel(f"ratio to {ratio_ref_label}", fontsize=7)
                    ax_ratio.grid(alpha=0.3)
                else:
                    ax.set_xlabel(xlabel)

            fig.suptitle(f"Track-parameter resolutions (scipy Gaussian fit sigma vs {var_key})")
            outer.tight_layout(fig, rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)

        if integrated_by_label or raw_residuals_by_label:
            fig, axes = plt.subplots(3, 2, figsize=(11, 12))
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            labels = list((raw_residuals_by_label or integrated_by_label).keys())
            colors = {label: color_cycle[i % len(color_cycle)] for i, label in enumerate(labels)}
            density = bool(raw_residuals_by_label)  # match normalization to the raw histogram
            for ax, param in zip(axes.flat, RESOLUTION_PARAMS):
                title, xlabel = RESOLUTION_PARAMS[param]
                half_width = RESOLUTION_SANITY_MAX[param]
                for label in labels:
                    color = colors[label]
                    if raw_residuals_by_label:
                        raw = raw_residuals_by_label[label][param]
                        edges, counts = raw["edges"], raw["counts"]
                        total = counts.sum()
                        binwidth = edges[1] - edges[0]
                        norm_counts = counts / (total * binwidth) if total > 0 else counts
                        ax.stairs(norm_counts, edges, fill=True, alpha=0.35, color=color)
                        xlabel = raw["xlabel"]
                    if integrated_by_label:
                        fit = integrated_by_label[label][param]
                        mean, sigma = fit["mean"], fit["sigma"]
                        x = np.linspace(-half_width, half_width, 400)
                        if sigma > 0:
                            y = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
                            if density:
                                y /= sigma * np.sqrt(2 * np.pi)
                        else:
                            y = np.zeros_like(x)
                        ax.plot(
                            x, y, color=color,
                            label=f"{label}: mean={mean:.3g}+-{fit['meanErr']:.2g}, "
                            f"sigma={sigma:.3g}+-{fit['sigmaErr']:.2g}",
                        )
                ax.axvline(0.0, color="grey", linewidth=0.8, linestyle=":")
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("density" if density else "normalized")
                ax.grid(alpha=0.3)
                if integrated_by_label:
                    ax.legend(fontsize=7)

            subtitle = (
                "Residual histogram (filled) with Gaussian fit overlay, integrated over eta and pT"
                if raw_residuals_by_label and integrated_by_label
                else "Residual histogram, integrated over eta and pT" if raw_residuals_by_label
                else "Residual distribution, integrated over eta and pT\n"
                "(fitted Gaussian shape, not a raw histogram)"
            )
            fig.suptitle(subtitle, fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)

        if summary is not None:
            _render_summary_page(pdf, summary)


def extract_integrated_resolution(hists: dict) -> dict:
    """``{param: {"mean", "meanErr", "sigma", "sigmaErr"}}`` from a writer built with
    ``_add_integrated_fitter_writer`` -- eta axis collapsed to one bin, so
    ``resmean/reswidth_<param>_vs_eta`` hold a single point: the residual fit integrated over the
    full eta AND pT range (``_vs_eta`` always marginalizes fully over pT). Picklable.
    """
    out = {}
    for param in RESOLUTION_PARAMS:
        mean_h = hists[f"resmean_{param}_vs_eta"].histogram
        width_h = hists[f"reswidth_{param}_vs_eta"].histogram
        out[param] = {
            "mean": float(np.asarray(mean_h.values())[0]),
            "meanErr": float(np.asarray(mean_h.errors())[0]),
            "sigma": float(np.asarray(width_h.values())[0]),
            "sigmaErr": float(np.asarray(width_h.errors())[0]),
        }
    return out


def extract_track_counts(finder_writer, total_particles: int) -> dict:
    """``{"total", "matched"}`` track counts.

    ``total_particles`` is the exact count from :class:`TrackCountCollector` -- NOT
    ``trackeff_vs_eta``'s own ``.total`` marginal, which undercounts by ~3x (see
    :class:`TrackCountCollector`). Its matched/total *ratio* is still reliable, so ``matched`` is
    derived by applying that ratio to the exact total.
    """
    eff = finder_writer.histograms()["trackeff_vs_eta"]
    eff_total = np.asarray(eff.total.values()).sum()
    eff_accepted = np.asarray(eff.accepted.values()).sum()
    ratio = eff_accepted / eff_total if eff_total > 0 else 0.0
    return {"total": total_particles, "matched": round(total_particles * ratio)}


def _format_report_line(label: str, counts: dict, integrated: dict) -> str:
    """One result-report line: match rate + eta/pT-integrated fit-sigma per physical param."""
    match_pct = 100.0 * counts["matched"] / counts["total"] if counts["total"] else 0.0
    params = "  ".join(
        f"{p}_sigma={integrated[p]['sigma']:.4g}+/-{integrated[p]['sigmaErr']:.2g}" for p in PARAM_NAMES
    )
    return f"{label:20s} matched {counts['matched']:.0f}/{counts['total']:.0f} ({match_pct:.1f}%)  {params}"


def _render_summary_page(pdf, summary: dict) -> None:
    """Text-only PDF page listing the run's selection cuts, track counts, B field, input files."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 12))
    ax.axis("off")
    lines = ["Run summary", ""]
    for section, rows in summary.items():
        lines.append(section)
        for row in rows:
            lines.append(f"  {row}")
        lines.append("")
    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", family="monospace", fontsize=8, wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _add_finder_writer(s, *, tracks_key, track_particle_matching, particle_track_matching):
    """``PythonPatternRecognitionPerformanceWriter`` against ``particles_selected``."""
    cfg = acts.examples.PythonPatternRecognitionPerformanceWriter.Config()
    cfg.inputTracks = tracks_key
    cfg.inputParticles = "particles_selected"
    cfg.inputTrackParticleMatching = track_particle_matching
    cfg.inputParticleTrackMatching = particle_track_matching
    cfg.inputParticleMeasurementsMap = "particle_measurements_map"
    writer = acts.examples.PythonPatternRecognitionPerformanceWriter(cfg, acts.logging.INFO)
    s.addWriter(writer)
    return writer


def _fitter_writer_config(*, tracks_key, track_particle_matching):
    """Shared ``PythonTrackParameterPerformanceWriter.Config`` builder against
    ``particles_selected``.

    ``fitFunction`` is the ROOT-free scipy Gaussian-fit backend, so resolution profiles are fit
    sigmas, not raw RMS. ``resPlotToolConfig``'s residual windows are widened per
    ``RESOLUTION_RESIDUAL_WINDOWS`` (same for SSM and KF, so the two stay comparable).
    """
    cfg = acts.examples.PythonTrackParameterPerformanceWriter.Config()
    cfg.inputTracks = tracks_key
    cfg.inputParticles = "particles_selected"
    cfg.inputTrackParticleMatching = track_particle_matching
    cfg.fitFunction = acts_scipy.makeScipyHistogramFitFunction()

    res_plot_cfg = cfg.resPlotToolConfig
    binning = dict(res_plot_cfg.varBinning)
    for param, (nbins, half_width) in RESOLUTION_RESIDUAL_WINDOWS.items():
        key = f"Residual_{param}"
        binning[key] = acts.Axis.regular(nbins, -half_width, half_width, binning[key].label)
    res_plot_cfg.varBinning = binning
    cfg.resPlotToolConfig = res_plot_cfg
    return cfg


def _add_fitter_writer(s, *, tracks_key, track_particle_matching):
    """``PythonTrackParameterPerformanceWriter`` binned normally vs eta/pT."""
    cfg = _fitter_writer_config(tracks_key=tracks_key, track_particle_matching=track_particle_matching)
    writer = acts.examples.PythonTrackParameterPerformanceWriter(cfg, acts.logging.INFO)
    s.addWriter(writer)
    return writer


def _add_integrated_fitter_writer(s, *, tracks_key, track_particle_matching):
    """Same as ``_add_fitter_writer``, but with the Eta axis collapsed to a single bin so
    ``resmean/reswidth_<param>_vs_eta`` become a single point: the residual fit integrated over
    the full eta AND pT range. See ``extract_integrated_resolution``."""
    cfg = _fitter_writer_config(tracks_key=tracks_key, track_particle_matching=track_particle_matching)
    binning = dict(cfg.resPlotToolConfig.varBinning)
    eta_axis = binning["Eta"]
    binning["Eta"] = acts.Axis.regular(1, eta_axis.edges[0], eta_axis.edges[-1], eta_axis.label)
    res_plot_cfg = cfg.resPlotToolConfig
    res_plot_cfg.varBinning = binning
    cfg.resPlotToolConfig = res_plot_cfg
    writer = acts.examples.PythonTrackParameterPerformanceWriter(cfg, acts.logging.INFO)
    s.addWriter(writer)
    return writer


def _add_track_system(s, *, tracks_key: str, out_prefix: str):
    """Post-fit selection + truth matching + the three performance writers for one track
    population (SSM, KF, or KF-beamspot) -- identical steps, just different input/output keys.

    Denominator = "particles_selected" (post pt/eta/hits cuts) for all three writers below, not
    raw "particles" -- matching itself still runs against full "particles". Deliberately NOT
    "truth_seeded_particles" (isolates KF-fit quality from seeding, N/A for SSM); using the same
    denominator for every system keeps efficiency numbers directly comparable.

    Returns ``(selected_tracks_key, finder_writer, fitter_writer, integrated_writer)``.
    """
    selected_key = f"{out_prefix}_tracks_selected"
    track_particle_matching = f"{out_prefix}_track_particle_matching"
    particle_track_matching = f"{out_prefix}_particle_track_matching"

    s.addAlgorithm(
        acts.examples.TrackSelectorAlgorithm(
            level=acts.logging.INFO,
            inputTracks=tracks_key,
            outputTracks=selected_key,
            selectorConfig=TRACK_SELECTOR_CUTS,
        )
    )
    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=acts.logging.INFO,
            inputTracks=selected_key,
            inputParticles="particles",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching=track_particle_matching,
            outputParticleTrackMatching=particle_track_matching,
            doubleMatching=True,
        )
    )
    finder_writer = _add_finder_writer(
        s, tracks_key=selected_key,
        track_particle_matching=track_particle_matching, particle_track_matching=particle_track_matching,
    )
    fitter_writer = _add_fitter_writer(s, tracks_key=selected_key, track_particle_matching=track_particle_matching)
    integrated_writer = _add_integrated_fitter_writer(
        s, tracks_key=selected_key, track_particle_matching=track_particle_matching,
    )
    return selected_key, finder_writer, fitter_writer, integrated_writer


class TrackCountCollector(acts.examples.IAlgorithm):
    """Exact per-run counts via ``len()`` on ``SimParticleContainer``/``ConstTrackContainer`` --
    NOT via ``trackeff_vs_eta``, whose ``.total`` marginal undercounts the true denominator by
    ~3x (cause not tracked down; this counter matches the writer's own printed log lines).

    ``track_keys`` maps a short label (e.g. ``"ssm"``, ``"kf"``) to the whiteboard key of that
    system's post-selection ``ConstTrackContainer``. After ``s.run()``, ``.counts`` holds
    ``{"particles": N, **{label: N for label in track_keys}}`` summed over all events.
    """

    def __init__(self, *, particles_key: str, track_keys: dict) -> None:
        acts.examples.IAlgorithm.__init__(self, "TrackCountCollector", acts.logging.INFO)
        self.particles = acts.examples.ReadDataHandle(self, acts.examples.SimParticleContainer, "Particles")
        self.particles.initialize(particles_key)
        self.track_handles = {}
        for label, key in track_keys.items():
            h = acts.examples.ReadDataHandle(self, acts.examples.ConstTrackContainer, f"Tracks_{label}")
            h.initialize(key)
            self.track_handles[label] = h
        self.counts = {"particles": 0, **{label: 0 for label in track_keys}}

    def execute(self, context):
        self.counts["particles"] += len(self.particles(context.eventStore))
        for label, h in self.track_handles.items():
            self.counts[label] += len(h(context.eventStore))
        return acts.examples.ProcessCode.SUCCESS


class ResidualCollector(acts.examples.IAlgorithm):
    """Collect per-particle (truth, SSM, KF) perigee parameters for the legacy-style
    offline plots (--dump-residuals).

    Association: the SSM track container preserves ``TruthTrackFinder`` prototrack
    order 1:1 (one prototrack per particle of ``particles_selected``, nothing
    dropped -- enforced with a hard check), so SSM track i == particle i; the
    post-fit d0/z0 selector windows are applied here identically.  KF tracks
    (which lose ~25% to seeding/fit failures, order not preserved) are matched by
    unique best Delta-R < 0.05 to the selected particles -- unambiguous at mrad
    resolutions vs >~0.1 separations.  Truth perigees are recomputed from
    (vertex, momentum, charge) at 3 T with track_regression.perigee.
    """

    def __init__(self, *, ssm_key: str, kf_key: str | None) -> None:
        acts.examples.IAlgorithm.__init__(self, "ResidualCollector", acts.logging.INFO)
        def read(Type, label, key):
            h = acts.examples.ReadDataHandle(self, Type, label)
            h.initialize(key)
            return h
        self.particles = read(acts.examples.SimParticleContainer, "Particles", "particles_selected")
        self.prototracks = read(acts.examples.ProtoTrackContainer, "Prototracks", "prototracks")
        self.ssm = read(acts.examples.ConstTrackContainer, "SsmTracks", ssm_key)
        self.kf = read(acts.examples.ConstTrackContainer, "KfTracks", kf_key) if kf_key else None
        self.rows_truth, self.rows_ssm, self.rows_kf = [], [], []

    def execute(self, context):
        from track_regression.perigee import truth_perigee

        parts = list(self.particles(context.eventStore))
        protos = self.prototracks(context.eventStore)
        n = len(parts)
        if len(protos) != n:  # association contract violated -- refuse silently wrong output
            raise RuntimeError(f"prototracks ({len(protos)}) != selected particles ({n})")
        pos = np.array([np.asarray(p.position) for p in parts], dtype=np.float64).reshape(n, 3)
        mom = np.array([np.asarray(p.momentum) for p in parts], dtype=np.float64).reshape(n, 3)
        q = np.array([p.charge for p in parts], dtype=np.float64)
        d0, z0, phi, theta, qop = truth_perigee(pos[:, 0], pos[:, 1], pos[:, 2],
                                                mom[:, 0], mom[:, 1], mom[:, 2], q, Bz=3.0)
        truth = np.stack([d0, z0, phi, theta, qop], axis=1)
        t_eta = -np.log(np.tan(np.clip(theta, 1e-8, np.pi - 1e-8) / 2.0))

        # SSM: raw container is 1:1 with prototracks in order; apply the d0/z0
        # selector windows here (identical to TrackSelectorAlgorithm's cuts).
        ssm = np.full((n, 5), np.nan)
        ssm_tracks = list(self.ssm(context.eventStore))
        if len(ssm_tracks) != n:
            raise RuntimeError(f"SSM tracks ({len(ssm_tracks)}) != prototracks ({n})")
        for i, track in enumerate(ssm_tracks):
            par = np.asarray(track.parameters)[:5]
            if abs(par[0]) <= TRACK_SELECTOR_D0_MAX_MM and abs(par[1]) <= TRACK_SELECTOR_Z0_MAX_MM:
                ssm[i] = par

        # KF: unique best Delta-R match to the selected particles.
        kf = np.full((n, 5), np.nan)
        if self.kf is not None:
            best = {}
            for track in self.kf(context.eventStore):
                par = np.asarray(track.parameters)[:5]
                k_eta = -np.log(np.tan(np.clip(par[3], 1e-8, np.pi - 1e-8) / 2.0))
                dphi = np.remainder(phi - par[2] + np.pi, 2.0 * np.pi) - np.pi
                dr2 = (t_eta - k_eta) ** 2 + dphi ** 2
                j = int(np.argmin(dr2))
                if dr2[j] < 0.05 ** 2 and dr2[j] < best.get(j, (np.inf, None))[0]:
                    best[j] = (float(dr2[j]), par)
            for j, (_, par) in best.items():
                kf[j] = par
        self.rows_truth.append(truth); self.rows_ssm.append(ssm); self.rows_kf.append(kf)
        return acts.examples.ProcessCode.SUCCESS

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            truth=np.concatenate(self.rows_truth, axis=0),
            ssm=np.concatenate(self.rows_ssm, axis=0),
            kf=np.concatenate(self.rows_kf, axis=0),
        )


class _SurfaceMaterialDecoratorVisitor(acts.TrackingGeometryMutableVisitor):
    def __init__(self, decorator) -> None:
        super().__init__()
        self.decorator = decorator

    def visitSurface(self, surface):
        self.decorator.decorate(surface)


def build_sequencer(
    *,
    particles_dir: Path,
    hits_dir: Path,
    geo_json: Path,
    material_json: Path,
    geoid_map_csv: Path,
    seeding_config: Path,
    ckpt_path: Path,
    model_config_path: Path,
    variant: str,
    events: int,
    output_dir: Path,
    with_baseline: bool = True,
    with_beamspot: bool = False,
    field_strength_t: float = 3.0,
    sort_by_s: bool = True,
    sort_key: str | None = None,
    seed_residual_features: bool = False,
    pt_min_gev: float = 0.5,
    pt_max_gev: float | None = None,
    hit_bounds_tolerance_mm: float = 5.0,
    dump_residuals: bool = False,
):
    field_strength = field_strength_t * u.T

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- geometry: gen3 topology (JSON) + JSON material map ---------------------
    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
    material_decorator = JsonMaterialDecorator(
        rConfig=MaterialMapJsonConverter.Config(),
        jFileName=str(material_json),
        level=acts.logging.INFO,
    )
    tracking_geometry = TrackingGeometryJsonConverter().fromJson(gctx, Path(geo_json).absolute())
    tracking_geometry.apply(_SurfaceMaterialDecoratorVisitor(material_decorator))
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, field_strength))

    # -- model --------------------------------------------------------------
    import common  # scripts/perf/common.py

    common.ensure_src_on_path()
    from track_regression.mamba_short import apply_variant

    cfg = common._load_merged_config(Path(model_config_path).resolve())
    model = common._instantiate(cfg["model"]["model"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model = model.eval().cuda()
    if variant != "v0":
        model = apply_variant(model, variant) or model

    # -- sequencer ------------------------------------------------------------
    s = acts.examples.Sequencer(
        events=events,
        numThreads=1,  # a Python IAlgorithm holding a CUDA model is not thread-safe
        logLevel=acts.logging.INFO,
        outputDir=str(output_dir),
        failOnUnmaskedFpe=False,
    )

    s.addReader(
        ParquetReader(
            level=acts.logging.INFO,
            collections={"cml_particles": str(particles_dir), "cml_hits": str(hits_dir)},
            expectedSchemas={
                "cml_particles": ColliderMLRelease1InputConverter.particleSchema(),
                "cml_hits": ColliderMLRelease1InputConverter.hitSchema(),
            },
        )
    )

    s.addAlgorithm(
        ColliderMLRelease1InputConverter(
            level=acts.logging.INFO,
            inputParticlesTable="cml_particles",
            inputHitsTable="cml_hits",
            outputParticles="particles",
            outputSimHits="simhits",
            outputMeasurements="measurements",
            outputMeasurementSubset="measurement_subset",
            outputClusters="clusters",
            outputMeasSimHitsMap="measurement_simhits_map",
            outputMeasParticlesMap="measurement_particles_map",
            outputParticleMeasurementsMap="particle_measurements_map",
            trackingGeometry=tracking_geometry,
            geoIdMapPath=str(geoid_map_csv),
            geoIdMapSourcePrefix="gen1",
            geoIdMapTargetPrefix="gen3",
            hitBoundsTolerance=hit_bounds_tolerance_mm,
        )
    )

    # Particle-level cuts mirroring core_kf_hits where ParticleSelectorConfig can express them --
    # d0/z0/hard_scatter cuts NOT reproduced, see module docstring.
    s.addWhiteboardAlias("particles_simulated_selected", "particles")
    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            # v2 campaign trains on 1-110 GeV (test where you train); the legacy
            # default (0.5, None) is kept unless --pt-min/--pt-max override it.
            pt=(pt_min_gev * u.GeV, None if pt_max_gev is None else pt_max_gev * u.GeV),
            eta=(-3.0, 3.0),
            measurements=(6, 20),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    s.addAlgorithm(
        acts.examples.TruthTrackFinder(
            level=acts.logging.INFO,
            inputParticles="particles_selected",
            inputMeasurements="measurements",
            inputParticleMeasurementsMap="particle_measurements_map",
            inputSimHits="simhits",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            outputProtoTracks="prototracks",
        )
    )

    ssm_fitter = add_ssm_algorithm(
        s, model=model, geoid_csv=str(geoid_map_csv), tracking_geometry=tracking_geometry, sort_by_s=sort_by_s,
        sort_key=sort_key, seed_residual_features=seed_residual_features,
    )
    _, ssm_finder_writer, ssm_fitter_writer, ssm_integrated_writer = _add_track_system(
        s, tracks_key="ssm_tracks", out_prefix="ssm",
    )
    writers = {"SSM": (ssm_finder_writer, ssm_fitter_writer, ssm_integrated_writer)}
    track_keys = {"ssm": "ssm_tracks_selected"}

    if with_baseline:
        addSeeding(
            s,
            trackingGeometry=tracking_geometry,
            field=field,
            rnd=acts.examples.RandomNumbers(seed=42),
            seedingAlgorithm=SeedingAlgorithm.TruthEstimated,
            selectedParticles="particles_selected",
            geoSelectionConfigFile=str(seeding_config),
            initialSigmas=[1 * u.mm, 1 * u.mm, 1 * u.degree, 1 * u.degree, 0 / u.GeV, 1 * u.ns],
            initialSigmaQoverPt=0.1 / u.GeV,
            initialSigmaPtRel=0.1,
            initialVarInflation=[1e0] * 6,
            logLevel=acts.logging.INFO,
        )
        addKalmanTracks(s, trackingGeometry=tracking_geometry, field=field, logLevel=acts.logging.INFO)

        # "kf_sel" (not "kf"): addKalmanTracks() already writes "kf_track_particle_matching"/
        # "kf_particle_track_matching" internally, matching raw pre-selection "kf_tracks" -- a
        # different prefix avoids colliding with those.
        _, kf_finder_writer, kf_fitter_writer, kf_integrated_writer = _add_track_system(
            s, tracks_key="kf_tracks", out_prefix="kf_sel",
        )
        writers["ACTS KF"] = (kf_finder_writer, kf_fitter_writer, kf_integrated_writer)
        track_keys["kf"] = "kf_sel_tracks_selected"

        if with_beamspot:
            # Beam-constrained refit (not applied to SSM -- see BEAMSPOT_*_STDDEV_MM): RefittingAlgorithm
            # reuses each track's own filtered/smoothed states from the KF fit above and injects an
            # extra pseudo-measurement at the perigee with this covariance, tightening d0/z0.
            # Off by default: broken in current acts main.
            beamspot_constraint = acts.SquareMatrix2.Zero()
            beamspot_constraint[0, 0] = (BEAMSPOT_D0_STDDEV_MM * u.mm) ** 2
            beamspot_constraint[1, 1] = (BEAMSPOT_Z0_STDDEV_MM * u.mm) ** 2
            s.addAlgorithm(
                acts.examples.RefittingAlgorithm(
                    level=acts.logging.INFO,
                    inputTracks="kf_tracks",
                    outputTracks="kf_refit_tracks",
                    fit=acts.examples.makeKalmanFitterFunction(
                        tracking_geometry, field,
                        multipleScattering=True, energyLoss=True,
                        reverseFilteringMomThreshold=0 * u.GeV, reverseFilteringCovarianceScaling=100.0,
                        freeToBoundCorrection=acts.examples.FreeToBoundCorrection(False),
                        chi2Cut=float("inf"), useJosephFormulation=False, level=acts.logging.INFO,
                    ),
                    beamSpotConstraint=beamspot_constraint,
                )
            )
            _, kfbs_finder_writer, kfbs_fitter_writer, kfbs_integrated_writer = _add_track_system(
                s, tracks_key="kf_refit_tracks", out_prefix="kf_beamspot",
            )
            writers["ACTS KF (beamspot)"] = (kfbs_finder_writer, kfbs_fitter_writer, kfbs_integrated_writer)
            track_keys["kf_beamspot"] = "kf_beamspot_tracks_selected"

    counter = TrackCountCollector(particles_key="particles_selected", track_keys=track_keys)
    s.addAlgorithm(counter)

    residuals = None
    if dump_residuals:
        residuals = ResidualCollector(
            ssm_key="ssm_tracks",
            kf_key="kf_sel_tracks_selected" if with_baseline else None,
        )
        s.addAlgorithm(residuals)

    return s, writers, ssm_fitter, counter, residuals


def main() -> None:
    global TRACK_SELECTOR_D0_MAX_MM, TRACK_SELECTOR_Z0_MAX_MM, TRACK_SELECTOR_CUTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles-dir", type=Path)
    ap.add_argument("--hits-dir", type=Path)
    ap.add_argument("--geo-json", type=Path)
    ap.add_argument("--material-json", type=Path)
    ap.add_argument("--geoid-map-csv", type=Path)
    ap.add_argument("--seeding-config", type=Path)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--config", type=Path, help="Model YAML config")
    ap.add_argument("--variant", default="v0", choices=("v0", "v2p", "v3", "v3c", "v4", "v5", "v5p", "v5pc"))
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--output", "-o", type=Path, default=Path.cwd() / "acts_validation_output")
    ap.add_argument("--no-baseline", action="store_true", help="Skip the ACTS KF baseline fit")
    ap.add_argument(
        "--beamspot", action="store_true",
        help="Also run the KF beamspot-constrained refit (off by default: broken in current acts main)",
    )
    ap.add_argument(
        "--no-sort-by-s", action="store_true",
        help="Skip the default re-sort of each prototrack's hits by s=sqrt(x^2+y^2+z^2) -- use "
        "TruthTrackFinder's truth-time order instead (the mismatched-with-training ordering).",
    )
    ap.add_argument(
        "--from-pickle", type=Path, default=None,
        help="Skip the ACTS run entirely and re-plot resolutions.pdf from a previous run's "
        "resolution_data.pkl (all the ACTS/model/data args above are ignored)",
    )
    # --- v2 (campaign-2) checkpoint support --------------------------------
    ap.add_argument(
        "--sort-key", choices=("s", "geometry", "none"), default=None,
        help="Hit ordering fed to the model. v2 checkpoints need 'geometry' (detector order, "
        "= the v2 stores); legacy checkpoints 's'. Overrides --no-sort-by-s when given.",
    )
    ap.add_argument(
        "--seed-residual-features", action="store_true",
        help="Append the 3 per-hit residuals to the ACTS three-pixel-point seed helix as input "
        "features 12-14 (v2 P'-style checkpoints, input_dim 15). Seed anchors for anchored "
        "heads are detected automatically from the checkpoint's loss module either way.",
    )
    ap.add_argument("--d0-max", type=float, default=TRACK_SELECTOR_D0_MAX_MM,
                    help="Post-fit |d0| selector window [mm] (v2 datasets: 7.1)")
    ap.add_argument("--z0-max", type=float, default=TRACK_SELECTOR_Z0_MAX_MM,
                    help="Post-fit |z0| selector window [mm] (v2 datasets: 270)")
    ap.add_argument("--pt-min", type=float, default=0.5, help="Particle-level pT floor [GeV] (v2: 1.0)")
    ap.add_argument("--pt-max", type=float, default=None, help="Particle-level pT ceiling [GeV] (v2: 110)")
    ap.add_argument("--residual-window", action="append", default=None, metavar="PARAM:NBINS:HALF",
                    help="Override a Residual_<param> histogram window, e.g. qop:120:0.006 "
                         "(repeatable; stock windows are too coarse for fixed-pT muon residuals)")
    ap.add_argument("--dump-residuals", action="store_true",
                    help="Also save per-particle (truth, SSM, KF) perigee params to "
                         "matched_residuals.npz in the output dir (for the legacy-style plots)")
    ap.add_argument("--hit-bounds-tolerance", type=float, default=5.0,
                    help="Converter sensor-bounds tolerance [mm]; one v2 hit beyond it aborts the whole "
                         "run (converter throws), so productions use a loose 25")
    args = ap.parse_args()

    override_residual_windows(args.residual_window)
    if (args.d0_max, args.z0_max) != (TRACK_SELECTOR_D0_MAX_MM, TRACK_SELECTOR_Z0_MAX_MM):
        TRACK_SELECTOR_D0_MAX_MM, TRACK_SELECTOR_Z0_MAX_MM = args.d0_max, args.z0_max
        TRACK_SELECTOR_CUTS = _make_track_selector_cuts()

    if args.from_pickle is not None:
        with open(args.from_pickle, "rb") as f:
            bundle = pickle.load(f)
        args.output.mkdir(parents=True, exist_ok=True)
        resolutions_pdf = args.output / "resolutions.pdf"
        plot_resolutions(
            bundle["data"], resolutions_pdf,
            integrated_by_label=bundle.get("integrated"),
            raw_residuals_by_label=bundle.get("raw_residuals"),
            summary=bundle.get("summary"),
        )
        print(f"[runner] resolution plots re-plotted from {args.from_pickle} to {resolutions_pdf}", flush=True)
        return

    required = {
        "--particles-dir": args.particles_dir, "--hits-dir": args.hits_dir, "--geo-json": args.geo_json,
        "--material-json": args.material_json, "--geoid-map-csv": args.geoid_map_csv,
        "--seeding-config": args.seeding_config, "--ckpt": args.ckpt, "--config": args.config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        ap.error(f"the following arguments are required (unless --from-pickle is given): {', '.join(missing)}")

    field_strength_t = 3.0  # build_sequencer's default -- matches ColliderML's ttbar_pu200 field
    s, writers, ssm_fitter, counter, residuals = build_sequencer(
        particles_dir=args.particles_dir,
        hits_dir=args.hits_dir,
        geo_json=args.geo_json,
        material_json=args.material_json,
        geoid_map_csv=args.geoid_map_csv,
        seeding_config=args.seeding_config,
        ckpt_path=args.ckpt,
        model_config_path=args.config,
        variant=args.variant,
        events=args.events,
        output_dir=args.output,
        with_baseline=not args.no_baseline,
        with_beamspot=args.beamspot,
        field_strength_t=field_strength_t,
        sort_by_s=not args.no_sort_by_s,
        sort_key=args.sort_key,
        seed_residual_features=args.seed_residual_features,
        pt_min_gev=args.pt_min,
        pt_max_gev=args.pt_max,
        hit_bounds_tolerance_mm=args.hit_bounds_tolerance,
        dump_residuals=args.dump_residuals,
    )
    s.run()
    if residuals is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        residuals.save(args.output / "matched_residuals.npz")
        print(f"[runner] matched residuals saved to {args.output / 'matched_residuals.npz'}", flush=True)

    print(
        f"[runner] SSM tracks seen={ssm_fitter.n_tracks_seen} kept={ssm_fitter.n_tracks_kept} "
        f"dropped_hitcount={ssm_fitter.n_tracks_dropped_hitcount} "
        f"dropped_geoid={ssm_fitter.n_tracks_dropped_geoid}",
        flush=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    data_by_label, integrated_by_label, raw_residuals_by_label, counts_by_label = {}, {}, {}, {}
    report_lines = []
    for label, (finder_w, fitter_w, integrated_w) in writers.items():
        data_by_label[label] = extract_resolution_data(fitter_w.histograms())
        integrated_by_label[label] = extract_integrated_resolution(integrated_w.histograms())
        raw_residuals_by_label[label] = extract_raw_residual_data(integrated_w.histograms())
        counts_by_label[label] = extract_track_counts(finder_w, counter.counts["particles"])
        report_lines.append(_format_report_line(label, counts_by_label[label], integrated_by_label[label]))

    report = "\n".join(report_lines)
    print(report, flush=True)
    (args.output / "report.txt").write_text(report + "\n")

    summary = {
        "Input files": [
            f"particles-dir: {args.particles_dir}",
            f"hits-dir: {args.hits_dir}",
            f"geo-json: {args.geo_json}",
            f"material-json: {args.material_json}",
            f"geoid-map-csv: {args.geoid_map_csv}",
            f"seeding-config: {args.seeding_config}",
            f"ckpt: {args.ckpt}",
            f"config: {args.config}",
            f"variant: {args.variant}    events: {args.events}",
        ],
        "Particle selection (ParticleSelectorConfig, upstream of both SSM and KF)": [
            f"pt in [{args.pt_min}, {args.pt_max if args.pt_max is not None else 'inf'}] GeV, |eta| <= 3.0, measurements in [6, 20]",
            "charged only, primary only",
            "NOT reproduced here: core_kf_hits's |d0|<=2.5mm/|z0|<=200mm/hard_scatter windows",
        ],
        "Track selection (TrackSelectorAlgorithm, identical cuts for every system post-fit)": [
            f"minMeasurements: {TRACK_SELECTOR_MIN_MEASUREMENTS}",
            f"|d0| (loc0) <= {TRACK_SELECTOR_D0_MAX_MM} mm",
            f"|z0| (loc1) <= {TRACK_SELECTOR_Z0_MAX_MM} mm",
        ],
        "KF baseline field": [f"Constant B_z = {field_strength_t} T"],
        **({
            "KF-beamspot refit (RefittingAlgorithm, separate branch off the raw KF fit)": [
                f"d0 stddev: {BEAMSPOT_D0_STDDEV_MM} mm, z0 stddev: {BEAMSPOT_Z0_STDDEV_MM} mm",
                "not applied to SSM tracks -- no fitted covariance to refit from",
            ],
        } if args.beamspot else {}),
        "Track counts (post-selection, matched via TrackTruthMatcher)": [
            f"{label}: {c['matched']:.0f} matched / {c['total']:.0f} total"
            for label, c in counts_by_label.items()
        ],
    }

    bundle = {
        "data": data_by_label, "integrated": integrated_by_label,
        "raw_residuals": raw_residuals_by_label, "summary": summary,
    }
    pickle_path = args.output / "resolution_data.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[runner] resolution data pickled to {pickle_path}", flush=True)

    resolutions_pdf = args.output / "resolutions.pdf"
    plot_resolutions(
        data_by_label, resolutions_pdf, integrated_by_label=integrated_by_label,
        raw_residuals_by_label=raw_residuals_by_label, summary=summary,
    )
    print(f"[runner] resolution plots written to {resolutions_pdf}", flush=True)


if __name__ == "__main__":
    main()
