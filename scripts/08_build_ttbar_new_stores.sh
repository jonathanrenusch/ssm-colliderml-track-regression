#!/bin/bash
# Build the stores from the NEW NERSC ttbar runs (CLAUDE.md §4.7) once scripts/06_fetch_nersc_ttbar.sh is done:
#   * eval store  ttbar_new_pt1  = runs 6-45,  pT >= 1 GeV (user's ttbar testing limit), truth-KF side-cars
#   * train store ttbar_new      = runs 46-784, standard core selection (pT >= 0.5), 4 runs per part
#   * eval-farm entry ICLR_eval_geom/ttbar_new_pt1/test (union of the store's splits, as 05_rebuild does)
#   * KF baselines for the new eval store
#   * mixed training store ICLR_retraining_geom_mixed8M = uniform muons + first ttbar_new train parts up to 8 M tracks
#   bash scripts/08_build_ttbar_new_stores.sh [eval_first=6] [eval_last=45] [train_last=784] [extra_max_tracks=8000000]
set -euo pipefail
E0="${1:-6}"; E1="${2:-45}"; T1="${3:-784}"; MAXX="${4:-8000000}"
RAW=/scratch/colliderml/drift_beamspot/ttbar/v1/runs
OUT="${OUT:-/scratch/colliderml/ICLR_retraining_geom}"; EV="${EV:-/scratch/colliderml/ICLR_eval_geom}"; BZ="${BZ:-2.0}"
MIXED="${MIXED:-/scratch/colliderml/ICLR_retraining_geom_mixed8M}"; BOUT="${BOUT:-eval_plots/baselines_KF_ttbar_new_pt1}"
REPO=/shared/tracking/ssm-colliderml-track-regression
missing=0; for n in $(seq $E0 $T1); do [ -f $RAW/$n/.fetched ] || { missing=$((missing+1)); }; done
echo "runs $E0-$T1: $missing not fetched"; [ $missing -eq 0 ] || { echo "fetch incomplete — abort"; exit 1; }
farm () { local d=$1; shift; rm -rf "$d"; mkdir -p "$d"; for n in "$@"; do ln -sfn "$RAW/$n" "$d/$n"; done; }
farm /scratch/colliderml/drift_beamspot/ttbar_new_eval/v1/runs  $(seq $E0 $E1)
farm /scratch/colliderml/drift_beamspot/ttbar_new_train/v1/runs $(seq $((E1+1)) $T1)
cd $REPO/src/track_regression
echo "=== preprocess eval store (pT >= 1) $(date)"
pixi run -e default python scripts/preprocess_flat.py --data-dir /scratch/colliderml/drift_beamspot/ttbar_new_eval/v1 \
    --output-dir $OUT/ttbar_new_pt1 --num-workers 20 --pt-min 1.0 --bz $BZ 2>&1 | grep -E "DONE|ERROR|Traceback|shards ->"
echo "=== preprocess train store $(date)"
pixi run -e default python scripts/preprocess_flat.py --data-dir /scratch/colliderml/drift_beamspot/ttbar_new_train/v1 \
    --output-dir $OUT/ttbar_new --num-workers 40 --shards-per-part 4 --bz $BZ 2>&1 | grep -E "DONE|ERROR|Traceback|shards ->"
echo "=== eval farm entry $(date)"
OUT="$OUT" EV="$EV" pixi run -e default python - <<'PY'
import json, os
from pathlib import Path
OUT=Path(os.environ['OUT']); EV=Path(os.environ['EV']); ds='ttbar_new_pt1'
d=EV/ds; (d/'test').mkdir(parents=True, exist_ok=True)
for sp in ('train','val'):
    if not (d/sp).exists() and (OUT/ds/sp).exists(): os.symlink(OUT/ds/sp, d/sp)
parts=[]; k=0; man=None
for sp in ('train','val','test'):
    if not (OUT/ds/sp/'manifest.json').exists(): continue
    m=json.load(open(OUT/ds/sp/'manifest.json')); man=man or m
    for p in m['parts']:
        name=f'part_{k:04d}'; k+=1
        if not (d/'test'/name).exists(): os.symlink(OUT/ds/sp/p['name'], d/'test'/name)
        parts.append({'name':name,'n_tracks':p['n_tracks'],'n_hits':p['n_hits']})
man['parts']=parts; man['n_tracks']=sum(p['n_tracks'] for p in parts); man['n_hits']=sum(p['n_hits'] for p in parts)
json.dump(man, open(d/'test'/'manifest.json','w'), indent=1); print(ds, man['n_tracks'], 'eval tracks')
PY
echo "=== KF baselines $(date)"
pixi run -e default python scripts/kf_baselines.py --stores ttbar_new_pt1="$EV/ttbar_new_pt1/test" --out-dir $REPO/$BOUT 2>&1 | tail -3
echo "=== mixed store $(date)"
pixi run -e default python $REPO/scripts/07_build_mixed_store.py --base $OUT/single_muon_uniform --extra $OUT/ttbar_new \
    --extra-max-tracks $MAXX --out $MIXED
echo "ALL DONE $(date)"
