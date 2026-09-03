#!/bin/bash
# The raw ttbar tree now holds runs 0-784; the legacy 'ttbar' eval sample is runs 0-5 only.
# Rebuild ICLR_retraining_geom_B3/ttbar from the runs-0-5 farm (3 T targets) and redo its eval-farm entry + KF baseline.
set -euo pipefail
OUT=/scratch/colliderml/ICLR_retraining_geom_B3; EV=/scratch/colliderml/ICLR_eval_geom_B3; REPO=/shared/tracking/ssm-colliderml-track-regression
cd $REPO/src/track_regression
rm -rf $OUT/ttbar $EV/ttbar
pixi run -e default python scripts/preprocess_flat.py --sort-key geometry --bz 3.0 --data-dir /scratch/colliderml/drift_beamspot/ttbar_r0-5/v1 \
    --output-dir $OUT/ttbar --num-workers 12 2>&1 | grep -E "DONE|ERROR|Traceback"
OUT="$OUT" EV="$EV" pixi run -e default python - <<'PY'
import json, os
from pathlib import Path
OUT=Path(os.environ['OUT']); EV=Path(os.environ['EV']); ds='ttbar'
d=EV/ds; (d/'test').mkdir(parents=True, exist_ok=True)
for sp in ('train','val'):
    if not (d/sp).exists(): os.symlink(OUT/ds/sp, d/sp)
parts=[]; k=0; man=None
for sp in ('train','val','test'):
    m=json.load(open(OUT/ds/sp/'manifest.json')); man=man or m
    for p in m['parts']:
        name=f'part_{k:04d}'; k+=1; os.symlink(OUT/ds/sp/p['name'], d/'test'/name)
        parts.append({'name':name,'n_tracks':p['n_tracks'],'n_hits':p['n_hits']})
man['parts']=parts; man['n_tracks']=sum(p['n_tracks'] for p in parts); man['n_hits']=sum(p['n_hits'] for p in parts)
json.dump(man, open(d/'test'/'manifest.json','w'), indent=1); print(ds, man['n_tracks'], 'eval tracks')
PY
pixi run -e default python scripts/kf_baselines.py --stores ttbar="$EV/ttbar/test" --out-dir $REPO/eval_plots/baselines_KF_geom_B3/ttbar_r0-5 2>&1 | tail -2
echo "B3 ttbar (runs 0-5) DONE $(date)"
