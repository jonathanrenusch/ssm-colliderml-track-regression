#!/bin/bash
# Rebuild the five drift_beamspot stores with the truth targets transported in the MEASURED
# 3 T field (CLAUDE.md §4.8) into NEW roots; the 2 T stores stay for the running sweep-3 runs.
# Same seed/split/sort as 05_rebuild_iclr_stores.sh, truth-KF side-cars now written by
# preprocess_flat itself (per-shard join), eval farm + KF baselines as in 05.
#   bash scripts/09_rebuild_stores_bz3.sh [store_root] [eval_root] [baseline_out]
set -euo pipefail
OUT="${1:-/scratch/colliderml/ICLR_retraining_geom_B3}"; EV="${2:-/scratch/colliderml/ICLR_eval_geom_B3}"
BOUT="${3:-eval_plots/baselines_KF_geom_B3}"; BZ=3.0
REPO=/shared/tracking/ssm-colliderml-track-regression
cd $REPO/src/track_regression; mkdir -p "$OUT" "$EV"
for ds in single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar single_muon_uniform; do
  echo "=== preprocess $ds (geometry, Bz=$BZ) $(date)"
  pixi run -e default python scripts/preprocess_flat.py --sort-key geometry --bz $BZ \
      --data-dir /scratch/colliderml/drift_beamspot/$ds/v1 --output-dir "$OUT/$ds" --num-workers 40 2>&1 | grep -E "DONE|ERROR|Traceback"
done
echo "=== rebalance uniform to 192/5/5 $(date)"
OUT="$OUT" pixi run -e default python - <<'PY'
import json, shutil, os
from pathlib import Path
root = Path(os.environ['OUT']) / 'single_muon_uniform'
moves = [(f'val/part_{i:04d}', f'train/part_{182+i-5:04d}') for i in range(5,10)] + \
        [(f'test/part_{i:04d}', f'train/part_{187+i-5:04d}') for i in range(5,10)]
for src, dst in moves:
    assert (root/src).exists() and not (root/dst).exists(), (src, dst)
    shutil.move(str(root/src), str(root/dst))
for sp in ('train','val','test'):
    man = json.load(open(root/sp/'manifest.json')); parts = []
    for d in sorted((root/sp).glob('part_*')):
        m = json.load(open(d/'meta.json')); parts.append({'name': d.name, 'n_tracks': m['n_tracks'], 'n_hits': m['n_hits']})
    man['parts'] = parts; man['n_tracks'] = sum(p['n_tracks'] for p in parts); man['n_hits'] = sum(p['n_hits'] for p in parts)
    json.dump(man, open(root/sp/'manifest.json','w'), indent=1); print(sp, len(parts), 'parts', man['n_tracks'], 'tracks')
meta = json.load(open(root/'dataset_meta.json')); meta['split_rebalanced'] = {'moved_to_train': [f'{s}->{d}' for s,d in moves]}
json.dump(meta, open(root/'dataset_meta.json','w'), indent=1)
PY
echo "=== eval farm $(date)"
ln -sfn "$OUT/single_muon_uniform" "$EV/single_muon_uniform"
OUT="$OUT" EV="$EV" pixi run -e default python - <<'PY'
import json, os
from pathlib import Path
OUT=Path(os.environ['OUT']); EV=Path(os.environ['EV'])
for ds in ('single_muon_2GeV','single_muon_10GeV','single_muon_100GeV','ttbar'):
    d=EV/ds; (d/'test').mkdir(parents=True, exist_ok=True)
    for sp in ('train','val'):
        if not (d/sp).exists(): os.symlink(OUT/ds/sp, d/sp)
    parts=[]; k=0; man=None
    for sp in ('train','val','test'):
        m=json.load(open(OUT/ds/sp/'manifest.json')); man=man or m
        for p in m['parts']:
            name=f'part_{k:04d}'; k+=1
            if not (d/'test'/name).exists(): os.symlink(OUT/ds/sp/p['name'], d/'test'/name)
            parts.append({'name':name,'n_tracks':p['n_tracks'],'n_hits':p['n_hits']})
    man['parts']=parts; man['n_tracks']=sum(p['n_tracks'] for p in parts); man['n_hits']=sum(p['n_hits'] for p in parts)
    json.dump(man, open(d/'test'/'manifest.json','w'), indent=1); print(ds, man['n_tracks'], 'eval tracks')
PY
echo "=== truth-KF coverage check $(date)"
OUT="$OUT" pixi run -e default python - <<'PY'
import json, os, numpy as np
from pathlib import Path
OUT=Path(os.environ['OUT'])
for ds in ('single_muon_uniform','single_muon_2GeV','single_muon_10GeV','single_muon_100GeV','ttbar'):
    for sp in ('test',):
        man=json.load(open(OUT/ds/sp/'manifest.json')); n=ok=0
        for p in man['parts']:
            f=OUT/ds/sp/p['name']/'truth_kf_reco.npy'
            if f.exists(): a=np.load(f, mmap_mode='r'); n+=len(a); ok+=int(np.isfinite(a[:,0]).sum())
        print(f"{ds}/{sp}: truth-KF side-car on {n:,} tracks, matched {100*ok/max(n,1):.2f} %")
PY
echo "=== KF baselines $(date)"
pixi run -e default python scripts/kf_baselines.py --stores single_muon_uniform="$EV/single_muon_uniform/test" \
   single_muon_2GeV="$EV/single_muon_2GeV/test" single_muon_10GeV="$EV/single_muon_10GeV/test" \
   single_muon_100GeV="$EV/single_muon_100GeV/test" ttbar="$EV/ttbar/test" --out-dir "$REPO/$BOUT" 2>&1 | tail -2
echo "ALL DONE $(date)"
