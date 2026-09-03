#!/bin/bash
# Rebuild the five ICLR (drift_beamspot) flat stores with scripts/preprocess_flat.py
# (event_id-value join, chosen --sort-key), reproduce the 192/5/5 uniform split
# of the original stores, build the eval symlink farm (fixed-pT and ttbar
# test/ = union of all parts), attach truth-KF side-cars and recompute the KF
# baselines.  This is the script that produced ICLR_retraining_{ssort,geom} on
# 2026-08-25 (CLAUDE.md §0.1).  The eval farm is symlinks only, so it is NOT
# copied to /eos — re-run the "eval farm" block (or this script with the
# preprocess step skipped, since completed parts are reused via _complete) to
# recreate it after restoring a store from /eos.
#
#   bash scripts/05_rebuild_iclr_stores.sh <s|geometry> <store_root> <eval_root> <baseline_out_dir>
#   e.g. bash scripts/05_rebuild_iclr_stores.sh s /scratch/colliderml/ICLR_retraining_ssort \
#            /scratch/colliderml/ICLR_eval_ssort eval_plots/baselines_KF_rebuilt_ssort
set -uo pipefail
SORT="$1"; OUT="$2"; EV="$3"; BOUT="$4"
cd /shared/tracking/ssm-colliderml-track-regression/src/track_regression
mkdir -p "$OUT" "$EV"
for ds in single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar single_muon_uniform; do
  echo "=== preprocess $ds ($SORT) $(date)"
  pixi run -e default python scripts/preprocess_flat.py --sort-key "$SORT" \
      --data-dir /scratch/colliderml/drift_beamspot/$ds/v1 --output-dir "$OUT/$ds" --num-workers 40 2>&1 | grep -E "DONE|ERROR|Traceback"
done
echo "=== rebalance uniform to 192/5/5 (identical part moves to the original store) $(date)"
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
echo "=== truth-KF side-cars $(date)"
pixi run -e default python scripts/extract_truth_kf.py --store "$OUT/single_muon_uniform/test" \
   --truth-tracks-glob '/scratch/colliderml/drift_beamspot/single_muon_uniform/v1/parquet/reco/truth_tracks/*.parquet' 2>&1 | tail -2
for ds in single_muon_2GeV single_muon_10GeV single_muon_100GeV; do for sp in train val test; do
  pixi run -e default python scripts/extract_truth_kf.py --store "$OUT/$ds/$sp" --drop-event-range 0 10000 \
     --truth-tracks-glob "/scratch/colliderml/drift_beamspot/$ds/v1/parquet/reco/truth_tracks/*.parquet" 2>&1 | tail -1
done; done
echo "=== KF baselines $(date)"
pixi run -e default python scripts/kf_baselines.py --stores single_muon_uniform="$EV/single_muon_uniform/test" \
   single_muon_2GeV="$EV/single_muon_2GeV/test" single_muon_10GeV="$EV/single_muon_10GeV/test" \
   single_muon_100GeV="$EV/single_muon_100GeV/test" ttbar="$EV/ttbar/test" --out-dir "$BOUT" 2>&1 | tail -2
echo "=== verify $(date)"
OUT="$OUT" SORT="$SORT" pixi run -e default python - <<'PY'
import numpy as np, json, os
OUT=os.environ['OUT']; SORT=os.environ['SORT']
from track_regression.hit_sorting import geometry_keys
for ds, sp in [('single_muon_uniform','train'),('single_muon_uniform','test'),('ttbar','test'),('single_muon_2GeV','train')]:
    man=json.load(open(f'{OUT}/{ds}/{sp}/manifest.json')); p=f'{OUT}/{ds}/{sp}/{man["parts"][-1]["name"]}/'
    hits=np.load(p+'hits.npy',mmap_mode='r'); off=np.load(p+'offsets.npy'); ln=np.load(p+'lengths.npy'); tg=np.load(p+'targets.npy')
    n=min(20000,len(ln)); ok=0; mis=0
    for i in range(n):
        h=np.asarray(hits[off[i]:off[i]+ln[i]])
        if SORT=='s': ok+=np.all(np.diff(h[:,6])>=0)
        else:
            pr,se=geometry_keys(h[:,:3],h[:,7]); ok+=np.array_equal(np.lexsort((se,pr)),np.arange(len(h)))
        inner=h[np.argmin(h[:,3])]; dphi=np.angle(np.exp(1j*(np.arctan2(inner[1],inner[0])-tg[i,2]))); mis+=abs(dphi)>0.7
    print(f'{ds}/{sp}: hit_sort_key={man.get("hit_sort_key")} parts={len(man["parts"])} tracks={man["n_tracks"]:,} | order ok {ok}/{n} | mislabel proxy (|dphi|>0.7) {100*mis/n:.2f}%')
meta=json.load(open(f'{OUT}/single_muon_uniform/dataset_meta.json')); print('uniform splits', {k:v['n_tracks'] for k,v in meta['splits'].items()}, 'rebalanced', 'split_rebalanced' in meta)
old=np.concatenate([np.load(f'/scratch/colliderml/ICLR_retraining/single_muon_uniform/test/part_{i:04d}/track_event_ids.npy') for i in range(5)])
new=np.concatenate([np.load(f'{OUT}/single_muon_uniform/test/part_{i:04d}/track_event_ids.npy') for i in range(5)])
print('test split: same event set as the original store:', np.array_equal(np.sort(old), np.sort(new)), len(old), len(new))
PY
echo "REBUILD DONE ($SORT) $(date)"
