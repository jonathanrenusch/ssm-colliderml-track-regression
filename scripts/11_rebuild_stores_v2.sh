#!/bin/bash
# v2 stores (2026-08-28 night): re-produced NERSC data (drift_beamspot_v2), hits in ACTS TRUE-TIME order
# (tracker_simhits.true_time; the digitised time still has no strip times), 3 T targets, |d0| <= 7.1 mm and
# |z0| <= 270 mm (the target normalisation ranges of the configs), ttbar 1 <= pT <= 110 GeV, truth-KF side-cars.
#   bash scripts/11_rebuild_stores_v2.sh [raw=/scratch/colliderml/drift_beamspot_v2] [out=/scratch/colliderml/ICLR_retraining_v2] [eval=/scratch/colliderml/ICLR_eval_v2]
set -euo pipefail
RAW="${1:-/scratch/colliderml/drift_beamspot_v2}"; OUT="${2:-/scratch/colliderml/ICLR_retraining_v2}"; EV="${3:-/scratch/colliderml/ICLR_eval_v2}"
REPO=/shared/tracking/ssm-colliderml-track-regression; cd $REPO/src/track_regression
COMMON="--sort-key true_time --bz 3.0 --apply-d0z0-windows --d0-window 7.1 --z0-window 270"
mkdir -p "$OUT" "$EV"
for ds in single_muon_2GeV single_muon_10GeV single_muon_100GeV single_muon_uniform; do
  echo "=== preprocess $ds $(date)"
  pixi run -e default python scripts/preprocess_flat.py $COMMON --data-dir $RAW/$ds/v1 --output-dir "$OUT/$ds" --num-workers 40 2>&1 | grep -E "DONE|ERROR|Traceback|shards ->|true_time\]"
done
# ttbar: legacy 6-run sample (runs 0-5, now re-produced) = eval 'ttbar'; runs 6-45 = eval 'ttbar_new_pt1' ; runs 46-784 = training 'ttbar_new_pt1_tr'
farm () { local d=$1; shift; rm -rf "$d"; mkdir -p "$d"; for n in "$@"; do ln -sfn "$RAW/ttbar/v1/runs/$n" "$d/$n"; done; }
farm $RAW/ttbar_r0-5/v1/runs $(seq 0 5); farm $RAW/ttbar_new_eval/v1/runs $(seq 6 45); farm $RAW/ttbar_new_train/v1/runs $(seq 46 784)
echo "=== preprocess ttbar (runs 0-5, 1-110 GeV) $(date)"
pixi run -e default python scripts/preprocess_flat.py $COMMON --pt-min 1.0 --pt-max 110 --data-dir $RAW/ttbar_r0-5/v1 --output-dir "$OUT/ttbar" --num-workers 12 2>&1 | grep -E "DONE|ERROR|Traceback|shards ->"
echo "=== preprocess ttbar_new_pt1 (runs 6-45, eval) $(date)"
pixi run -e default python scripts/preprocess_flat.py $COMMON --pt-min 1.0 --pt-max 110 --data-dir $RAW/ttbar_new_eval/v1 --output-dir "$OUT/ttbar_new_pt1" --num-workers 20 2>&1 | grep -E "DONE|ERROR|Traceback|shards ->"
echo "=== preprocess ttbar_new_pt1_tr (runs 46-784, training) $(date)"
pixi run -e default python scripts/preprocess_flat.py $COMMON --pt-min 1.0 --pt-max 110 --data-dir $RAW/ttbar_new_train/v1 --output-dir "$OUT/ttbar_new_pt1_tr" --num-workers 40 --shards-per-part 4 2>&1 | grep -E "DONE|ERROR|Traceback|shards ->"
echo "=== rebalance uniform to 192/5/5 $(date)"
OUT="$OUT" pixi run -e default python - <<'PY'
import json, shutil, os
from pathlib import Path
root = Path(os.environ['OUT']) / 'single_muon_uniform'
moves = [(f'val/part_{i:04d}', f'train/part_{182+i-5:04d}') for i in range(5,10)] + [(f'test/part_{i:04d}', f'train/part_{187+i-5:04d}') for i in range(5,10)]
for src, dst in moves:
    if (root/src).exists() and not (root/dst).exists(): shutil.move(str(root/src), str(root/dst))
for sp in ('train','val','test'):
    man = json.load(open(root/sp/'manifest.json')); parts = []
    for d in sorted((root/sp).glob('part_*')):
        m = json.load(open(d/'meta.json')); parts.append({'name': d.name, 'n_tracks': m['n_tracks'], 'n_hits': m['n_hits']})
    man['parts'] = parts; man['n_tracks'] = sum(p['n_tracks'] for p in parts); man['n_hits'] = sum(p['n_hits'] for p in parts)
    json.dump(man, open(root/sp/'manifest.json','w'), indent=1)
meta = json.load(open(root/'dataset_meta.json')); meta['split_rebalanced'] = {'moved_to_train': [f'{s}->{d}' for s,d in moves]}
json.dump(meta, open(root/'dataset_meta.json','w'), indent=1)
PY
echo "=== eval farm + KF baselines $(date)"
pixi run -e default python $REPO/scripts/build_eval_farm.py --store-root $OUT --eval-root $EV --union single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar ttbar_new_pt1 --link single_muon_uniform
pixi run -e default python scripts/kf_baselines.py --stores single_muon_uniform="$EV/single_muon_uniform/test" single_muon_2GeV="$EV/single_muon_2GeV/test" \
   single_muon_10GeV="$EV/single_muon_10GeV/test" single_muon_100GeV="$EV/single_muon_100GeV/test" ttbar="$EV/ttbar/test" ttbar_new_pt1="$EV/ttbar_new_pt1/test" \
   --out-dir $REPO/eval_plots/baselines_KF_v2 2>&1 | tail -2
echo "=== mixed store (all ttbar_new_pt1_tr train parts + its val) $(date)"
pixi run -e default python $REPO/scripts/07_build_mixed_store.py --base $OUT/single_muon_uniform --extra $OUT/ttbar_new_pt1_tr --extra-max-tracks 100000000 --extra-val --out ${OUT}_mixed
echo "=== COUNTS $(date)"
OUT="$OUT" pixi run -e default python - <<'PY'
import json, os
from pathlib import Path
OUT = Path(os.environ['OUT'])
print(f"{'dataset':20s} {'train':>12s} {'val':>10s} {'test':>10s} {'total':>12s}   sort / bz / windows / pt")
for ds in ('single_muon_uniform','single_muon_2GeV','single_muon_10GeV','single_muon_100GeV','ttbar','ttbar_new_pt1','ttbar_new_pt1_tr'):
    m = json.load(open(OUT/ds/'dataset_meta.json')); sp = {k: json.load(open(OUT/ds/k/'manifest.json'))['n_tracks'] if (OUT/ds/k/'manifest.json').exists() else 0 for k in ('train','val','test')}
    s = m['selection']; print(f"{ds:20s} {sp['train']:>12,} {sp['val']:>10,} {sp['test']:>10,} {sum(sp.values()):>12,}   {m['hit_sort_key']} / {m['bz']} T / |d0|<={s['d0_max']} |z0|<={s['z0_max']} / pT {s['pt_min']}-{s.get('pt_max','inf')}")
mm = json.load(open(str(OUT)+'_mixed/train/manifest.json')); mv = json.load(open(str(OUT)+'_mixed/val/manifest.json'))
print(f"{'MIXED (train store)':20s} {mm['n_tracks']:>12,} {mv['n_tracks']:>10,} {'(muon test)':>10s}   {mm.get('mixed_from')}")
PY
echo "ALL DONE $(date)"
