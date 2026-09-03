#!/bin/bash
# Stage the DATA v2 stores (true-time order, 3 T, in-range, ttbar 1-110 GeV) on a machine's /scratch (from /eos), build the eval farm and the
# mixed muon+ttbar training store for Q'.  Run on the target machine (e.g. sess5).
#   bash scripts/10_stage_B3_on_sess5.sh [extra_max_tracks=all]
set -euo pipefail
MAXX="${1:-100000000}"   # all ttbar train parts (15.8 M at pT >= 1)
EOS=/eos/project/e/end-to-end-colliderml/data/ICLR_retraining_v2
OUT=/scratch/colliderml/ICLR_retraining_v2; EV=/scratch/colliderml/ICLR_eval_v2
REPO=/shared/tracking/ssm-colliderml-track-regression
for ds in single_muon_uniform single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar ttbar_new_pt1 ttbar_new_pt1_tr; do
  if [ -f "$OUT/$ds/dataset_meta.json" ] && [ -f "$OUT/$ds/train/manifest.json" ]; then echo "$ds already on /scratch"; continue; fi
  [ -f "$EOS/$ds/dataset_meta.json" ] || { echo "$ds not on /eos yet — abort"; exit 1; }
  echo "=== copy $ds $(date)"; bash $REPO/scripts/copy_dataset.sh "$EOS/$ds" "$OUT/$ds" 16
done
cd $REPO/src/track_regression
pixi run -e default python $REPO/scripts/build_eval_farm.py --store-root $OUT --eval-root $EV \
   --union single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar ttbar_new_pt1 --link single_muon_uniform
pixi run -e default python $REPO/scripts/07_build_mixed_store.py --base $OUT/single_muon_uniform --extra $OUT/ttbar_new_pt1_tr \
   --extra-max-tracks $MAXX --extra-val --out /scratch/colliderml/ICLR_retraining_v2_mixed
echo "STAGED $(date)"
