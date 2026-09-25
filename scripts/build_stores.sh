#!/bin/bash
# Build the paper's training and evaluation stores from the raw parquet tables.
#
#   bash scripts/build_stores.sh <raw_root> <store_root> <eval_root> [workers=16]
#
# <raw_root> holds the datasets downloaded with scripts/fetch_data.sh:
#   single_muon_uniform, single_muon_loguniform, single_muon_2GeV, single_muon_10GeV,
#   single_muon_50GeV, ttbar (runs 6-784).
# Output:
#   <store_root>/<dataset>   one flat store per muon sample
#   <store_root>/ttbar       ttbar runs 46-784, 1 <= pT <= 110 GeV (training)
#   <store_root>/ttbar_bench ttbar runs 6-45, same cuts (throughput benchmark)
#   <store_root>/mixLU       uniform-pT + log-uniform-pT muons            (symlinks)
#   <store_root>/mix3        mixLU + ttbar = the training store           (symlinks)
#   <eval_root>/<dataset>    2 / 10 / 50 GeV muons and ttbar_bench (all splits as test),
#                            uniform-pT muons (its test split)
# Datasets whose store already exists (dataset_meta.json) are not rebuilt.
set -euo pipefail
if [ $# -lt 3 ]; then sed -n '2,17p' "$0"; exit 2; fi
RAW=$(realpath "$1"); OUT=$(realpath -m "$2"); EV=$(realpath -m "$3"); WORKERS=${4:-16}
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT" "$EV"

# hits in simulated-time order, 3 T perigee targets, |d0| <= 7.1 mm, |z0| <= 270 mm
# (these are the preprocess_flat.py defaults, spelled out here as the recipe)
COMMON="--sort-key true_time --bz 3.0 --d0-max 7.1 --z0-max 270 --num-workers $WORKERS"
preprocess () {   # <dataset dir> <store name> [extra args]
  local src=$1 name=$2; shift 2
  if [ -f "$OUT/$name/dataset_meta.json" ]; then echo "=== $name exists, skipped"; return; fi
  echo "=== preprocess $name $(date)"
  $PY scripts/preprocess_flat.py $COMMON --data-dir "$src" --output-dir "$OUT/$name" "$@"
}

for ds in single_muon_2GeV single_muon_10GeV single_muon_50GeV single_muon_loguniform single_muon_uniform; do
  preprocess "$RAW/$ds/v1" "$ds"
done

# ttbar: runs 46-784 = training sample (4 runs per part), runs 6-45 = throughput benchmark;
# both 1 <= pT <= 110 GeV
ttbar_runs () {   # <farm name> <first run> <last run>
  mkdir -p "$RAW/$1/v1/runs"
  for n in $(seq "$2" "$3"); do ln -sfn "$RAW/ttbar/v1/runs/$n" "$RAW/$1/v1/runs/$n"; done
}
ttbar_runs ttbar_train 46 784; ttbar_runs ttbar_bench 6 45
preprocess "$RAW/ttbar_train/v1" ttbar --pt-min 1.0 --pt-max 110 --shards-per-part 4
preprocess "$RAW/ttbar_bench/v1" ttbar_bench --pt-min 1.0 --pt-max 110

# The uniform-pT sample has 202 shards -> 182 / 10 / 10 train / val / test parts.  Keep 5
# val and 5 test parts (5 M tracks each) and move the others to the end of train.
OUT="$OUT" $PY - <<'PY'
import json, os, shutil
from pathlib import Path
root = Path(os.environ["OUT"]) / "single_muon_uniform"
meta = json.load(open(root / "dataset_meta.json"))
if "split_rebalanced" not in meta:
    k, moves = len(list((root / "train").glob("part_*"))), []
    for sp in ("val", "test"):
        for d in sorted((root / sp).glob("part_*"))[5:]:
            moves.append(f"{sp}/{d.name}->train/part_{k:04d}")
            shutil.move(str(d), str(root / "train" / f"part_{k:04d}")); k += 1
    for sp in ("train", "val", "test"):
        man = json.load(open(root / sp / "manifest.json")); parts = []
        for d in sorted((root / sp).glob("part_*")):
            m = json.load(open(d / "meta.json"))
            parts.append({"name": d.name, "n_tracks": m["n_tracks"], "n_hits": m["n_hits"]})
        man.update(parts=parts, n_tracks=sum(p["n_tracks"] for p in parts), n_hits=sum(p["n_hits"] for p in parts))
        json.dump(man, open(root / sp / "manifest.json", "w"), indent=1)
    meta["split_rebalanced"] = {"moved_to_train": moves}
    json.dump(meta, open(root / "dataset_meta.json", "w"), indent=1)
    print(f"single_muon_uniform: moved {len(moves)} parts to train")
PY

echo "=== mixed training stores $(date)"
$PY scripts/build_mixed_store.py --base "$OUT/single_muon_uniform" --extra "$OUT/single_muon_loguniform" --out "$OUT/mixLU"
$PY scripts/build_mixed_store.py --base "$OUT/mixLU" --extra "$OUT/ttbar" --out "$OUT/mix3"

echo "=== eval farm $(date)"
$PY scripts/build_eval_farm.py --store-root "$OUT" --eval-root "$EV" \
    --union single_muon_2GeV single_muon_10GeV single_muon_50GeV ttbar_bench --link single_muon_uniform
echo "ALL DONE $(date)"
