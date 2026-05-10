#!/usr/bin/env bash
# Reference: how to build the two preprocessed dataset variants used by
# the shipped configs. This script is intentionally a thin guide, not a
# do-everything one-shot — read the README's Dataset section first and
# adapt to your storage / parallelism preferences.
#
# Two variants are required:
#   $DATA_ROOT/p0_core_pretrain               — used by all pretrain configs
#   $DATA_ROOT/p200_core_kf_matched_finetune  — used by all fine-tune
#                                               configs and by paper plots
set -euo pipefail

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "ERROR: DATA_ROOT is not set." >&2
  echo "       e.g. export DATA_ROOT=/path/to/large/scratch" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_WORKERS="${NUM_WORKERS:-8}"

cat <<EOF
==========================================================================
Step 1 — download raw parquet shards from HuggingFace (CERN/ColliderML-Release-1)
         via the colliderml CLI (installed by 'pixi install --locked').

    pixi run -e default colliderml download --config ttbar_pu0_particles    --out \$DATA_ROOT/raw/p0
    pixi run -e default colliderml download --config ttbar_pu0_tracker_hits --out \$DATA_ROOT/raw/p0
    pixi run -e default colliderml download --config ttbar_pu0_tracks       --out \$DATA_ROOT/raw/p0
    pixi run -e default colliderml download --config ttbar_pu200_particles    --out \$DATA_ROOT/raw/p200
    pixi run -e default colliderml download --config ttbar_pu200_tracker_hits --out \$DATA_ROOT/raw/p200
    pixi run -e default colliderml download --config ttbar_pu200_tracks       --out \$DATA_ROOT/raw/p200

==========================================================================
Step 2 — preprocess the raw shards (apply track selection, pack compact CSR,
         augment with ACTS-CKF reco + DM mask). One invocation per variant:

    cd \$REPO_ROOT/src/hepattn/experiments/colliderml_regr
    pixi run -e default python -m hepattn.experiments.colliderml_regr.scripts.preprocess_colliderml_compact \\
        --data-dir   \$DATA_ROOT/raw/p0 \\
        --output-dir \$DATA_ROOT/p0_core_pretrain \\
        --selection-file utils/selection_p200_datasets.yaml \\
        --selection-variant core \\
        --selection '{"hard_scatter": true}' \\
        --num-workers $NUM_WORKERS \\
        --augment-acts

    pixi run -e default python -m hepattn.experiments.colliderml_regr.scripts.preprocess_colliderml_compact \\
        --data-dir   \$DATA_ROOT/raw/p200 \\
        --output-dir \$DATA_ROOT/p200_core_kf_matched_finetune \\
        --selection-file utils/selection_p200_datasets.yaml \\
        --selection-variant core_kf_matched \\
        --selection '{"hard_scatter": false}' \\
        --num-workers $NUM_WORKERS \\
        --augment-acts

==========================================================================
Step 3 — split each preprocessed variant 90/5/5 train/val/test
         (writes split.json next to the shards):

    pixi run -e default python -m hepattn.experiments.colliderml_regr.scripts.create_split \\
        --preprocessed-dir \$DATA_ROOT/p0_core_pretrain
    pixi run -e default python -m hepattn.experiments.colliderml_regr.scripts.create_split \\
        --preprocessed-dir \$DATA_ROOT/p200_core_kf_matched_finetune
==========================================================================

The raw parquets at \$DATA_ROOT/raw/ are no longer needed once step 2
finishes and can be deleted to free ~600 GB.
EOF
