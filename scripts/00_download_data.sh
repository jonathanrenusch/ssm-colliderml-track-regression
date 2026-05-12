#!/usr/bin/env bash
# Reference: how to build the two preprocessed dataset variants used by
# the shipped configs. This script is intentionally a thin guide, not a
# do-everything one-shot — read the README's Dataset section first and
# adapt to your storage / parallelism preferences.
#
# Output paths are absolute (``/scratch/colliderml/arxiv_retraining/...``)
# to match every shipped config. Edit the path here AND in the configs
# under ``src/track_regression/config/`` if your scratch lives elsewhere.
#
# Two variants are required:
#   /scratch/colliderml/arxiv_retraining/p0_core_pretrain
#       — used by all pretrain configs
#   /scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune
#       — used by all fine-tune configs and by paper plots
set -euo pipefail

NUM_WORKERS="${NUM_WORKERS:-8}"

cat <<'EOF'
==========================================================================
Step 1 — download raw parquet shards from HuggingFace (CERN/ColliderML-Release-1)
         via the colliderml CLI (installed by 'pixi install --locked').

    pixi run -e default colliderml download --config ttbar_pu0_particles    --out /scratch/colliderml/arxiv_retraining/raw/p0
    pixi run -e default colliderml download --config ttbar_pu0_tracker_hits --out /scratch/colliderml/arxiv_retraining/raw/p0
    pixi run -e default colliderml download --config ttbar_pu0_tracks       --out /scratch/colliderml/arxiv_retraining/raw/p0
    pixi run -e default colliderml download --config ttbar_pu200_particles    --out /scratch/colliderml/arxiv_retraining/raw/p200
    pixi run -e default colliderml download --config ttbar_pu200_tracker_hits --out /scratch/colliderml/arxiv_retraining/raw/p200
    pixi run -e default colliderml download --config ttbar_pu200_tracks       --out /scratch/colliderml/arxiv_retraining/raw/p200

==========================================================================
Step 2 — preprocess the raw shards (apply track selection, pack compact CSR,
         augment with ACTS-CKF reco + DM mask, write hit_times.npy time-sort
         sidecar, auto-build train/val/test split.json at 90/5/5):

    cd $REPO_ROOT/src/track_regression
    pixi run -e default python -m track_regression.scripts.preprocess_colliderml_compact \
        --data-dir   /scratch/colliderml/arxiv_retraining/raw/p0 \
        --output-dir /scratch/colliderml/arxiv_retraining/p0_core_pretrain \
        --selection-file selection_p200_datasets.yaml \
        --selection-variant core \
        --selection '{"hard_scatter": true}' \
        --num-workers $NUM_WORKERS

    pixi run -e default python -m track_regression.scripts.preprocess_colliderml_compact \
        --data-dir   /scratch/colliderml/arxiv_retraining/raw/p200 \
        --output-dir /scratch/colliderml/arxiv_retraining/p200_core_kf_matched_finetune \
        --selection-file selection_p200_datasets.yaml \
        --selection-variant core_kf_matched \
        --selection '{"hard_scatter": false}' \
        --num-workers $NUM_WORKERS

(``--augment-acts`` is the default; pass ``--no-acts`` to disable. The
preprocessor automatically writes ``split.json`` (90/5/5) at the end of
every run; pass ``--no-split`` if you want to defer that.)
==========================================================================

The raw parquets at /scratch/colliderml/arxiv_retraining/raw/ are no longer
needed once step 2 finishes and can be deleted to free ~600 GB.
EOF
