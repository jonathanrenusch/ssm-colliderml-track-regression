#!/usr/bin/env bash
# Launch all four preprocessings in parallel under nohup.
# Each job writes its own .log under <OUT>/_logs/<dataset>.log
# PIDs are echoed at the end and saved to <OUT>/_logs/pids.txt for later reference.

set -u

OUT=/eos/project/e/end-to-end-colliderml/data/arxiv_retraining
LOGS=$OUT/_logs
mkdir -p "$LOGS"

RAW_P0=/eos/project/e/end-to-end-muon-tracking/tracking/colliderml/p0/CERN__ColliderML-Release-1
RAW_P200=/eos/project/n/ngt2-4/data/ColliderML-Release-1.old/data
SEL=/shared/tracking/ssm-track-regression/src/hepattn/experiments/colliderml_regr/utils/selection_p200_datasets.yaml

cd /shared/tracking/ssm-track-regression || exit 1

run() {
    local name=$1; shift
    echo "[$(date +%T)] launching $name → $LOGS/$name.log"
    nohup pixi run -e default python -m hepattn.experiments.colliderml_regr.scripts.preprocess_colliderml_compact \
        "$@" --num-workers 8 \
        > "$LOGS/$name.log" 2>&1 &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$name $pid" >> "$LOGS/pids.txt"
    printf "  PID %s  LOG %s\n" "$pid" "$LOGS/$name.log"
}

: > "$LOGS/pids.txt"

run p0_core_pretrain \
    --data-dir   "$RAW_P0" \
    --output-dir "$OUT/p0_core_pretrain" \
    --particles-subdir ttbar_pu0_particles_recorded_only \
    --hits-subdir      ttbar_pu0_tracker_hits \
    --tracks-subdir    ttbar_pu0_tracks \
    --selection-file "$SEL" \
    --selection-variant core \
    --selection '{"hard_scatter": true}'

run p200_core_kf_matched_finetune \
    --data-dir   "$RAW_P200" \
    --output-dir "$OUT/p200_core_kf_matched_finetune" \
    --particles-subdir ttbar_pu200_particles_recorded_only \
    --hits-subdir      ttbar_pu200_tracker_hits \
    --tracks-subdir    ttbar_pu200_tracks \
    --selection-file "$SEL" \
    --selection-variant core_kf_matched \
    --selection '{"hard_scatter": false}'

run p0_core_kf_hits_pretrain \
    --data-dir   "$RAW_P0" \
    --output-dir "$OUT/p0_core_kf_hits_pretrain" \
    --particles-subdir ttbar_pu0_particles_recorded_only \
    --hits-subdir      ttbar_pu0_tracker_hits \
    --tracks-subdir    ttbar_pu0_tracks \
    --selection-file "$SEL" \
    --selection-variant core_kf_hits \
    --selection '{"hard_scatter": true}'

run p200_core_kf_hits_finetune \
    --data-dir   "$RAW_P200" \
    --output-dir "$OUT/p200_core_kf_hits_finetune" \
    --particles-subdir ttbar_pu200_particles_recorded_only \
    --hits-subdir      ttbar_pu200_tracker_hits \
    --tracks-subdir    ttbar_pu200_tracks \
    --selection-file "$SEL" \
    --selection-variant core_kf_hits \
    --selection '{"hard_scatter": false}'

echo
echo "All 4 jobs detached. Tail any log with:"
echo "  tail -f $LOGS/<dataset>.log"
echo "PIDs recorded in: $LOGS/pids.txt"
