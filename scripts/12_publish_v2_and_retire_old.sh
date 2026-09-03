#!/bin/bash
# After scripts/11_rebuild_stores_v2.sh has finished and its COUNTS were checked:
#  1) copy the v2 stores (and the v2 raw parquet) to /eos (permanent, per-file verified),
#  2) retire the superseded datasets on /eos and /scratch (user decision 2026-08-28).
#   bash scripts/12_publish_v2_and_retire_old.sh publish     # step 1
#   bash scripts/12_publish_v2_and_retire_old.sh retire      # step 2 (only after step 1 reported 0 mismatches)
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression; EOS=/eos/project/e/end-to-end-colliderml/data
OUT=/scratch/colliderml/ICLR_retraining_v2; RAW=/scratch/colliderml/drift_beamspot_v2
case "${1:-}" in
  publish)
    for ds in single_muon_uniform single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar ttbar_new_pt1 ttbar_new_pt1_tr; do
      echo "=== store $ds -> eos $(date)"; bash $REPO/scripts/copy_dataset.sh $OUT/$ds $EOS/ICLR_retraining_v2/$ds 16 2>&1 | tail -1
    done
    for ds in single_muon_uniform single_muon_2GeV single_muon_10GeV single_muon_100GeV ttbar; do
      echo "=== raw $ds -> eos $(date)"; bash $REPO/scripts/copy_dataset.sh $RAW/$ds $EOS/drift_beamspot_v2/$ds 16 2>&1 | tail -1
    done
    echo "PUBLISH DONE $(date)";;
  retire)
    echo "retiring superseded datasets $(date)"
    for d in $EOS/ICLR_retraining $EOS/ICLR_retraining_geom $EOS/ICLR_retraining_ssort $EOS/ICLR_retraining_geom_B3; do [ -d "$d" ] && { du -sh "$d" | cut -f1 | tr '\n' ' '; rm -rf "$d"; echo "removed $d"; }; done
    for d in /scratch/colliderml/drift_beamspot /scratch/colliderml/ICLR_retraining_geom /scratch/colliderml/ICLR_retraining_geom_B3 /scratch/colliderml/ICLR_eval_geom /scratch/colliderml/ICLR_eval_geom_B3 \
             /scratch/colliderml/ICLR_retraining_geom_mixed8M /scratch/colliderml/ICLR_retraining_geom_B3_mixed8M /scratch/colliderml/ICLR_retraining_geom_B3_mixed13M /scratch/colliderml/ICLR_retraining_geom_B3_mixed16M; do
      [ -e "$d" ] && { du -sh "$d" 2>/dev/null | cut -f1 | tr '\n' ' '; rm -rf "$d"; echo "removed $d"; }; done
    df -h /scratch | tail -1; echo "RETIRE DONE $(date)";;
  *) echo "usage: $0 publish|retire"; exit 1;;
esac
