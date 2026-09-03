#!/bin/bash
# Unattended chain for the night of 2026-08-28 (user: "off you go"):
#   fetch verified -> 11_rebuild_stores_v2 -> counts -> sweep-4 dry runs -> launch 4 trainings -> publish v2 to /eos -> retire old data (only if publish verified)
set -uo pipefail
REPO=/shared/tracking/ssm-colliderml-track-regression; cd $REPO; L=$REPO/launch_logs/data
log () { echo "[$(date '+%F %T')] $*"; }
# 1) wait for the muon fetcher to finish and verify
while ! grep -q "single_muon_uniform DONE" $L/fetch_v2_muons_*.log; do sleep 60; done
mm=$(grep "single_muon_uniform DONE" $L/fetch_v2_muons_*.log | grep -o "[0-9]* size mismatches" | awk '{print $1}')
tt=$(grep -c "INCOMPLETE" $L/fetch_v2_ttbar_*.log)
log "fetch done: uniform size mismatches=$mm, ttbar incomplete runs=$tt"
[ "$mm" = "0" ] && [ "$tt" = "0" ] || { log "FETCH NOT CLEAN — stopping before preprocessing"; exit 1; }
# 2) rebuild stores
log "rebuilding v2 stores"; bash scripts/11_rebuild_stores_v2.sh > $L/rebuild_v2_$(date +%Y%m%d_%H%M%S).log 2>&1; rc=$?
log "rebuild rc=$rc"; grep -A12 "=== COUNTS" $L/rebuild_v2_*.log | tail -13
[ $rc -eq 0 ] && [ -f /scratch/colliderml/ICLR_retraining_v2_mixed/train/manifest.json ] || { log "REBUILD FAILED — stopping"; exit 1; }
# 3) dry runs, then launch
log "sweep-4 dry runs"; DRY=1 bash src/track_regression/config/ssm_cls/ICLR_sweep4/launch_sess3.sh > $L/sweep4_dry_$(date +%Y%m%d_%H%M%S).log 2>&1
grep "^dry" $L/sweep4_dry_*.log | tail -5
if grep "^dry" $L/sweep4_dry_*.log | tail -5 | grep -qv "rc=0"; then log "DRY RUN FAILED — not launching"; exit 1; fi
if grep "^dry" $L/sweep4_dry_*.log | tail -5 | grep -q "val, errors: [1-9]"; then
  # 'errors' counts also match 'calibration_error' lines; check for real tracebacks instead
  if grep -l "Traceback" $REPO/launch_logs/sweep4/dryrun/*.log 2>/dev/null | grep -q .; then log "DRY RUN TRACEBACK — not launching"; exit 1; fi
fi
log "launching sweep 4"; bash src/track_regression/config/ssm_cls/ICLR_sweep4/launch_sess3.sh > $L/sweep4_launch_$(date +%Y%m%d_%H%M%S).log 2>&1; cat $L/sweep4_launch_*.log | tail -6
# 4) publish v2 to /eos, then retire old data only if every copy verified
log "publishing v2 to /eos"; bash scripts/12_publish_v2_and_retire_old.sh publish > $L/publish_v2_$(date +%Y%m%d_%H%M%S).log 2>&1
nv=$(grep -c "^verified" $L/publish_v2_*.log); bad=$(grep -ci "mismatch\|missing\|error" $L/publish_v2_*.log)
log "publish: $nv datasets verified, $bad problem lines"
if [ "$nv" -ge 12 ] && [ "$bad" = "0" ]; then log "retiring old datasets"; bash scripts/12_publish_v2_and_retire_old.sh retire > $L/retire_old_$(date +%Y%m%d_%H%M%S).log 2>&1; tail -3 $L/retire_old_*.log; else log "PUBLISH NOT CLEAN — old data NOT deleted"; fi
log "NIGHT CHAIN DONE"
