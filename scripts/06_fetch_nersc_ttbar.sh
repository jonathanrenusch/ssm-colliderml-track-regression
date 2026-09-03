#!/bin/bash
# Fetch the drift_beamspot ttbar runs from the NERSC portal (parquet tables only:
# particles, tracker_hits, tracks, truth_tracks — no tracker_simhits, no ROOT).
#   bash scripts/06_fetch_nersc_ttbar.sh <first_run> <last_run> [dest_root] [parallel]
# dest layout = <dest_root>/ttbar/v1/runs/<N>/<table>/*.parquet (same as the local runs 0-5).
# Resumable (wget -c); a run is marked done with <run>/.fetched once all four tables are present.
set -uo pipefail
FIRST=$1; LAST=$2; DEST="${3:-/scratch/colliderml/drift_beamspot}"; PAR="${4:-8}"
BASE=https://portal.nersc.gov/cfs/m4958/ColliderML/drift_beamspot/ttbar/v1/runs
fetch_run () {
  local n=$1
  local d="$DEST/ttbar/v1/runs/$n"   # two statements: `local a=$1 b=$a` expands $a BEFORE a is assigned
  [ -f "$d/.fetched${TAG:-}" ] && return 0
  for t in ${TABLES:-particles tracker_hits tracks truth_tracks}; do
    # -r -np -l1 -A parquet: only the parquet files of this table's index page; -nH --cut-dirs=8 -> <table>/<file>
    wget -q -c -r -np -nH -l1 --cut-dirs=8 -A "*.parquet" -R "index.html*" -P "$d" "$BASE/$n/$t/" || echo "WARN run $n table $t: wget rc=$?"
  done
  ok=1; for t in ${TABLES:-particles tracker_hits tracks truth_tracks}; do [ -n "$(ls "$d"/$t/*.parquet 2>/dev/null)" ] || ok=0; done
  if [ $ok = 1 ]; then
    touch "$d/.fetched${TAG:-}"; echo "run $n done $(du -sh "$d" | cut -f1) $(date +%H:%M:%S)"
  else
    echo "run $n INCOMPLETE $(date +%H:%M:%S)"
  fi
}
export -f fetch_run; export DEST BASE TABLES TAG
seq "$FIRST" "$LAST" | xargs -P "$PAR" -I{} bash -c 'fetch_run {}'
echo "ALL DONE $(date)  fetched=$(ls -d $DEST/ttbar/v1/runs/*/.fetched 2>/dev/null | wc -l)"
