#!/bin/bash
# Fetch a drift_beamspot dataset in the FLAT parquet layout from the NERSC portal:
#   <dest>/<ds>/v1/parquet/{truth/particles,reco/tracker_hits,reco/tracks,reco/truth_tracks,truth/tracker_simhits}/*.parquet
#   bash scripts/06b_fetch_nersc_flat.sh <dataset> [dest_root=/scratch/colliderml/drift_beamspot_v2] [parallel=8] [tables...]
# Resumable (wget -c); verifies every file's size against the portal's Content-Length at the end.
set -uo pipefail
DS=$1; DEST="${2:-/scratch/colliderml/drift_beamspot_v2}"; PAR="${3:-8}"; shift 3 2>/dev/null || shift $#
TABLES=("$@"); [ ${#TABLES[@]} -eq 0 ] && TABLES=(truth/particles reco/tracker_hits reco/tracks reco/truth_tracks truth/tracker_simhits)
BASE=https://portal.nersc.gov/cfs/m4958/ColliderML/drift_beamspot/$DS/v1/parquet
LIST=$(mktemp)
for t in "${TABLES[@]}"; do
  mkdir -p "$DEST/$DS/v1/parquet/$t"
  curl -s "$BASE/$t/" | grep -o 'href="[^"]*\.parquet"' | cut -d'"' -f2 | while read -r f; do echo "$t $f"; done
done > "$LIST"
echo "$DS: $(wc -l < "$LIST") files to fetch into $DEST/$DS ($PAR streams) $(date)"
fetch_one () { local t=$1 f=$2; wget -q -c -O "$DEST/$DS/v1/parquet/$t/$f" "$BASE/$t/$f" || echo "WARN $t/$f rc=$?"; }
export -f fetch_one; export DEST DS BASE
xargs -P "$PAR" -n 2 bash -c 'fetch_one "$0" "$1"' < "$LIST"
# verify sizes
bad=0; while read -r t f; do want=$(curl -sI "$BASE/$t/$f" | grep -i content-length | awk '{print $2}' | tr -d '\r'); have=$(stat -c %s "$DEST/$DS/v1/parquet/$t/$f" 2>/dev/null || echo 0); [ "$want" = "$have" ] || { echo "SIZE MISMATCH $t/$f want $want have $have"; bad=$((bad+1)); }; done < "$LIST"
echo "$DS DONE $(date): $(wc -l < "$LIST") files, $bad size mismatches"; rm -f "$LIST"
