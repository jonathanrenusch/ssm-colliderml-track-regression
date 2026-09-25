#!/bin/bash
# Download one ColliderML drift_beamspot dataset (the parquet tables the preprocessing
# reads) from the public NERSC portal, mirroring the portal layout under <raw_root>:
#
#   bash scripts/fetch_data.sh <dataset> <raw_root> [parallel=8]
#   bash scripts/fetch_data.sh ttbar <raw_root> [parallel=8] [first_run=6 last_run=784]
#
#   single-particle guns: <raw_root>/<dataset>/v1/parquet/{truth/particles, truth/tracker_simhits,
#                         reco/tracker_hits, reco/tracks, reco/truth_tracks}/*.parquet
#   ttbar:                <raw_root>/ttbar/v1/runs/<N>/{particles, tracker_simhits, tracker_hits,
#                         tracks, truth_tracks}/*.parquet
#
# Resumable (wget -c).  Every file's size is checked against the portal's Content-Length;
# the script exits non-zero if any file is missing or has the wrong size.
set -uo pipefail
if [ $# -lt 2 ]; then sed -n '2,13p' "$0"; exit 2; fi
DS=$1; RAW=$2; PAR=${3:-8}; FIRST=${4:-6}; LAST=${5:-784}
URL=https://portal.nersc.gov/cfs/m4958/ColliderML/drift_beamspot/$DS/v1

if [ "$DS" = ttbar ]; then
  DIRS=$(for n in $(seq "$FIRST" "$LAST"); do
           for t in particles tracker_hits tracks truth_tracks tracker_simhits; do echo "runs/$n/$t"; done
         done)
else
  DIRS=$(printf 'parquet/%s\n' truth/particles reco/tracker_hits reco/tracks reco/truth_tracks truth/tracker_simhits)
fi

# <dir> <file> for every parquet file listed in the portal's directory indices
list_dir () { curl -sf "$URL/$1/" | grep -o 'href="[^"/]*\.parquet"' | cut -d'"' -f2 | sed "s|^|$1 |"; }
# download one file, then compare its size with the portal's
fetch_one () {
  local dir=$1 f=$2 want have
  mkdir -p "$RAW/$DS/v1/$dir"
  wget -q -c -O "$RAW/$DS/v1/$dir/$f" "$URL/$dir/$f" || echo "BAD $dir/$f: wget rc=$?"
  want=$(curl -sfI "$URL/$dir/$f" | grep -i '^content-length' | awk '{print $2}' | tr -d '\r')
  have=$(stat -c %s "$RAW/$DS/v1/$dir/$f" 2>/dev/null || echo 0)
  [ -n "$want" ] && [ "$want" = "$have" ] || echo "BAD $dir/$f: size $have, portal $want"
}
export -f list_dir fetch_one; export URL RAW DS

LIST=$(mktemp); LOG=$(mktemp)
echo "$DIRS" | xargs -P "$PAR" -I{} bash -c 'list_dir {}' > "$LIST"
n_dirs=$(echo "$DIRS" | wc -l); n_listed=$(cut -d' ' -f1 "$LIST" | sort -u | wc -l)
echo "$DS: $(wc -l < "$LIST") files in $n_listed of $n_dirs table directories -> $RAW/$DS ($PAR streams)"
xargs -r -P "$PAR" -n 2 bash -c 'fetch_one "$0" "$1"' < "$LIST" | tee "$LOG"
bad=$(grep -c '^BAD' "$LOG"); missing=$((n_dirs - n_listed))
echo "$DS: $(wc -l < "$LIST") files, $bad bad, $missing table directories without files"
rm -f "$LIST" "$LOG"
[ "$bad" -eq 0 ] && [ "$missing" -eq 0 ]
