#!/bin/bash
# Copy a preprocessed flat dataset between /eos and /scratch, either direction,
# with parallel streams.
#
#   # before training: permanent storage -> fast local disk
#   ./copy_dataset.sh /eos/project/e/end-to-end-colliderml/data/ICLR_retraining/single_muon_uniform \
#                     /scratch/colliderml/ICLR_retraining/single_muon_uniform
#
#   # after preprocessing: fast local disk -> permanent storage
#   ./copy_dataset.sh /scratch/colliderml/ICLR_retraining/ttbar \
#                     /eos/project/e/end-to-end-colliderml/data/ICLR_retraining/ttbar
#
#   ./copy_dataset.sh <src_dir> <dst_dir> [parallel_streams]     (default 16)
#
# EOS throughput scales with the number of files in flight, not with file size:
# one 2 GB file caps at ~147 MB/s no matter how many streams.  The flat format
# is written with ~640 MB parts precisely so this parallelises.  Measured on
# this project at -P 16, cold:
#
#     /scratch -> /eos   655 MB/s   (full 145 GB muon set, ~4 min)
#     /eos -> /scratch   982 MB/s   (22 GiB / 384 files, so ~2.5 min for 145 GB)
#
# Both numbers need enough *large* files to fill the streams: the 5-part val
# store only reaches 261 MB/s because 5 files carry almost all the bytes.
#
# Re-running is safe and cheap: files already present at the destination with
# the same size are skipped, so an interrupted copy resumes instead of
# restarting.  Set FORCE=1 to recopy everything.
set -euo pipefail
SRC="${1:?usage: copy_dataset.sh <src> <dst> [streams]}"
DST="${2:?usage: copy_dataset.sh <src> <dst> [streams]}"
P="${3:-16}"
FORCE="${FORCE:-0}"

[ -d "$SRC" ] || { echo "no such source: $SRC" >&2; exit 1; }
mkdir -p "$DST"
cd "$SRC"
find . -type d -exec mkdir -p "$DST/{}" \;
# Apparent size, not allocated blocks: EOS and ext4 use different block sizes,
# so du -sb totals never match across the two filesystems.
BYTES=$(find . -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
N=$(find . -type f | wc -l)
echo "copying $N files, $((BYTES/1073741824)) GiB, $P streams: $SRC -> $DST"
START=$(date +%s)
# Skip files already present at the same size, so an interrupted copy resumes.
if [ "$FORCE" = "1" ]; then
  find . -type f
else
  find . -type f -printf '%s %p\n' | while read -r sz f; do
    dsz=$(stat -c%s "$DST/$f" 2>/dev/null || echo -1)
    [ "$dsz" = "$sz" ] || printf '%s\n' "$f"
  done
fi | xargs -r -P "$P" -I{} cp -f {} "$DST/{}"
ELAPSED=$(( $(date +%s) - START ))
echo "done in ${ELAPSED}s ($(( BYTES / 1048576 / (ELAPSED>0?ELAPSED:1) )) MB/s)"

# verify: same file count and same total bytes
DN=$(find "$DST" -type f | wc -l)
DB=$(find "$DST" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
if [ "$DN" = "$N" ] && [ "$DB" = "$BYTES" ]; then
  echo "verified: $DN files, $DB bytes match"
else
  echo "MISMATCH: src $N files/$BYTES bytes vs dst $DN files/$DB bytes" >&2
  exit 1
fi
