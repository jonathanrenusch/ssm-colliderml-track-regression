#!/usr/bin/env bash
# ============================================================================
# Parallel copy of the arxiv_retraining preprocessed datasets to scratch
# ============================================================================
#
# Source:  ${SRC_ROOT}                              (override with --src)
# Target:  /scratch/colliderml/arxiv_retraining/<dataset>
#
# Set SRC_ROOT below (or pass --src) to point at wherever your
# preprocessed datasets live (e.g. a slow NFS/EOS mount). The script
# parallel-copies them into a fast local scratch tree.
#
# Four preprocessed datasets are managed by this script:
#
#   pretrain pair (used by every pretrain config)
#     p0_core_pretrain                 (variant=core,         hard_scatter=true)
#     p0_core_kf_hits_pretrain         (variant=core_kf_hits, hard_scatter=true)
#
#   finetune pair (used by every fine-tune config)
#     p200_core_kf_matched_finetune    (variant=core_kf_matched, hard_scatter=false)
#     p200_core_kf_hits_finetune       (variant=core_kf_hits,    hard_scatter=false)
#
# Each preprocessed dataset is laid out as
#   <root>/manifest.json
#   <root>/split.json                  (90/5/5, written automatically by preprocess)
#   <root>/shard_XXXX/hits.npy
#   <root>/shard_XXXX/hit_times.npy    (v2 truth-time sort sidecar)
#   <root>/shard_XXXX/_complete        (sentinel)
#   <root>/shard_XXXX/selected_tracks/<files>.npy
#
# Files that already exist on the target with identical size are skipped.
# Uses GNU parallel for the actual `cp` calls.
#
# ----------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------
#
#   # Pretraining datasets only (p0_core_pretrain + p0_core_kf_hits_pretrain)
#   ./parallel_copy_to_scratch.sh pretrain
#
#   # Fine-tuning datasets only
#   ./parallel_copy_to_scratch.sh finetune
#
#   # Everything (all 4)
#   ./parallel_copy_to_scratch.sh all
#
#   # One specific dataset
#   ./parallel_copy_to_scratch.sh p0_core_pretrain
#   ./parallel_copy_to_scratch.sh p200_core_kf_matched_finetune
#
#   # Tuning knobs
#   ./parallel_copy_to_scratch.sh pretrain --jobs 30        # raise parallel cp count
#   ./parallel_copy_to_scratch.sh pretrain --dry-run        # show what would be copied
#   ./parallel_copy_to_scratch.sh pretrain --scratch /tmp   # alt scratch root
#
#   # Recommended: run via nohup so the copy survives ssh disconnects
#   nohup ./parallel_copy_to_scratch.sh pretrain \
#       > /scratch/colliderml/arxiv_retraining/_logs/copy_pretrain.log 2>&1 &
#
# ----------------------------------------------------------------------------
set -euo pipefail

# ── Defaults ──
JOBS=20
DRY_RUN=false
SCRATCH="/scratch"

# ── Source / target roots ──
# Set SRC_ROOT to wherever your preprocessed datasets live (e.g. a slow
# NFS / EOS / S3 mount). Override at the CLI with --src. The shipped
# default below is a generic placeholder; edit it to fit your system.
SRC_ROOT="${COLLIDERML_SRC_ROOT:-/data/colliderml/arxiv_retraining}"
DST_SUBDIR="colliderml/arxiv_retraining"   # appended to $SCRATCH

# ── Dataset groups ──
PRETRAIN_DATASETS=(p0_core_pretrain p0_core_kf_hits_pretrain)
FINETUNE_DATASETS=(p200_core_kf_matched_finetune p200_core_kf_hits_finetune)
ALL_DATASETS=("${PRETRAIN_DATASETS[@]}" "${FINETUNE_DATASETS[@]}")

# ── Argument parsing ──
print_help() {
    cat <<EOF
Usage: $0 {pretrain|finetune|all|<dataset_name>} [options]

Targets:
  pretrain   copy ${PRETRAIN_DATASETS[*]}
  finetune   copy ${FINETUNE_DATASETS[*]}
  all        copy all 4 datasets
  <name>     copy a single dataset (one of the names above)

Options:
  -j, --jobs N    number of parallel cp jobs (default ${JOBS})
      --dry-run   list files without copying
      --scratch P override scratch root (default ${SCRATCH})
      --src P     override EOS source root (default ${SRC_ROOT})
  -h, --help      show this message and exit

Examples:
  $0 pretrain --jobs 30
  $0 p200_core_kf_matched_finetune --dry-run
  nohup $0 all > /scratch/colliderml/arxiv_retraining/_logs/copy_all.log 2>&1 &
EOF
}

# A bare ``-h`` / ``--help`` (no other args) prints usage and exits 0.
if [[ $# -eq 0 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

TARGET="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs|-j) JOBS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --scratch) SCRATCH="$2"; shift 2 ;;
        --src)     SRC_ROOT="$2"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; print_help >&2; exit 1 ;;
        *)   echo "Unknown positional arg: $1" >&2; print_help >&2; exit 1 ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    print_help >&2
    exit 1
fi

# Resolve dataset list from group names
case "$TARGET" in
    pretrain) DATASETS=("${PRETRAIN_DATASETS[@]}") ;;
    finetune) DATASETS=("${FINETUNE_DATASETS[@]}") ;;
    all)      DATASETS=("${ALL_DATASETS[@]}") ;;
    *)
        # Single-dataset mode — must be one of the known datasets
        match=false
        for ds in "${ALL_DATASETS[@]}"; do
            if [[ "$ds" == "$TARGET" ]]; then match=true; break; fi
        done
        if ! $match; then
            echo "Unknown dataset: $TARGET" >&2
            echo "Valid datasets: ${ALL_DATASETS[*]}" >&2
            exit 1
        fi
        DATASETS=("$TARGET")
        ;;
esac

DST_ROOT="${SCRATCH}/${DST_SUBDIR}"
mkdir -p "$DST_ROOT"

# ── Helper: copy one preprocessed dataset (shard tree + manifests) ──
copy_dataset() {
    local name="$1"
    local src="${SRC_ROOT}/${name}"
    local dst="${DST_ROOT}/${name}"

    if [[ ! -d "$src" ]]; then
        echo "  SKIP ${name} — source not found: ${src}"
        return
    fi

    mkdir -p "$dst"

    # ── Top-level metadata (manifest.json + split.json) ──────────────────────
    # Both must be present for downstream training:
    #   - manifest.json carries the preprocessing schema (format_version,
    #     sort_key, hit_feature_names, …) — read by data.py at shard open.
    #   - split.json maps shard indices to train / val / test — required by
    #     ColliderMLRegrDataModule.setup; missing it raises FileNotFoundError.
    # We always overwrite (small, cheap, must stay in sync with the source)
    # AND we fail loudly if either is missing on the source — this block has
    # historically been silent and led to a "split.json not found" surprise
    # at training time.
    local meta_missing=()
    for meta in manifest.json split.json; do
        if [[ -f "${src}/${meta}" ]]; then
            cp -f "${src}/${meta}" "${dst}/${meta}"
            echo "    [metadata] copied ${meta}  ($(stat --format='%s' "${dst}/${meta}") bytes)"
        else
            meta_missing+=("$meta")
        fi
    done
    if [[ ${#meta_missing[@]} -gt 0 ]]; then
        echo "    ERROR: missing on source: ${meta_missing[*]}" >&2
        echo "    Source: ${src}" >&2
        echo "    Re-run preprocessing (which auto-builds split.json) before copying." >&2
        return 1
    fi

    # ── Build the per-file copy list ──
    local file_list
    file_list=$(mktemp /tmp/copy_list_${name}_XXXXXX.txt)

    local total=0 skipped=0 to_copy=0 bytes_to_copy=0

    # Iterate shards
    for shard_dir in "${src}"/shard_*; do
        [[ -d "$shard_dir" ]] || continue
        local shard_name
        shard_name=$(basename "$shard_dir")
        mkdir -p "${dst}/${shard_name}"

        # Files at shard root: hits.npy, hit_times.npy, _complete sentinel
        for f in "${shard_dir}"/*.npy "${shard_dir}"/_complete; do
            [[ -f "$f" ]] || continue
            total=$((total + 1))
            local base; base=$(basename "$f")
            local target_file="${dst}/${shard_name}/${base}"
            local src_size; src_size=$(stat --format="%s" "$f")
            if [[ -f "$target_file" ]]; then
                local dst_size; dst_size=$(stat --format="%s" "$target_file")
                if [[ "$src_size" == "$dst_size" ]]; then
                    skipped=$((skipped + 1))
                    continue
                fi
            fi
            echo "$f ${target_file}" >> "$file_list"
            to_copy=$((to_copy + 1))
            bytes_to_copy=$((bytes_to_copy + src_size))
        done

        # Files in subdirectories (selected_tracks/)
        for subdir in "${shard_dir}"/*/; do
            [[ -d "$subdir" ]] || continue
            local sub_name; sub_name=$(basename "$subdir")
            mkdir -p "${dst}/${shard_name}/${sub_name}"
            for f in "${subdir}"*.npy; do
                [[ -f "$f" ]] || continue
                total=$((total + 1))
                local base; base=$(basename "$f")
                local target_file="${dst}/${shard_name}/${sub_name}/${base}"
                local src_size; src_size=$(stat --format="%s" "$f")
                if [[ -f "$target_file" ]]; then
                    local dst_size; dst_size=$(stat --format="%s" "$target_file")
                    if [[ "$src_size" == "$dst_size" ]]; then
                        skipped=$((skipped + 1))
                        continue
                    fi
                fi
                echo "$f ${target_file}" >> "$file_list"
                to_copy=$((to_copy + 1))
                bytes_to_copy=$((bytes_to_copy + src_size))
            done
        done
    done

    local gb_to_copy; gb_to_copy=$(awk "BEGIN {printf \"%.1f\", ${bytes_to_copy}/1e9}")
    echo "  ${name}: ${total} files total, ${skipped} already on scratch, ${to_copy} to copy (${gb_to_copy} GB)"

    if [[ "$to_copy" -eq 0 ]]; then
        rm -f "$file_list"
        return
    fi

    if $DRY_RUN; then
        echo "    [DRY RUN] Would copy ${to_copy} files with ${JOBS} parallel jobs."
        head -5 "$file_list" | while read -r src dst; do echo "      $(basename "$src")"; done
        [[ "$to_copy" -gt 5 ]] && echo "      ... and $((to_copy - 5)) more"
        rm -f "$file_list"
        return
    fi

    echo "    Copying with ${JOBS} parallel jobs..."
    local t_start; t_start=$(date +%s)

    # GNU parallel: each line of the file list is "src dst". Avoid ``--bar``:
    # it writes carriage-return-updated progress to stdout, which floods the
    # log file when running under nohup (no TTY). The per-dataset summary
    # line below is enough.
    cat "$file_list" | parallel --jobs "$JOBS" --colsep ' ' \
        "cp {1} {2}" > /dev/null

    local t_end; t_end=$(date +%s)
    local elapsed=$((t_end - t_start))
    local rate; rate=$(awk "BEGIN {r=${bytes_to_copy}/${elapsed:-1}/1e6; printf \"%.0f\", r}")
    echo "    Done: ${to_copy} files in ${elapsed}s (${rate} MB/s)"

    # ── Post-copy verification ───────────────────────────────────────────────
    # Confirm manifest.json + split.json actually landed on the target and
    # that every source shard has a corresponding destination shard. A
    # silent skip here used to surface only at training time as
    # ``FileNotFoundError: split.json not found`` — explicit verification
    # makes any future regression visible at copy time instead.
    local n_shards_src; n_shards_src=$(find "${src}" -maxdepth 1 -name 'shard_*' -type d | wc -l)
    local n_shards_dst; n_shards_dst=$(find "${dst}" -maxdepth 1 -name 'shard_*' -type d | wc -l)
    local verify_ok=true
    for meta in manifest.json split.json; do
        if [[ ! -f "${dst}/${meta}" ]]; then
            echo "    VERIFY FAIL: ${dst}/${meta} missing after copy" >&2
            verify_ok=false
        fi
    done
    if [[ "$n_shards_dst" -ne "$n_shards_src" ]]; then
        echo "    VERIFY WARN: ${name} shards on dst (${n_shards_dst}) != src (${n_shards_src})" >&2
    fi
    if $verify_ok; then
        echo "    [verify] manifest.json + split.json + ${n_shards_dst}/${n_shards_src} shards present on ${dst}"
    fi
    echo

    rm -f "$file_list"
}

# ── Main ──
echo "============================================================================"
echo "Parallel copy: arxiv_retraining EOS → scratch"
echo "============================================================================"
echo "  Source:    $SRC_ROOT"
echo "  Target:    $DST_ROOT"
echo "  Datasets:  ${DATASETS[*]}"
echo "  Parallel:  $JOBS"
echo "  Dry run:   $DRY_RUN"
echo "  Scratch free: $(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"
echo

for ds in "${DATASETS[@]}"; do
    copy_dataset "$ds"
done

echo "============================================================================"
echo "All done."
echo "============================================================================"
df -h "$SCRATCH" | tail -1 | awk '{print "  Scratch used: "$3" / "$2"  free: "$4}'
