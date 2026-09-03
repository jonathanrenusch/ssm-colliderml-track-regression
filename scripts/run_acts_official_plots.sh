#!/bin/bash
# Official ACTS-pipeline performance plots for the paper model (YZ-3L-FT) on the
# v2 datasets: shim the parquet to the Release-1 layout, then run the collaborators'
# acts_integration.py per dataset -> eval_plots/paper/acts_official/<ds>/resolutions.pdf
set -u
R=/shared/tracking/ssm-colliderml-track-regression
PY=/shared/tracking/pyacts_env/bin/python
C=/scratch/colliderml/acts_compat
RAW=/scratch/colliderml/drift_beamspot_v2
GEO="--geo-json /scratch/colliderml/odd-json/odd.json --material-json /scratch/colliderml/odd-json/gen3_material_map_map.json --geoid-map-csv /scratch/colliderml/odd-json/geoid_map.csv --seeding-config /scratch/colliderml/odd-json/odd-seeding-config-gen3.json"
# Override the model with MODEL_CKPT / MODEL_CFG / OUT_TAG env vars (default: YZ-3L-FT).
MODEL="--ckpt ${MODEL_CKPT:-$R/eval_plots/round6/YZ3LFT/ckpts/model.ckpt} --config ${MODEL_CFG:-$R/eval_plots/paper/acts_official/YZ3LFT_config_short.yaml} --variant v5pc"
OUT_TAG="${OUT_TAG:-acts_official}"
V2="--sort-key geometry --seed-residual-features --d0-max 7.1 --z0-max 270 --pt-min 1.0 --pt-max 110 --hit-bounds-tolerance 25 --dump-residuals"
rw() {  # per-dataset Residual_<param> windows: the stock bins are coarser than the sharp cores
  case "$1" in
    ttbar)               echo "--residual-window qop:150:0.03" ;;
    single_muon_uniform) echo "--residual-window qop:150:0.03" ;;
    single_muon_2GeV)    echo "--residual-window qop:120:0.02 --residual-window phi:120:0.02" ;;
    single_muon_10GeV)   echo "--residual-window qop:120:0.006 --residual-window phi:120:0.008 --residual-window theta:120:0.004 --residual-window d0:120:0.2" ;;
    single_muon_100GeV)  echo "--residual-window qop:120:0.002 --residual-window phi:120:0.004 --residual-window theta:120:0.002 --residual-window d0:120:0.2" ;;
  esac
}
GPU="${1:-1}"; SETS="${2:-all}"
cd $R; mkdir -p launch_logs/sweep7/acts

shim() { [ -f "$2/tracker_hits/part_000000.parquet" ] || $PY scripts/make_acts_compat_parquet.py "$1" "$2"; }

run_ds() {  # name compat_dir n_events
  echo "=== $1 ($3 events) $(date)"
  # per-process Triton cache: parallel runs race on the shared AFS ~/.triton
  export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}_gpu${GPU}
  mkdir -p "$TRITON_CACHE_DIR"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/acts_integration.py \
    --particles-dir "$2/particles" --hits-dir "$2/tracker_hits" \
    $GEO $MODEL $V2 $(rw "$1") --events "$3" -o "eval_plots/paper/$OUT_TAG/$1" \
    > "launch_logs/sweep7/acts/${OUT_TAG}_$1.log" 2>&1
  grep -m1 "tracks seen" "launch_logs/sweep7/acts/${OUT_TAG}_$1.log"
  # legacy-style pages (campaign design, clipped stats in legends) next to resolutions.pdf
  /shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/bin/python \
    scripts/acts_legacy_style_plots.py "eval_plots/paper/$OUT_TAG/$1" "$1" "ACTS KF" \
    >> "launch_logs/sweep7/acts/${OUT_TAG}_$1.log" 2>&1
  # band pages from the writers' fit-sigma errors; vs-pT + ratio only for the uniform sample
  WPT=""; [ "$1" = single_muon_uniform ] && WPT="--with-pt"
  /shared/tracking/ssm-colliderml-track-regression/.pixi/envs/default/bin/python \
    scripts/acts_band_plots.py "eval_plots/paper/$OUT_TAG/$1" "$1" $WPT \
    >> "launch_logs/sweep7/acts/${OUT_TAG}_$1.log" 2>&1
}

if [ "$SETS" = all ] || [ "$SETS" = ttbar ]; then
  # ttbar: eval runs 6-15 (disjoint from training runs 46-784), combined
  mkdir -p $C/ttbar_r6-15/particles $C/ttbar_r6-15/tracker_hits
  for n in 6 7 8 9 10 11 12 13 14 15; do
    shim $RAW/ttbar/v1/runs/$n $C/ttbar_run$n
    ln -sf $(readlink -f $C/ttbar_run$n/particles)/*.parquet $C/ttbar_r6-15/particles/run${n}_particles.parquet
    ln -sf $C/ttbar_run$n/tracker_hits/part_000000.parquet $C/ttbar_r6-15/tracker_hits/run${n}_hits.parquet
  done
  run_ds ttbar $C/ttbar_r6-15 12800
fi

if [ "$SETS" = single_muon_uniform ]; then
  # uniform: 200k-event slice (enough for smooth vs-pT profiles; full raw = 200M events)
  [ -f "$C/single_muon_uniform_200k/tracker_hits/part_000000.parquet" ] || \
    MAX_EVENTS=200000 $PY scripts/make_acts_compat_parquet.py $RAW/single_muon_uniform $C/single_muon_uniform_200k
  run_ds single_muon_uniform $C/single_muon_uniform_200k 200000
fi

for ds in single_muon_2GeV single_muon_10GeV single_muon_100GeV; do
  if [ "$SETS" = all ] || [ "$SETS" = "$ds" ]; then
    shim $RAW/$ds $C/$ds
    NEV=$($PY - <<EOF
import pyarrow.parquet as pq
print(pq.read_metadata("$C/$ds/tracker_hits/part_000000.parquet").num_rows)
EOF
)
    run_ds $ds $C/$ds "$NEV"
  fi
done
echo "ALL DONE $(date)"
