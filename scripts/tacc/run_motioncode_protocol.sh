#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

DATASETS_DEFAULT="PronunciationAudio ECGFiveDays FreezerSmallTrain HouseTwenty InsectEPGRegularTrain ItalyPowerDemand Lightning7 MoteStrain PowerCons SonyAIBORobotSurface2"
SEEDS_DEFAULT="42 43 44 45 46"
PREFIX_DEFAULT="0.8 0.6"

read -r -a DATASETS <<< "${ADAWARP_DATASETS:-$DATASETS_DEFAULT}"
read -r -a SEEDS <<< "${ADAWARP_SEEDS:-$SEEDS_DEFAULT}"
read -r -a PREFIXES <<< "${ADAWARP_PREFIX_FRACTIONS:-$PREFIX_DEFAULT}"

DEVICE="${ADAWARP_DEVICE:-auto}"
EPOCHS="${ADAWARP_EPOCHS:-50}"
STEPS_PER_EPOCH="${ADAWARP_STEPS_PER_EPOCH:-4}"
OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results}"

REFIT_ARG=()
if [[ "${ADAWARP_REFIT:-1}" == "1" ]]; then
  REFIT_ARG=(--refit)
fi

mkdir -p "$OUTPUT_ROOT/logs/motioncode_protocol"

if [[ "${ADAWARP_SKIP_MOTION_CODE:-0}" != "1" ]]; then
  bash scripts/tacc/check_motion_code_env.sh
  echo "[Motion Code] matched rerun"
  python benchmark_motion_code_forecasting.py \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --prefix-fractions "${PREFIXES[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --skip-existing \
    > "$OUTPUT_ROOT/logs/motioncode_protocol/MotionCode_matched.stdout.log" \
    2> "$OUTPUT_ROOT/logs/motioncode_protocol/MotionCode_matched.stderr.log"
else
  echo "[Motion Code] skipped because ADAWARP_SKIP_MOTION_CODE=1"
fi

for prefix in "${PREFIXES[@]}"; do
  tag="${prefix/./p}"
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "[AdaWarp] dataset=$dataset seed=$seed prefix=$prefix"
      python benchmark_awp_forecasting.py \
        --dataset "$dataset" \
        --seed "$seed" \
        --observed-fraction "$prefix" \
        --output-dir "$OUTPUT_ROOT/adawarp_motioncode_protocol/prefix_$tag" \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        --steps-per-epoch "$STEPS_PER_EPOCH" \
        --save-raw-predictions \
        "${REFIT_ARG[@]}" \
        > "$OUTPUT_ROOT/logs/motioncode_protocol/AdaWarp_${dataset}_prefix${tag}_seed${seed}.stdout.log" \
        2> "$OUTPUT_ROOT/logs/motioncode_protocol/AdaWarp_${dataset}_prefix${tag}_seed${seed}.stderr.log"
    done
  done
done

BASELINE_MODELS_DEFAULT="persistence moving_average seasonal_naive exponential_smoothing arima state_space tbats dynamics_simplex dynamics_equal dynamics_no_class dynamics_local_simplex dynamics_best_head"
read -r -a BASELINE_MODELS <<< "${ADAWARP_PROTOCOL_BASELINES:-$BASELINE_MODELS_DEFAULT}"
python benchmark_adawarp_protocol_baselines.py \
  --datasets "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --prefix-fractions "${PREFIXES[@]}" \
  --models "${BASELINE_MODELS[@]}" \
  --output-root "$OUTPUT_ROOT"

if [[ "${ADAWARP_RUN_SHORT_NEURAL:-1}" == "1" ]]; then
  for prefix in "${PREFIXES[@]}"; do
    tag="${prefix/./p}"
    python benchmark_tslibrary_neural_forecasting.py \
      --models ${ADAWARP_SHORT_NEURAL_MODELS:-DLinear NLinear PatchTST TimesNet iTransformer TimeMixer N-BEATS N-HiTS VPNet Autoformer FEDformer Informer ETSformer Pyraformer} \
      --datasets "${DATASETS[@]}" \
      --seeds "${SEEDS[@]}" \
      --observed-fraction "$prefix" \
      --output-dir "$OUTPUT_ROOT/tslibrary_short_forecasting/prefix_$tag" \
      --device "${ADAWARP_NEURAL_DEVICE:-cuda}" \
      --epochs "${ADAWARP_SHORT_NEURAL_EPOCHS:-10}" \
      --save-raw-predictions
  done
fi

echo "[motioncode_protocol] complete"
