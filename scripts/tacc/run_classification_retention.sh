#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

DATASETS_DEFAULT="PDSetting1 PDSetting2 PronunciationAudio ECGFiveDays FreezerSmallTrain HouseTwenty InsectEPGRegularTrain ItalyPowerDemand Lightning7 MoteStrain PowerCons SonyAIBORobotSurface2 UWaveGestureLibraryAll"
SEEDS_DEFAULT="42 43 44 45 46"

read -r -a DATASETS <<< "${ADAWARP_CLASSIFICATION_DATASETS:-$DATASETS_DEFAULT}"
read -r -a SEEDS <<< "${ADAWARP_SEEDS:-$SEEDS_DEFAULT}"

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results}"
DEVICE="${ADAWARP_DEVICE:-auto}"
EPOCHS="${ADAWARP_CLASSIFICATION_EPOCHS:-50}"
STEPS_PER_EPOCH="${ADAWARP_STEPS_PER_EPOCH:-4}"

mkdir -p "$OUTPUT_ROOT/logs/classification_retention" "$OUTPUT_ROOT/audit"

if [[ "${ADAWARP_SKIP_MOTION_CODE:-0}" != "1" ]]; then
  bash scripts/tacc/check_motion_code_env.sh
  python benchmark_motion_code_classification.py \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --skip-existing \
    > "$OUTPUT_ROOT/logs/classification_retention/MotionCode_classification.stdout.log" \
    2> "$OUTPUT_ROOT/logs/classification_retention/MotionCode_classification.stderr.log"
else
  echo "[Motion Code classification] skipped because ADAWARP_SKIP_MOTION_CODE=1"
fi

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python benchmark_awp_motion_code.py \
      --dataset "$dataset" \
      --seed "$seed" \
      --output-dir "$OUTPUT_ROOT/adawarp_classification" \
      --device "$DEVICE" \
      --epochs "$EPOCHS" \
      --steps-per-epoch "$STEPS_PER_EPOCH" \
      --refit \
      > "$OUTPUT_ROOT/logs/classification_retention/AdaWarp_${dataset}_seed${seed}.stdout.log" \
      2> "$OUTPUT_ROOT/logs/classification_retention/AdaWarp_${dataset}_seed${seed}.stderr.log"
  done
done

if [[ "${ADAWARP_RUN_TSLIB_CLASSIFICATION:-1}" == "1" ]]; then
  python scripts/tacc/run_tslibrary_classification.py \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --models ${ADAWARP_TSLIB_CLASSIFICATION_MODELS:-Informer Autoformer FEDformer ETSformer LightTS PatchTST Crossformer DLinear TimesNet iTransformer Mamba} \
    --output-root "$OUTPUT_ROOT" \
    --batch-size "${ADAWARP_TSLIB_CLASSIFICATION_BATCH_SIZE:-16}" \
    --train-epochs "${ADAWARP_TSLIB_CLASSIFICATION_EPOCHS:-100}" \
    --timesnet-epochs "${ADAWARP_TSLIB_CLASSIFICATION_TIMESNET_EPOCHS:-30}" \
    --patience "${ADAWARP_TSLIB_CLASSIFICATION_PATIENCE:-10}" \
    --num-workers "${ADAWARP_TSLIB_CLASSIFICATION_NUM_WORKERS:-4}" \
    --gpu "${ADAWARP_GPU:-0}" \
    --missing-policy "${ADAWARP_TSLIB_CLASSIFICATION_MISSING_POLICY:-fail}" \
    --skip-existing \
    > "$OUTPUT_ROOT/logs/classification_retention/tslibrary_classification.stdout.log" \
    2> "$OUTPUT_ROOT/logs/classification_retention/tslibrary_classification.stderr.log"
fi

if [[ "${ADAWARP_RUN_MODERN_TSC:-1}" == "1" ]]; then
  python benchmark_modern_tsc_classifiers.py \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --models ${ADAWARP_MODERN_TSC_MODELS:-MiniROCKET MultiROCKET Hydra InceptionTime} \
    --output-root "$OUTPUT_ROOT" \
    --n-jobs "${ADAWARP_TSC_N_JOBS:-1}" \
    --inception-epochs "${ADAWARP_INCEPTION_EPOCHS:-150}" \
    --batch-size "${ADAWARP_INCEPTION_BATCH_SIZE:-64}" \
    --missing-policy "${ADAWARP_MODERN_TSC_MISSING_POLICY:-fail}" \
    --skip-existing \
    > "$OUTPUT_ROOT/logs/classification_retention/modern_tsc.stdout.log" \
    2> "$OUTPUT_ROOT/logs/classification_retention/modern_tsc.stderr.log"
fi

echo "[classification_retention] complete"
