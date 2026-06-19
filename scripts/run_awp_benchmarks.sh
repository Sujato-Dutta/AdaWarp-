#!/usr/bin/env bash
set -euo pipefail

# Sequential launcher for an interactive GPU allocation.
# Override values as needed, for example:
#   SEEDS="42 43 44 45 46" DEVICE="cuda:0" bash scripts/run_awp_benchmarks.sh

DATASETS=(
  "PDSetting1"
  "PDSetting2"
  "PronunciationAudio"
  "ECGFiveDays"
  "FreezerSmallTrain"
  "HouseTwenty"
  "InsectEPGRegularTrain"
  "ItalyPowerDemand"
  "Lightning7"
  "MoteStrain"
  "PowerCons"
  "SonyAIBORobotSurface2"
)

SEEDS="${SEEDS:-42 43 44 45 46}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-out/awp_motion_code}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for dataset in "${DATASETS[@]}"; do
  for seed in ${SEEDS}; do
    "${PYTHON_BIN}" -u benchmark_awp_motion_code.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --output-dir "${OUTPUT_DIR}" \
      ${EXTRA_ARGS}
  done
done

"${PYTHON_BIN}" -u aggregate_awp_results.py --output-dir "${OUTPUT_DIR}"
