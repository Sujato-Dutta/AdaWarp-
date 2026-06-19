#!/usr/bin/env bash
#SBATCH --job-name=awp-mc
#SBATCH --array=0-59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=out/awp_motion_code/logs/%A_%a.out
#SBATCH --error=out/awp_motion_code/logs/%A_%a.err

set -euo pipefail

# This array launches 12 datasets x 5 seeds. Adjust the environment setup for
# your cluster before submitting. Use the administrator-provided Python
# environment as-is: this launcher does not install or upgrade packages.
# Running one compact model per GPU gives better throughput than replicating a
# single tiny dataset across multiple GPUs.

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
SEEDS=(42 43 44 45 46)

DATASET_COUNT="${#DATASETS[@]}"
DATASET_INDEX="$((SLURM_ARRAY_TASK_ID % DATASET_COUNT))"
SEED_INDEX="$((SLURM_ARRAY_TASK_ID / DATASET_COUNT))"
DATASET="${DATASETS[${DATASET_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"
OUTPUT_DIR="${OUTPUT_DIR:-out/awp_motion_code}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUTPUT_DIR}/logs"

"${PYTHON_BIN}" -u benchmark_awp_motion_code.py \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS}
