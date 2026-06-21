#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/mvpf_by_dataset slurm results

DATASETS_DEFAULT="ETTh1 ETTh2 Weather Electricity Traffic"
read -r -a DATASETS <<< "${ADAWARP_MVPF_DATASETS_TO_SUBMIT:-${ADAWARP_LTSF_DATASETS:-$DATASETS_DEFAULT}}"

ACCOUNT="${ADAWARP_ALLOCATION:-IRI23021}"
PARTITION="${ADAWARP_PARTITION:-gh}"
WALLTIME="${ADAWARP_WALLTIME:-24:00:00}"
JOB_PARENT="${ADAWARP_MVPF_JOB_PARENT:-$WORK/motion_code-master/results/vista_ltsf_adawarp_mvpf_by_dataset}"

echo "[submit_adawarp_mvpf] account=$ACCOUNT partition=$PARTITION walltime=$WALLTIME"
echo "[submit_adawarp_mvpf] parent=$JOB_PARENT"
echo "[submit_adawarp_mvpf] datasets=${DATASETS[*]}"

for DATASET in "${DATASETS[@]}"; do
  DATASET_SAFE="$(printf "%s" "$DATASET" | tr -c 'A-Za-z0-9_' '_' | sed 's/_*$//')"
  OUTPUT_ROOT="$JOB_PARENT/${DATASET_SAFE}/AdaWarp_MVPF"
  JOB_NAME="mvpf_${DATASET_SAFE}"
  sbatch \
    -A "$ACCOUNT" \
    -p "$PARTITION" \
    -t "$WALLTIME" \
    -J "$JOB_NAME" \
    slurm/adawarp_mvpf_ltsf_vista.sbatch "$DATASET" "$OUTPUT_ROOT"
done