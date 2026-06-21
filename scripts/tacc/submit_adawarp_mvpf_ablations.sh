#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/mvpf_ablation slurm results

ABLATIONS_DEFAULT="no_prototype_memory no_adaptive_shifts single_scale_16 fixed_radius no_frequency_gate equal_component_weights no_trend_residual no_reconstruction_aux"
read -r -a ABLATIONS <<< "${ADAWARP_MVPF_ABLATIONS_TO_SUBMIT:-${ADAWARP_MVPF_ABLATIONS:-$ABLATIONS_DEFAULT}}"

ACCOUNT="${ADAWARP_ALLOCATION:-IRI23021}"
PARTITION="${ADAWARP_PARTITION:-gh}"
WALLTIME="${ADAWARP_WALLTIME:-24:00:00}"
MAX_SUBMITS="${ADAWARP_MAX_SUBMITS:-20}"
JOB_PARENT="${ADAWARP_MVPF_ABLATION_JOB_PARENT:-$WORK/motion_code-master/results/vista_ltsf_adawarp_mvpf_ablations}"

if (( ${#ABLATIONS[@]} > MAX_SUBMITS )); then
  echo "[submit_mvpf_ablation] refusing to submit ${#ABLATIONS[@]} jobs because ADAWARP_MAX_SUBMITS=$MAX_SUBMITS" >&2
  echo "[submit_mvpf_ablation] reduce ADAWARP_MVPF_ABLATIONS_TO_SUBMIT or raise ADAWARP_MAX_SUBMITS intentionally" >&2
  exit 2
fi

echo "[submit_mvpf_ablation] account=$ACCOUNT partition=$PARTITION walltime=$WALLTIME max_submits=$MAX_SUBMITS"
echo "[submit_mvpf_ablation] parent=$JOB_PARENT"
echo "[submit_mvpf_ablation] ablations=${ABLATIONS[*]}"
echo "[submit_mvpf_ablation] datasets=${ADAWARP_LTSF_DATASETS:-ETTh1 ETTh2 Weather Electricity Traffic}"

for ABLATION in "${ABLATIONS[@]}"; do
  ABLATION_SAFE="$(printf "%s" "$ABLATION" | tr -c 'A-Za-z0-9_' '_' | sed 's/_*$//')"
  OUTPUT_ROOT="$JOB_PARENT/$ABLATION_SAFE"
  JOB_NAME="mvpfab_${ABLATION_SAFE}"
  sbatch \
    -A "$ACCOUNT" \
    -p "$PARTITION" \
    -t "$WALLTIME" \
    -J "$JOB_NAME" \
    slurm/adawarp_mvpf_ablation_vista.sbatch "$ABLATION" "$OUTPUT_ROOT"
done
