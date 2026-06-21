#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/ltsf_by_model slurm results

MODELS_DEFAULT="AdaWarp-U AdaWarp-Global AdaWarp-Cluster DLinear PatchTST TimesNet iTransformer TimeMixer FEDformer VPNet AdaWarp-VPF"
read -r -a MODELS <<< "${ADAWARP_LTSF_MODELS_TO_SUBMIT:-$MODELS_DEFAULT}"

ACCOUNT="${ADAWARP_ALLOCATION:-IRI23021}"
PARTITION="${ADAWARP_PARTITION:-gh}"
WALLTIME="${ADAWARP_WALLTIME:-24:00:00}"
JOB_PARENT="${ADAWARP_LTSF_JOB_PARENT:-$WORK/motion_code-master/results/vista_ltsf_by_model}"

SUBMITTED=0
MAX_SUBMISSIONS="${ADAWARP_MAX_SUBMISSIONS:-0}"

echo "[submit_ltsf_by_model] account=$ACCOUNT partition=$PARTITION walltime=$WALLTIME"
echo "[submit_ltsf_by_model] parent=$JOB_PARENT"
echo "[submit_ltsf_by_model] models=${MODELS[*]}"

for MODEL in "${MODELS[@]}"; do
  if [[ "$MAX_SUBMISSIONS" != "0" && "$SUBMITTED" -ge "$MAX_SUBMISSIONS" ]]; then
    echo "[submit_ltsf_by_model] reached ADAWARP_MAX_SUBMISSIONS=$MAX_SUBMISSIONS"
    break
  fi

  MODEL_SAFE="$(printf "%s" "$MODEL" | tr -c 'A-Za-z0-9_' '_' | sed 's/_*$//')"
  OUTPUT_ROOT="$JOB_PARENT/${MODEL_SAFE}"
  JOB_NAME="ltsf_${MODEL_SAFE}"

  if command -v squeue >/dev/null 2>&1 && [[ "${ADAWARP_ALLOW_DUPLICATE_JOBS:-0}" != "1" ]]; then
    EXISTING="$(squeue -h -u "${USER:-}" -n "$JOB_NAME" -o "%i" 2>/dev/null | tr '\n' ' ' || true)"
    if [[ -n "${EXISTING//[[:space:]]/}" ]]; then
      echo "[submit_ltsf_by_model] skip $MODEL; existing queued/running job(s): $EXISTING"
      continue
    fi
  fi

  sbatch \
    -A "$ACCOUNT" \
    -p "$PARTITION" \
    -t "$WALLTIME" \
    -J "$JOB_NAME" \
    slurm/ltsf_single_model_vista.sbatch "$MODEL" "$OUTPUT_ROOT"
  SUBMITTED=$((SUBMITTED + 1))
done

echo "[submit_ltsf_by_model] submitted $SUBMITTED jobs"