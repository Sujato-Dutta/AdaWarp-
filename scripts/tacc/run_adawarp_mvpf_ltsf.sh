#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

DATASETS_DEFAULT="ETTh1 ETTh2 Weather Electricity Traffic"
HORIZONS_DEFAULT="96 192 336 720"
SEEDS_DEFAULT="42"
PATCH_LENS_DEFAULT="8 16 32"

read -r -a DATASETS <<< "${ADAWARP_LTSF_DATASETS:-$DATASETS_DEFAULT}"
read -r -a HORIZONS <<< "${ADAWARP_LTSF_HORIZONS:-$HORIZONS_DEFAULT}"
read -r -a SEEDS <<< "${ADAWARP_LTSF_SEEDS:-$SEEDS_DEFAULT}"
read -r -a PATCH_LENS <<< "${ADAWARP_MVPF_PATCH_LENS:-$PATCH_LENS_DEFAULT}"

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results/adawarp_mvpf_ltsf}"
mkdir -p "$OUTPUT_ROOT/audit"

{
  echo "model=AdaWarp-MVPF"
  echo "output_root=$OUTPUT_ROOT"
  echo "datasets=${DATASETS[*]}"
  echo "horizons=${HORIZONS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "patch_lens=${PATCH_LENS[*]}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$(command -v python)"
  python - <<'PY'
import sys
print("python_version=" + sys.version.replace("\n", " "))
try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda=" + str(torch.version.cuda))
    print("cuda_available=" + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("cuda_device=" + torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_error=" + repr(exc))
PY
} > "$OUTPUT_ROOT/audit/adawarp_mvpf_config.txt"

python benchmark_adawarp_mvpf_ltsf.py \
  --datasets "${DATASETS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --seeds "${SEEDS[@]}" \
  --seq-len "${ADAWARP_LTSF_SEQ_LEN:-96}" \
  --epochs "${ADAWARP_MVPF_EPOCHS:-10}" \
  --batch-size "${ADAWARP_MVPF_BATCH_SIZE:-16}" \
  --eval-batch-size "${ADAWARP_MVPF_EVAL_BATCH_SIZE:-16}" \
  --learning-rate "${ADAWARP_MVPF_LR:-0.0007}" \
  --weight-decay "${ADAWARP_MVPF_WEIGHT_DECAY:-0.0001}" \
  --d-model "${ADAWARP_MVPF_D_MODEL:-128}" \
  --depth "${ADAWARP_MVPF_DEPTH:-2}" \
  --dropout "${ADAWARP_MVPF_DROPOUT:-0.05}" \
  --patch-lens "${PATCH_LENS[@]}" \
  --num-prototypes "${ADAWARP_MVPF_NUM_PROTOTYPES:-8}" \
  --max-shift "${ADAWARP_MVPF_MAX_SHIFT:-2}" \
  --reconstruction-weight "${ADAWARP_MVPF_RECONSTRUCTION_WEIGHT:-0.03}" \
  --max-train-windows "${ADAWARP_LTSF_MAX_TRAIN_WINDOWS:-2048}" \
  --max-eval-windows "${ADAWARP_LTSF_MAX_EVAL_WINDOWS:-2048}" \
  --data-root "${ADAWARP_LTSF_DATA_ROOT:-TSLibrary/dataset}" \
  --output-root "$OUTPUT_ROOT" \
  --device "${ADAWARP_DEVICE:-auto}"

python aggregate_adawarp_experiments.py --output-root "$OUTPUT_ROOT"
echo "[run_adawarp_mvpf_ltsf] complete output_root=$OUTPUT_ROOT"