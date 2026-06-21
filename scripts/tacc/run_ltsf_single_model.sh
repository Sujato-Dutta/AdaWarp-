#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/tacc/run_ltsf_single_model.sh MODEL" >&2
  exit 2
fi

MODEL="$1"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

DATASETS_DEFAULT="ETTh1 ETTh2 Weather Electricity Traffic"
HORIZONS_DEFAULT="96 192 336 720"
SEEDS_DEFAULT="42"

read -r -a DATASETS <<< "${ADAWARP_LTSF_DATASETS:-$DATASETS_DEFAULT}"
read -r -a HORIZONS <<< "${ADAWARP_LTSF_HORIZONS:-$HORIZONS_DEFAULT}"
read -r -a SEEDS <<< "${ADAWARP_LTSF_SEEDS:-$SEEDS_DEFAULT}"

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results/ltsf_single_model}"
mkdir -p "$OUTPUT_ROOT/audit"

{
  echo "model=$MODEL"
  echo "output_root=$OUTPUT_ROOT"
  echo "datasets=${DATASETS[*]}"
  echo "horizons=${HORIZONS[*]}"
  echo "seeds=${SEEDS[*]}"
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
} > "$OUTPUT_ROOT/audit/ltsf_single_model_config.txt"

run_adawarp_family() {
  python benchmark_adawarp_ltsf.py \
    --datasets "${DATASETS[@]}" \
    --horizons "${HORIZONS[@]}" \
    --seeds "${SEEDS[@]}" \
    --models "$MODEL" \
    --seq-len "${ADAWARP_LTSF_SEQ_LEN:-96}" \
    --ridge "${ADAWARP_LTSF_RIDGE:-0.02}" \
    --num-clusters "${ADAWARP_LTSF_NUM_CLUSTERS:-8}" \
    --max-train-windows "${ADAWARP_LTSF_MAX_TRAIN_WINDOWS:-2048}" \
    --max-eval-windows "${ADAWARP_LTSF_MAX_EVAL_WINDOWS:-2048}" \
    --data-root "${ADAWARP_LTSF_DATA_ROOT:-TSLibrary/dataset}" \
    --output-root "$OUTPUT_ROOT"
}

run_tslibrary_family() {
  python scripts/tacc/run_tslibrary_ltsf_baselines.py \
    --datasets "${DATASETS[@]}" \
    --horizons "${HORIZONS[@]}" \
    --seeds "${SEEDS[@]}" \
    --models "$MODEL" \
    --seq-len "${ADAWARP_LTSF_SEQ_LEN:-96}" \
    --label-len "${ADAWARP_TSLIB_LABEL_LEN:-48}" \
    --train-epochs "${ADAWARP_TSLIB_EPOCHS:-10}" \
    --batch-size "${ADAWARP_TSLIB_BATCH_SIZE:-32}" \
    --learning-rate "${ADAWARP_TSLIB_LR:-0.0001}" \
    --d-model "${ADAWARP_TSLIB_D_MODEL:-128}" \
    --d-ff "${ADAWARP_TSLIB_D_FF:-256}" \
    --n-heads "${ADAWARP_TSLIB_N_HEADS:-8}" \
    --e-layers "${ADAWARP_TSLIB_E_LAYERS:-2}" \
    --d-layers "${ADAWARP_TSLIB_D_LAYERS:-1}" \
    --patience "${ADAWARP_TSLIB_PATIENCE:-3}" \
    --output-root "$OUTPUT_ROOT"
}

run_custom_family() {
  python benchmark_custom_neural_ltsf.py \
    --datasets "${DATASETS[@]}" \
    --horizons "${HORIZONS[@]}" \
    --seeds "${SEEDS[@]}" \
    --models "$MODEL" \
    --seq-len "${ADAWARP_LTSF_SEQ_LEN:-96}" \
    --epochs "${ADAWARP_CUSTOM_LTSF_EPOCHS:-10}" \
    --batch-size "${ADAWARP_CUSTOM_LTSF_BATCH_SIZE:-16}" \
    --eval-batch-size "${ADAWARP_CUSTOM_LTSF_EVAL_BATCH_SIZE:-16}" \
    --learning-rate "${ADAWARP_CUSTOM_LTSF_LR:-0.001}" \
    --weight-decay "${ADAWARP_CUSTOM_LTSF_WEIGHT_DECAY:-0.0001}" \
    --d-model "${ADAWARP_CUSTOM_LTSF_D_MODEL:-256}" \
    --depth "${ADAWARP_CUSTOM_LTSF_DEPTH:-2}" \
    --blocks "${ADAWARP_CUSTOM_LTSF_BLOCKS:-4}" \
    --dropout "${ADAWARP_CUSTOM_LTSF_DROPOUT:-0.05}" \
    --vpnet-patch-len "${ADAWARP_VPNET_PATCH_LEN:-16}" \
    --max-train-windows "${ADAWARP_LTSF_MAX_TRAIN_WINDOWS:-2048}" \
    --max-eval-windows "${ADAWARP_LTSF_MAX_EVAL_WINDOWS:-2048}" \
    --data-root "${ADAWARP_LTSF_DATA_ROOT:-TSLibrary/dataset}" \
    --output-root "$OUTPUT_ROOT" \
    --device "${ADAWARP_DEVICE:-auto}"
}

case "$MODEL" in
  AdaWarp-U|AdaWarp-Global|AdaWarp-Cluster|persistence|seasonal_naive)
    run_adawarp_family
    ;;
  DLinear|PatchTST|TimesNet|iTransformer|TimeMixer|Autoformer|FEDformer|Informer|ETSformer|Pyraformer)
    run_tslibrary_family
    ;;
  NLinear|N-BEATS|N-HiTS|VPNet|AdaWarp-VPF)
    run_custom_family
    ;;
  *)
    echo "unknown LTSF model: $MODEL" >&2
    exit 2
    ;;
esac

python aggregate_adawarp_experiments.py --output-root "$OUTPUT_ROOT"
echo "[run_ltsf_single_model] complete model=$MODEL output_root=$OUTPUT_ROOT"
