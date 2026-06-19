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
MODELS_DEFAULT="AdaWarp-U AdaWarp-Global AdaWarp-Cluster persistence seasonal_naive"

read -r -a DATASETS <<< "${ADAWARP_LTSF_DATASETS:-$DATASETS_DEFAULT}"
read -r -a HORIZONS <<< "${ADAWARP_LTSF_HORIZONS:-$HORIZONS_DEFAULT}"
read -r -a SEEDS <<< "${ADAWARP_LTSF_SEEDS:-$SEEDS_DEFAULT}"
read -r -a MODELS <<< "${ADAWARP_LTSF_MODELS:-$MODELS_DEFAULT}"

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results}"

python benchmark_adawarp_ltsf.py \
  --datasets "${DATASETS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --seeds "${SEEDS[@]}" \
  --models "${MODELS[@]}" \
  --seq-len "${ADAWARP_LTSF_SEQ_LEN:-96}" \
  --max-train-windows "${ADAWARP_LTSF_MAX_TRAIN_WINDOWS:-2048}" \
  --max-eval-windows "${ADAWARP_LTSF_MAX_EVAL_WINDOWS:-2048}" \
  --data-root "${ADAWARP_LTSF_DATA_ROOT:-TSLibrary/dataset}" \
  --output-root "$OUTPUT_ROOT"

if [[ "${ADAWARP_RUN_TSLIB_LTSF:-1}" == "1" ]]; then
  python scripts/tacc/run_tslibrary_ltsf_baselines.py \
    --datasets "${DATASETS[@]}" \
    --horizons "${HORIZONS[@]}" \
    --seeds "${SEEDS[@]}" \
    --models ${ADAWARP_TSLIB_LTSF_MODELS:-DLinear PatchTST TimesNet iTransformer TimeMixer Autoformer FEDformer Informer ETSformer Pyraformer} \
    --train-epochs "${ADAWARP_TSLIB_EPOCHS:-10}" \
    --batch-size "${ADAWARP_TSLIB_BATCH_SIZE:-32}" \
    --output-root "$OUTPUT_ROOT"
fi

if [[ "${ADAWARP_RUN_CUSTOM_LTSF:-1}" == "1" ]]; then
  python benchmark_custom_neural_ltsf.py \
    --datasets "${DATASETS[@]}" \
    --horizons "${HORIZONS[@]}" \
    --seeds "${SEEDS[@]}" \
    --models ${ADAWARP_CUSTOM_LTSF_MODELS:-NLinear N-BEATS N-HiTS VPNet} \
    --epochs "${ADAWARP_CUSTOM_LTSF_EPOCHS:-10}" \
    --batch-size "${ADAWARP_CUSTOM_LTSF_BATCH_SIZE:-16}" \
    --eval-batch-size "${ADAWARP_CUSTOM_LTSF_EVAL_BATCH_SIZE:-16}" \
    --device "${ADAWARP_DEVICE:-auto}" \
    --data-root "${ADAWARP_LTSF_DATA_ROOT:-TSLibrary/dataset}" \
    --output-root "$OUTPUT_ROOT"
fi

echo "[ltsf_main5] complete"
