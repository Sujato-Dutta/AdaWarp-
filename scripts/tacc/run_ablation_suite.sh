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

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results}"
ABLATION_MODELS_DEFAULT="gp_zero_mean_rbf gp_linear_trend_mean gp_periodic_plus_linear gp_local_linear_trend gp_class_mean_residual gp_prefix_validated_mean dynamics_without_gp_prototype_template dynamics_simplex dynamics_equal dynamics_unconstrained_ls dynamics_no_class dynamics_local_simplex dynamics_best_head dynamics_earliest_split dynamics_all_rolling_splits dynamics_no_rolling_validation head_last head_mean3 head_mean6 head_season6 head_season12 head_season24 head_drift010 head_drift025 head_holt_damped head_ar4 head_ar8 head_ar16 head_fourier3 head_global_ar2 head_global_ar4 head_global_ar8 head_global_ar12 head_global_ar16 head_class_ar2 head_class_ar4 head_class_ar8 head_class_ar12 head_class_ar16 leave_one_last leave_one_mean3 leave_one_mean6 leave_one_season6 leave_one_season12 leave_one_season24 leave_one_drift010 leave_one_drift025 leave_one_holt_damped leave_one_ar4 leave_one_ar8 leave_one_ar16 leave_one_fourier3 leave_one_global_ar2 leave_one_global_ar4 leave_one_global_ar8 leave_one_global_ar12 leave_one_global_ar16 leave_one_class_ar2 leave_one_class_ar4 leave_one_class_ar8 leave_one_class_ar12 leave_one_class_ar16"
read -r -a ABLATION_MODELS <<< "${ADAWARP_DYNAMICS_ABLATIONS:-$ABLATION_MODELS_DEFAULT}"

python benchmark_adawarp_protocol_baselines.py \
  --datasets "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --prefix-fractions "${PREFIXES[@]}" \
  --models "${ABLATION_MODELS[@]}" \
  --output-root "$OUTPUT_ROOT"

if [[ "${ADAWARP_RUN_STRUCTURAL_ABLATIONS:-1}" == "1" ]]; then
  DEVICE="${ADAWARP_DEVICE:-auto}"
  EPOCHS="${ADAWARP_EPOCHS:-50}"
  STEPS_PER_EPOCH="${ADAWARP_STEPS_PER_EPOCH:-4}"
  mkdir -p "$OUTPUT_ROOT/logs/structural_ablations"
  for prefix in "${PREFIXES[@]}"; do
    tag="${prefix/./p}"
    for dataset in "${DATASETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for variant in full_blend_with_gp_candidates no_warp no_residual no_warp_no_residual no_affine no_generative gp_only gp_residual_ar4 gp_residual_ar8 gp_residual_ar16; do
          extra=()
          case "$variant" in
            full_blend_with_gp_candidates) extra=(--forecast-include-gp-candidates) ;;
            no_warp) extra=(--no-use-sample-warp) ;;
            no_residual) extra=(--no-use-adaptive-residual) ;;
            no_warp_no_residual) extra=(--no-use-sample-warp --no-use-adaptive-residual) ;;
            no_affine) extra=(--no-use-affine-alignment) ;;
            no_generative) extra=(--generative-weight 0.0) ;;
            gp_only) extra=(--forecast-calibration head --forecast-head gp) ;;
            gp_residual_ar4) extra=(--forecast-calibration head --forecast-head gp_residual_ar4) ;;
            gp_residual_ar8) extra=(--forecast-calibration head --forecast-head gp_residual_ar8) ;;
            gp_residual_ar16) extra=(--forecast-calibration head --forecast-head gp_residual_ar16) ;;
          esac
          python benchmark_awp_forecasting.py \
            --dataset "$dataset" \
            --seed "$seed" \
            --observed-fraction "$prefix" \
            --output-dir "$OUTPUT_ROOT/structural_ablations/$variant/prefix_$tag" \
            --device "$DEVICE" \
            --epochs "$EPOCHS" \
            --steps-per-epoch "$STEPS_PER_EPOCH" \
            --save-raw-predictions \
            "${extra[@]}" \
            > "$OUTPUT_ROOT/logs/structural_ablations/${variant}_${dataset}_prefix${tag}_seed${seed}.stdout.log" \
            2> "$OUTPUT_ROOT/logs/structural_ablations/${variant}_${dataset}_prefix${tag}_seed${seed}.stderr.log"
        done
      done
    done
  done
fi

if [[ "${ADAWARP_RUN_HELDOUT:-1}" == "1" ]]; then
  python benchmark_adawarp_heldout_forecasting.py \
    --datasets ECGFiveDays FreezerSmallTrain HouseTwenty InsectEPGRegularTrain ItalyPowerDemand Lightning7 MoteStrain PowerCons SonyAIBORobotSurface2 \
    --seeds "${SEEDS[@]}" \
    --prefix-fractions "${PREFIXES[@]}" \
    --device "${ADAWARP_DEVICE:-auto}" \
    --epochs "${ADAWARP_EPOCHS:-50}" \
    --steps-per-epoch "${ADAWARP_STEPS_PER_EPOCH:-4}" \
    --output-root "$OUTPUT_ROOT"
fi

echo "[ablation_suite] complete"
