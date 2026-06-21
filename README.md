# AdaWarp: Adaptive Warped Prototype Models for Time-Series Classification and Forecasting

This repository contains the AdaWarp experimental codebase built on top of the original Motion Code repository. The current project studies AdaWarp as a family of models for time-series classification and forecasting:

- **AdaWarp-SGP** for short, mostly univariate, class-conditioned forecasting and classification retention. It uses sparse Gaussian-process class prototypes, bounded adaptive warps, and prefix-validated continuation dynamics.
- **AdaWarp-MVPF** for long-term multivariate forecasting. It uses multiscale variate-patch fields with local cross-variate mixing, adaptive patch shifts, trend/residual structure, and reconstruction regularization.

The original Motion Code implementation is still present and is used as a matched baseline, but the main code paths in this repository are the AdaWarp experiment runners and TACC scripts.

## Current TACC Pointer

The matched reruns used for the current internal paper draft were run on **TACC Vista**. The working TACC path used during development was:

```bash
/work/11617/sujato_ts/vista/motion_code-master
```

Important result folders on TACC are expected under that path, especially:

```bash
results/main
results/classification
results/motion code
results/ltsf_5
```

These result folders are intentionally not tracked by Git because they are large and environment-specific. For independent repeatability, use the GitHub commit plus the TACC pointer above, then rerun the scripts below or inspect the saved TACC result folders.

## What Is Included

### Core AdaWarp code

```text
awp_motion_code.py                  # AdaWarp-SGP / short-protocol model
awp_forecasting_utils.py            # prefix-to-suffix continuation heads and forecast utilities
awp_datasets.py                     # dataset loading helpers for short protocols
adawarp_neural_baselines.py         # repo-native neural baselines including VPNet/AdaWarp-VPF
adawarp_mvpf.py                     # AdaWarp-MVPF long-term forecasting model
adawarp_experiment_utils.py         # shared experiment utilities and result writing
```

### Main experiment runners

```text
benchmark_adawarp_protocol_baselines.py   # short class-conditioned forecasting baselines and ablations
benchmark_awp_motion_code.py              # AdaWarp-SGP classification/forecasting entry point
benchmark_motion_code_forecasting.py      # matched original Motion Code forecasting rerun
benchmark_motion_code_classification.py   # matched original Motion Code classification rerun
benchmark_modern_tsc_classifiers.py       # MiniROCKET, MultiROCKET, Hydra matched classification reruns
benchmark_tslibrary_neural_forecasting.py # TSLibrary neural forecasting bridge
benchmark_custom_neural_ltsf.py           # custom neural LTSF baselines
benchmark_adawarp_mvpf_ltsf.py            # AdaWarp-MVPF LTSF runs
benchmark_adawarp_mvpf_ablation.py        # AdaWarp-MVPF LTSF ablations
aggregate_adawarp_experiments.py          # aggregate matched experiment results
aggregate_adawarp_mvpf_ablations.py       # aggregate MVPF ablation results
```

### Configs

```text
configs/motioncode_protocol/default.json       # short prefix-to-suffix protocol
configs/classification_retention/default.json  # classification retention protocol
configs/ltsf_main5/default.json                # ETTh1/ETTh2/Weather/Electricity/Traffic LTSF protocol
configs/ablations/default.json                 # short-protocol ablation plan
configs/baselines/default.json                 # baseline registry
```

### TACC scripts

```text
scripts/tacc/setup_env.sh                    # conservative TACC virtualenv setup
scripts/tacc/run_motioncode_protocol.sh      # short AdaWarp-SGP + protocol baselines
scripts/tacc/run_classification_retention.sh # AdaWarp/Motion Code/TSLibrary classification reruns
scripts/tacc/run_ablation_suite.sh           # short-protocol ablations
scripts/tacc/run_ltsf_single_model.sh        # one LTSF model per job
scripts/tacc/submit_ltsf_by_model.sh         # submit LTSF baseline jobs by model
scripts/tacc/run_adawarp_mvpf_ltsf.sh        # AdaWarp-MVPF LTSF for one dataset/model job
scripts/tacc/submit_adawarp_mvpf_by_dataset.sh
scripts/tacc/run_adawarp_mvpf_ablation.sh
scripts/tacc/submit_adawarp_mvpf_ablations.sh
scripts/tacc/aggregate_all.sh
scripts/tacc/aggregate_ltsf_by_model.py
```

## Data Layout

Datasets are not committed. The expected local/TACC layout is:

```text
data/                         # short protocol datasets
TSLibrary/dataset/             # TSLibrary classification/LTSF datasets
TSLibrary/dataset/ETT-small/    # ETTh1, ETTh2, ETTm1, ETTm2 when used
TSLibrary/dataset/weather/      # weather.csv
TSLibrary/dataset/LR_Datasets/  # long-range CSV copies used by some runners
```

The following paths are ignored by Git:

```text
data/*
TSLibrary/dataset/*
TSLibrary/checkpoints/*
TSLibrary/results/*
results/*
out/*
logs/*
paper/
```

## Local Setup

For quick CPU smoke checks on a local machine:

```bash
python -m venv .venv-adawarp
.venv-adawarp\Scripts\activate      # Windows cmd/powershell style may differ
pip install -r requirements.txt
pip install -r requirements-tacc-extra.txt
```

The full neural runs are intended for TACC/GPU. Local CPU runs are useful only for import checks, small smoke tests, and result aggregation.

## TACC Setup Notes

The TACC environment should not blindly overwrite cluster-provided accelerator packages. In particular, avoid forcing new Torch, CUDA, NumPy, or JAX versions unless the module/container choice has been checked.

A typical Vista setup was:

```bash
cd $WORK/motion_code-master
module load gcc/14.2.0 cuda/12.6 python3/3.11.8
python3 -m venv --system-site-packages .venv-adawarp
source .venv-adawarp/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tacc-extra.txt
```

Then verify the accelerator stack from an allocated GPU job, not only from a login node:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda build', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Reproducing the Main Experiments

Set an output root before running jobs:

```bash
export ADAWARP_OUTPUT_ROOT=$WORK/motion_code-master/results/<run_name>
export ADAWARP_DEVICE=cuda
```

### 1. Short class-conditioned forecasting

```bash
export ADAWARP_EPOCHS=50
export ADAWARP_STEPS_PER_EPOCH=4
export ADAWARP_SEEDS="42 43 44 45 46"
export ADAWARP_PREFIX_FRACTIONS="0.8 0.6"
bash scripts/tacc/run_motioncode_protocol.sh
bash scripts/tacc/aggregate_all.sh
```

Matched original Motion Code reruns require JAX. If JAX is unavailable or unstable, document the omission and rerun once the environment is fixed.

### 2. Classification retention

```bash
export ADAWARP_SEEDS="42"
bash scripts/tacc/run_classification_retention.sh
bash scripts/tacc/aggregate_all.sh
```

Modern scalable TSC methods such as MiniROCKET, MultiROCKET, and Hydra require `aeon` or compatible `sktime` implementations. InceptionTime usually requires TensorFlow/Keras. If a required package is missing, the missing row should be audited rather than silently replaced.

### 3. Long-term forecasting baselines

The LTSF panel uses ETTh1, ETTh2, Weather, Electricity, and Traffic at horizons 96, 192, 336, and 720. To run one model per job:

```bash
export ADAWARP_LTSF_MODELS="DLinear PatchTST TimesNet iTransformer TimeMixer FEDformer VPNet"
bash scripts/tacc/submit_ltsf_by_model.sh
```

### 4. AdaWarp-MVPF long-term forecasting

```bash
bash scripts/tacc/submit_adawarp_mvpf_by_dataset.sh
```

Aggregate after jobs complete:

```bash
bash scripts/tacc/aggregate_all.sh
python scripts/tacc/aggregate_ltsf_by_model.py --root results/ltsf_5
```

### 5. AdaWarp-MVPF ablations

```bash
bash scripts/tacc/submit_adawarp_mvpf_ablations.sh
```

Then aggregate:

```bash
python aggregate_adawarp_mvpf_ablations.py --root results/ltsf_5/AdaWarp-MVPF/ablations
```

## Current Evidence Summary

The current matched result set supports the following conservative claims:

- **Short class-conditioned forecasting:** AdaWarp-SGP has the best mean rank across the matched short-protocol comparator panel and improves over the matched Motion Code rerun on 9/10 datasets.
- **Classification:** AdaWarp-SGP retains Motion Code-style classification performance but is not presented as a new classification SOTA result.
- **Long-term forecasting:** AdaWarp-MVPF is competitive against matched neural baselines on ETTh1, ETTh2, Weather, Electricity, and Traffic. It has strong rank/win behavior, while PatchTST remains strongest by some raw mean-error summaries.
- **Ablations:** Dynamics variants are ablations only. They are used to test whether prefix-validated continuation dynamics explain the short forecasting gain and should not be treated as main external baselines.

All final claims should come from matched reruns under `results/`, not from stale `out/` files, released-paper values, or old TSLibrary result folders.

## Result Provenance

Expected matched result locations:

```text
results/main/metrics/                         # short protocol AdaWarp and baselines
results/motion code/metrics/                  # matched original Motion Code reruns
results/classification/metrics/               # AdaWarp + TSLibrary classification reruns
results/classification/rocket_hydra/metrics/  # MiniROCKET/MultiROCKET/Hydra reruns
results/ltsf_5/                               # LTSF baselines and AdaWarp-MVPF
results/ablations/                            # short-protocol ablations
```

Because `results/` is ignored, a reviewer or another student should either use the TACC pointer above or rerun the jobs using the scripts in `scripts/tacc/`.

## Notes for Independent Checkers

1. Confirm the Git commit hash.
2. Confirm the TACC working directory.
3. Confirm the loaded modules and Python environment.
4. Confirm that `results/` paths are generated by matched reruns.
5. Confirm that short-protocol neural windows do not cross into final suffixes.
6. Confirm that dynamics variants are reported only as ablations.
7. Confirm that paper-facing tables are generated from matched CSVs, not stale artifacts.

## Legacy Motion Code Files

The original Motion Code files remain available for baseline reruns and backward compatibility:

```text
motion_code.py
motion_code_utils.py
sparse_gp.py
benchmarks.py
visualize.py
```

For new AdaWarp experiments, prefer the AdaWarp scripts and configs listed above.
