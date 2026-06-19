# AdaWarp Forecasting Leakage Audit

Strict rule for every evaluated trajectory:

> Only its observed prefix may influence any fitted object used to forecast its suffix, unless the protocol is explicitly transductive and labeled as such.

This audit is an experiment artifact. Do not report a metric in the final paper unless the corresponding runner also writes audit metadata and raw predictions.

| Component | Data visible while fitting/selecting | Evaluation suffix visible? | Split used | Notes |
|---|---|---:|---|---|
| Raw data cleaning | Raw trajectory files and labels needed to construct train/test assets | No target suffix values used for any fitted forecasting object | Dataset release split, then prefix/suffix split | Cleaning must not drop or alter examples based on future-suffix error. Any manual exclusion must be documented separately. |
| Train/validation/test split | Official split for classification; Motion Code-style forecasting collection prefixes; LTSF chronological train/val/test split | No | Official split or chronological split | Held-out forecasting trains on UCR train split and evaluates UCR test prefixes. |
| Normalization | Motion Code-style forecasting uses observed prefixes only; held-out and LTSF use training split only | No | Prefix-only or training-split-only | The evaluated trajectory's suffix must never enter median/IQR/mean/std statistics. |
| Class labels | Oracle class mode may read the evaluated prefix label by protocol; predicted and soft modes use prefix-only classifier output | No suffix | Explicit class-mode setting | Oracle class results must be labeled as class-conditioned. |
| Prefix extraction | Times/values up to the requested prefix fraction | No | Per-trajectory prefix/suffix split | `observed_fraction` creates the boundary before any fitting. |
| Template grids | Training/support prefixes only | No | Prefix only | Template interpolation may use prefix times/values and class labels, never suffix values. |
| GP inducing/prototype construction | Support/training prefixes, labels, model hyperparameters, ordered landmark parameters | No | Prefix only | Motion Code-style prototypes may be transductive over collection prefixes; held-out prototypes train only on training split prefixes. |
| Residual/warp adapter fitting | Training/support prefixes and validation prefixes only | No | Train/validation prefix split | Adapter parameters are learned before test suffix scoring. |
| Dynamics-head fitting | Rolling origins inside observed prefixes; optional pooled dynamics across observed prefixes | No | Internal prefix-to-prefix-tail tasks | `benchmark_adawarp_protocol_baselines.py` and `calibrate_forecast_blend` create targets only from prefix tails. |
| Blend-weight fitting | Candidate-head predictions and targets from rolling origins inside prefixes | No | Internal rolling origins | Simplex, equal, unconstrained, best-head, earliest-split, and leave-one-head variants are fit from the same prefix-only tasks. |
| GP forecast-center ablations | Prefix times/values and fixed kernel choices; class-mean residual uses class prefixes only | No | Prefix only | These are internal GP design ablations, not external paper baselines. |
| Conformal/uncertainty scaling | Prefix-internal residuals and GP variances | No | Internal prefix split | Scaling is diagnostic only; it must not use final suffix residuals. |
| Neural baseline preprocessing | Sliding windows whose inputs and targets are fully contained in observed prefixes | No | Prefix-only windows | `benchmark_tslibrary_neural_forecasting.py` trains one model per known class from prefix windows. |
| Neural baseline checkpoint selection | Training-window loss and validation windows from training/prefix data only | No | Prefix-only or LTSF validation split | Test suffix/window metrics are read after checkpoint selection. |
| LTSF windows | Standard train/validation/test chronological windows | No test targets during fitting | Standard LTSF split | Normalization must be fitted on the training partition only. |
| Motion Code matched rerun | Same split and prefix visibility as AdaWarp matched protocol | No | Prefix only | Requires JAX. If skipped, Motion Code cannot appear as a matched baseline. |
| TSLibrary classification baselines | Official classification training split; seed-specific training run | Test labels only for final scoring | Official classification split | Informer, Autoformer, FEDformer, ETSformer, LightTS, PatchTST, Crossformer, DLinear, TimesNet, iTransformer, and Mamba must be rerun by `scripts/tacc/run_tslibrary_classification.py`; old cached folders are not final evidence. |
| Modern TSC classifiers | Official classification training split; optional validation internal to classifier | Test labels only for final scoring | Official classification split | MiniROCKET, MultiROCKET, Hydra, and InceptionTime must come from real aeon/sktime implementations or be marked missing. |
| Checkpoint selection | Training and validation prefixes/splits only | No | Validation split | No model may choose epoch/head using final test suffix metrics. |
| Test metric computation | Frozen predictions and hidden suffix/labels | Yes, for scoring only | Final evaluation | Metrics scripts may read suffixes only after all fitted objects and checkpoints are fixed. |
| Aggregation/statistical tests | Completed metric files and raw prediction artifacts | Already-scored artifacts only | Post hoc | Aggregation must not alter predictions. |

Required audit artifacts after a full TACC run:

- `results/audit/environment_setup.json`
- `results/audit/environment_protocol_baselines.json`
- `results/audit/environment_motion_code_classification.json`
- `results/audit/environment_tslibrary_classification.json`
- `results/audit/environment_modern_tsc_classification.json`
- `results/audit/environment_heldout_forecasting.json`
- `results/audit/environment_ltsf_adawarp.json`
- `results/audit/environment_aggregate.json`
- `results/audit/result_manifest.csv`

If any baseline is unavailable because an official/compatible implementation is missing, record it in `results/audit/missing_required_classification_baselines.csv` or the corresponding missing-baseline audit file. Do not replace it with a homemade proxy under the same name.
