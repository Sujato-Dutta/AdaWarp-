@echo off
setlocal EnableExtensions

set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=short"
if not "%~1"=="" shift

if /I "%PROFILE%"=="smoke" goto profile_smoke
if /I "%PROFILE%"=="short" goto profile_short
if /I "%PROFILE%"=="full" goto profile_full

echo Usage: scripts\run_awp_forecasting_cpu.cmd [smoke^|short^|full] [dataset ...]
exit /b 2

:profile_smoke
set "PROFILE_ARGS=--epochs 2 --steps-per-epoch 1 --no-refit --max-support-per-class 8 --max-query-per-class 4 --eval-batch-size 32"
goto profile_ready

:profile_short
set "PROFILE_ARGS=--epochs 30 --steps-per-epoch 1 --no-refit --max-support-per-class 16 --max-query-per-class 8 --eval-batch-size 64"
goto profile_ready

:profile_full
set "PROFILE_ARGS="

:profile_ready
if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=out\awp_forecasting_cpu"
if "%SEEDS%"=="" set "SEEDS=42"

set "DATASETS="
:collect_datasets
if "%~1"=="" goto datasets_ready
set "DATASETS=%DATASETS% %~1"
shift
goto collect_datasets

:datasets_ready
if defined DATASETS goto run
set "DATASETS=PronunciationAudio ECGFiveDays FreezerSmallTrain HouseTwenty InsectEPGRegularTrain ItalyPowerDemand Lightning7 MoteStrain PowerCons SonyAIBORobotSurface2"

:run
for %%D in (%DATASETS%) do (
    for %%S in (%SEEDS%) do (
        "%PYTHON_BIN%" -u benchmark_awp_forecasting.py --dataset "%%D" --seed "%%S" --device cpu --output-dir "%OUTPUT_DIR%" %PROFILE_ARGS%
        if errorlevel 1 exit /b 1
    )
)

"%PYTHON_BIN%" -u aggregate_awp_forecasts.py --output-dir "%OUTPUT_DIR%"
if errorlevel 1 exit /b 1

endlocal
