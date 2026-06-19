@echo off
setlocal EnableExtensions

if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=out\tslibrary_neural_forecasting_10datasets_seed42"
if "%SEEDS%"=="" set "SEEDS=42"

"%PYTHON_BIN%" -B -u benchmark_tslibrary_neural_forecasting.py --device cpu --output-dir "%OUTPUT_DIR%" --seeds %SEEDS% %*
if errorlevel 1 exit /b 1

endlocal
