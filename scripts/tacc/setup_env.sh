#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ADAWARP_VENV:-.venv-adawarp}"

echo "[setup] repo=$REPO_ROOT"
echo "[setup] python=$($PYTHON_BIN --version)"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python - <<'PY'
import importlib.util
import sys
print("[setup] venv python", sys.version.replace("\n", " "))
for name in ["torch", "numpy", "pandas", "scipy", "sklearn", "sktime", "aeon", "tensorflow", "jax", "mamba_ssm"]:
    spec = importlib.util.find_spec(name)
    print(f"[setup] {name}: {'present' if spec else 'missing'}")
if importlib.util.find_spec("jax") is None:
    print("[setup] WARNING: original Motion Code matched reruns require jax. Load a TACC module or install it only after confirming it will not replace the cluster NumPy stack.")
if importlib.util.find_spec("aeon") is None:
    print("[setup] WARNING: MiniROCKET/MultiROCKET/Hydra matched classification reruns need aeon or compatible sktime classifiers.")
if importlib.util.find_spec("tensorflow") is None:
    print("[setup] WARNING: InceptionTime matched classification rerun usually needs TensorFlow/Keras.")
if importlib.util.find_spec("mamba_ssm") is None:
    print("[setup] WARNING: TSLibrary Mamba classification rerun requires mamba_ssm. Use ADAWARP_TSLIB_CLASSIFICATION_MODELS to omit it only if the omission is audited.")
PY

if [[ "${ADAWARP_INSTALL_EXTRAS:-1}" == "1" ]]; then
  echo "[setup] installing extra non-torch requirements"
  python -m pip install -r requirements-tacc-extra.txt
else
  echo "[setup] ADAWARP_INSTALL_EXTRAS=0, skipping pip install"
fi

mkdir -p results/audit
python - <<'PY'
from adawarp_experiment_utils import save_environment
save_environment("results/audit/environment_setup.json")
PY

echo "[setup] complete"
