#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

python - <<'PY'
import importlib.util
import sys

missing = []
for name in ["jax", "jaxlib", "scipy"]:
    if importlib.util.find_spec(name) is None:
        missing.append(name)

if missing:
    print("[motion-code-env] missing:", ", ".join(missing))
    print("[motion-code-env] original Motion Code will not run until these are available.")
    print("[motion-code-env] preferred: load a TACC-provided JAX module if one exists.")
    print("[motion-code-env] fallback after confirming NumPy compatibility:")
    print("  python -m pip install --no-deps jax jaxlib")
    sys.exit(1)

import jax
import jax.numpy as jnp
import scipy
import motion_code

print("[motion-code-env] ok")
print("[motion-code-env] jax", jax.__version__)
print("[motion-code-env] scipy", scipy.__version__)
print("[motion-code-env] x64", jax.config.jax_enable_x64)
print("[motion-code-env] devices", jax.devices())
PY
