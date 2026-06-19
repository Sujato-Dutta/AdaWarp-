#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "${ADAWARP_VENV:-.venv-adawarp}" ]]; then
  source "${ADAWARP_VENV:-.venv-adawarp}/bin/activate"
fi

OUTPUT_ROOT="${ADAWARP_OUTPUT_ROOT:-results}"
python measure_adawarp_efficiency.py --output-root "$OUTPUT_ROOT"
python aggregate_adawarp_experiments.py --output-root "$OUTPUT_ROOT"
echo "[aggregate_all] complete"
