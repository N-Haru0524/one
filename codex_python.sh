#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo ".venv Python is missing." >&2
    echo "Create it with: uv venv --python 3.12 .venv" >&2
    echo "Then install deps with: uv pip install --python .venv/bin/python -e ." >&2
    exit 1
fi

export PYTHONNOUSERSITE=1

exec "${VENV_PYTHON}" "$@"
