#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo ".venv is missing. Create it first with: python -m venv .venv" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"

export PYGLET_HEADLESS=1
export PYGLET_SHADOW_WINDOW=0

if [[ $# -eq 0 ]]; then
    exec "${SHELL:-/bin/bash}" -i
fi

exec "$@"
