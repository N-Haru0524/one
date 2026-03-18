#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
USER_ID="$(id -u)"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo ".venv is missing. Create it first with: uv venv --python 3.12 .venv" >&2
    exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is not set. Start from a GUI/X11 session or export DISPLAY first." >&2
    exit 1
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
    for candidate in \
        "${HOME}/.Xauthority" \
        "/run/user/${USER_ID}/gdm/Xauthority" \
        "/run/user/${USER_ID}/.mutter-Xwaylandauth."* \
        "/tmp/xauth-"* \
        ; do
        if [[ -f "${candidate}" ]]; then
            export XAUTHORITY="${candidate}"
            break
        fi
    done
fi

source "${VENV_DIR}/bin/activate"

unset PYGLET_HEADLESS
unset PYGLET_SHADOW_WINDOW
export PYTHONNOUSERSITE=1

if command -v xset >/dev/null 2>&1; then
    if ! xset q >/dev/null 2>&1; then
        echo "Cannot open X11 display '${DISPLAY}'." >&2
        echo "Resolved XAUTHORITY='${XAUTHORITY:-<unset>}'." >&2
        echo "Run this from the active GUI session, or fix DISPLAY/XAUTHORITY/X11 permissions." >&2
        exit 1
    fi
fi

if [[ $# -eq 0 ]]; then
    exec "${SHELL:-/bin/bash}" -i
fi

exec "$@"
