#!/usr/bin/env bash
# Always uses .venv — avoids "ModuleNotFoundError" when `streamlit` on PATH is not the venv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "No .venv found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" -m streamlit run "$ROOT/dashboard/app.py" "$@"
