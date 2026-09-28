#!/usr/bin/env bash
# Run the Kaggle CLI (Python 3.13 tools venv) with KAGGLE_API_TOKEN taken from .env.
#   training/kaggle.sh kernels list --mine
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$ROOT/.env" ]; then
  KAGGLE_API_TOKEN="$( (grep -E '^[[:space:]]*KAGGLE_API_TOKEN[[:space:]]*=' "$ROOT/.env" || true) | head -1 | cut -d= -f2- | tr -d ' \r"'"'")"
  export KAGGLE_API_TOKEN
fi
[ -n "${KAGGLE_API_TOKEN:-}" ] || { echo "Set KAGGLE_API_TOKEN in .env (kaggle.com > Settings > API)" >&2; exit 1; }
for exe in "$ROOT/.venv-tools/Scripts/kaggle.exe" "$ROOT/.venv-tools/bin/kaggle"; do
  [ -x "$exe" ] && exec "$exe" "$@"
done
echo "Kaggle CLI not found: py -3.13 -m venv .venv-tools && .venv-tools/Scripts/pip install kaggle" >&2
exit 1
