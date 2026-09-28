#!/usr/bin/env bash
# Upload the app to a Hugging Face Space: deploy/deploy_space.sh <user>/<space>
# Needs `hf auth login` (or an HF_TOKEN env var) with write access.
set -euo pipefail

SPACE="${1:?usage: deploy/deploy_space.sh <user>/<space>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$ROOT"
cp -r app.py riverguard models examples requirements.txt packages.txt LICENSE "$STAGE"/
cp deploy/space_README.md "$STAGE"/README.md
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +

hf repo create "$SPACE" --repo-type space --space_sdk gradio --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --delete "*" \
  --commit-message "Deploy $(git rev-parse --short HEAD 2>/dev/null || echo local)"
echo "Deployed: https://huggingface.co/spaces/$SPACE"
