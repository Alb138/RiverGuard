# RiverGuard

YOLO11s plastic-waste detector with Telegram alerts (`riverguard/`), a Gradio web demo (`app.py`, deployed as a
Hugging Face Space), a local monitor CLI (`run_camera.py`) and a retraining pipeline (`training/`).

## Commands

```bash
.venv/Scripts/python -m pytest -q               # tests (Windows venv, Python 3.10)
.venv/Scripts/python app.py                     # web app on http://127.0.0.1:7860
.venv/Scripts/python run_camera.py --source examples/canal-litter.jpg --no-show
.venv/Scripts/python training/evaluate.py --tags datasets/tier-b/tags.csv models/best.pt --tiled
training/kaggle.sh kernels list --mine          # Kaggle CLI (needs Python 3.11+, lives in .venv-tools)
```

## Rules

- **Never commit `_original/`.** It holds the course deliverables: a report with teammates' phone numbers, IDs and
  birthdates, a notebook with an old Roboflow key, and a 500 MB video. It's gitignored; keep it that way.
- Secrets live only in `.env` (`TELEGRAM_*`, `ROBOFLOW_API_KEY`, `KAGGLE_API_TOKEN`). Notepad tends to save it as
  `.env.txt`; `.env.*` is ignored, so check the real filename if a key "isn't set".
- The public Space must never get Telegram secrets: visitors only see an alert *preview*.
- Credit teammates as "a 3-person group project", without names.
- `models/best.pt` is v1. Replace it only when a new model beats it on all three test sets in `training/README.md`.
- Images added to `examples/`, `docs/` or training data need an entry in the matching `ATTRIBUTION.md`, from a
  license that allows reuse (no NC/ND).

## Gotchas

- Files have CRLF line endings. Python heredoc edits that insert `"\n"` inside string literals break the file;
  use the Edit tool for multi-line changes.
- Kaggle output downloads through the CLI often fail with `IncompleteRead`. List the file URLs with the SDK and
  fetch them with `curl -C -`, stripping `\r` from any URL list written by Python on Windows.
- Wikimedia Commons rate-limits hard (HTTP 429 with `Retry-After: 600`). Use a descriptive User-Agent and at least
  3 s between requests. Original files are on `upload.wikimedia.org`, not `thumb.wikimedia.org`.
- `datasets/` and `runs/` are gitignored and large. Rebuild with `training/build_dataset.py`, which reads
  `training/sources.yaml`.
