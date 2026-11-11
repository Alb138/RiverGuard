# RiverGuard

YOLO11s plastic-waste detector with Telegram alerts (`riverguard/`), a local Gradio web app (`app.py`), the public
Streamlit demo (`demo/streamlit_app.py`, on Streamlit Community Cloud), a local monitor CLI (`run_camera.py`) and a
retraining pipeline (`training/`).

## Commands

```bash
.venv/Scripts/python -m pytest -q               # tests (Windows venv, Python 3.10)
.venv/Scripts/python app.py                     # web app on http://127.0.0.1:7860
.venv-demo/Scripts/python -m streamlit run demo/streamlit_app.py   # public demo (Python 3.13 venv, demo/requirements.txt)
.venv-demo/Scripts/python -m pytest -q tests/test_demo.py          # demo smoke test (skipped in the 3.10 venv)
.venv/Scripts/python run_camera.py --source examples/river-bottles.jpg --no-show
.venv/Scripts/python training/evaluate.py --tags datasets/tier-b/tags.csv models/best.pt --tiled
training/kaggle.sh kernels list --mine          # Kaggle CLI (needs Python 3.11+, lives in .venv-tools)
```

## Rules

- **Never commit `_original/`.** It holds the course deliverables: a report with teammates' phone numbers, IDs and
  birthdates, a notebook with an old Roboflow key, and a 500 MB video. It's gitignored; keep it that way.
- Secrets live only in `.env` (`TELEGRAM_*`, `ROBOFLOW_API_KEY`, `KAGGLE_API_TOKEN`). Notepad tends to save it as
  `.env.txt`; `.env.*` is ignored, so check the real filename if a key "isn't set".
- The public demo must never send Telegram alerts: visitors only see an alert *preview*. `demo/streamlit_app.py` has
  no alerter at all; keep it that way and never add Telegram secrets to Streamlit Cloud.
- Credit teammates as "a 3-person group project", without names.
- `models/best.pt` is v1 (item detector); `models/density.pt` is the v3b dense-trash classifier (threshold 0.5).
  Replace either only when a new model beats it on the test sets in `training/README.md`, and when replacing
  `density.pt`, copy that build's `datasets/density-*/ATTRIBUTION.md` to `training/DENSITY_ATTRIBUTION.md`.
- `models/face_detection_yunet_2023mar.onnx` (YuNet, MIT, licence file next to it) drops trash boxes that cover a
  face: the v1 detector never saw people and boxed faces as trash.
- v4 (people negatives + `other_trash`, `runs/kaggle-v4/v4.pt`, results in `training/README.md`) was shipped on
  2026-11-11 and rolled back to v1 the same day at the user's request. Don't re-ship it without asking.
- The live demo runs `models/*.onnx`, not the `.pt` files. Re-export after replacing a `.pt` (commands in the README's
  Deployment section); `test_onnx_exports_match_pytorch_models` fails until you do.
- Keep demo examples (`examples/`) free of trash-carpeted scenes, and check each one doesn't trigger a false alarm.
- Images added to `examples/`, `docs/` or training data need an entry in the matching `ATTRIBUTION.md`, from a
  license that allows reuse (no NC/ND).

## Gotchas

- Files have CRLF line endings. Python heredoc edits that insert `"\n"` inside string literals break the file;
  use the Edit tool for multi-line changes.
- Kaggle output downloads through the CLI often fail with `IncompleteRead`. List the file URLs with the SDK and
  fetch them with `curl -C -`, stripping `\r` from any URL list written by Python on Windows.
- Wikimedia Commons rate-limits hard (HTTP 429 with `Retry-After: 600`). Use a descriptive User-Agent and at least
  3 s between requests. Original files are on `upload.wikimedia.org`, not `thumb.wikimedia.org`.
- Streamlit Community Cloud installs `demo/requirements.txt` (it sits next to the entrypoint), not the root one. Keep
  PyTorch out of it. Streamlit 1.66+ needs Python 3.11+, so the demo test runs in its own CI job.
- Hugging Face Gradio/Docker Spaces need PRO and Render's free plan needs a card; that's why the demo is on Streamlit.
- Keep video decode/encode at one thread (`riverguard/web.py`): x264 and OpenCV default to one thread per host core,
  which pushed peak memory past 512 MB. When measuring memory on Windows, the venv `python.exe` is a launcher (track
  its child), and idle working sets get trimmed: use `peak_wset` or private bytes, not current RSS.
- `datasets/` and `runs/` are gitignored and large. Rebuild with `training/build_dataset.py`, which reads
  `training/sources.yaml`.
