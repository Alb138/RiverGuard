# 🌊 RiverGuard

**Real-time plastic waste detection for rivers, with Telegram early-warning alerts.**

RiverGuard watches a camera feed of a river or floodgate. A fine-tuned YOLO11 detector counts floating plastic and
other trash, and a second model checks whether the water is carpeted with trash too dense to count. When either says the river is over
the limit, it sends a Telegram alert with an evidence photo so cleanup crews can act *before* the build-up blocks a
floodgate.

**[▶ Try the live demo](https://riverguard.streamlit.app/)** (free hosting: if it has been idle, click the wake-up
button and give it about a minute)

<p align="center">
  <img src="docs/demo-detection.jpg" width="58%" alt="RiverGuard detecting plastic bottles and bags in a river">
  <img src="docs/telegram-alert.png" width="40%" alt="Telegram alert sent by the RiverGuard bot">
</p>

## Why

Urban rivers in Indonesia carry large amounts of macroplastic. When it piles up at floodgates it blocks the flow and
makes floods worse. Many floodgates already have CCTV, but it's used for passive recording, and operators have to
watch the screens to notice a build-up. RiverGuard turns that existing camera into an active sensor that notifies
the crew when the waste count passes a safe limit.

## How it works

```mermaid
flowchart LR
    A[Camera / video / image] --> B[YOLO11s detector<br/>4 plastic classes + other trash]
    A --> H[YOLO11s-cls<br/>dense-trash check]
    B --> C{count &gt; threshold<br/>or dense?}
    H --> C
    C -- no --> D[SAFE banner]
    C -- yes --> E[DANGER banner]
    E --> F{cooldown elapsed?}
    F -- yes --> G[Telegram sendPhoto<br/>annotated frame + caption]
```

1. Each frame is run through the detector (confidence ≥ 0.35 by default). Large photos are also scanned as
   overlapping 640 px tiles, so small, distant items keep enough pixels to be recognised. A box cut off by a
   tile's edge is dropped, because the full frame or the neighbouring tile sees that item whole, and duplicate
   boxes where tiles overlap are merged. Boxes that cover a face are dropped too (see Limitations).
2. The whole frame also goes through a small classifier that answers one question: *is the water carpeted with
   trash?* That catches scenes where items are packed too tightly for the detector to count.
3. Detections are counted per class and drawn on the frame with a SAFE / DANGER status banner.
4. If the count is **above** the threshold (default 5), **or** the dense-trash score is at least 50%, the annotated
   frame is sent to a Telegram chat, at most once per cooldown window (default 20 s) so a single build-up doesn't
   flood the chat.

## Models

| | Item detector ([`models/best.pt`](models/best.pt), 19 MB) | Dense-trash check ([`models/density.pt`](models/density.pt), 11 MB) |
|---|---|---|
| Base model | YOLO11s (COCO-pretrained), fine-tuned twice (v1, then v4) | YOLO11s-cls (ImageNet-pretrained) |
| Output | Boxes: `other_plastic`, `plastic_bag`, `plastic_bottle`, `plastic_package`, `other_trash` (cans, metal) | P(water carpeted with trash) |
| Data | Four public river/harbour/lake trash datasets from Roboflow Universe (~9,400 images) plus 511 openly licensed photos of people with nothing to box | ~1,100 openly licensed photos from 160+ photographers (Openverse, Wikimedia Commons, Roboflow Universe), tagged dense / not dense |
| Training | v1: 64 epochs on Colab; v4: 73 more epochs from v1, 640 px, Kaggle T4 | 384 px, Kaggle T4, split by photographer |
| Test | mAP50 0.81 on the original held-out test split · 83% of held-out cans labelled `other_trash` · boxes on 5% of held-out people photos (v1: 65%) | Threshold 0.5 chosen on held-out photographers |

### How well does it work on real rivers?

The detector's validation score comes from the same dataset it trained on. To measure the real use case, I built
three extra test sets of river photos that no model trained on. Each is scored by whether RiverGuard makes the
right SAFE/DANGER call:

| Test set | Detector only | **Detector + dense check (shipped)** |
|---|---|---|
| Trash-carpeted rivers (66 photos, 50 should alert) | 37.9% · 41 missed | **80.3% · 13 missed · 0 false alarms** |
| Unseen photographers (108 photos, 10 should alert) | 91.7% · 9 missed | **97.2% · 1 missed · 2 false alarms** |
| Real river scenes, 0–30 items (17 photos) | 82.4% · 2 missed · 1 false alarm | 82.4% · 1 missed · 2 false alarms |
| Original test split (273 images) | 97.1% · 3 false alarms | 94.5% · 10 false alarms |

**The dense check turns RiverGuard from blind to trash-carpeted rivers into catching four in five**, at the
cost of a few extra false alarms on close-up photos of litter. For a flood early-warning system a missed blockage is
the expensive mistake, and the alert cooldown keeps occasional false alarms from spamming the crew. The full
experiment history, including the retraining attempts that failed and why, is in [training/README.md](training/README.md).
Compared with v1, the shipped v4 detector stops boxing people, recognises cans, and is equal or better on three of
the four sets; on the 17 real river scenes it raises one extra false alarm (coins on a lily pad, see Limitations).

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/Alb138/RiverGuard.git
cd RiverGuard
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Local monitor** (OpenCV window, sends real alerts if configured):

```bash
python run_camera.py --source 0                         # webcam
python run_camera.py --source examples/versailles-pond.jpg  # image
python run_camera.py --source river.mp4 --threshold 8   # video file
python run_camera.py --source rtsp://camera/stream --no-show   # headless next to a CCTV
```

Press `q` to quit. Run `python run_camera.py --help` for all options.

**Web app** (Gradio; sends real alerts if configured, unlike the public demo):

```bash
python app.py        # then open http://127.0.0.1:7860
```

### Telegram alerts

1. Create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`) and copy the token.
2. Send your bot a message, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
   For a group, add the bot to the group first; group IDs start with `-`.
3. `cp .env.example .env` and fill in `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`.

Without these, everything still runs. Alerts are skipped, and the web app shows a preview of the message instead.

## Project structure

```
riverguard/        core package: config, detector + model runtimes, drawing, Telegram alerts, web helpers
run_camera.py      local monitor CLI (webcam / video / image / RTSP)
app.py             Gradio web app (local, with real alerts)
demo/              the public demo: Streamlit app + its slim requirements
models/            best.pt (item detector) + density.pt (dense-trash check), and their ONNX exports
examples/          sample images (see examples/ATTRIBUTION.md)
training/          data collection, labeling tools, dataset builders, Kaggle notebooks, evaluation
tests/             pytest suite (pytest -q)
```

## Deployment

The [live demo](https://riverguard.streamlit.app/) is [`demo/streamlit_app.py`](demo/streamlit_app.py) on the free
[Streamlit Community Cloud](https://streamlit.io/cloud), which updates it on every push to `main`. It installs only
[`demo/requirements.txt`](demo/requirements.txt): no PyTorch, because it runs ONNX exports of both models on plain
onnxruntime ([`riverguard/backends.py`](riverguard/backends.py)), which uses far less memory and starts faster.
The [tests](.github/workflows/tests.yml) check that the exports give the same boxes and scores as the `.pt` models,
and load the demo page with exactly the packages the cloud installs. After 12 hours without visitors the app goes to
sleep; the next visitor wakes it with one click, which takes about a minute.

To run the demo locally (Python 3.11+):

```bash
pip install -r demo/requirements.txt
streamlit run demo/streamlit_app.py
```

After replacing a model, re-export the ONNX files (needs the full `requirements.txt` plus `pip install onnx onnxslim`):

```bash
python -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', dynamic=True, simplify=True)"
python -c "from ultralytics import YOLO; YOLO('models/density.pt').export(format='onnx', imgsz=384, simplify=True)"
```

The public demo never sends Telegram alerts (it doesn't read credentials at all), so visitors see an alert preview
and never message a real chat.

## Limitations & next steps

- **The dense check still misses about one in five trash-carpeted scenes**, including piles of trash on a river bank
  (as opposed to a carpet on the water), and sometimes fires on close-up photos of a few items. It was trained on
  only ~100 dense photos, and openly licensed photos of trash piles at rivers are rare; more varied dense examples
  are the main lever (see [training/README.md](training/README.md)). Murky water and night-time footage are untested.
- **Small white shapes on water can still look like plastic** to the item detector: coins on a lily pad become
  several "bags", and when large photos are scanned in tiles a small cloud can get a `plastic_bag` box. A retrain
  with 330 look-alike photos (clouds, birds, sails, ice, foam) fixed the cloud but recognised fewer cans and lost
  accuracy elsewhere, so it wasn't shipped. A drink can photographed close up can still come out as
  `plastic_bottle`; held-out cans in the training datasets' style are labelled `other_trash` 83% of the time.
- The v1 detector boxed faces as trash (12 of 16 test selfies) because it had never seen a person; v4 learned from
  photos of people and boxes 1 of 16. A small face detector
  ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), MIT licence) still drops any
  box that covers a face, as a safety net for the webcam.
- The original proposal also described a three-level scale (safe / warning / danger) and GPS-tagged alerts; neither
  is implemented yet.
- Counting is per frame, so the same bottle is re-counted as it drifts past. Tracking (e.g. ByteTrack) would allow
  counting unique items over time.

## Background

RiverGuard started as a 3-person group project for the Artificial Intelligence course at BINUS University
(3rd semester, January 2026). This repository is a cleaned-up and deployed version of that prototype.

## License

[AGPL-3.0](LICENSE), inherited from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). The YuNet face
model is MIT-licensed ([its licence](models/face_detection_yunet_2023mar.LICENSE)). Images are licensed separately:
see [examples/ATTRIBUTION.md](examples/ATTRIBUTION.md) and the training credits in
[training/ATTRIBUTION.md](training/ATTRIBUTION.md), [training/DENSITY_ATTRIBUTION.md](training/DENSITY_ATTRIBUTION.md)
and [training/NEGATIVES_ATTRIBUTION.md](training/NEGATIVES_ATTRIBUTION.md) (the photos of people).
