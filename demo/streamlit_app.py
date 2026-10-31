"""RiverGuard hosted demo on Streamlit Community Cloud.

It runs the ONNX exports on onnxruntime (no PyTorch, see demo/requirements.txt) and never sends Telegram
alerts: visitors only see a preview. For real alerts, run app.py locally.

    pip install -r demo/requirements.txt
    streamlit run demo/streamlit_app.py
"""

import hashlib
import io
import os
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # Streamlit only puts this file's folder on the import path
os.environ.setdefault("RIVERGUARD_THREADS", "2")  # the free tier gives at most 2 cores; more threads just wait


def code_version() -> str:
    """Changes whenever the detection code or a model changes.

    Streamlit Cloud updates a running app's files in place after a push, but only reloads modules in this folder,
    and its caches can't see inside imported code. Without this, the app kept serving the old detector's results."""
    h = hashlib.sha1()
    for path in sorted((ROOT / "riverguard").glob("*.py")):
        h.update(path.read_bytes())
    for path in sorted((ROOT / "models").glob("*.onnx")):
        h.update(f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:12]


VERSION = code_version()
if "riverguard" in sys.modules and getattr(sys.modules["riverguard"], "demo_version", None) != VERSION:
    for name in [n for n in sys.modules if n.split(".")[0] == "riverguard"]:
        del sys.modules[name]  # the code changed under the running app (or predates this check): import it afresh

import riverguard  # noqa: E402

riverguard.demo_version = VERSION
from riverguard import DetectionResult, Settings, TrashDetector, annotate  # noqa: E402
from riverguard.detector import MAX_SIDE, fit_within  # noqa: E402
from riverguard.web import FOOTER, INTRO, alert_preview_html, alert_rule, annotate_video, status_html  # noqa: E402

DEFAULTS = Settings()  # thresholds only: this app never reads Telegram credentials
EXAMPLES = sorted((ROOT / "examples").glob("*.jpg"))
FIRST_EXAMPLE = ROOT / "examples" / "river-bank-bottles.jpg"  # shown on arrival: dense trash, so it alerts
EXAMPLE_LABELS = {"river-bank-bottles.jpg": "River bank", "danube-bottle.jpg": "Danube",
                  "versailles-pond.jpg": "Pond"}  # short, so all three fit on one line on a phone
VIDEO_MAX_SECONDS = 10
VIDEO_FPS = 2
PREVIEW_NOTE = "👀 Demo mode: this is a preview of the Telegram alert. The public demo never sends one."

st.set_page_config(page_title="RiverGuard", page_icon="🌊", layout="wide")


@st.cache_resource(max_entries=1, show_spinner="Loading the models…")  # a new version replaces the old detector
def load_models(version: str) -> tuple[TrashDetector, threading.Lock]:
    detector = TrashDetector(str(ROOT / "models" / "best.onnx"), conf=DEFAULTS.conf_threshold,
                             density_model_path=str(ROOT / "models" / "density.onnx"),
                             dense_threshold=DEFAULTS.dense_threshold,
                             face_model_path=str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"))
    return detector, threading.Lock()  # one model run at a time: parallel runs only add memory on a small server


def detect(frame_bgr: np.ndarray, threshold: int, conf: float, tiled: bool = False) -> DetectionResult:
    detector, lock = load_models(VERSION)
    with lock:
        return detector.detect(frame_bgr, threshold, conf, tiled=tiled)


def decode(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEGs decode straight at a smaller size: phone photos are huge
    img = ImageOps.exif_transpose(img).convert("RGB")  # phone photos are often stored sideways
    return fit_within(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))


@st.cache_data(max_entries=64, show_spinner="Looking for plastic…")
def analyze(data: bytes, conf: float, threshold: int, tiled: bool, version: str) -> tuple[np.ndarray, DetectionResult]:
    """Annotated BGR frame and result. Cached across visitors, so the examples are instant after the first run.
    `version` keeps results from older code out of the cache."""
    frame = decode(data)
    result = detect(frame, threshold, conf, tiled=tiled)
    return annotate(frame, result), result


def show_result(annotated: np.ndarray, result: DetectionResult, column) -> None:
    with column:
        st.image(annotated, channels="BGR", width="stretch")
        st.html(status_html(result, RULE))
    st.html(alert_preview_html(annotated, result, PREVIEW_NOTE))


def choose_example() -> None:
    name = st.session_state.example
    if name is not None:  # None when the visitor clicks the selected example again
        st.session_state.image = (name, (ROOT / "examples" / name).read_bytes())


def choose_upload() -> None:
    upload = st.session_state.upload
    if upload is not None:
        st.session_state.image = (upload.name, upload.getvalue())
        st.session_state.example = None  # so clicking an example switches back to it


RULE = alert_rule(load_models(VERSION)[0])
if "image" not in st.session_state:
    st.session_state.example = FIRST_EXAMPLE.name
    choose_example()

st.markdown(INTRO)
left, right = st.columns(2)
conf = left.slider("Detection confidence", 0.05, 0.9, DEFAULTS.conf_threshold, 0.05)
threshold = right.slider("Alert threshold (alert when count is above this)", 0, 30, DEFAULTS.alert_threshold, 1)

image_tab, video_tab, camera_tab = st.tabs(["Image", "Video", "Webcam"], key="tab", on_change="rerun")

with image_tab:
    if image_tab.open:  # only the open tab runs, so a hidden webcam never asks for camera access
        left, right = st.columns(2)
        with left:
            st.pills("Try an example", [p.name for p in EXAMPLES], key="example",
                     format_func=lambda name: EXAMPLE_LABELS.get(name, Path(name).stem),
                     on_change=choose_example)
            st.file_uploader("…or upload your own river photo", type=["jpg", "jpeg", "png", "webp"],
                             key="upload", on_change=choose_upload)
            tiled = st.checkbox("Tiled detection (slower, finds more small items)", value=True,
                                help="Also scans large photos as overlapping 640 px tiles, so small and distant "
                                     "items keep enough pixels to be recognised.")
        name, data = st.session_state.image
        try:
            annotated, result = analyze(data, conf, threshold, tiled, VERSION)
        except Exception:  # noqa: BLE001 - a broken upload should not take the page down
            right.error(f"Could not read {name} as an image.")
        else:
            show_result(annotated, result, right)

with video_tab:
    if video_tab.open:
        st.markdown(f"The first {VIDEO_MAX_SECONDS} seconds are analysed at {VIDEO_FPS} frames per second. "
                    "The alert uses the frame with the most items.")
        left, right = st.columns(2)
        with left:
            video = st.file_uploader("River video", type=["mp4", "mov", "avi", "webm", "mkv"], key="video")
            start = st.button("Analyse video", type="primary", disabled=video is None)
        if start and video is not None:
            with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as f:
                f.write(video.getvalue())
            bar = right.progress(0.0, "Analysing frames…")
            try:
                out_path, result, annotated, t, seconds = annotate_video(
                    f.name, lambda frame: detect(frame, threshold, conf), VIDEO_MAX_SECONDS, VIDEO_FPS,
                    progress=lambda fraction: bar.progress(min(fraction, 1.0), "Analysing frames…"))
            except ValueError as e:
                right.error(str(e))
            else:
                summary = status_html(result, RULE, prefix="peak of ") + (
                    f"<p style='opacity:.75'>Peak at {t:.1f}s. Analysed the first {seconds:.0f}s "
                    f"at {VIDEO_FPS} fps.</p>")
                st.session_state.video_result = ((video.file_id, conf, threshold), Path(out_path).read_bytes(),
                                                 summary, alert_preview_html(annotated, result, PREVIEW_NOTE))
                os.remove(out_path)
            finally:
                os.remove(f.name)
                bar.empty()
        saved = st.session_state.get("video_result")
        if video is not None and saved and saved[0] == (video.file_id, conf, threshold):
            _, mp4, summary, preview = saved
            with right:
                st.video(mp4)
                st.html(summary)
            st.html(preview)

with camera_tab:
    if camera_tab.open:
        st.markdown("Take a photo of some plastic bottles or bags (a photo on another screen works too). "
                    "Faces are ignored: the model only learned rivers, so it used to mistake them for trash.")
        if st.toggle("Turn on the camera"):
            left, right = st.columns(2)
            with left:
                shot = st.camera_input("Webcam", label_visibility="collapsed")
            if shot is not None:
                annotated, result = analyze(shot.getvalue(), conf, threshold, False, VERSION)
                show_result(annotated, result, right)

st.divider()
st.markdown(FOOTER)
st.caption("Free hosting: after a quiet day the app sleeps, and waking it up takes about a minute.")
