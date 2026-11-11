"""Pieces shared by the two web demos: app.py (Gradio, local) and demo/streamlit_app.py (the hosted demo)."""

import base64
import html
import tempfile
from collections.abc import Callable

import cv2
import imageio_ffmpeg
import numpy as np

from .alerts import build_caption
from .detector import PATCH_CLASS, DetectionResult, TrashDetector, fit_within, items
from .draw import annotate

INTRO = """
# 🌊 RiverGuard: plastic waste detection with early-warning alerts
A YOLO11s model fine-tuned to spot **plastic bottles, bags, packaging and other plastic** in rivers, plus a
second model that checks whether the water is **carpeted with trash** too dense to count.
When more items than the safe limit are visible, or the water is carpeted, RiverGuard sends a
**Telegram alert with a photo** to cleanup crews. This public demo shows a *preview* of that alert instead of sending it.
"""

FOOTER = (
    "Models: YOLO11s item detector fine-tuned on a public Roboflow river-trash dataset "
    "(P 0.85 · R 0.72 · mAP50 0.83), and a YOLO11s-cls dense-trash classifier "
    "(79% correct alarms on trash-carpeted test rivers). "
    "[Source code on GitHub](https://github.com/Alb138/RiverGuard) · Originally a university group project."
)


def alert_rule(detector: TrashDetector) -> str:
    """Extra alarm condition to mention next to the item limit, if the loaded models can see dense trash."""
    return ", or when dense trash is visible" if detector.has_density or PATCH_CLASS in detector.class_names else ""


def to_data_uri(frame_bgr: np.ndarray, width: int = 480) -> str:
    small = fit_within(frame_bgr, width)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def status_html(result: DetectionResult, rule: str = "", prefix: str = "") -> str:
    danger = result.over_threshold
    color = "#dc2626" if danger else "#16a34a"
    label = "DANGER" if danger else "SAFE"
    rows = "".join(
        f"<li><code>{html.escape(name)}</code> × {n}</li>" for name, n in result.per_class.items()
    )
    if result.patches:
        rows = f"<li><b>Dense trash</b> × {len(result.patches)} (too packed to count)</li>" + rows
    if result.is_dense:
        rows = f"<li><b>Dense trash</b> covering the water ({result.dense_score:.0%} confidence)</li>" + rows
    rows = rows or "<li>No plastic waste detected</li>"
    dense = result.is_dense or result.patches
    headline = f"dense trash + {items(result.count)}" if dense else f"{prefix}{items(result.count)}"
    return f"""
<div style="border:1px solid var(--border-color-primary, rgba(128,128,128,.35));border-left:6px solid {color};
            border-radius:8px;padding:12px 16px">
  <div style="font-size:1.25rem;font-weight:700;color:{color}">{label}: {headline}</div>
  <div style="opacity:.8;margin:2px 0 8px">Alert fires above {items(result.threshold)}{rule}</div>
  <ul style="margin:0;padding-left:1.2rem">{rows}</ul>
</div>"""


def alert_preview_html(annotated_bgr: np.ndarray, result: DetectionResult, note: str = "") -> str:
    """The Telegram message RiverGuard would send, drawn as a chat bubble, followed by `note`."""
    if not result.over_threshold:
        return (f"<div style='opacity:.75;padding:12px'>No alert: {items(result.count)} is within the "
                f"safe limit of {result.threshold}. Raise the item count or lower the threshold to see one.</div>")
    caption = build_caption(result).replace("\n", "<br>")
    return f"""
<div style="background:#9bbf8f;border-radius:12px;padding:14px;max-width:420px;font-family:system-ui,sans-serif">
  <div style="background:#fff;color:#000;border-radius:12px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.2)">
    <img src="{to_data_uri(annotated_bgr)}" style="display:block;width:100%" alt="Alert photo">
    <div style="padding:8px 12px 10px;font-size:14px;line-height:1.45">{caption}</div>
  </div>
  <div style="color:#1f3a18;font-size:12px;margin-top:8px">RiverGuard bot · alert preview</div>
</div>
<p style="margin-top:8px">{note}</p>"""


def annotate_video(path: str, detect: Callable[[np.ndarray], DetectionResult], max_seconds: float, fps: float,
                   progress: Callable[[float], None] = lambda fraction: None):
    """Detect on `fps` frames per second of the first `max_seconds` of a video and write an annotated H.264 mp4.

    Returns (mp4 path, peak result, its annotated frame, peak time in s, seconds analysed). The peak is the frame
    with the most items; the alert uses it. Raises ValueError if no frame can be read."""
    # One decoder and one encoder thread: both buffer frames per thread, and they default to one thread per host
    # core, which pushed peak memory past 512 MB on a many-core host.
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(src_fps * max_seconds)
    step = max(1, round(src_fps / fps))
    last_frame = min(total, int(src_fps * max_seconds))
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    writer, peak, idx = None, None, 0
    try:
        while idx < last_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frame = fit_within(frame)
                frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]  # H.264 needs even sizes
                result = detect(frame)
                annotated = annotate(frame, result)
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = imageio_ffmpeg.write_frames(
                        out_path, (w, h), fps=src_fps / step, codec="libx264", pix_fmt_in="bgr24",
                        macro_block_size=2,
                        output_params=["-movflags", "+faststart", "-preset", "veryfast", "-threads", "1"],
                    )
                    writer.send(None)
                writer.send(np.ascontiguousarray(annotated).tobytes())
                if peak is None or result.count > peak[0].count:
                    peak = (result, annotated, idx / src_fps)
                progress(idx / last_frame)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if peak is None:
        raise ValueError("Could not read any frames from that video.")
    return out_path, *peak, min(idx, last_frame) / src_fps
