"""RiverGuard web demo (Gradio). This is the Hugging Face Space entrypoint.

Real Telegram alerts are sent only when TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are set
(locally via .env, or as Space secrets). Without them the app shows a preview of the
alert it would have sent, so a public demo can never spam anyone's chat.
"""

import base64
import html
import logging
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np

from riverguard import DetectionResult, Settings, TelegramAlerter, TrashDetector, annotate, build_caption, items
from riverguard.detector import PATCH_CLASS, fit_within

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # Gradio's own HTTP calls

VIDEO_MAX_SECONDS = 20  # keep free-CPU processing time reasonable
VIDEO_FPS = 5  # frames analysed per second of video

settings = Settings.from_env()
detector = TrashDetector(settings.model_path, conf=settings.conf_threshold,
                         density_model_path=settings.density_model_path, dense_threshold=settings.dense_threshold)
alerter = TelegramAlerter(settings.telegram_token, settings.telegram_chat_id, settings.alert_cooldown)
# With the density classifier (or a trash_patch model, see training/) the app also alerts on dense,
# uncountable trash.
PATCH_RULE = ", or when dense trash is visible" if detector.has_density or PATCH_CLASS in detector.class_names else ""
EXAMPLES = sorted(str(p) for p in Path(__file__).parent.joinpath("examples").glob("*.jpg"))


# ---------- helpers ----------

def to_data_uri(frame_bgr: np.ndarray, width: int = 480) -> str:
    small = fit_within(frame_bgr, width)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def status_html(result: DetectionResult, prefix: str = "") -> str:
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
<div style="border:1px solid var(--border-color-primary);border-left:6px solid {color};
            border-radius:8px;padding:12px 16px">
  <div style="font-size:1.25rem;font-weight:700;color:{color}">{label}: {headline}</div>
  <div style="opacity:.8;margin:2px 0 8px">Alert fires above {items(result.threshold)}{PATCH_RULE}</div>
  <ul style="margin:0;padding-left:1.2rem">{rows}</ul>
</div>"""


OUTCOME_TEXT = {
    "queued": "✅ Sent to the configured Telegram chat.",
    "sent": "✅ Sent to the configured Telegram chat.",
    "cooldown": "⏳ Threshold exceeded, but an alert was sent less than "
                f"{settings.alert_cooldown:.0f}s ago (cooldown).",
    "failed": "❌ Telegram rejected the alert, check the server logs.",
    "disabled": "👀 Demo mode: this is a preview. No bot token is configured, so nothing was sent.",
}


def telegram_preview_html(annotated_bgr: np.ndarray, result: DetectionResult, outcome: str) -> str:
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
<p style="margin-top:8px">{OUTCOME_TEXT.get(outcome, "")}</p>"""


# ---------- handlers ----------

def send_alert(annotated_bgr: np.ndarray, result: DetectionResult) -> str:
    # In demo mode every visitor gets the preview; the shared cooldown only throttles real sends.
    return alerter.maybe_alert(annotated_bgr, result) if alerter.enabled else "disabled"


def analyze_image(image_rgb, conf, threshold, tiled=True):
    if image_rgb is None:
        return None, "", ""
    frame = fit_within(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    result = detector.detect(frame, int(threshold), conf, tiled=tiled)
    annotated = annotate(frame, result)
    outcome = send_alert(annotated, result)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), status_html(result), telegram_preview_html(annotated, result, outcome)


def analyze_stream(image_rgb, conf, threshold):
    if image_rgb is None:
        return None, ""
    annotated_rgb, status, _ = analyze_image(image_rgb, conf, threshold, tiled=False)  # keep live video responsive
    return annotated_rgb, status


def analyze_video(path, conf, threshold, progress=gr.Progress()):
    if not path:
        return None, "", ""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(src_fps * VIDEO_MAX_SECONDS)
    step = max(1, round(src_fps / VIDEO_FPS))
    last_frame = min(total, int(src_fps * VIDEO_MAX_SECONDS))
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
                result = detector.detect(frame, int(threshold), conf)
                annotated = annotate(frame, result)
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = imageio_ffmpeg.write_frames(
                        out_path, (w, h), fps=src_fps / step, codec="libx264",
                        pix_fmt_in="bgr24", macro_block_size=2, output_params=["-movflags", "+faststart"],
                    )
                    writer.send(None)
                writer.send(np.ascontiguousarray(annotated).tobytes())
                if peak is None or result.count > peak[0].count:
                    peak = (result, annotated, idx / src_fps)
                progress(idx / last_frame, desc="Analysing frames")
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if peak is None:
        raise gr.Error("Could not read any frames from that video.")
    result, annotated, t = peak
    outcome = send_alert(annotated, result)
    summary = status_html(result, prefix="peak of ") + (
        f"<p style='opacity:.75'>Peak at {t:.1f}s. Analysed the first {min(idx, last_frame) / src_fps:.0f}s "
        f"at {VIDEO_FPS} fps.</p>"
    )
    return out_path, summary, telegram_preview_html(annotated, result, outcome)


# ---------- UI ----------

INTRO = """
# 🌊 RiverGuard: plastic waste detection with early-warning alerts
A YOLO11s model fine-tuned to spot **plastic bottles, bags, packaging and other plastic** in rivers, plus a
second model that checks whether the water is **carpeted with trash** too dense to count.
When more items than the safe limit are visible, or the water is carpeted, RiverGuard sends a
**Telegram alert with a photo** to cleanup crews. This public demo shows a *preview* of that alert instead of sending it.
"""

with gr.Blocks(title="RiverGuard") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        conf = gr.Slider(0.05, 0.9, value=settings.conf_threshold, step=0.05, label="Detection confidence")
        threshold = gr.Slider(0, 30, value=settings.alert_threshold, step=1,
                              label="Alert threshold (alert when count is above this)")

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="numpy", label="River photo", sources=["upload", "clipboard"])
                tiled = gr.Checkbox(value=True, label="Tiled detection: also scan large photos in 640 px tiles "
                                    "(slower, finds more small and distant items)")
                img_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                img_out = gr.Image(label="Detections")
                img_status = gr.HTML()
        img_alert = gr.HTML()
        if EXAMPLES:
            gr.Examples(
                examples=[[p, settings.conf_threshold, settings.alert_threshold, True] for p in EXAMPLES],
                inputs=[img_in, conf, threshold, tiled], outputs=[img_out, img_status, img_alert],
                fn=analyze_image, run_on_click=True, cache_examples=False,
            )
        img_btn.click(analyze_image, [img_in, conf, threshold, tiled], [img_out, img_status, img_alert],
                      api_name="detect_image")

    with gr.Tab("Video"):
        gr.Markdown(f"The first {VIDEO_MAX_SECONDS} seconds are analysed at {VIDEO_FPS} frames per second. "
                    "The alert uses the frame with the most items.")
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="River video", sources=["upload"])
                vid_btn = gr.Button("Analyse video", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="Annotated video")
                vid_status = gr.HTML()
        vid_alert = gr.HTML()
        vid_btn.click(analyze_video, [vid_in, conf, threshold], [vid_out, vid_status, vid_alert],
                      concurrency_limit=1, api_name="detect_video")

    with gr.Tab("Webcam"):
        gr.Markdown("Point your camera at some plastic bottles or bags (a photo on another screen works too).")
        with gr.Row():
            cam_in = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
            with gr.Column():
                cam_out = gr.Image(label="Live detections")
                cam_status = gr.HTML()
        cam_in.stream(analyze_stream, [cam_in, conf, threshold], [cam_out, cam_status],
                      stream_every=0.5, time_limit=120, concurrency_limit=4, api_name="detect_webcam")

    gr.Markdown(
        "Models: YOLO11s item detector fine-tuned on a public Roboflow river-trash dataset "
        "(P 0.85 · R 0.72 · mAP50 0.83), and a YOLO11s-cls dense-trash classifier "
        "(79% correct alarms on trash-carpeted test rivers). "
        "[Source code on GitHub](https://github.com/Alb138/RiverGuard) · Originally a university group project."
    )

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
