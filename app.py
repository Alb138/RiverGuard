"""RiverGuard web app (Gradio), for running locally with real alerts. The hosted demo is demo/streamlit_app.py.

Real Telegram alerts are sent only when TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are set
(locally via .env). Without them the app shows a preview of the alert it would have sent,
so a public deployment can never spam anyone's chat.
"""

import logging
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from riverguard import DetectionResult, Settings, TelegramAlerter, TrashDetector, annotate
from riverguard.detector import fit_within
from riverguard.web import FOOTER, INTRO, alert_preview_html, alert_rule, annotate_video, status_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # Gradio's own HTTP calls

VIDEO_MAX_SECONDS = 20  # keep CPU processing time reasonable
VIDEO_FPS = 5  # frames analysed per second of video

settings = Settings.from_env()
detector = TrashDetector(settings.model_path, conf=settings.conf_threshold,
                         density_model_path=settings.density_model_path, dense_threshold=settings.dense_threshold,
                         face_model_path=settings.face_model_path)
# One model run at a time across all visitors: parallel runs on a small server only add memory, not speed.
model_lock = threading.Lock()
alerter = TelegramAlerter(settings.telegram_token, settings.telegram_chat_id, settings.alert_cooldown)
# With the density classifier (or a trash_patch model, see training/) the app also alerts on dense,
# uncountable trash.
RULE = alert_rule(detector)
EXAMPLES = sorted(str(p) for p in Path(__file__).parent.joinpath("examples").glob("*.jpg"))

OUTCOME_TEXT = {
    "queued": "✅ Sent to the configured Telegram chat.",
    "sent": "✅ Sent to the configured Telegram chat.",
    "cooldown": "⏳ Threshold exceeded, but an alert was sent less than "
                f"{settings.alert_cooldown:.0f}s ago (cooldown).",
    "failed": "❌ Telegram rejected the alert, check the server logs.",
    "disabled": "👀 Demo mode: this is a preview. No bot token is configured, so nothing was sent.",
}


# ---------- handlers ----------

def detect(frame_bgr: np.ndarray, threshold, conf, tiled: bool = False) -> DetectionResult:
    with model_lock:
        return detector.detect(frame_bgr, int(threshold), conf, tiled=tiled)


def send_alert(annotated_bgr: np.ndarray, result: DetectionResult) -> str:
    # In demo mode every visitor gets the preview; the shared cooldown only throttles real sends.
    return alerter.maybe_alert(annotated_bgr, result) if alerter.enabled else "disabled"


def preview(annotated_bgr: np.ndarray, result: DetectionResult) -> str:
    return alert_preview_html(annotated_bgr, result, OUTCOME_TEXT.get(send_alert(annotated_bgr, result), ""))


def analyze_image(image_rgb, conf, threshold, tiled=True):
    if image_rgb is None:
        return None, "", ""
    frame = fit_within(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    result = detect(frame, threshold, conf, tiled=tiled)
    annotated = annotate(frame, result)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), status_html(result, RULE), preview(annotated, result)


def analyze_stream(image_rgb, conf, threshold):
    if image_rgb is None:
        return None, ""
    annotated_rgb, status, _ = analyze_image(image_rgb, conf, threshold, tiled=False)  # keep live video responsive
    return annotated_rgb, status


def analyze_video(path, conf, threshold, progress=gr.Progress()):
    if not path:
        return None, "", ""
    try:
        out_path, result, annotated, t, seconds = annotate_video(
            path, lambda frame: detect(frame, threshold, conf), VIDEO_MAX_SECONDS, VIDEO_FPS,
            progress=lambda fraction: progress(fraction, desc="Analysing frames"))
    except ValueError as e:
        raise gr.Error(str(e)) from e
    summary = status_html(result, RULE, prefix="peak of ") + (
        f"<p style='opacity:.75'>Peak at {t:.1f}s. Analysed the first {seconds:.0f}s at {VIDEO_FPS} fps.</p>"
    )
    return out_path, summary, preview(annotated, result)


# ---------- UI ----------

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
                img_out = gr.Image(label="Detections", format="jpeg")  # ~100x cheaper to encode than webp
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
                cam_out = gr.Image(label="Live detections", format="jpeg")
                cam_status = gr.HTML()
        cam_in.stream(analyze_stream, [cam_in, conf, threshold], [cam_out, cam_status],
                      stream_every=0.5, time_limit=120, concurrency_limit=4, api_name="detect_webcam")

    gr.Markdown(FOOTER)

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
