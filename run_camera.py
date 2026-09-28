"""RiverGuard local monitor: watch a webcam, video file, image or RTSP stream and alert via Telegram.

    python run_camera.py --source 0                  # default webcam
    python run_camera.py --source river.mp4
    python run_camera.py --source examples/canal-litter.jpg --threshold 3
    python run_camera.py --source rtsp://... --no-show  # headless, e.g. on a server next to a CCTV
"""

import argparse
import logging
import time
from pathlib import Path

import cv2

from riverguard import Settings, TelegramAlerter, TrashDetector, annotate, status_text

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "RiverGuard"

log = logging.getLogger("riverguard")


def parse_args(settings: Settings) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index, video/image path, or stream URL (default: 0)")
    p.add_argument("--model", default=settings.model_path)
    p.add_argument("--conf", type=float, default=settings.conf_threshold, help="detection confidence")
    p.add_argument("--threshold", type=int, default=settings.alert_threshold,
                   help="alert when more than this many items are visible")
    p.add_argument("--cooldown", type=float, default=settings.alert_cooldown, help="seconds between alerts")
    p.add_argument("--no-show", action="store_true", help="don't open a preview window")
    p.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=None,
                   help="also scan large frames in 640 px tiles: finds more small/distant items, ~6x slower "
                        "(default: on for images, off for video and streams)")
    return p.parse_args()


def handle_alert(alerter: TelegramAlerter, frame, result) -> None:
    outcome = alerter.maybe_alert(frame, result)
    if outcome == "queued":
        log.warning("%s -> sending Telegram alert", status_text(result))
    elif outcome == "disabled":
        log.warning("%s (Telegram not configured, alert skipped)", status_text(result))


def run_image(path: str, detector: TrashDetector, alerter: TelegramAlerter, args) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    result = detector.detect(frame, args.threshold, args.conf, tiled=args.tiled is not False)
    annotated = annotate(frame, result)
    log.info("%s %s", status_text(result), result.per_class)
    handle_alert(alerter, annotated, result)
    if not args.no_show:
        cv2.imshow(WINDOW, annotated)
        log.info("Press any key in the window to close")
        cv2.waitKey(0)


def run_stream(source, detector: TrashDetector, alerter: TelegramAlerter, args) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    log.info("Monitoring %s. Alert when more than %d items. Press q to quit.", source, args.threshold)
    last_log = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.detect(frame, args.threshold, args.conf, tiled=bool(args.tiled))
            blink_on = int(time.time() * 5) % 2 == 0
            annotated = annotate(frame, result, blink_on=blink_on)
            if result.over_threshold:
                handle_alert(alerter, annotated, result)

            if args.no_show:
                if time.monotonic() - last_log > 5:
                    log.info(status_text(result))
                    last_log = time.monotonic()
                continue
            cv2.imshow(WINDOW, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    settings = Settings.from_env()
    args = parse_args(settings)

    log.info("Loading model %s", args.model)
    detector = TrashDetector(args.model, conf=args.conf)
    alerter = TelegramAlerter(settings.telegram_token, settings.telegram_chat_id, args.cooldown)
    if not alerter.enabled:
        log.info("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set: running without alerts (see .env.example)")

    if args.source.isdigit():
        run_stream(int(args.source), detector, alerter, args)
    elif Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(args.source, detector, alerter, args)
    else:
        run_stream(args.source, detector, alerter, args)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
