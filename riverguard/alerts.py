"""Telegram alerts: caption formatting, cooldown, and sending the evidence photo."""

import html
import logging
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests

from .detector import DetectionResult, items

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendPhoto"


def build_caption(result: DetectionResult, when: datetime | None = None) -> str:
    """Telegram caption (HTML parse mode). Mirrors the original alert, in English."""
    when = when or datetime.now()
    breakdown = ", ".join(f"{html.escape(label)} ×{n}" for label, n in result.per_class.items())
    patches = len(result.patches)
    return (
        "🚨 <b>ALERT: PLASTIC WASTE BUILD-UP</b> 🚨\n\n"
        + (f"🟧 Dense trash: {patches} area{'s' if patches != 1 else ''} too packed to count\n" if patches else "")
        + f"🔢 Detected: {items(result.count)}\n"
        + (f"🧴 {breakdown}\n" if breakdown else "")
        + f"🎯 Safe limit: {items(result.threshold)}\n"
        + f"📅 Time: {when:%Y-%m-%d %H:%M:%S}\n"
        + "<i>Please follow up.</i>"
    )


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame as JPEG")
    return buf.tobytes()


class Cooldown:
    def __init__(self, seconds: float, clock=time.monotonic):
        self.seconds = seconds
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """True (and restart the timer) if the cooldown has elapsed."""
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < self.seconds:
                return False
            self._last = now
            return True


class TelegramAlerter:
    def __init__(self, token: str | None, chat_id: str | None, cooldown_s: float = 20.0,
                 session: requests.Session | None = None):
        self.token = token
        self.chat_id = chat_id
        self.cooldown = Cooldown(cooldown_s)
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def maybe_alert(self, annotated_bgr: np.ndarray, result: DetectionResult, background: bool = True) -> str:
        """Send an alert if the result is over threshold and the cooldown allows it.

        Returns one of: "not_triggered", "cooldown", "disabled" (would have sent, but no credentials),
        "sent"/"failed", or "queued" when sending in the background.
        """
        if not result.over_threshold:
            return "not_triggered"
        if not self.cooldown.try_acquire():
            return "cooldown"
        if not self.enabled:
            return "disabled"

        caption = build_caption(result)
        photo = encode_jpeg(annotated_bgr)
        if background:
            threading.Thread(target=self.send, args=(photo, caption), name="telegram-alert").start()
            return "queued"
        return "sent" if self.send(photo, caption) else "failed"

    def send(self, photo_jpeg: bytes, caption: str) -> bool:
        try:
            resp = self.session.post(
                TELEGRAM_API.format(token=self.token),
                data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"photo": ("riverguard.jpg", photo_jpeg, "image/jpeg")},
                timeout=10,
            )
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.error("Telegram request failed: %s", exc)
            return False
        if not body.get("ok"):
            log.error("Telegram rejected the alert: %s", body.get("description", resp.status_code))
            return False
        log.info("Telegram alert sent")
        return True
