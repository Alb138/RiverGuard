"""Draw detections and the SAFE / DANGER status banner onto a BGR frame."""

import cv2
import numpy as np

from .detector import DetectionResult, items

RED = (0, 0, 255)
GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def status_text(result: DetectionResult) -> str:
    if result.is_dense:
        return f"DANGER: dense trash ({result.dense_score:.0%}) + {items(result.count)}"
    if result.patches:
        return f"DANGER: dense trash + {items(result.count)}"
    if result.over_threshold:
        return f"DANGER: {items(result.count)} (limit {result.threshold})"
    return f"SAFE: {items(result.count)} (limit {result.threshold})"


def annotate(frame_bgr: np.ndarray, result: DetectionResult, blink_on: bool = True) -> np.ndarray:
    """Return a copy of the frame with boxes and the status banner drawn on it.

    `blink_on` lets live video flash the DANGER text like the original demo;
    still images leave it at True so the text is always visible.
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    scale = max(0.5, min(w, h) / 720)
    thick = max(1, round(2 * scale))

    label_scale, label_thick = 0.45 * scale, max(1, thick - 1)
    for det in sorted(result.detections, key=lambda d: not d.is_patch):  # patches underneath items
        x1, y1, x2, y2 = det.box
        color = ORANGE if det.is_patch else RED
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        label = f"{'dense trash' if det.is_patch else det.label} {int(det.confidence * 100)}%"
        (tw, th), base = cv2.getTextSize(label, FONT, label_scale, label_thick)
        top = y1 - th - base - 4 if y1 - th - base - 4 >= 0 else y1  # tag above the box, or inside if no room
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 2), FONT, label_scale, WHITE, label_thick, cv2.LINE_AA)

    title, status = "RiverGuard", status_text(result)
    title_size = cv2.getTextSize(title, FONT, 0.7 * scale, thick)[0]
    status_size = cv2.getTextSize(status, FONT, 0.6 * scale, thick)[0]
    pad = int(10 * scale)
    banner_w = max(title_size[0], status_size[0]) + 2 * pad
    banner_h = title_size[1] + status_size[1] + 4 * pad
    cv2.rectangle(frame, (0, 0), (banner_w, banner_h), BLACK, -1)
    cv2.putText(frame, title, (pad, pad + title_size[1]), FONT, 0.7 * scale, WHITE, thick)

    color = RED if result.over_threshold else GREEN
    if result.over_threshold and not blink_on:
        color = WHITE
    cv2.putText(frame, status, (pad, banner_h - pad), FONT, 0.6 * scale, color, thick)
    return frame
