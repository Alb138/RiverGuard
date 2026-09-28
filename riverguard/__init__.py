"""RiverGuard: plastic-waste detection with Telegram early-warning alerts."""

from .alerts import TelegramAlerter, build_caption
from .config import Settings
from .detector import Detection, DetectionResult, TrashDetector, is_over_threshold, items
from .draw import annotate, status_text

__all__ = [
    "Detection",
    "DetectionResult",
    "Settings",
    "TelegramAlerter",
    "TrashDetector",
    "annotate",
    "build_caption",
    "is_over_threshold",
    "items",
    "status_text",
]
