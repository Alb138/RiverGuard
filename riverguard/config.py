"""Runtime settings, read from environment variables (and a local .env file if present)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "best.pt"
DEFAULT_DENSITY_PATH = ROOT / "models" / "density.pt"  # optional whole-photo "dense trash" classifier

load_dotenv(ROOT / ".env")


def _default_density() -> str | None:
    env = os.getenv("RIVERGUARD_DENSITY_MODEL")
    if env is not None:
        return env or None  # set it to an empty string to disable the density check
    return str(DEFAULT_DENSITY_PATH) if DEFAULT_DENSITY_PATH.exists() else None


@dataclass(frozen=True)
class Settings:
    model_path: str = str(DEFAULT_MODEL_PATH)
    density_model_path: str | None = None
    conf_threshold: float = 0.35  # minimum YOLO confidence for a detection to count
    alert_threshold: int = 5  # alert when MORE than this many items are visible
    dense_threshold: float = 0.5  # alert when the density classifier is at least this sure
    alert_cooldown: float = 20.0  # seconds between two Telegram alerts
    telegram_token: str | None = None
    telegram_chat_id: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=os.getenv("RIVERGUARD_MODEL", str(DEFAULT_MODEL_PATH)),
            density_model_path=_default_density(),
            conf_threshold=float(os.getenv("RIVERGUARD_CONF", cls.conf_threshold)),
            alert_threshold=int(os.getenv("RIVERGUARD_THRESHOLD", cls.alert_threshold)),
            dense_threshold=float(os.getenv("RIVERGUARD_DENSE_THRESHOLD", cls.dense_threshold)),
            alert_cooldown=float(os.getenv("RIVERGUARD_COOLDOWN", cls.alert_cooldown)),
            telegram_token=os.getenv("TELEGRAM_TOKEN") or None,
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)
