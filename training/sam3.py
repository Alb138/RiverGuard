"""Text-prompted object finding with SAM 3 on Roboflow's Serverless API (0.5 credits per 1,000 images).

Used to draft labels: each prompt returns polygons, which we turn into boxes. Responses are cached
under datasets/sam3-cache/ so re-running never pays twice for the same image and prompts.
"""

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import requests

from training import rf

URL = "https://serverless.roboflow.com/sam3/concept_segment"
CACHE = rf.ROOT / "datasets" / "sam3-cache"


def segment(image_path: Path, prompts: list[str]) -> dict:
    """Raw SAM 3 response for an image and a list of text prompts (cached)."""
    data = Path(image_path).read_bytes()
    key = hashlib.md5(data + json.dumps(prompts).encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    try:
        resp = requests.post(URL, headers={"Authorization": f"Bearer {rf.api_key()}"}, timeout=180, json={
            "image": {"type": "base64", "value": base64.b64encode(buf.tobytes()).decode()},
            "prompts": [{"type": "text", "text": p} for p in prompts]})
    except requests.RequestException as exc:
        raise rf.RoboflowError(f"SAM 3 request failed ({type(exc).__name__})") from None
    body = rf._check(resp, "SAM 3 segmentation")
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(body))
    return body


def boxes(response: dict, min_conf: float = 0.5) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """(prompt, confidence, x1 y1 x2 y2) for every polygon above min_conf."""
    out = []
    for result in response.get("prompt_results", []):
        prompt = result["echo"]["text"]
        for pred in result.get("predictions", []):
            if pred.get("confidence", 0) < min_conf:
                continue
            pts = np.array([p for poly in pred.get("masks", []) for p in poly], dtype=float).reshape(-1, 2)
            if len(pts):
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                out.append((prompt, float(pred["confidence"]), (int(x1), int(y1), int(x2), int(y2))))
    return out
