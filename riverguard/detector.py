"""YOLO11 plastic-waste detector and the counting/threshold logic around it."""

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

MAX_SIDE = 1280  # the app downscales larger inputs to this before detection

# A region of trash packed too densely to count item by item (v2+ models). Its presence alone
# means DANGER: nobody needs to know whether a carpet of trash holds 180 or 240 items.
PATCH_CLASS = "trash_patch"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels

    @property
    def is_patch(self) -> bool:
        return self.label == PATCH_CLASS


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    threshold: int

    @property
    def patches(self) -> list[Detection]:
        return [d for d in self.detections if d.is_patch]

    @property
    def count(self) -> int:
        """Countable items only; dense patches are reported separately."""
        return sum(not d.is_patch for d in self.detections)

    @property
    def over_threshold(self) -> bool:
        return is_over_threshold(self.count, self.threshold) or bool(self.patches)

    @property
    def per_class(self) -> dict[str, int]:
        return dict(Counter(d.label for d in self.detections if not d.is_patch).most_common())


def items(n: int) -> str:
    return f"{n} item" if n == 1 else f"{n} items"


def is_over_threshold(count: int, threshold: int) -> bool:
    """The original system alerts only when the count is strictly above the limit."""
    return count > threshold


def fit_within(frame: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame


TILE = 640  # the model's training resolution
TILE_OVERLAP = 0.2


def tile_origins(h: int, w: int, size: int = TILE, overlap: float = TILE_OVERLAP) -> list[tuple[int, int]]:
    """Top-left corners of overlapping size x size tiles that cover an h x w image."""
    step = max(1, int(size * (1 - overlap)))

    def axis(n: int) -> list[int]:
        if n <= size:
            return [0]
        starts = list(range(0, n - size + 1, step))
        if starts[-1] + size < n:
            starts.append(n - size)
        return starts

    return [(x, y) for y in axis(h) for x in axis(w)]


def merge_boxes(boxes: np.ndarray, scores: np.ndarray, ios_thresh: float = 0.6) -> list[int]:
    """Greedy, class-agnostic suppression by intersection-over-smaller-box.

    Plain IoU misses the duplicates tiling creates: a bottle cut by a tile edge yields a small partial
    box inside the full one. Intersection over the smaller box catches those."""
    order = np.argsort(-scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    for i in order:
        if keep:
            k = np.array(keep)
            ix1 = np.maximum(boxes[i, 0], boxes[k, 0])
            iy1 = np.maximum(boxes[i, 1], boxes[k, 1])
            ix2 = np.minimum(boxes[i, 2], boxes[k, 2])
            iy2 = np.minimum(boxes[i, 3], boxes[k, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            ios = inter / np.maximum(np.minimum(areas[i], areas[k]), 1e-9)
            if np.any(ios > ios_thresh):
                continue
        keep.append(int(i))
    return keep


class TrashDetector:
    def __init__(self, model_path: str, conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    @property
    def class_names(self) -> list[str]:
        return list(self.model.names.values())

    def detect(self, frame_bgr: np.ndarray, threshold: int, conf: float | None = None,
               tiled: bool = False) -> DetectionResult:
        """Detect plastic in a BGR frame.

        With `tiled=True`, frames much larger than the model's 640 px input are also scanned as
        overlapping 640 px tiles, so small, distant items keep enough pixels to be recognised.
        """
        conf = self.conf if conf is None else conf
        h, w = frame_bgr.shape[:2]
        crops, offsets = [frame_bgr], [(0, 0)]
        if tiled and max(h, w) > TILE * 1.5:
            for x, y in tile_origins(h, w):
                crops.append(frame_bgr[y:y + TILE, x:x + TILE])
                offsets.append((x, y))

        boxes, scores, classes = [], [], []
        for result, (ox, oy) in zip(self.model(crops, conf=conf, verbose=False), offsets):
            xyxy = result.boxes.xyxy.cpu().numpy()
            boxes.append(xyxy + np.array([ox, oy, ox, oy], dtype=xyxy.dtype))
            scores.append(result.boxes.conf.cpu().numpy())
            classes.append(result.boxes.cls.cpu().numpy())
        boxes, scores, classes = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
        if len(crops) > 1:
            # Merge patches and items separately: an item inside a patch is not a duplicate of it.
            is_patch = np.array([self.model.names[int(c)] == PATCH_CLASS for c in classes], dtype=bool)
            keep = []
            for group in (np.flatnonzero(is_patch), np.flatnonzero(~is_patch)):
                keep += [int(group[i]) for i in merge_boxes(boxes[group], scores[group])] if len(group) else []
        else:
            keep = range(len(boxes))

        detections = [
            Detection(label=self.model.names[int(classes[i])], confidence=float(scores[i]),
                      box=tuple(int(v) for v in boxes[i]))
            for i in keep
        ]
        return DetectionResult(detections=detections, threshold=threshold)
