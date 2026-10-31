"""YOLO11 plastic-waste detector and the counting/threshold logic around it."""

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .backends import FaceDetector, load_classifier, load_detector

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
    # Probability from the optional whole-photo density classifier that the water is carpeted with
    # trash (None when no density model is loaded).
    dense_score: float | None = None
    dense_threshold: float = 0.5

    @property
    def patches(self) -> list[Detection]:
        return [d for d in self.detections if d.is_patch]

    @property
    def is_dense(self) -> bool:
        return self.dense_score is not None and self.dense_score >= self.dense_threshold

    @property
    def count(self) -> int:
        """Countable items only; dense patches are reported separately."""
        return sum(not d.is_patch for d in self.detections)

    @property
    def over_threshold(self) -> bool:
        return is_over_threshold(self.count, self.threshold) or bool(self.patches) or self.is_dense

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


def cut_by_tile_edge(boxes: np.ndarray, crop_hw: tuple[int, int], origin: tuple[int, int],
                     image_hw: tuple[int, int], tol: float = 2) -> np.ndarray:
    """Which tile boxes (in tile coordinates) touch an edge where the tile ends but the image goes on.

    Such an object is only partly inside the tile, so its box is a fragment: one bottle cut in two by a tile
    edge came back as two "bottles". The full frame or the overlapping neighbour tile sees more of it."""
    (ch, cw), (ox, oy), (h, w) = crop_hw, origin, image_hw
    return (((ox > 0) & (boxes[:, 0] <= tol)) | ((oy > 0) & (boxes[:, 1] <= tol))
            | ((ox + cw < w) & (boxes[:, 2] >= cw - tol)) | ((oy + ch < h) & (boxes[:, 3] >= ch - tol)))


def merge_boxes(boxes: np.ndarray, scores: np.ndarray, ios_thresh: float = 0.6) -> list[int]:
    """Greedy, class-agnostic suppression by intersection-over-smaller-box.

    Plain IoU misses the duplicates tiling creates: a tile may box only part of an item that another
    pass boxes whole, and that small box sits inside the big one. Intersection over the smaller box catches it."""
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


def covers_face(box: tuple[int, int, int, int], faces: np.ndarray, min_share: float = 0.5) -> bool:
    """True if at least `min_share` of some face lies inside the box.

    The detector was trained on river photos only and has never seen a person, so it often boxes a face
    (in a webcam, nearly always) as other_plastic or plastic_bottle. Items held next to a face don't cover it."""
    if not len(faces):
        return False
    x1, y1, x2, y2 = box
    iw = np.clip(np.minimum(x2, faces[:, 2]) - np.maximum(x1, faces[:, 0]), 0, None)
    ih = np.clip(np.minimum(y2, faces[:, 3]) - np.maximum(y1, faces[:, 1]), 0, None)
    areas = np.maximum((faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1]), 1e-9)
    return bool(np.any(iw * ih / areas >= min_share))


DENSE_CLASS = "dense"  # the density classifier's positive class (its other class is "not_dense")


class TrashDetector:
    """Item detector plus optional density classifier and face filter. Each model is a .pt checkpoint (runs on
    Ultralytics/PyTorch) or an .onnx export (runs on onnxruntime alone, see riverguard/backends.py)."""

    def __init__(self, model_path: str, conf: float = 0.35, density_model_path: str | None = None,
                 dense_threshold: float = 0.5, face_model_path: str | None = None):
        self.model = load_detector(model_path)
        self.faces = FaceDetector(face_model_path) if face_model_path else None
        self.conf = conf
        self.density = load_classifier(density_model_path) if density_model_path else None
        self.dense_index = next((i for i, n in self.density.names.items() if n == DENSE_CLASS), None) \
            if self.density else None
        if self.density is not None and self.dense_index is None:
            raise ValueError(f"{density_model_path} has no '{DENSE_CLASS}' class: {self.density.names}")
        self.dense_threshold = dense_threshold

    @property
    def has_density(self) -> bool:
        return self.density is not None

    @property
    def class_names(self) -> list[str]:
        return list(self.model.names.values())

    def detect(self, frame_bgr: np.ndarray, threshold: int, conf: float | None = None,
               tiled: bool = False) -> DetectionResult:
        """Detect plastic in a BGR frame.

        With `tiled=True`, frames much larger than the model's 640 px input are also scanned as
        overlapping 640 px tiles, so small, distant items keep enough pixels to be recognised.
        Tile boxes cut off by an inner tile edge are dropped, and duplicates are merged. With a face model,
        boxes covering a face are dropped too.
        """
        conf = self.conf if conf is None else conf
        h, w = frame_bgr.shape[:2]
        crops, offsets = [frame_bgr], [(0, 0)]
        if tiled and max(h, w) > TILE * 1.5:
            for x, y in tile_origins(h, w):
                crops.append(frame_bgr[y:y + TILE, x:x + TILE])
                offsets.append((x, y))

        boxes, scores, classes = [], [], []
        predictions = self.model.predict(crops, conf)
        for i, ((xyxy, crop_scores, crop_classes), (ox, oy)) in enumerate(zip(predictions, offsets)):
            if i > 0:  # a tile (crop 0 is the full frame)
                whole = ~cut_by_tile_edge(xyxy, crops[i].shape[:2], (ox, oy), (h, w))
                xyxy, crop_scores, crop_classes = xyxy[whole], crop_scores[whole], crop_classes[whole]
            boxes.append(xyxy + np.array([ox, oy, ox, oy], dtype=xyxy.dtype))
            scores.append(crop_scores)
            classes.append(crop_classes)
        boxes, scores, classes = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
        if len(crops) > 1:
            # Merge patches and items separately: an item inside a patch is not a duplicate of it.
            is_patch = np.array([self.model.names[int(c)] == PATCH_CLASS for c in classes], dtype=bool)
            keep = []
            for group in (np.flatnonzero(is_patch), np.flatnonzero(~is_patch)):
                keep += [int(group[i]) for i in merge_boxes(boxes[group], scores[group])] if len(group) else []
        else:
            keep = range(len(boxes))
        if self.faces is not None:
            faces = self.faces(frame_bgr)
            keep = [i for i in keep if not covers_face(boxes[i], faces)]

        detections = [
            Detection(label=self.model.names[int(classes[i])], confidence=float(scores[i]),
                      box=tuple(int(v) for v in boxes[i]))
            for i in keep
        ]
        return DetectionResult(detections=detections, threshold=threshold, dense_score=self.dense_score(frame_bgr),
                               dense_threshold=self.dense_threshold)

    def dense_score(self, frame_bgr: np.ndarray) -> float | None:
        """P(water carpeted with trash) for the whole frame, or None without a density model."""
        if self.density is None:
            return None
        return float(self.density.probs(frame_bgr)[self.dense_index])
