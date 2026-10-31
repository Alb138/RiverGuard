"""Model runtimes: Ultralytics for .pt checkpoints, or plain onnxruntime for .onnx exports.

The ONNX path needs no PyTorch, so the public demo (demo/streamlit_app.py) stays small and starts fast.
Its pre- and post-processing reproduce Ultralytics' predict step, so both runtimes return the same boxes.
Export with `YOLO("models/best.pt").export(format="onnx", dynamic=True, simplify=True)`.
"""

import ast
import os

import cv2
import numpy as np
from PIL import Image

# One detector/classifier output per image: boxes (n, 4) as x1, y1, x2, y2 pixels, scores (n,), class ids (n,).
Boxes = tuple[np.ndarray, np.ndarray, np.ndarray]


def load_detector(path: str):
    return OnnxDetector(path) if str(path).endswith(".onnx") else UltralyticsDetector(path)


def load_classifier(path: str):
    return OnnxClassifier(path) if str(path).endswith(".onnx") else UltralyticsClassifier(path)


class FaceDetector:
    """OpenCV's YuNet face detector (models/face_detection_yunet_2023mar.onnx, MIT licence, from opencv_zoo)."""

    # At 0.8 it still caught every face the trash model boxed in 16 test selfies, while on 174 river test photos
    # the only trash boxes it removed were two false alarms (a house and a kayaker).
    def __init__(self, path: str, score: float = 0.8):
        self.net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 50)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Face boxes as (n, 4) x1, y1, x2, y2 pixels."""
        h, w = image_bgr.shape[:2]
        self.net.setInputSize((w, h))
        _, faces = self.net.detect(image_bgr)
        if faces is None:
            return np.zeros((0, 4), np.float32)
        return np.column_stack([faces[:, 0], faces[:, 1], faces[:, 0] + faces[:, 2], faces[:, 1] + faces[:, 3]])


# ---------- Ultralytics (.pt) ----------

class UltralyticsDetector:
    def __init__(self, path: str):
        from ultralytics import YOLO  # imported here so the ONNX runtime works without torch installed

        self.model = YOLO(path)
        self.names: dict[int, str] = self.model.names

    def predict(self, images: list[np.ndarray], conf: float) -> list[Boxes]:
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
                for r in self.model(images, conf=conf, verbose=False)]


class UltralyticsClassifier:
    def __init__(self, path: str):
        from ultralytics import YOLO

        self.model = YOLO(path, task="classify")
        self.names: dict[int, str] = self.model.names

    def probs(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.model(image_bgr, verbose=False)[0].probs.data.cpu().numpy()


# ---------- onnxruntime (.onnx) ----------

def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # Small shared servers report the host's cores but only grant a slice of one; extra threads just wait.
    opts.intra_op_num_threads = int(os.getenv("RIVERGUARD_THREADS", "0"))
    opts.enable_cpu_mem_arena = False  # input sizes vary, and a growing arena would hold on to peak memory
    session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map  # written by Ultralytics' exporter
    return session, ast.literal_eval(meta["names"]), ast.literal_eval(meta["imgsz"]), int(meta.get("stride", 32))


def letterbox(image: np.ndarray, size: int, auto: bool, stride: int) -> tuple[np.ndarray, float, int, int]:
    """Ultralytics' LetterBox: fit inside size x size, keep the aspect ratio, pad with grey (114).

    With `auto`, padding only rounds each side up to a multiple of the stride (a smaller rectangle).
    Returns the padded image, the scale and the left/top padding."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = size - new_w, size - new_h
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    if (w, h) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Standard greedy non-maximum suppression (same rule as torchvision.ops.nms)."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou_thresh]
    return keep


class OnnxDetector:
    IOU = 0.7  # Ultralytics' default NMS IoU
    MAX_DET = 300
    MAX_WH = 7680  # class offset so NMS never compares boxes of different classes

    def __init__(self, path: str):
        self.session, self.names, imgsz, self.stride = _session(path)
        self.size = max(imgsz)

    def predict(self, images: list[np.ndarray], conf: float) -> list[Boxes]:
        # Like Ultralytics, use the tighter rectangle padding only when every image in the call has the same shape.
        auto = len({im.shape for im in images}) == 1
        return [self._predict_one(im, conf, auto) for im in images]

    def _predict_one(self, image: np.ndarray, conf: float, auto: bool) -> Boxes:
        h, w = image.shape[:2]
        padded, r, left, top = letterbox(image, self.size, auto, self.stride)
        blob = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255  # BGR->RGB
        pred = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0][0].T  # (anchors, 4 + classes)

        cls_scores = pred[:, 4:]
        classes = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(pred)), classes]
        mask = scores > conf
        xywh, scores, classes = pred[mask, :4], scores[mask], classes[mask]
        boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)

        keep = nms(boxes + classes[:, None] * self.MAX_WH, scores, self.IOU)[: self.MAX_DET]
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        boxes = (boxes - [left, top, left, top]) / r  # undo the letterbox
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes.astype(np.float32), scores.astype(np.float32), classes.astype(int)


class OnnxClassifier:
    def __init__(self, path: str):
        self.session, self.names, imgsz, _ = _session(path)
        self.size = max(imgsz)

    def probs(self, image_bgr: np.ndarray) -> np.ndarray:
        # Ultralytics' classify transform: shorter side to `size` (PIL bilinear), then a centre crop.
        img = Image.fromarray(image_bgr)
        w, h = img.size
        short, long = (w, h) if w <= h else (h, w)
        new_long = int(self.size * long / short)
        img = img.resize((self.size, new_long) if w <= h else (new_long, self.size), Image.BILINEAR)
        left, top = int(round((img.width - self.size) / 2)), int(round((img.height - self.size) / 2))
        img = np.asarray(img.crop((left, top, left + self.size, top + self.size)))
        blob = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255  # BGR->RGB
        return self.session.run(None, {self.session.get_inputs()[0].name: blob})[0][0]  # softmax is in the graph
