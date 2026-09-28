"""Draft RiverGuard labels for a folder of images, then apply hand-made patch boxes and corrections.

    python training/autolabel.py datasets/hard-raw/marilao            # draft + review sheets
    python training/autolabel.py datasets/hard-raw/marilao --grid     # also write grid overlays for drawing patches

Items come from SAM 3 text prompts, topped up with the current model's confident boxes that SAM missed.
Two optional files in the image folder refine the result (both keyed by image file name):
- patches.json: {"img.jpg": [[x1, y1, x2, y2], ...]}. trash_patch boxes as 0-1 fractions of width/height.
- edits.json:   {"img.jpg": {"drop": [3, 7], "relabel": {"5": "plastic_bag"}, "add": [[cls, x1, y1, x2, y2]]}}.
  Indices refer to the numbers drawn on the review image; "add" uses 0-1 fractions.
Items whose centre lies inside a patch are removed (LABELING.md: don't box items inside a patch).

Output: <folder>/labels/*.txt (YOLO, RiverGuard's 5 classes) and <folder>/review/*.jpg (numbered boxes).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import sam3  # noqa: E402

CLASSES = ["other_plastic", "plastic_bag", "plastic_bottle", "plastic_package", "trash_patch"]
PROMPT_CLASS = {"plastic bottle": "plastic_bottle", "plastic bag": "plastic_bag", "food wrapper": "plastic_package",
                "plastic cup": "plastic_package", "styrofoam": "other_plastic"}
COLORS = {"other_plastic": (255, 200, 0), "plastic_bag": (0, 200, 0), "plastic_bottle": (0, 0, 255),
          "plastic_package": (255, 0, 255), "trash_patch": (0, 140, 255)}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def draft(path: Path, v1: YOLO, sam_conf: float, v1_conf: float) -> list[tuple[str, float, tuple]]:
    """Merged item boxes: SAM 3 first (dedupe across prompts), then v1 boxes SAM didn't cover."""
    found = sorted(((PROMPT_CLASS[p], c, b) for p, c, b in sam3.boxes(sam3.segment(path, list(PROMPT_CLASS)), sam_conf)),
                   key=lambda t: -t[1])
    keep: list[tuple[str, float, tuple]] = []
    for cls, conf, box in found:
        if all(iou(box, k[2]) < 0.6 for k in keep):
            keep.append((cls, conf, box))
    r = v1(str(path), conf=v1_conf, verbose=False)[0]
    for xyxy, conf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        box = tuple(int(v) for v in xyxy)
        if all(iou(box, k[2]) < 0.3 for k in keep):
            keep.append((v1.names[int(c)], float(conf), box))
    return keep


def grid_overlay(img: np.ndarray) -> np.ndarray:
    """Lines every 10% with labels, for estimating patch boxes as fractions."""
    out = img.copy()
    h, w = out.shape[:2]
    for k in range(1, 10):
        x, y = int(w * k / 10), int(h * k / 10)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(out, f".{k}", (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(out, f".{k}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--sam-conf", type=float, default=0.5)
    ap.add_argument("--v1-conf", type=float, default=0.5)
    ap.add_argument("--grid", action="store_true", help="write grid overlays to <folder>/grid/")
    ap.add_argument("--push", metavar="PROJECT", help="upload the final labels to this Roboflow project, replacing "
                    "the drafts (image ids come from prelabel.py's manifest)")
    args = ap.parse_args()

    patches = json.loads((args.folder / "patches.json").read_text()) if (args.folder / "patches.json").exists() else {}
    edits = json.loads((args.folder / "edits.json").read_text()) if (args.folder / "edits.json").exists() else {}
    v1 = YOLO(args.model)
    for sub in ("labels", "review") + (("grid",) if args.grid else ()):
        (args.folder / sub).mkdir(exist_ok=True)

    for path in sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMG_EXTS):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        if args.grid:
            cv2.imwrite(str(args.folder / "grid" / path.name), grid_overlay(img), [cv2.IMWRITE_JPEG_QUALITY, 85])
        items = draft(path, v1, args.sam_conf, args.v1_conf)

        # Tag each box with its draft index so review numbers stay stable across edits.
        e = edits.get(path.name, {})
        items = [(str(i), e.get("relabel", {}).get(str(i), cls), box) for i, (cls, conf, box) in enumerate(items)
                 if i not in set(e.get("drop", []))]
        items += [(f"a{j}", cls, (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
                  for j, (cls, x1, y1, x2, y2) in enumerate(e.get("add", []))]
        patch_boxes = [(int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)) for x1, y1, x2, y2 in patches.get(path.name, [])]
        inside = lambda b: any(p[0] <= (b[0] + b[2]) / 2 <= p[2] and p[1] <= (b[1] + b[3]) / 2 <= p[3] for p in patch_boxes)
        items = [t for t in items if not inside(t[2])]
        final = items + [("", "trash_patch", p) for p in patch_boxes]

        lines = []
        for _, cls, (x1, y1, x2, y2) in final:
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                lines.append(f"{CLASSES.index(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (args.folder / "labels" / f"{path.stem}.txt").write_text("\n".join(lines))

        review = img.copy()
        for tag, cls, (x1, y1, x2, y2) in final:
            cv2.rectangle(review, (x1, y1), (x2, y2), COLORS[cls], 3 if cls == "trash_patch" else 2)
            if tag:
                cv2.putText(review, tag, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(review, tag, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 1)
        cv2.imwrite(str(args.folder / "review" / path.name), review, [cv2.IMWRITE_JPEG_QUALITY, 85])
        n_patch = len(patch_boxes)
        print(f"{path.name}: {len(final) - n_patch} items, {n_patch} patches")
        if args.push:
            from training import rf
            manifest = json.loads((args.folder / ".prelabel_manifest.json").read_text())
            rf.upload_annotation(args.push, manifest[path.name], f"{path.stem}.txt", "\n".join(lines),
                                 dict(enumerate(CLASSES)), prediction=False, overwrite=True)


if __name__ == "__main__":
    main()
