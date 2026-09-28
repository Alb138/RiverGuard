"""Pre-label images with the current model and upload them to a Roboflow project for review.

    python training/prelabel.py datasets/hard-raw --project riverguard-hard
    python training/prelabel.py datasets/test-raw --project riverguard-test --split test
    python training/prelabel.py datasets/hard-raw --dry-run runs/prelabel-preview   # no upload

Boxes are sent as model predictions, so in Roboflow you only fix, delete, or add boxes instead
of drawing every one from scratch. Already-uploaded files are skipped, so re-running is safe.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riverguard import DetectionResult, Detection, annotate  # noqa: E402
from training import rf  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST = ".prelabel_manifest.json"


def yolo_lines(result) -> str:
    lines = []
    for cls_id, (x, y, w, h) in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist()):
        lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, help="folder of images")
    ap.add_argument("--project", help="Roboflow project slug in your workspace (create it in the web UI first)")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--batch", help="Roboflow batch name (default: folder name)")
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="higher = fewer wrong boxes to delete, more missed ones to add")
    ap.add_argument("--dry-run", type=Path, metavar="OUT_DIR", help="write previews and labels here instead")
    args = ap.parse_args()
    if not args.dry_run and not args.project:
        ap.error("--project is required unless --dry-run is given")

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXTS)
    model = YOLO(args.model)
    labelmap = {int(k): v for k, v in model.names.items()}
    manifest_path = args.images / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.dry_run:
        args.dry_run.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(images, 1):
        key = str(path.relative_to(args.images))
        if not args.dry_run and key in manifest:
            continue
        result = model(str(path), conf=args.conf, verbose=False)[0]
        txt = yolo_lines(result)
        if args.dry_run:
            img = cv2.imread(str(path))
            dets = [Detection(model.names[int(c)], float(s), tuple(int(v) for v in b))
                    for b, s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                       result.boxes.cls.tolist())]
            cv2.imwrite(str(args.dry_run / f"{path.stem}.jpg"), annotate(img, DetectionResult(dets, threshold=5)))
            (args.dry_run / f"{path.stem}.txt").write_text(txt)
        else:
            image_id = rf.upload_image(args.project, path, split=args.split, batch=args.batch or args.images.name)
            if txt:
                rf.upload_annotation(args.project, image_id, f"{path.stem}.txt", txt, labelmap)
            manifest[key] = image_id
            manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{i}/{len(images)}] {key}: {len(result.boxes)} boxes")


if __name__ == "__main__":
    main()
