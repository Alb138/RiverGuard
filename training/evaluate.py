"""Compare RiverGuard models on held-out test sets, both as a detector and as an alarm.

Tier A: a YOLO test split with full boxes -> mAP + count error + SAFE/DANGER accuracy.
    python training/evaluate.py --data datasets/riverguard-v2/data-test-base.yaml models/best.pt runs/v2/best.pt

Tier B: images with one tag each (safe / danger / none), e.g. dense scenes too crowded to box -> alarm only.
    python training/evaluate.py --tags datasets/tier-b/tags.csv models/best.pt runs/v2/best.pt

Add --tiled to also score each model with tiled inference. Alarm metrics see images the way the app
does (downscaled to 1280 px, same confidence), so they predict what users will experience.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from riverguard import TrashDetector  # noqa: E402
from riverguard.detector import PATCH_CLASS, fit_within  # noqa: E402


def tier_a_images(data: dict, split: str) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Images, ground-truth item counts, and whether each image contains a labeled dense-trash patch."""
    src = data.get(split)
    if not src:
        raise SystemExit(f"data.yaml has no '{split}' split")
    patch_id = next((i for i, n in data["names"].items() if n == PATCH_CLASS), None)
    folders = [Path(p) for p in (src if isinstance(src, list) else [src])]
    images = sorted(p for f in folders for p in f.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
    counts, patches = [], []
    for p in images:
        lb = Path(img2label_paths([str(p)])[0])
        ids = [int(line.split()[0]) for line in lb.read_text().splitlines() if line.strip()] if lb.exists() else []
        counts.append(sum(i != patch_id for i in ids))
        patches.append(patch_id in ids)
    return images, np.array(counts), np.array(patches)


def tier_b_images(tags_csv: Path) -> tuple[list[Path], np.ndarray]:
    """CSV with columns image,label (label: safe | danger | none). GT is encoded as a pseudo-count:
    danger = +inf above any threshold, safe/none = 0, which is all the alarm metrics need."""
    images, danger = [], []
    with open(tags_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            images.append((tags_csv.parent / row["image"]).resolve())
            danger.append(row["label"].strip().lower() == "danger")
    return images, np.array(danger)


def alarm_metrics(detector: TrashDetector, images: list[Path], gt_danger: np.ndarray, gt_count: np.ndarray | None,
                  conf: float, threshold: int, tiled: bool) -> dict:
    results = [detector.detect(fit_within(cv2.imread(str(p))), threshold, conf, tiled=tiled) for p in images]
    pred = np.array([r.count for r in results])
    danger_pred = np.array([r.over_threshold for r in results])  # items over the limit, or a dense patch
    tp = int(np.sum(gt_danger & danger_pred))
    fn = int(np.sum(gt_danger & ~danger_pred))
    fp = int(np.sum(~gt_danger & danger_pred))
    out = {
        "decision_accuracy": float(np.mean(gt_danger == danger_pred)),
        "missed_alerts": fn,  # DANGER scenes the model called SAFE (the costly error)
        "false_alerts": fp,
        "alert_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "alert_precision": tp / (tp + fp) if tp + fp else float("nan"),
    }
    if gt_count is not None:
        out["count_mae"] = float(np.mean(np.abs(pred - gt_count)))
        out["count_bias"] = float(np.mean(pred - gt_count))  # negative = undercounting
    out["per_image"] = [{"image": p.name, "pred": int(r.count), "patches": len(r.patches)}
                        for p, r in zip(images, results)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="one or more .pt files")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="tier A: YOLO data.yaml containing the test split")
    src.add_argument("--tags", type=Path, help="tier B: CSV of image,label (safe|danger|none)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.35, help="app operating confidence")
    ap.add_argument("--threshold", type=int, default=5, help="alert when count is above this")
    ap.add_argument("--tiled", action="store_true", help="also score each model with tiled inference")
    ap.add_argument("--out", default="runs/eval/results.json")
    args = ap.parse_args()

    if args.data:
        data = check_det_dataset(args.data)
        images, gt_count, gt_patch = tier_a_images(data, args.split)
        gt_danger = (gt_count > args.threshold) | gt_patch
        print(f"Tier A: {len(images)} images, {gt_count.sum()} boxes, {int(gt_danger.sum())} DANGER scenes\n")
    else:
        images, gt_danger = tier_b_images(args.tags)
        gt_count = None
        print(f"Tier B: {len(images)} tagged images, {int(gt_danger.sum())} DANGER scenes\n")

    results = {}
    for weights in args.models:
        detector = TrashDetector(weights)
        row = {}
        if args.data:
            names = list(data["names"].values())
            if detector.class_names != names[: len(detector.class_names)]:
                # v1 (4 classes) may be scored on a v2 test set (4 classes + trash_patch): its classes must be a prefix.
                raise SystemExit(f"{weights}: classes {detector.class_names} don't match the test set's {names}")
            val = YOLO(weights).val(data=args.data, split=args.split, conf=0.001, plots=False, verbose=False,
                                    workers=0)
            row = {"mAP50": float(val.box.map50), "mAP50-95": float(val.box.map),
                   "precision": float(val.box.mp), "recall": float(val.box.mr),
                   "per_class_mAP50-95": dict(zip(data["names"].values(), map(float, val.box.maps)))}
        results[weights] = {**row, **alarm_metrics(detector, images, gt_danger, gt_count, args.conf,
                                                   args.threshold, tiled=False)}
        if args.tiled:
            results[f"{weights} (tiled)"] = alarm_metrics(detector, images, gt_danger, gt_count, args.conf,
                                                          args.threshold, tiled=True)

    cols = ["mAP50", "mAP50-95", "count_mae", "count_bias", "decision_accuracy", "missed_alerts", "false_alerts"]
    cols = [c for c in cols if any(c in r for r in results.values())]
    print("| model | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, r in results.items():
        cells = [("—" if c not in r else f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for c in cols]
        print(f"| {name} | " + " | ".join(cells) + " |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "results": results}, indent=2))
    print(f"\nFull results (incl. per-image counts): {out}")


if __name__ == "__main__":
    main()
