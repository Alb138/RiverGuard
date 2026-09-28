"""Cut dense, high-resolution photos into 640 px crops that can be labeled completely.

    python training/make_crops.py datasets/hard-src datasets/hard-raw --per-image 4

A whole photo of a trash-carpeted river holds hundreds of items: nobody can box them all, and a
half-labeled image teaches the model that unboxed trash is background. A 640 px crop of the same
scene holds a manageable 10-40 items, and it matches the model's input size, so nothing is lost
to downscaling. Near-uniform crops (sky, empty water) are skipped. Crops keep their source name,
so attribution can be traced back (see crops.csv).
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = 640


def texture(crop: np.ndarray) -> float:
    """Mean gradient magnitude: low for sky and calm water, high for cluttered trash."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return float(np.mean(np.hypot(gx, gy)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--per-image", type=int, default=4, help="max crops kept per photo (most textured first)")
    ap.add_argument("--min-texture", type=float, default=40.0, help="skip calmer crops (sky, open water)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for path in sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXTS):
        img = cv2.imread(str(path))
        if img is None or max(img.shape[:2]) < SIZE:
            continue
        h, w = img.shape[:2]
        candidates = []
        for y in range(0, h - SIZE + 1, SIZE):  # no overlap, so no item is labeled twice
            for x in range(0, w - SIZE + 1, SIZE):
                crop = img[y:y + SIZE, x:x + SIZE]
                t = texture(crop)
                if t >= args.min_texture:
                    candidates.append((t, x, y, crop))
        candidates.sort(key=lambda c: -c[0])
        keep = candidates[: args.per_image]
        rng.shuffle(keep)
        for t, x, y, crop in keep:
            name = f"{path.stem[:50]}__x{x}_y{y}.jpg"
            cv2.imwrite(str(args.dst / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"crop": name, "source_image": path.name, "x": x, "y": y, "texture": round(t, 1)})
        print(f"{path.name}: {len(keep)} of {len(candidates)} textured crops")

    with open(args.dst / "crops.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["crop", "source_image", "x", "y", "texture"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} crops -> {args.dst}")


if __name__ == "__main__":
    main()
