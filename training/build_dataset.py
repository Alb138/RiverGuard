"""Merge the datasets listed in training/sources.yaml into one YOLO dataset for training v2.

    python training/build_dataset.py                 # -> datasets/riverguard-v2/ (+ .zip for Kaggle)
    python training/build_dataset.py --no-zip

What it does, per source: download the Roboflow version, remap its classes to RiverGuard's
four (dropping unmapped ones), and convert any polygon labels to boxes. Then across sources:
- `use: test` sources, and the `hold_out` splits of train sources (e.g. a dataset's own test split),
  become the test set, one subfolder per source so each can be scored on its own
  (`data-test-<source>.yaml`). Training images that are near-duplicates of a test image are dropped,
  so the test score can't be inflated by leakage.
- `max_groups` caps how many original photos a source contributes, so one domain can't dominate.
- `repeat` copies a small source's training images N times (validation and test are never repeated).
- `use: train` sources are pooled and split 90/10 into train/val. Roboflow's augmented copies
  of one photo (`name_jpg.rf.<hash>.jpg`) always land in the same split.
- Exact perceptual-hash duplicates are removed.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import rf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NEAR_DUP_BITS = 6  # phash Hamming distance treated as "same photo" for the test-leakage check


def group_key(name: str) -> str:
    """Roboflow names augmented copies `<orig>_jpg.rf.<hash>.jpg`; group them by `<orig>`."""
    return re.split(r"\.rf\.", Path(name).stem)[0]


def read_names(dataset_dir: Path) -> list[str]:
    names = yaml.safe_load((dataset_dir / "data.yaml").read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def convert_label(text: str, idx_map: dict[int, int | None]) -> list[str]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        target = idx_map[int(parts[0])]
        if target is None:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) == 4:
            x, y, w, h = coords
        else:  # polygon: x1 y1 x2 y2 ... -> bounding box
            xs, ys = coords[0::2], coords[1::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w > 0 and h > 0:
            out.append(f"{target} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return out


def phash(path: Path) -> int:
    with Image.open(path) as im:
        return int(str(imagehash.phash(im)), 16)


def hamming_to_set(h: int, others: np.ndarray) -> int:
    if others.size == 0:
        return 64
    x = np.bitwise_xor(others, np.uint64(h))
    return int(np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(axis=1).min())


def collect(source: dict, classes: list[str], raw_dir: Path) -> list[dict]:
    ws, proj, ver = source["roboflow"].split("/")
    src_dir = rf.download_version(ws, proj, int(ver), raw_dir / source["name"])
    names = read_names(src_dir)
    mapping = source.get("map", {})
    missing = [n for n in names if n not in mapping]
    if missing:
        raise SystemExit(f"{source['name']}: classes {missing} have no entry in `map` (use null to drop)")
    idx_map = {i: (classes.index(mapping[n]) if mapping[n] else None) for i, n in enumerate(names)}

    items = []
    for img in sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.parent.name == "images"):
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        lines = convert_label(label.read_text() if label.exists() else "", idx_map)
        items.append({"img": img, "lines": lines, "source": source["name"], "split": img.parent.parent.name,
                      "group": f"{source['name']}/{group_key(img.name)}"})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=ROOT / "training" / "sources.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "riverguard-v2")
    ap.add_argument("--raw", type=Path, default=ROOT / "datasets" / "raw")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.sources.read_text())
    classes = cfg["classes"]
    pools = {"train": [], "test": []}
    for source in cfg["sources"]:
        print(f"== {source['name']} ({source['roboflow']}, {source.get('license', 'license?')})")
        items = collect(source, classes, args.raw)
        if source["use"] == "test":
            held, items = items, []
        else:
            held = [i for i in items if i["split"] in source.get("hold_out", [])]
            items = [i for i in items if i["split"] not in source.get("hold_out", [])]
        if source.get("max_groups"):
            groups = sorted({i["group"] for i in items}, key=lambda g: hashlib.md5(g.encode()).hexdigest())
            keep = set(groups[: source["max_groups"]])
            items = [i for i in items if i["group"] in keep]
        for i in items:
            i["repeat"] = int(source.get("repeat", 1))
        pools["test"].extend(held)
        pools["train"].extend(items)
        print(f"   {len(items)} train/val images, {len(held)} held out for test")

    print("Hashing images for de-duplication...")
    for item in pools["train"] + pools["test"]:
        item["hash"] = phash(item["img"])
    test_hashes = np.array([i["hash"] for i in pools["test"]], dtype=np.uint64)

    seen, kept, dropped = set(), [], Counter()
    for item in pools["train"]:
        if item["hash"] in seen:
            dropped["exact duplicate"] += 1
            continue
        if hamming_to_set(item["hash"], test_hashes) <= NEAR_DUP_BITS:
            dropped["near-duplicate of a test image"] += 1
            continue
        seen.add(item["hash"])
        kept.append(item)

    def split_of(item: dict) -> str:
        bucket = int(hashlib.md5(item["group"].encode()).hexdigest(), 16) % 1000
        return "val" if bucket < args.val_fraction * 1000 else "train"

    if args.out.exists():
        shutil.rmtree(args.out)
    stats = defaultdict(lambda: {"images": 0, "boxes": Counter(), "sources": Counter()})
    for split, items in [("test", pools["test"])] + [(None, kept)]:
        for item in items:
            s = split or split_of(item)
            sub = Path(item["source"]) if s == "test" else Path()  # test/images/<source>/...
            # Some source names are 150+ chars; keep paths under Windows' 260-char limit.
            digest = hashlib.md5(item["img"].name.encode()).hexdigest()[:8]
            stem = f"{item['source']}__{item['img'].stem[:40]}_{digest}"
            (args.out / s / "images" / sub).mkdir(parents=True, exist_ok=True)
            (args.out / s / "labels" / sub).mkdir(parents=True, exist_ok=True)
            copies = item.get("repeat", 1) if s == "train" else 1  # oversample small sources in train only
            for k in range(copies):
                name = stem if k == 0 else f"{stem}_r{k}"
                shutil.copy2(item["img"], args.out / s / "images" / sub / f"{name}{item['img'].suffix.lower()}")
                (args.out / s / "labels" / sub / f"{name}.txt").write_text("\n".join(item["lines"]))
                stats[s]["images"] += 1
                stats[s]["sources"][item["source"]] += 1
                stats[s]["boxes"].update(classes[int(line.split()[0])] for line in item["lines"])

    base = {"train": "train/images", "val": "val/images", "test": "test/images", "names": dict(enumerate(classes))}
    (args.out / "data.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    for subset in sorted({i["source"] for i in pools["test"]}):
        (args.out / f"data-test-{subset}.yaml").write_text(
            yaml.safe_dump({**base, "test": f"test/images/{subset}"}, sort_keys=False))
    summary = {s: {"images": v["images"], "boxes": dict(v["boxes"]), "sources": dict(v["sources"])}
               for s, v in stats.items()}
    summary["dropped"] = dict(dropped)
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_zip:
        archive = shutil.make_archive(str(args.out), "zip", args.out)
        print(f"Zipped for Kaggle: {archive}")


if __name__ == "__main__":
    main()
