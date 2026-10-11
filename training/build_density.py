"""Build the dense-trash classification dataset from tagged candidates.

    python training/build_density.py              # -> datasets/density-v1/ (+ zip for Kaggle)

Layout (Ultralytics classification): datasets/density-v1/{train,val,test}/{dense,not_dense}/*.jpg
- dense = tagged "dense"; not_dense = "scattered" + "clean"; "reject" is skipped.
- Splits are by photographer, so one person's photos (often the same river) never straddle train and test.
  About 20% of photographers become the new tier C test set; the rest split ~80/20 into train/val.
- Tier A/B test photographers, and near-duplicates of tier A/B images, are excluded entirely.
- At most --cap photos per photographer, so a few prolific uploaders can't dominate. Labeled datasets
  (source "dataset") count as one photographer each, get --dataset-cap, and go to train only.
- Near-duplicates across candidates are dropped (dataset copies win), so a photo can't sit in two splits.
Also writes meta.csv, tier C as an evaluate.py tags file, and ATTRIBUTION.md (credits) into the dataset folder.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.build_dataset import NEAR_DUP_BITS, hamming_to_set, phash  # noqa: E402
from training.collect_candidates import OUT, THUMBS, UA, load_candidates  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL = OUT / "full"
LONG_SIDE = 640
# Photographers whose photos are also in a local dataset (Marilao), so any web copies stay in train too.
TRAIN_ONLY_PHOTOGRAPHERS = ("jfvelasquez floro",)


def bucket(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100


def creator_key(c: dict) -> str:
    name = re.sub(r"\s+", " ", (c.get("creator") or "").strip().lower())
    return f"{c.get('source')}:{name}" if name else f"anon:{c['cid']}"


def test_set_exclusions() -> tuple[set[str], np.ndarray]:
    """Photographer names and image hashes from the tier A/B test sets."""
    screening = json.loads((ROOT / "datasets/commons-screening/screening.json").read_text(encoding="utf-8"))
    p = screening["picks"]
    ids = set(p["tier_a_box_label"]) | set(sum(p["tier_b_tag_only"].values(), []))
    names = {re.sub(r"\s+", " ", screening["items"][str(i)]["artist"].strip().lower()) for i in ids
             if str(i) in screening["items"]}
    test_dirs = ("datasets/tier-b", "datasets/test-raw/commons",
                 "datasets/raw/float-plastic-waste-river-yfg5q/test")  # base test split = in-domain test set
    imgs = [f for d in test_dirs for f in (ROOT / d).rglob("*.jpg")]
    return names - {""}, np.array([phash(f) for f in imgs], dtype=np.uint64)


_wikimedia_blocked = False


def fetch_full(i: int, c: dict) -> Path | None:
    global _wikimedia_blocked
    f = FULL / f"{i:05d}.jpg"
    if f.exists():
        return f
    wiki = c.get("source") == "wikimedia"
    if wiki and _wikimedia_blocked:
        return None  # caller falls back to the screening thumbnail
    url = c["url"]
    if wiki and c.get("thumb"):
        url = re.sub(r"/\d+px-", "/1280px-", c["thumb"].split("?")[0])
    try:
        r = requests.get(url, headers=UA, timeout=90)
    except requests.RequestException:
        return None
    if wiki and r.status_code == 429:
        print("Wikimedia is rate-limiting: using thumbnails for the remaining Wikimedia photos", flush=True)
        _wikimedia_blocked = True
        return None
    time.sleep(3.0 if wiki else 0.4)
    if r.status_code != 200 or r.content[:2] != b"\xff\xd8":
        return None
    FULL.mkdir(parents=True, exist_ok=True)
    f.write_bytes(r.content)
    return f


def save_resized(src: Path, dst: Path) -> bool:
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    s = LONG_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "density-v1")
    ap.add_argument("--cap", type=int, default=10, help="max photos per photographer")
    ap.add_argument("--dataset-cap", type=int, default=80, help="max photos per labeled dataset (train only)")
    ap.add_argument("--test-pct", type=int, default=20)
    ap.add_argument("--val-pct", type=int, default=16)
    ap.add_argument("--dense-repeat", type=int, default=1, help="copies of each train/dense image (rebalancing)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    cands = load_candidates()
    tags = {int(r["idx"]): r["label"] for r in csv.DictReader(open(OUT / "tags.csv", encoding="utf-8"))}
    items = [(f"c{i:05d}", c, tags[i], i) for i, c in enumerate(cands) if tags.get(i, "reject") != "reject"]

    test_names, test_hashes = test_set_exclusions()
    kept, dropped, seen = [], Counter(), []
    items.sort(key=lambda it: it[1].get("source") != "dataset")  # dataset copies win over re-found web copies
    for key, c, label, ref in items:
        name = re.sub(r"\s+", " ", (c.get("creator") or "").strip().lower())
        if name and name in test_names:
            dropped["tier A/B photographer"] += 1
            continue
        thumb = THUMBS / f"{ref:05d}.jpg"
        h = phash(thumb) if thumb.exists() else None
        if h is not None and hamming_to_set(h, test_hashes) <= NEAR_DUP_BITS:
            dropped["near-duplicate of a test image"] += 1
            continue
        if h is not None and hamming_to_set(h, np.array(seen, dtype=np.uint64)) <= NEAR_DUP_BITS:
            dropped["near-duplicate of another candidate"] += 1
            continue
        if h is not None:
            seen.append(h)
        kept.append((key, c, label, ref))

    by_creator = defaultdict(list)
    for it in kept:
        by_creator[creator_key(it[1])].append(it)
    chosen = []
    for group, its in by_creator.items():
        cap = args.dataset_cap if group.startswith("dataset:") else args.cap
        its.sort(key=lambda it: (it[2] != "dense", hashlib.md5(it[0].encode()).hexdigest()))  # dense first
        chosen += [(group, *it) for it in its[:cap]]
        dropped["over photographer cap"] += max(0, len(its) - cap)

    if args.out.exists():
        shutil.rmtree(args.out)
    rows, stats = [], Counter()
    for group, key, c, label, ref in chosen:
        b = bucket(group)
        split = "test" if b < args.test_pct else "val" if b < args.test_pct + args.val_pct else "train"
        if group.startswith("dataset:") or any(n in group for n in TRAIN_ONLY_PHOTOGRAPHERS):
            split = "train"  # labeled datasets are one "photographer" each: keep val/test independent
        cls = "dense" if label == "dense" else "not_dense"
        src = ROOT / c["path"] if c.get("source") == "dataset" else fetch_full(ref, c)
        if src is None and (THUMBS / f"{ref:05d}.jpg").exists():
            src = THUMBS / f"{ref:05d}.jpg"  # full size unavailable (e.g. Wikimedia rate limit): 330 px thumbnail
            dropped["used 330 px thumbnail (kept)"] += 1
        if src is None or not save_resized(src, args.out / split / cls / f"{key}.jpg"):
            dropped["download failed"] += 1
            continue
        stats[(split, cls)] += 1
        rows.append({"file": f"{split}/{cls}/{key}.jpg", "split": split, "class": cls, "label": label,
                     "photographer": group, "creator": c.get("creator", ""), "license": c.get("license", ""),
                     "link": c.get("landing_url", ""), "title": c.get("title", "")})

    with open(args.out / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Tier C for evaluate.py --tags: dense must alarm, clean must not; scattered is ambiguous for the alarm.
    with open(args.out / "tierc_tags.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "source"])
        for r in rows:
            if r["split"] == "test" and r["label"] in ("dense", "clean"):
                w.writerow([r["file"], "danger" if r["label"] == "dense" else "none", r["link"]])

    used = sorted(rows, key=lambda r: (r["creator"].lower(), r["file"]))
    lines = ["# Density classifier image credits", "",
             "Photos used to train and test the dense-trash classifier (`training/build_density.py`). They are",
             "downscaled copies and are **not** covered by the repository's AGPL-3.0 license. Only licenses",
             "that allow reuse and modification were used.", "", "| File | Author | License | Source |", "|---|---|---|---|"]
    lines += [f"| `{r['file']}` | {r['creator'][:40].replace('|', '/') or 'unknown'} | {r['license']} | "
              f"[link]({r['link']}) |" for r in used]
    # Written next to the dataset; copy it to training/DENSITY_ATTRIBUTION.md when this build's model ships.
    (args.out / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    photographers = {s: len({r["photographer"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")}
    print(json.dumps({"images": {f"{s}/{c}": n for (s, c), n in sorted(stats.items())},
                      "photographers": photographers, "dropped": dict(dropped)}, indent=1))
    # Rebalance: repeat train/dense images (different augmentations each epoch); val/test are never repeated.
    for k in range(1, args.dense_repeat):
        for f in sorted((args.out / "train" / "dense").glob("*.jpg")):
            if "_r" not in f.stem:
                shutil.copy2(f, f.with_name(f"{f.stem}_r{k}.jpg"))
    if args.dense_repeat > 1:
        print("train/dense after repeat:", len(list((args.out / "train" / "dense").glob("*.jpg"))))
    if not args.no_zip:
        up = ROOT / "datasets" / "density-upload"
        up.mkdir(parents=True, exist_ok=True)
        print("zip:", shutil.make_archive(str(up / "riverguard-density"), "zip", args.out))


if __name__ == "__main__":
    main()
