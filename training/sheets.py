"""Numbered contact sheets for tagging density candidates, and per-label review sheets.

    python training/sheets.py screen                    # sheets of untagged candidates -> datasets/density/sheets/
    python training/sheets.py tag dense 12 15 40-44     # record tags (numbers = candidate index)
    python training/sheets.py tag reject --rest 0-179   # everything in 0-179 not tagged yet
    python training/sheets.py review                    # grouped sheets per label -> datasets/density/review/
    python training/sheets.py stats

Labels: dense (trash covers a substantial area of water, too much to count), scattered (visible, countable
litter), clean (water, no or negligible trash), reject (not a usable water scene).
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.collect_candidates import OUT, THUMBS, load_candidates  # noqa: E402

TAGS = OUT / "tags.csv"
LABELS = ("dense", "scattered", "clean", "reject")
CW, CH, COLS, ROWS = 320, 240, 6, 5


def read_tags() -> dict[int, str]:
    if not TAGS.exists():
        return {}
    return {int(r["idx"]): r["label"] for r in csv.DictReader(open(TAGS, encoding="utf-8"))}


def write_tags(tags: dict[int, str]) -> None:
    cands = load_candidates()
    with open(TAGS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "cid", "label"])
        for i in sorted(tags):
            w.writerow([i, cands[i]["cid"], tags[i]])


def parse_ranges(specs: list[str]) -> list[int]:
    out = []
    for s in specs:
        a, _, b = s.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def make_sheets(indices: list[int], out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    pages = 0
    for s in range(0, len(indices), per):
        sheet = np.full((ROWS * CH, COLS * CW, 3), 40, np.uint8)
        for k, i in enumerate(indices[s:s + per]):
            img = cv2.imread(str(THUMBS / f"{i:05d}.jpg"))
            if img is None:
                continue
            h, w = img.shape[:2]
            sc = min(CW / w, (CH - 2) / h)
            img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))))
            r, c = divmod(k, COLS)
            sheet[r * CH:r * CH + img.shape[0], c * CW:c * CW + img.shape[1]] = img
            cv2.rectangle(sheet, (c * CW, r * CH), (c * CW + 66, r * CH + 24), (0, 0, 0), -1)
            cv2.putText(sheet, str(i), (c * CW + 3, r * CH + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{prefix}_{s // per:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        pages += 1
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("screen")
    t = sub.add_parser("tag")
    t.add_argument("label", choices=LABELS)
    t.add_argument("indices", nargs="*")
    t.add_argument("--rest", help="range: tag every not-yet-tagged candidate in it")
    sub.add_parser("review")
    sub.add_parser("stats")
    args = ap.parse_args()

    tags = read_tags()
    have = {int(p.stem) for p in THUMBS.glob("*.jpg")}
    if args.cmd == "screen":
        todo = sorted(i for i in have if i not in tags)
        shutil.rmtree(OUT / "sheets", ignore_errors=True)
        n = make_sheets(todo, OUT / "sheets", "s")
        print(f"{len(todo)} untagged -> {n} sheets in {OUT / 'sheets'} (numbers {todo[0] if todo else '-'}..{todo[-1] if todo else '-'})")
    elif args.cmd == "tag":
        idx = parse_ranges(args.indices)
        if args.rest:
            idx += [i for i in parse_ranges([args.rest]) if i not in tags and i in have]
        for i in idx:
            tags[i] = args.label
        write_tags(tags)
        print(f"tagged {len(idx)} as {args.label}; totals {dict(Counter(tags.values()))}")
    elif args.cmd == "review":
        shutil.rmtree(OUT / "review", ignore_errors=True)
        for label in LABELS[:3]:
            n = make_sheets(sorted(i for i, l in tags.items() if l == label), OUT / "review", label)
            print(f"{label}: {n} sheets")
    else:
        cands = load_candidates()
        by = Counter(tags.values())
        creators = {l: len({cands[i]["creator"] for i, x in tags.items() if x == l}) for l in LABELS}
        print(f"candidates {len(cands)}, thumbnails {len(have)}, tagged {len(tags)}")
        for l in LABELS:
            print(f"  {l:9} {by.get(l, 0):4} photos from {creators[l]} photographers")


if __name__ == "__main__":
    main()
