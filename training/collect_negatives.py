"""Collect negative examples for the item detector: photos with nothing to detect.

v1 never saw a person, so it boxed faces as trash (set "people", v4). v4 then learned to call small white shapes
plastic: coins on a lily pad, a cloud (set "lookalikes", v4b: clouds, birds, sails, ice and lilies on water).
Training on such photos with *no* boxes teaches the model they aren't trash. Openly licensed (CC BY, BY-SA, CC0,
public domain) photos come from Wikimedia Commons searches.

    python training/collect_negatives.py search            # find candidates (datasets/negatives/candidates.jsonl)
    python training/collect_negatives.py fetch             # download thumbnails (960 px, saved at 640), politely
    python training/collect_negatives.py sheets            # numbered contact sheets for screening
    python training/collect_negatives.py build             # drop rejects, split by photographer, write credits
    python training/collect_negatives.py --set lookalikes search ...   # same steps -> datasets/negatives-lookalikes

Screening: list the numbers of photos to leave out in <set folder>/rejects.txt, one per line (anything showing
litter, bottles or packaging, which would teach the model to ignore real trash; drawings; duplicates).
"""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "RiverGuard-portfolio/0.4 (student project; negative examples for a trash detector; "
                    "https://github.com/Alb138/RiverGuard)"}
REUSABLE = re.compile(r"CC0|CC BY|Public domain|PD", re.I)

SETS = {
    # People in the situations RiverGuard meets them: in front of a webcam, near water, in a crowd.
    "people": ("negatives", [
        "selfie", "webcam portrait", "video call", "headshot photograph", "portrait photograph man",
        "portrait photograph woman", "child portrait", "elderly person portrait", "student portrait", "family portrait",
        "group photograph", "people indoors", "person using smartphone", "person using laptop", "office workers",
        "classroom students", "people at a cafe", "crowd of people", "festival crowd", "kayaking", "canoeing river",
        "fisherman boat river", "people on a bridge", "river ferry passengers", "rowing boat",
    ]),
    # Small white shapes on or above water that aren't trash.
    "lookalikes": ("negatives-lookalikes", [
        "cumulus clouds over lake", "clouds reflected in river", "sky with white clouds", "swans on river",
        "ducks on a pond", "seagulls on water", "egret river", "sailboats on lake", "ice floes river",
        "water lilies pond", "lily pads", "coins in fountain", "white water rapids", "river foam",
        "snow on riverbank", "geese on lake",
    ]),
}
# Set by main() from --set.
OUT = ROOT / "datasets" / "negatives"
CANDIDATES = OUT / "candidates.jsonl"
THUMBS = OUT / "thumbs"
QUERIES: list[str] = SETS["people"][1]


def load() -> list[dict]:
    return [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines()] \
        if CANDIDATES.exists() else []


def search(per_query: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {c["title"] for c in load()}
    with CANDIDATES.open("a", encoding="utf-8") as fh:
        for q in QUERIES:
            added, offset = 0, 0
            while added < per_query and offset is not None:
                params = {"action": "query", "format": "json", "generator": "search",
                          "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 50, "gsroffset": offset,
                          "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 960,
                          "iiextmetadatafilter": "LicenseShortName|Artist"}
                r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=UA, timeout=60)
                if r.status_code == 429:
                    sys.exit("Commons rate limit: run `search` again later (it resumes)")
                body = r.json()
                for p in sorted(body.get("query", {}).get("pages", {}).values(), key=lambda p: p["index"]):
                    ii = (p.get("imageinfo") or [{}])[0]
                    m = ii.get("extmetadata", {})
                    license_ = m.get("LicenseShortName", {}).get("value", "")
                    title = p["title"][5:]
                    if (title in seen or not REUSABLE.search(license_) or re.search("NC|ND", license_)
                            or (ii.get("width") or 0) < 480 or not ii.get("thumburl")):
                        continue
                    seen.add(title)
                    artist = re.sub("<[^>]+>", "", m.get("Artist", {}).get("value", "")).strip() or "unknown"
                    fh.write(json.dumps({"title": title, "thumb": ii["thumburl"], "link": ii["descriptionurl"],
                                         "creator": artist, "license": license_, "query": q}) + "\n")
                    added += 1
                    if added >= per_query:
                        break
                offset = body.get("continue", {}).get("gsroffset")
                time.sleep(3)
            print(f"{q!r}: +{added}", flush=True)
    print(f"{len(load())} candidates")


def thumb_path(i: int) -> Path:
    return THUMBS / f"n{i:04d}.jpg"


def get_thumb(url: str) -> bytes | None:
    """Wikimedia only renders standard thumbnail widths now (640 px answers HTTP 400). Try 960, and smaller standard
    widths for photos narrower than that, honouring Retry-After (it rate-limits hard)."""
    for width in (960, 500, 330):
        for _ in range(6):
            try:
                r = requests.get(re.sub(r"/\d+px-", f"/{width}px-", url), headers=UA, timeout=60)
            except requests.RequestException:
                time.sleep(10)
                continue
            if r.status_code == 429:
                time.sleep(min(int(r.headers.get("Retry-After", "10") or 10), 120))
                continue
            break
        if r.ok:
            return r.content
        time.sleep(1)
    return None


def fetch(max_seconds: float) -> None:
    """Download missing thumbnails one at a time, 3 s apart."""
    THUMBS.mkdir(parents=True, exist_ok=True)
    t0, got = time.time(), 0
    for i, c in enumerate(load()):
        if thumb_path(i).exists() or time.time() - t0 > max_seconds:
            continue
        data = get_thumb(c["thumb"])
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is not None:
            s = 640 / max(img.shape[:2])  # the detector trains at 640 px
            if s < 1:
                img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb_path(i)), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            got += 1
        if got % 50 == 0 and got:
            print(f"{got} fetched", flush=True)
        time.sleep(3)
    have = sum(thumb_path(i).exists() for i in range(len(load())))
    print(f"{got} new; {have} of {len(load())} on disk")


def sheets() -> None:
    """Contact sheets of 48 numbered thumbnails each (datasets/negatives/sheets/)."""
    out = OUT / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    ids = [i for i in range(len(load())) if thumb_path(i).exists()]
    for s in range(0, len(ids), 48):
        tiles = []
        for i in ids[s:s + 48]:
            img = cv2.imread(str(thumb_path(i)))
            h, w = img.shape[:2]
            img = cv2.resize(img, (200, 200 * h // w)) if w >= h else cv2.resize(img, (200 * w // h, 200))
            tile = np.full((210, 210, 3), 255, np.uint8)
            tile[5:5 + img.shape[0], 5:5 + img.shape[1]] = img
            cv2.rectangle(tile, (0, 0), (52, 22), (0, 0, 0), -1)
            cv2.putText(tile, str(i), (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        tiles += [np.full((210, 210, 3), 255, np.uint8)] * (-len(tiles) % 8)
        grid = np.vstack([np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)])
        cv2.imwrite(str(out / f"sheet_{s // 48:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(ids)} thumbnails on {-(-len(ids) // 48)} sheets in {out}")


def build(test_fraction: float, per_photographer: int) -> None:
    """Accepted photos -> datasets/negatives/{train,test}/ (split by photographer) plus meta.csv."""
    lines = (OUT / "rejects.txt").read_text().splitlines() if (OUT / "rejects.txt").exists() else []
    rejects = {int(x) for line in lines if not line.lstrip().startswith("#") for x in re.findall(r"\d+", line)}
    cands = load()
    keep = [i for i in range(len(cands)) if thumb_path(i).exists() and i not in rejects]
    # Same photographer -> same split, so the people test set checks photographers the model never saw.
    group = {i: re.sub(r"\W+", " ", cands[i]["creator"].lower()).strip() for i in keep}
    # Cap each photographer ("video call" alone returned dozens of near-identical government conference rooms).
    by_group: dict[str, list[int]] = {}
    for i in keep:
        by_group.setdefault(group[i], []).append(i)
    keep = sorted(i for ids in by_group.values() for i in random.sample(ids, min(per_photographer, len(ids))))
    test_groups = {g for g in set(group.values())
                   if int(hashlib.md5(g.encode()).hexdigest(), 16) % 1000 < test_fraction * 1000}
    rows = []
    for split in ("train", "test"):
        shutil.rmtree(OUT / split, ignore_errors=True)  # a rebuild must not keep photos rejected since
        (OUT / split).mkdir(parents=True)
    for i in keep:
        split = "test" if group[i] in test_groups else "train"
        dst = OUT / split / thumb_path(i).name
        dst.write_bytes(thumb_path(i).read_bytes())
        c = cands[i]
        rows.append({"file": f"{split}/{dst.name}", "split": split, "creator": c["creator"], "license": c["license"],
                     "link": c["link"], "title": c["title"]})
    with (OUT / "meta.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [f"# Negative examples credits: {OUT.name}", "",
          "Photos with nothing to detect (people, or white shapes on water that aren't trash), used to train and",
          "test the item detector so it stops boxing them (`training/collect_negatives.py`). They are downscaled",
          "copies and are **not** covered by the repository's AGPL-3.0 license. Only licenses that allow reuse and",
          "modification were used.", "",
          "| File | Author | License | Source |", "|---|---|---|---|"]
    md += [f"| `{r['file']}` | {r['creator'].replace('|', '/')} | {r['license']} | [link]({r['link']}) |" for r in rows]
    # Written next to the data; copy it to training/NEGATIVES_ATTRIBUTION.md when a model trained on it ships.
    (OUT / "ATTRIBUTION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    n_test = sum(r["split"] == "test" for r in rows)
    print(f"{len(rows)} photos ({len(rejects)} rejected): train {len(rows) - n_test}, test {n_test}, "
          f"{len({group[i] for i in keep})} photographers")


def main() -> None:
    global OUT, CANDIDATES, THUMBS, QUERIES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), default="people")
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("search").add_argument("--per-query", type=int, default=40)
    sub.add_parser("fetch").add_argument("--max-seconds", type=float, default=3600)
    sub.add_parser("sheets")
    b = sub.add_parser("build")
    b.add_argument("--test-fraction", type=float, default=0.15)
    b.add_argument("--per-photographer", type=int, default=12)
    args = ap.parse_args()
    folder, QUERIES = SETS[args.set]
    OUT = ROOT / "datasets" / folder
    CANDIDATES, THUMBS = OUT / "candidates.jsonl", OUT / "thumbs"
    random.seed(0)
    {"search": lambda: search(args.per_query), "fetch": lambda: fetch(args.max_seconds), "sheets": sheets,
     "build": lambda: build(args.test_fraction, args.per_photographer)}[args.step]()


if __name__ == "__main__":
    main()
