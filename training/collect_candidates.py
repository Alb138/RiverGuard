"""Collect candidate river photos (dense trash vs. not) for the density classifier.

    python training/collect_candidates.py openverse   # search Openverse; resumable, stops at the daily quota
    python training/collect_candidates.py commons     # add unscreened photos from the earlier Commons crawl
    python training/collect_candidates.py commons-search   # targeted Commons full-text searches
    python training/collect_candidates.py datasets    # labeled Roboflow datasets on disk (one photo per original)
    python training/collect_candidates.py thumbs      # download 320 px screening thumbnails

Only licenses that allow reuse and modification are kept (CC BY, BY-SA, CC0, public domain).
Candidates go to datasets/density/candidates.jsonl, one JSON object per line, never rewritten
in place, so contact-sheet numbers stay stable as more candidates are appended.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "density"
CANDIDATES = OUT / "candidates.jsonl"
STATE = OUT / "openverse_state.json"
THUMBS = OUT / "thumbs"
UA = {"User-Agent": "RiverGuard-portfolio/0.3 (non-commercial student project; dataset collection)"}

# Dense/trash terms first, then look-alike rivers from the same regions (negatives that share
# skylines, boats and water colour, so "Asian city" can't become a shortcut for "dense").
QUERIES = [
    "river garbage", "polluted river", "river trash", "floating garbage", "garbage canal", "river rubbish",
    "plastic pollution river", "dirty river", "polluted canal", "trash boom", "garbage water", "river litter",
    "sampah sungai", "sungai kotor", "kali sampah", "estero", "garbage creek", "plastic waste water",
    "Jakarta canal", "Jakarta river", "Manila river", "Pasig river", "Yamuna river", "Buriganga",
    "Bangkok khlong", "Saigon canal", "Mumbai creek", "city canal", "river bridge", "urban river",
]
REUSABLE = re.compile(r"^(by|by-sa|cc0|pdm)$", re.I)


def load_candidates() -> list[dict]:
    if not CANDIDATES.exists():
        return []
    return [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_candidates(new: list[dict]) -> int:
    known = load_candidates()
    ids = {c["cid"] for c in known}
    urls = {c["url"] for c in known}
    fresh = [c for c in new if c["cid"] not in ids and c["url"] not in urls]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES, "a", encoding="utf-8") as fh:
        for c in fresh:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            ids.add(c["cid"]); urls.add(c["url"])
    return len(fresh)


def openverse(max_requests: int) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    used = 0
    for q in QUERIES:
        page = state.get(q, 1)
        while page != "done" and used < max_requests:
            r = None
            for attempt in range(4):
                try:
                    r = requests.get("https://api.openverse.org/v1/images/", headers=UA, timeout=60, params={
                        "q": q, "license_type": "commercial,modification", "page_size": 20, "page": page,
                        "mature": "false"})
                    break
                except requests.RequestException as exc:
                    print(f"network error on {q!r} p{page}: {type(exc).__name__}; retrying", flush=True)
                    time.sleep(30 * (attempt + 1))
            if r is None:
                print("giving up for now; rerun to resume"); STATE.write_text(json.dumps(state, indent=1)); return
            used += 1
            left = r.headers.get("x-ratelimit-available-anon_sustained")
            if r.status_code == 429:
                print(f"daily quota reached at {q!r} p{page}; rerun tomorrow"); STATE.write_text(json.dumps(state, indent=1)); return
            if r.status_code != 200:  # anonymous access ends after ~12 pages (HTTP 401/400)
                print(f"{q!r}: stop at page {page} (HTTP {r.status_code})")
                state[q] = "done"; break
            body = r.json()
            new = [{
                "cid": f"ov:{x['id']}", "source": x.get("source"), "url": x["url"], "creator": x.get("creator") or "",
                "creator_url": x.get("creator_url") or "", "license": f"CC {x['license'].upper()} {x.get('license_version') or ''}".strip()
                if x["license"] not in ("cc0", "pdm") else x["license"].upper(),
                "landing_url": x.get("foreign_landing_url") or "", "title": x.get("title") or "",
                "width": x.get("width"), "height": x.get("height"), "query": q,
            } for x in body.get("results", []) if REUSABLE.match(x.get("license", "")) and (x.get("width") or 9999) >= 640]
            added = append_candidates(new)
            print(f"{q!r} p{page}/{body.get('page_count')}: +{added} new (quota left today: {left})", flush=True)
            page = "done" if page >= (body.get("page_count") or 0) else page + 1
            state[q] = page
            STATE.write_text(json.dumps(state, indent=1))
            if left is not None and int(left) <= 3:
                print("daily quota almost used up; rerun tomorrow"); return
            time.sleep(3.5)  # 20 requests/min burst limit
    print(f"done for now: {used} requests, {len(load_candidates())} candidates total")


def commons() -> None:
    """Unscreened photos from the Commons crawl in river categories (earlier screening covered others)."""
    meta = json.loads((ROOT / "datasets/commons-screening/commons_meta.json").read_text(encoding="utf-8"))
    screening = json.loads((ROOT / "datasets/commons-screening/screening.json").read_text(encoding="utf-8"))
    p = screening["picks"]  # photos already used as v2 training or test images
    used = set(p["tier_a_box_label"]) | set(p["hard_train_marilao"]) | set(sum(p["tier_b_tag_only"].values(), []))
    already = {screening["items"][str(i)]["title"] for i in used if str(i) in screening["items"]}
    cats = re.compile(r"Ciliwung|Rivers of Jakarta|Kali |Sewerage in Indonesia|Philippines|Bangladesh|India|"
                      r"East Timor|Nigeria|Cameroon|Nepal|Venezuela|Waste in Indonesia|Water pollution", re.I)
    lic = re.compile(r"CC0|CC BY|Public domain|PD", re.I)
    new = [{
        "cid": f"wc:{t[5:]}", "source": "wikimedia", "url": m["url"], "thumb": m["thumb"], "creator": m["artist"],
        "creator_url": "", "license": m["license"], "landing_url": m["url"], "title": t[5:],
        "width": m["width"], "height": m["height"], "query": f"commons:{m['category']}",
    } for t, m in meta.items() if t not in already and cats.search(m.get("category") or "")
        and lic.search(m["license"]) and not re.search("NC|ND", m["license"]) and m["width"] >= 640 and m.get("thumb")]
    print(f"commons: +{append_candidates(new)} new of {len(new)}")


COMMONS_QUERIES = [
    "river garbage", "river pollution plastic", "polluted river", "floating garbage", "garbage canal", "trash boom",
    "river trash", "plastic waste river", "sampah sungai", "sampah kali", "garbage river Indonesia",
    "garbage river Philippines", "garbage river India", "clogged canal garbage", "trash rack river",
]
COMMONS_STATE = OUT / "commons_search_state.json"


def commons_search(max_seconds: float) -> None:
    """Targeted Commons full-text searches (50 files per request, license read from the file page)."""
    lic = re.compile(r"CC0|CC BY|Public domain|PD", re.I)
    state = json.loads(COMMONS_STATE.read_text()) if COMMONS_STATE.exists() else {}
    t0 = time.time()
    for q in COMMONS_QUERIES:
        offset = state.get(q, 0)
        while offset != "done" and time.time() - t0 < max_seconds:
            params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": f"{q} filetype:bitmap",
                      "gsrnamespace": 6, "gsrlimit": 50, "gsroffset": offset, "prop": "imageinfo",
                      "iiprop": "url|size|extmetadata", "iiurlwidth": 330,
                      "iiextmetadatafilter": "LicenseShortName|Artist"}
            try:
                r = requests.get("https://commons.wikimedia.org/w/api.php", params=params, headers=UA, timeout=60)
            except requests.RequestException as exc:
                print(f"network error on {q!r}: {type(exc).__name__}; stopping"); break
            if r.status_code == 429:
                print("commons rate limit; rerun later"); COMMONS_STATE.write_text(json.dumps(state, indent=1)); return
            body = r.json()
            new = []
            for p in body.get("query", {}).get("pages", {}).values():
                ii = (p.get("imageinfo") or [{}])[0]
                m = ii.get("extmetadata", {})
                license_ = m.get("LicenseShortName", {}).get("value", "")
                if not lic.search(license_) or re.search("NC|ND", license_) or (ii.get("width") or 0) < 640:
                    continue
                new.append({"cid": f"wc:{p['title'][5:]}", "source": "wikimedia", "url": ii["descriptionurl"],
                            "thumb": ii.get("thumburl"), "creator": re.sub("<[^>]+>", "", m.get("Artist", {}).get("value", "")).strip(),
                            "creator_url": "", "license": license_, "landing_url": ii["descriptionurl"],
                            "title": p["title"][5:], "width": ii.get("width"), "height": ii.get("height"),
                            "query": f"commons-search:{q}"})
            added = append_candidates(new)
            offset = body.get("continue", {}).get("gsroffset", "done")
            state[q] = offset
            COMMONS_STATE.write_text(json.dumps(state, indent=1))
            print(f"commons {q!r}: +{added} new (next offset {offset})", flush=True)
            time.sleep(3.0)
    print(f"commons search: {len(load_candidates())} candidates total")


# Labeled Roboflow Universe datasets already downloaded (datasets/raw/...) plus the Marilao photos. Each is one
# "photographer" group in build_density.py and goes to training only, so tier C stays independent photographers.
LOCAL_DATASETS = [
    ("marilao", "datasets/hard-raw/marilao", "JFVelasquez Floro (Wikimedia Commons)", "CC0",
     "https://commons.wikimedia.org/wiki/Category:Garbage-plastic_pollution_in_Marilao_River", None),
    ("sampah-sungai", "datasets/raw/sampah-sungai", "Roboflow Universe: uav-7u7iy", "CC BY 4.0",
     "https://universe.roboflow.com/uav-7u7iy/sampah-sungai-kxtsx", None),
    ("waste-in-water", "datasets/raw/waste-in-water", "Roboflow Universe: chinatele", "CC BY 4.0",
     "https://universe.roboflow.com/chinatele/waste-in-water", None),
    ("river-waste", "datasets/raw/river-waste", "Roboflow Universe: class-czqdb", "MIT",
     "https://universe.roboflow.com/class-czqdb/river-waste", None),
    ("floating-trash", "datasets/raw/floating-trash-detection", "Roboflow Universe: waterqualityprediction",
     "CC BY 4.0", "https://universe.roboflow.com/waterqualityprediction/floating-trash-detection", None),
    ("floating-waste", "datasets/raw/floating-waste-r6ifo-hcjzg", "Roboflow Universe: aol-ai-cer7r", "CC BY 4.0",
     "https://universe.roboflow.com/aol-ai-cer7r/floating-waste-r6ifo-hcjzg", 150),  # video frames: sample
    ("base", "datasets/raw/float-plastic-waste-river-yfg5q", "Roboflow Universe: floatplasticwasteriver",
     "CC BY 4.0", "https://universe.roboflow.com/floatplasticwasteriver/floating-plastic-waste-river", 200),
]


def local_datasets() -> None:
    """Add one image per original photo (Roboflow exports hold augmented copies) from each local dataset."""
    import hashlib
    for name, folder, creator, license_, link, limit in LOCAL_DATASETS:
        root = ROOT / folder
        if not root.exists():
            print(f"{name}: {folder} not found, skipped"); continue
        groups = {}
        for p in sorted(x for x in root.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        and "review" not in x.parts and "grid" not in x.parts and "gridreview" not in x.parts):
            groups.setdefault(re.split(r"\.rf\.", p.stem)[0], p)
        keys = sorted(groups, key=lambda k: hashlib.md5(k.encode()).hexdigest())[:limit]
        new = [{"cid": f"ds:{name}/{k}", "source": "dataset", "dataset": name, "path": str(groups[k].relative_to(ROOT)),
                "url": f"local:{name}/{k}", "creator": creator, "creator_url": "", "license": license_,
                "landing_url": link, "title": k, "width": None, "height": None, "query": f"dataset:{name}"}
               for k in keys]
        print(f"{name}: {len(groups)} originals, +{append_candidates(new)} new candidates")


def closeups(limit: int, max_items: int = 8) -> None:
    """Close-up negatives: v1's own photos of a few items on ground or water (train/valid splits only; its test
    split is the in-domain test set). Tagged "scattered" straight from their box labels, so no screening needed."""
    import hashlib
    sys.path.insert(0, str(ROOT))
    from training.sheets import read_tags, write_tags
    root = ROOT / "datasets/raw/float-plastic-waste-river-yfg5q"
    name, _, creator, license_, link, _ = next(d for d in LOCAL_DATASETS if d[0] == "base")
    groups = {}
    for split in ("train", "valid"):
        for p in sorted((root / split / "images").glob("*")):
            label = root / split / "labels" / f"{p.stem}.txt"
            n = sum(1 for line in label.read_text().splitlines() if line.strip()) if label.exists() else 0
            if 1 <= n <= max_items:
                groups.setdefault(re.split(r"\.rf\.", p.stem)[0], p)
    keys = sorted(groups, key=lambda k: hashlib.md5(k.encode()).hexdigest())[:limit]
    append_candidates([{"cid": f"ds:{name}/{k}", "source": "dataset", "dataset": name,
                        "path": str(groups[k].relative_to(ROOT)), "url": f"local:{name}/{k}", "creator": creator,
                        "creator_url": "", "license": license_, "landing_url": link, "title": k, "width": None,
                        "height": None, "query": f"dataset:{name}"} for k in keys])
    wanted = {f"ds:{name}/{k}" for k in keys}
    tags = read_tags()
    new = {i: "scattered" for i, c in enumerate(load_candidates()) if c["cid"] in wanted and i not in tags}
    tags.update(new)
    write_tags(tags)
    print(f"closeups: {len(groups)} eligible (1-{max_items} items), {len(keys)} selected, {len(new)} newly tagged")


def thumb_url(c: dict) -> str:
    if c["source"] == "flickr":  # ..._b.jpg -> ..._n.jpg (320 px), straight from Flickr's CDN
        return re.sub(r"_[a-z]\.(jpg|png)$", r"_n.\1", c["url"]) if re.search(r"_[a-z]\.(jpg|png)$", c["url"]) else c["url"]
    u = (c.get("thumb") or c["url"]).split("?")[0]
    # Wikimedia: always ask for a 330 px thumbnail (originals are large and rate-limited much harder).
    m = re.match(r"https://upload\.wikimedia\.org/wikipedia/commons/(?:thumb/)?([0-9a-f])/([0-9a-f]{2})/([^/]+)", u)
    if m:
        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{m[1]}/{m[2]}/{m[3]}/330px-{m[3]}"
    return re.sub(r"/\d+px-", "/330px-", u)


def _fetch_thumb(job: tuple[int, dict]) -> str:
    i, c = job
    if c["source"] == "dataset":  # local file: resize instead of downloading
        import cv2
        img = cv2.imread(str(ROOT / c["path"]))
        if img is None:
            return "failed"
        s = 320 / max(img.shape[:2])
        cv2.imwrite(str(THUMBS / f"{i:05d}.jpg"), cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s))))
        return "ok"
    for attempt in range(6):  # Wikimedia answers 429 with Retry-After: ~10 s; honour it instead of giving up
        try:
            r = requests.get(thumb_url(c), headers=UA, timeout=60)
        except requests.RequestException:
            time.sleep(15)
            continue
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("retry-after") or 10) + 1)
    else:
        return "limited"
    # JPEG, PNG or WebP (OpenCV reads all three regardless of the .jpg file name)
    if r.status_code == 200 and (r.content[:2] == b"\xff\xd8" or r.content[:4] in (b"\x89PNG", b"RIFF")):
        (THUMBS / f"{i:05d}.jpg").write_bytes(r.content)
        return "ok"
    return "failed"


PRIORITY_TERMS = ("ciliwung", "jakarta", "sampah", "kotor", "indonesia", "estero", "india", "philippines",
                  "river garbage", "floating", "plastic pollution river", "river pollution plastic", "polluted river",
                  "rubbish", "garbage", "litter", "plastic", "waste", "boom", "river trash")
# Screening showed "river trash" on Commons is mostly street bins, bands and weddings: fetch it last.


def _priority(query: str) -> int:
    """Download likely-dense, target-region searches first, so screening starts on the best photos."""
    q = query.lower()
    return next((i for i, t in enumerate(PRIORITY_TERMS) if t in q), len(PRIORITY_TERMS))


def thumbs(max_seconds: float, max_priority: int | None = None) -> None:
    """Flickr and other CDNs in parallel; Wikimedia one at a time (it rate-limits hard), skipped once limited."""
    from concurrent.futures import ThreadPoolExecutor
    THUMBS.mkdir(parents=True, exist_ok=True)
    todo = [(i, c) for i, c in enumerate(load_candidates()) if not (THUMBS / f"{i:05d}.jpg").exists()]
    fast = [j for j in todo if j[1]["source"] != "wikimedia"]
    slow = sorted((j for j in todo if j[1]["source"] == "wikimedia"
                   # photos only: TIFF archive scans and SVG charts have no usable thumbnails
                   and j[1]["url"].split("?")[0].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                   and (max_priority is None or _priority(j[1]["query"]) < max_priority)),
                  key=lambda j: _priority(j[1]["query"]))
    t0, res = time.time(), Counter()
    with ThreadPoolExecutor(6) as pool:
        for k, outcome in enumerate(pool.map(_fetch_thumb, fast), 1):
            res[outcome] += 1
            if k % 200 == 0:
                print(f"thumbs: {k}/{len(fast)} fast-source done", flush=True)
    limited_run = 0
    for k, job in enumerate(slow, 1):
        if time.time() - t0 > max_seconds:
            break
        outcome = _fetch_thumb(job)
        res[outcome] += 1
        limited_run = limited_run + 1 if outcome == "limited" else 0
        if limited_run >= 5:
            print("wikimedia keeps rate-limiting; stopping (rerun later)")
            break
        if k % 100 == 0:
            print(f"thumbs: {k}/{len(slow)} wikimedia done ({dict(res)})", flush=True)
        time.sleep(2.0)
    print(f"thumbs: {dict(res)}, {len(list(THUMBS.glob('*.jpg')))} on disk")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["openverse", "commons", "commons-search", "datasets", "closeups", "thumbs"])
    ap.add_argument("--limit", type=int, default=800, help="closeups: how many photos to add")
    ap.add_argument("--max-priority", type=int, help="thumbs: only Wikimedia searches ranked below this in "
                    "PRIORITY_TERMS (e.g. 12 = everything before 'polluted river')")
    ap.add_argument("--max-requests", type=int, default=190, help="Openverse requests this run (anon: 200/day)")
    ap.add_argument("--max-seconds", type=float, default=3000, help="time budget for downloads / searches")
    args = ap.parse_args()
    {"openverse": lambda: openverse(args.max_requests), "commons": commons,
     "commons-search": lambda: commons_search(args.max_seconds), "datasets": local_datasets,
     "closeups": lambda: closeups(args.limit),
     "thumbs": lambda: thumbs(args.max_seconds, args.max_priority)}[args.step]()


if __name__ == "__main__":
    sys.exit(main())
