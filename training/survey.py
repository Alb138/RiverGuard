"""Print classes, size, license and versions of candidate Roboflow Universe datasets.

    python training/survey.py                      # the candidates below
    python training/survey.py some-ws/some-project # specific ones
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import rf  # noqa: E402

CANDIDATES = [
    "aol-ai-cer7r/floating-waste-r6ifo-hcjzg",  # the team's copy used in the original notebook
    "floatplasticwasteriver/floating-plastic-waste-river",
    "uav-7u7iy/sampah-sungai-kxtsx",
    "waterqualityprediction/floating-trash-detection",
    "try-uqrjj/trash-in-river",
    "class-czqdb/river-waste",
    "image-labelling-bevoh/water-surface-trash-detector",
    "chinatele/waste-in-water",
    "usg-dataset-hfwp5/waste-in-water-dvii5",
    "khrissabdelaadim/ocean-plastics-waste-detection-float-plastics-nmqra",
    "rakadata/river-xxqwa",
    "object-detetctindebris/deteksi-sampah-xi63h",
    "taco-t7kkz/taco-dataset-ql1ng",
]


def main() -> None:
    slugs = sys.argv[1:] or CANDIDATES
    report = {}
    for slug in slugs:
        ws, proj = slug.split("/")
        try:
            info = rf.project_info(ws, proj)
        except rf.RoboflowError as exc:
            print(f"\n## {slug}\n  ERROR {exc}")
            continue
        p = info.get("project", {})
        versions = info.get("versions", [])
        latest = max(versions, key=lambda v: int(str(v.get("id", "0")).rsplit("/", 1)[-1] or 0), default={})
        row = {
            "name": p.get("name"),
            "type": p.get("type"),
            "license": p.get("license") or info.get("license"),
            "images": p.get("images"),
            "classes": p.get("classes"),
            "versions": [str(v.get("id", "")).rsplit("/", 1)[-1] for v in versions],
            "latest_version_images": latest.get("images"),
            "latest_version_splits": latest.get("splits"),
            "latest_version_preprocessing": latest.get("preprocessing"),
            "latest_version_augmentation": latest.get("augmentation"),
        }
        report[slug] = row
        print(f"\n## {slug}: {row['name']} [{row['type']}] license={row['license']} images={row['images']}")
        print(f"   versions={row['versions']} latest: images={row['latest_version_images']} "
              f"splits={row['latest_version_splits']}")
        print(f"   classes={row['classes']}")
    out = Path("datasets/survey.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
