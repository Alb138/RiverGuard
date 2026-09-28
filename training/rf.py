"""Minimal Roboflow REST client.

We don't use the `roboflow` SDK because it pins an old OpenCV that clashes with the app's.
The API key travels as a query parameter, so errors here never include the request URL.
"""

import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

API = "https://api.roboflow.com"
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class RoboflowError(RuntimeError):
    pass


def api_key() -> str:
    key = (os.getenv("ROBOFLOW_API_KEY") or "").strip()
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env (Roboflow > Settings > API Keys).")
    return key


def _request(method: str, url: str, what: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, url, **kwargs)
    except requests.RequestException as exc:
        # requests' messages embed the full URL (including ?api_key=...), so don't chain them.
        raise RoboflowError(f"{what}: network error ({type(exc).__name__})") from None


def _check(resp: requests.Response, what: str) -> dict:
    try:
        body = resp.json()
    except ValueError:
        body = {"raw": resp.text[:300]}
    if resp.status_code not in (200, 202) or body.get("error"):
        err = body.get("error", body)
        if isinstance(err, dict):
            err = err.get("message", err)
        raise RoboflowError(f"{what} failed (HTTP {resp.status_code}): {err}")
    return body


def project_info(workspace: str, project: str) -> dict:
    what = f"project info for {workspace}/{project}"
    resp = _request("GET", f"{API}/{workspace}/{project}", what, params={"api_key": api_key()}, timeout=60)
    return _check(resp, what)


def generate_version(workspace: str, project: str) -> int:
    """Create a plain dataset version (auto-orient only, no resize or augmentation). Returns its number."""
    what = f"version generation for {workspace}/{project}"
    resp = _request("POST", f"{API}/{workspace}/{project}/generate", what, params={"api_key": api_key()},
                    json={"preprocessing": {"auto-orient": True}, "augmentation": {}}, timeout=120)
    return int(_check(resp, what)["version"])


def download_version(workspace: str, project: str, version: int, dest: Path, fmt: str = "yolov11") -> Path:
    """Export a dataset version (waiting for Roboflow to generate it) and extract it into `dest`."""
    if dest.exists() and any(dest.iterdir()):
        return dest
    url = f"{API}/{workspace}/{project}/{version}/{fmt}"
    while True:
        what = f"export of {workspace}/{project}/{version}"
        resp = _request("GET", url, what, params={"api_key": api_key(), "nocache": "true"}, timeout=120)
        if resp.status_code == 400 and "still generating" in resp.text:
            print("  waiting for the version to finish generating")
            time.sleep(10)
            continue
        body = _check(resp, what)
        if resp.status_code == 200:
            link = body["export"]["link"]
            break
        print(f"  waiting for export ({float(body.get('progress') or 0):.0%})")
        time.sleep(5)

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "export.zip"
        with requests.get(link, stream=True, timeout=600) as r:
            if r.status_code != 200:
                raise RoboflowError(f"download of {workspace}/{project}/{version} failed (HTTP {r.status_code})")
            with open(zpath, "wb") as fh:
                shutil.copyfileobj(r.raw, fh)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
    return dest


def upload_image(project: str, path: Path, split: str = "train", batch: str | None = None,
                 tags: tuple[str, ...] = ()) -> str:
    """Upload one image to a project in the key's workspace. Returns the Roboflow image id."""
    params = {"api_key": api_key(), "name": path.name, "split": split}
    if batch:
        params["batch"] = batch
    if tags:
        params["tag"] = list(tags)
    with open(path, "rb") as fh:
        resp = _request("POST", f"{API}/dataset/{project}/upload", f"upload of {path.name}", params=params,
                        files={"file": (path.name, fh, "image/jpeg")}, timeout=300)
    body = _check(resp, f"upload of {path.name}")
    if not (body.get("success") or body.get("duplicate")):
        raise RoboflowError(f"upload of {path.name} failed: {body}")
    return body["id"]


def upload_annotation(project: str, image_id: str, name: str, yolo_txt: str, labelmap: dict[int, str],
                      prediction: bool = True, overwrite: bool = False) -> None:
    """Attach YOLO-format boxes to an uploaded image. `prediction=True` shows them in Roboflow as
    model suggestions to review, rather than as finished ground truth."""
    params = {"api_key": api_key(), "name": name}
    if prediction:
        params["prediction"] = "true"
    if overwrite:
        params["overwrite"] = "true"
    resp = _request("POST", f"{API}/dataset/{project}/annotate/{image_id}", f"annotation of {name}",
                    params=params, json={"annotationFile": yolo_txt, "labelmap": labelmap}, timeout=60)
    if resp.status_code == 409:
        return  # already annotated
    _check(resp, f"annotation of {name}")
