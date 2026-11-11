"""Smoke test for the hosted Streamlit demo (Streamlit Community Cloud redeploys every push to main)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "best.onnx").exists(), reason="ONNX models not present")


def html_bodies(at) -> str:
    return "\n".join(e.proto.body for e in at.get("html"))


def test_demo_alerts_on_the_dense_example_and_stays_safe_on_the_pond(monkeypatch):
    pytest.importorskip("onnxruntime")
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("TELEGRAM_TOKEN", "")  # the demo must not need (or use) credentials

    at = testing.AppTest.from_file(str(ROOT / "demo" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Image", "Video", "Webcam"]
    page = html_bodies(at)
    assert "DANGER: dense trash + " in page and "RiverGuard bot · alert preview" in page
    assert "The public demo never sends one." in page

    at.button_group(key="example").set_value("river-bottles.jpg").run()
    assert not at.exception
    page = html_bodies(at)
    assert "SAFE: 3 items" in page and "No alert" in page
