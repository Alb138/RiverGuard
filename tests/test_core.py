from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from riverguard import Detection, DetectionResult, TelegramAlerter, annotate, build_caption, is_over_threshold
from riverguard.alerts import Cooldown
from riverguard.config import DEFAULT_MODEL_PATH


def make_result(labels, threshold=5):
    dets = [Detection(label=l, confidence=0.9, box=(10, 10, 50, 50)) for l in labels]
    return DetectionResult(detections=dets, threshold=threshold)


def test_threshold_is_strictly_greater():
    assert not is_over_threshold(5, 5)
    assert is_over_threshold(6, 5)
    assert not make_result(["plastic_bottle"] * 5).over_threshold
    assert make_result(["plastic_bottle"] * 6).over_threshold


def test_per_class_counts_sorted_by_frequency():
    result = make_result(["plastic_bag", "plastic_bottle", "plastic_bottle"])
    assert result.per_class == {"plastic_bottle": 2, "plastic_bag": 1}


def test_caption_contents():
    result = make_result(["plastic_bottle"] * 7 + ["other_plastic"] * 2)
    caption = build_caption(result, when=datetime(2026, 1, 3, 21, 38, 16))
    assert "Detected: 9 items" in caption
    assert "Safe limit: 5 items" in caption
    assert "plastic_bottle ×7, other_plastic ×2" in caption
    assert "2026-01-03 21:38:16" in caption


def test_cooldown_blocks_until_elapsed():
    now = [100.0]
    cd = Cooldown(20, clock=lambda: now[0])
    assert cd.try_acquire()
    now[0] = 119.9
    assert not cd.try_acquire()
    now[0] = 120.0
    assert cd.try_acquire()


def test_alerter_without_credentials_never_posts():
    session = MagicMock()
    alerter = TelegramAlerter(None, None, session=session)
    frame = np.zeros((64, 64, 3), np.uint8)
    assert alerter.maybe_alert(frame, make_result(["x"] * 9), background=False) == "disabled"
    # "disabled" means "would have sent now", so it is rate-limited like a real alert
    assert alerter.maybe_alert(frame, make_result(["x"] * 9), background=False) == "cooldown"
    session.post.assert_not_called()


def test_alerter_sends_once_per_cooldown():
    session = MagicMock()
    session.post.return_value.json.return_value = {"ok": True}
    alerter = TelegramAlerter("123:abc", "42", cooldown_s=60, session=session)
    frame = np.zeros((64, 64, 3), np.uint8)
    over = make_result(["plastic_bottle"] * 9)

    assert alerter.maybe_alert(frame, make_result(["x"]), background=False) == "not_triggered"
    assert alerter.maybe_alert(frame, over, background=False) == "sent"
    assert alerter.maybe_alert(frame, over, background=False) == "cooldown"
    assert session.post.call_count == 1

    kwargs = session.post.call_args.kwargs
    assert kwargs["data"]["chat_id"] == "42"
    assert kwargs["data"]["parse_mode"] == "HTML"
    assert kwargs["timeout"] == 10


def test_alerter_reports_telegram_error():
    session = MagicMock()
    session.post.return_value.json.return_value = {"ok": False, "description": "Unauthorized"}
    alerter = TelegramAlerter("bad", "42", session=session)
    frame = np.zeros((64, 64, 3), np.uint8)
    assert alerter.maybe_alert(frame, make_result(["x"] * 9), background=False) == "failed"


def test_annotate_keeps_shape_and_draws():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = annotate(frame, make_result(["plastic_bag"] * 6))
    assert out.shape == frame.shape
    assert out.any() and not frame.any()


@pytest.mark.skipif(not Path(DEFAULT_MODEL_PATH).exists(), reason="model weights not present")
def test_model_loads_with_expected_classes():
    from riverguard import TrashDetector

    detector = TrashDetector(str(DEFAULT_MODEL_PATH))
    assert detector.class_names == ["other_plastic", "plastic_bag", "plastic_bottle", "plastic_package"]
    result = detector.detect(np.zeros((640, 640, 3), np.uint8), threshold=5)
    assert result.count == 0 and not result.over_threshold


def test_tile_origins_cover_image_with_overlap():
    from riverguard.detector import TILE, tile_origins

    origins = tile_origins(960, 1280)
    assert (0, 0) in origins and (1280 - TILE, 960 - TILE) in origins  # both corners covered
    assert tile_origins(500, 600) == [(0, 0)]  # smaller than one tile


def test_merge_boxes_drops_partial_duplicates_but_keeps_neighbours():
    from riverguard.detector import merge_boxes

    boxes = np.array([
        [100, 100, 200, 300],  # full bottle
        [100, 100, 200, 180],  # same bottle cut by a tile edge (inside the full box)
        [210, 100, 300, 300],  # a different bottle right next to it
    ], dtype=float)
    scores = np.array([0.9, 0.6, 0.8])
    assert sorted(merge_boxes(boxes, scores)) == [0, 2]


def test_dense_patch_triggers_danger_and_is_not_counted_as_an_item():
    from riverguard import status_text

    result = make_result(["plastic_bottle", "trash_patch"])
    assert result.count == 1
    assert result.per_class == {"plastic_bottle": 1}
    assert len(result.patches) == 1
    assert result.over_threshold  # a patch alone means DANGER
    assert status_text(result) == "DANGER: dense trash + 1 item"
    assert not make_result(["plastic_bottle"] * 5).over_threshold


def test_caption_and_drawing_handle_patches():
    result = make_result(["trash_patch", "trash_patch"])
    caption = build_caption(result)
    assert "Dense trash: 2 areas too packed to count" in caption
    assert "Detected: 0 items" in caption and "🧴" not in caption  # no empty breakdown line
    frame = np.zeros((480, 640, 3), np.uint8)
    assert annotate(frame, result).any()


def test_density_score_triggers_danger_only_above_threshold():
    from riverguard import status_text

    few = [Detection(label="plastic_bottle", confidence=0.9, box=(10, 10, 50, 50))] * 2
    dense = DetectionResult(detections=few, threshold=5, dense_score=0.87, dense_threshold=0.5)
    assert dense.is_dense and dense.over_threshold
    assert status_text(dense) == "DANGER: dense trash (87%) + 2 items"
    assert "Dense trash covering the water (87% confidence)" in build_caption(dense)

    unsure = DetectionResult(detections=few, threshold=5, dense_score=0.3, dense_threshold=0.5)
    assert not unsure.is_dense and not unsure.over_threshold

    no_model = DetectionResult(detections=few, threshold=5)  # dense_score None -> v1 behaviour
    assert not no_model.is_dense and not no_model.over_threshold
    assert "Dense trash" not in build_caption(make_result(["x"] * 9))


@pytest.mark.skipif(not (Path(DEFAULT_MODEL_PATH).parent / "density.pt").exists(), reason="density model not present")
def test_density_model_loads_and_scores_a_frame():
    from riverguard import TrashDetector

    detector = TrashDetector(str(DEFAULT_MODEL_PATH), density_model_path=str(Path(DEFAULT_MODEL_PATH).parent / "density.pt"))
    assert detector.has_density and detector.density.names[detector.dense_index] == "dense"
    result = detector.detect(np.zeros((480, 640, 3), np.uint8), threshold=5)
    assert 0.0 <= result.dense_score <= 1.0


MODELS = Path(DEFAULT_MODEL_PATH).parent


@pytest.mark.skipif(not all((MODELS / f).exists() for f in ("best.pt", "density.pt", "best.onnx", "density.onnx")),
                    reason="model weights not present")
def test_onnx_exports_match_pytorch_models():
    """The hosted demo runs the .onnx exports. They must give the same answers as the .pt models
    (re-export both after replacing a .pt, see riverguard/backends.py)."""
    pytest.importorskip("onnxruntime")
    import cv2

    from riverguard import TrashDetector
    from riverguard.detector import fit_within

    pt = TrashDetector(str(MODELS / "best.pt"), density_model_path=str(MODELS / "density.pt"))
    onnx = TrashDetector(str(MODELS / "best.onnx"), density_model_path=str(MODELS / "density.onnx"))
    assert onnx.class_names == pt.class_names
    frame = fit_within(cv2.imread(str(Path(__file__).parent.parent / "examples" / "river-bank-bottles.jpg")))
    for tiled in (False, True):
        a, b = pt.detect(frame, 5, tiled=tiled), onnx.detect(frame, 5, tiled=tiled)
        assert a.count > 5 and b.per_class == a.per_class
        boxes_a, boxes_b = (np.array(sorted(d.box for d in r.detections)) for r in (a, b))
        assert np.abs(boxes_a - boxes_b).max() <= 1  # pixels
        assert b.dense_score == pytest.approx(a.dense_score, abs=1e-3)
