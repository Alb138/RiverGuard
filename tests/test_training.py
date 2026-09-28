import numpy as np
import pytest

pytest.importorskip("imagehash")
from training.build_dataset import convert_label, group_key, hamming_to_set  # noqa: E402


def test_group_key_groups_roboflow_augmentations():
    assert group_key("IMG_0042_jpg.rf.3f9a1c.jpg") == "IMG_0042_jpg"
    assert group_key("IMG_0042_jpg.rf.77b2e0.jpg") == "IMG_0042_jpg"
    assert group_key("plain_name.jpg") == "plain_name"


def test_convert_label_remaps_and_drops_classes():
    idx_map = {0: 2, 1: None, 2: 0}  # bottle -> plastic_bottle, can -> dropped, cup -> other_plastic
    text = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n2 0.3 0.3 0.1 0.1\n"
    assert convert_label(text, idx_map) == [
        "2 0.500000 0.500000 0.200000 0.200000",
        "0 0.300000 0.300000 0.100000 0.100000",
    ]


def test_convert_label_turns_polygons_into_boxes():
    # triangle spanning x 0.2..0.6, y 0.1..0.5
    assert convert_label("0 0.2 0.1 0.6 0.1 0.4 0.5", {0: 1}) == ["1 0.400000 0.300000 0.400000 0.400000"]


def test_convert_label_skips_degenerate_and_blank_lines():
    assert convert_label("\n0 0.5 0.5 0 0.1\n", {0: 0}) == []


def test_hamming_to_set():
    hashes = np.array([0b1111, 0], dtype=np.uint64)
    assert hamming_to_set(0b0111, hashes) == 1
    assert hamming_to_set(0, np.array([], dtype=np.uint64)) == 64
