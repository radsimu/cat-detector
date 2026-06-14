"""
Unit tests for detection logic — US-2 · Cat Detection with Human Guard (#6105).
These tests do NOT require a camera or GPU — they use synthetic numpy frames.
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub ultralytics before importing detector (torchvision needs _lzma on 3.14)
_ultra_stub = types.ModuleType("ultralytics")
_ultra_stub.YOLO = MagicMock()
sys.modules.setdefault("ultralytics", _ultra_stub)

from cat_detector.detector import (  # noqa: E402
    DetectionResult,
    _cross_class_nms,
    _iou,
    should_trigger_webhook,
)


@pytest.fixture(autouse=True)
def _no_roi_mask():
    """Tests assume the whole frame is in-bounds. Save/restore the runtime
    setting so a user-configured polygon can't filter test fixture boxes."""
    from cat_detector.config import settings
    saved = settings.roi_polygon
    settings.roi_polygon = ""
    yield
    settings.roi_polygon = saved


def _result(cats: int, humans: int, confidence: float = 0.9, boxes=None,
            dogs: int = 0) -> DetectionResult:
    return DetectionResult(
        cat_count=cats,
        dog_count=dogs,
        human_count=humans,
        max_cat_confidence=confidence,
        max_animal_confidence=confidence,
        annotated_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        boxes=boxes or [],
    )


# ---------------------------------------------------------------------------
# Webhook trigger logic (ACs 2 & 3)
# ---------------------------------------------------------------------------


class TestWebhookTriggerLogic:
    def test_trigger_when_cat_only(self):
        trigger, reason = should_trigger_webhook(_result(cats=1, humans=0))
        assert trigger is True
        assert reason is None

    def test_trigger_when_multiple_cats(self):
        trigger, reason = should_trigger_webhook(_result(cats=3, humans=0))
        assert trigger is True

    def test_no_trigger_when_no_cats(self):
        trigger, reason = should_trigger_webhook(_result(cats=0, humans=0))
        assert trigger is False
        assert reason == "no_cats"

    def test_no_trigger_when_human_present(self):
        trigger, reason = should_trigger_webhook(_result(cats=2, humans=1))
        assert trigger is False
        assert reason == "human_present"

    def test_no_trigger_when_human_only(self):
        trigger, reason = should_trigger_webhook(_result(cats=0, humans=1))
        assert trigger is False
        assert reason == "no_cats"


# ---------------------------------------------------------------------------
# DetectionResult.boxes field (AC5)
# ---------------------------------------------------------------------------


class TestDetectionResultBoxes:
    def test_boxes_default_empty(self):
        r = _result(cats=0, humans=0)
        assert r.boxes == []

    def test_boxes_preserved(self):
        boxes = [
            {"class_id": 15, "class_name": "cat", "confidence": 0.92, "bbox": [10, 20, 200, 300]},
            {"class_id": 0, "class_name": "person", "confidence": 0.85, "bbox": [50, 60, 400, 500]},
        ]
        r = _result(cats=1, humans=1, boxes=boxes)
        assert len(r.boxes) == 2
        assert r.boxes[0]["class_name"] == "cat"
        assert r.boxes[1]["class_name"] == "person"

    def test_boxes_serializable_to_json(self):
        boxes = [
            {"class_id": 15, "class_name": "cat", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]},
        ]
        r = _result(cats=1, humans=0, boxes=boxes)
        serialized = json.dumps(r.boxes)
        deserialized = json.loads(serialized)
        assert deserialized == boxes


# ---------------------------------------------------------------------------
# detect() function — YOLO integration (ACs 1 & 5)
# ---------------------------------------------------------------------------


class TestDetectFunction:
    """Tests for the detect() function with mocked YOLO model."""

    def _make_mock_box(self, cls_id: int, conf: float, xyxy: list[float]):
        box = MagicMock()
        box.cls = MagicMock()
        box.cls.__getitem__ = lambda self, idx: cls_id
        box.conf = MagicMock()
        box.conf.__getitem__ = lambda self, idx: conf
        box.xyxy = MagicMock()
        box.xyxy.__getitem__ = lambda self, idx: MagicMock(tolist=lambda: xyxy)
        return box

    def _make_mock_result(self, boxes_data):
        mock_result = MagicMock()
        mock_boxes = []
        for cls_id, conf, xyxy in boxes_data:
            mock_boxes.append(self._make_mock_box(cls_id, conf, xyxy))
        mock_result.boxes = mock_boxes
        mock_result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        return mock_result

    def test_detect_cat_only(self):
        from cat_detector.detector import detect

        mock_model = MagicMock()
        result_obj = self._make_mock_result([
            (15, 0.95, [10.0, 20.0, 200.0, 300.0]),  # cat
        ])
        mock_model.return_value = [result_obj]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("cat_detector.detector.get_model", return_value=mock_model):
            result = detect(frame)

        assert result.cat_count == 1
        assert result.human_count == 0
        assert result.max_cat_confidence == 0.95
        assert len(result.boxes) == 1
        assert result.boxes[0]["class_id"] == 15
        assert result.boxes[0]["class_name"] == "cat"
        assert result.boxes[0]["confidence"] == 0.95
        assert result.boxes[0]["bbox"] == [10.0, 20.0, 200.0, 300.0]

    def test_detect_cat_and_human(self):
        from cat_detector.detector import detect

        mock_model = MagicMock()
        result_obj = self._make_mock_result([
            (15, 0.92, [10.0, 20.0, 200.0, 300.0]),   # cat
            (0, 0.88, [300.0, 50.0, 500.0, 400.0]),    # person
        ])
        mock_model.return_value = [result_obj]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("cat_detector.detector.get_model", return_value=mock_model):
            result = detect(frame)

        assert result.cat_count == 1
        assert result.human_count == 1
        assert len(result.boxes) == 2
        cat_box = next(b for b in result.boxes if b["class_name"] == "cat")
        person_box = next(b for b in result.boxes if b["class_name"] == "person")
        assert cat_box["confidence"] == 0.92
        assert person_box["confidence"] == 0.88

    def test_detect_tracks_dog_and_cat_separately(self):
        from cat_detector.detector import detect

        mock_model = MagicMock()
        # Non-overlapping cat and dog boxes — cross-class NMS keeps both
        result_obj = self._make_mock_result([
            (15, 0.9, [10.0, 20.0, 200.0, 300.0]),    # cat
            (16, 0.8, [300.0, 400.0, 500.0, 600.0]),  # dog (far away)
        ])
        mock_model.return_value = [result_obj]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("cat_detector.detector.get_model", return_value=mock_model):
            result = detect(frame)

        assert result.cat_count == 1
        assert result.dog_count == 1
        assert result.human_count == 0
        assert len(result.boxes) == 2

    def test_detect_empty_frame(self):
        from cat_detector.detector import detect

        mock_model = MagicMock()
        result_obj = self._make_mock_result([])
        mock_model.return_value = [result_obj]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("cat_detector.detector.get_model", return_value=mock_model):
            result = detect(frame)

        assert result.cat_count == 0
        assert result.human_count == 0
        assert result.max_cat_confidence == 0.0
        assert result.boxes == []

    def test_detect_uses_configured_confidence(self):
        from cat_detector.detector import detect

        mock_model = MagicMock()
        result_obj = self._make_mock_result([])
        mock_model.return_value = [result_obj]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with (
            patch("cat_detector.detector.get_model", return_value=mock_model),
            patch("cat_detector.detector.settings") as mock_settings,
        ):
            mock_settings.detection_confidence = 0.75
            detect(frame)
            call_kwargs = mock_model.call_args.kwargs
            assert call_kwargs["conf"] == 0.75
            assert call_kwargs["verbose"] is False
            assert call_kwargs["classes"] == [0, 15, 16]


# ---------------------------------------------------------------------------
# Cross-class NMS (suppresses overlapping boxes of different classes)
# ---------------------------------------------------------------------------


class TestCrossClassNMS:
    def test_iou_identical_boxes(self):
        assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_iou_no_overlap(self):
        assert _iou([0, 0, 10, 10], [100, 100, 200, 200]) == 0.0

    def test_iou_half_overlap(self):
        # Two 10x10 boxes sharing 5x10 strip — inter=50, areas=100 each, union=150
        assert _iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)

    def test_nms_keeps_both_when_non_overlapping(self):
        boxes = [
            {"class_id": 15, "class_name": "cat", "confidence": 0.8, "bbox": [0, 0, 10, 10]},
            {"class_id": 16, "class_name": "dog", "confidence": 0.7, "bbox": [100, 100, 200, 200]},
        ]
        result = _cross_class_nms(boxes)
        assert len(result) == 2

    def test_nms_suppresses_lower_confidence_duplicate(self):
        # Same bbox, two classes — should keep the higher-confidence one
        boxes = [
            {"class_id": 0, "class_name": "person", "confidence": 0.45, "bbox": [10, 10, 100, 200]},
            {"class_id": 15, "class_name": "cat", "confidence": 0.37, "bbox": [10, 10, 100, 200]},
        ]
        result = _cross_class_nms(boxes)
        assert len(result) == 1
        assert result[0]["class_name"] == "person"

    def test_nms_keeps_higher_when_cat_wins(self):
        boxes = [
            {"class_id": 0, "class_name": "person", "confidence": 0.40, "bbox": [10, 10, 100, 200]},
            {"class_id": 15, "class_name": "cat", "confidence": 0.85, "bbox": [10, 10, 100, 200]},
        ]
        result = _cross_class_nms(boxes)
        assert len(result) == 1
        assert result[0]["class_name"] == "cat"
