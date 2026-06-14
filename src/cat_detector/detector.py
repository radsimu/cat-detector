"""
Cat/human detection using YOLOv8.

YOLO COCO class IDs used here:
  0  = person
  15 = cat
  16 = dog (safety net — cats can be misclassified as dogs)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cat_detector.config import settings

# IoU threshold above which two boxes of different classes are considered
# the same object — keep only the highest-confidence one.
_CROSS_CLASS_IOU_THRESHOLD = 0.7

# BGR colors for drawing boxes by class
_CLASS_COLORS = {
    0: (0, 165, 255),    # person — orange
    15: (0, 200, 0),     # cat — green
    16: (180, 100, 180), # dog — purple
}

# COCO class indices
_PERSON_CLASS = 0
_CAT_CLASS = 15
_DOG_CLASS = 16

# Classes passed to YOLO so only these are detected — keeps annotated frames
# free of chair/phone/etc. boxes and makes the actual animal boxes visible.
_TRACKED_CLASSES = [_PERSON_CLASS, _CAT_CLASS, _DOG_CLASS]

# Human-readable class names for the classes we care about
_CLASS_NAMES = {
    _PERSON_CLASS: "person",
    _CAT_CLASS: "cat",
    _DOG_CLASS: "dog",
}


@dataclass
class DetectionResult:
    cat_count: int
    dog_count: int
    human_count: int
    max_cat_confidence: float
    # Highest confidence across cat/dog in this frame. Used by the
    # consecutive-frame gate so a cat misclassified as dog mid-sequence
    # doesn't break the streak.
    max_animal_confidence: float
    annotated_frame: np.ndarray  # BGR frame with bounding boxes drawn
    boxes: list[dict[str, Any]] = field(default_factory=list)  # per-box details


_model: Any | None = None


def get_model() -> Any:
    """Lazy-load the ultralytics YOLO model. Imported lazily so the
    Hailo-only deployment can skip torch/ultralytics if it wants to."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(settings.detection_model)
    return _model


def _run_inference_ultralytics(frame: np.ndarray) -> list[dict[str, Any]]:
    """Run YOLO on CPU/GPU via ultralytics and return raw box dicts."""
    model = get_model()
    results = model(
        frame,
        conf=settings.detection_confidence,
        classes=_TRACKED_CLASSES,
        verbose=False,
    )
    result = results[0]
    raw: list[dict[str, Any]] = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = [round(c, 1) for c in box.xyxy[0].tolist()]
        raw.append({
            "class_id": cls,
            "class_name": _CLASS_NAMES.get(cls, str(cls)),
            "confidence": round(conf, 4),
            "bbox": xyxy,
        })
    return raw


def _run_inference(frame: np.ndarray) -> list[dict[str, Any]]:
    """Dispatch to the configured backend. Both backends return the same
    list-of-dicts shape so the rest of detect() stays backend-agnostic."""
    if settings.detection_backend == "hailo":
        from cat_detector import detector_hailo
        return detector_hailo.run_inference(
            frame,
            hef_path=settings.detection_hef_path,
            tracked_classes=_TRACKED_CLASSES,
            class_names=_CLASS_NAMES,
            min_confidence=settings.detection_confidence,
        )
    return _run_inference_ultralytics(frame)


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cross_class_nms(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suppress boxes across classes when they overlap heavily — keeps only
    the highest-confidence box per cluster. YOLO's built-in NMS runs per
    class; this handles the case where the model emits both 'person' and
    'cat' predictions on the same object."""
    sorted_boxes = sorted(raw, key=lambda b: -b["confidence"])
    kept: list[dict[str, Any]] = []
    for box in sorted_boxes:
        if any(_iou(box["bbox"], k["bbox"]) > _CROSS_CLASS_IOU_THRESHOLD for k in kept):
            continue
        kept.append(box)
    return kept


def _parse_roi_polygon(width: int, height: int) -> np.ndarray | None:
    """Parse settings.roi_polygon (normalized 0..1 vertices) into a pixel-space
    array suitable for cv2.pointPolygonTest / cv2.polylines. Returns None if
    no mask is configured or the JSON is unusable."""
    raw = settings.roi_polygon
    if not raw or not isinstance(raw, str):
        return None
    try:
        pts = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(pts, list) or len(pts) < 3:
        return None
    arr = np.array(
        [(int(round(x * width)), int(round(y * height))) for x, y in pts],
        dtype=np.int32,
    )
    return arr


def _box_inside_polygon(bbox: list[float], polygon: np.ndarray) -> bool:
    """True if the box center is inside (or on the edge of) the polygon."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0


def _draw_polygon(frame: np.ndarray, polygon: np.ndarray) -> None:
    """Draw the ROI polygon outline on the frame in-place."""
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)


def _draw_boxes(frame: np.ndarray, boxes: list[dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of the frame."""
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b["bbox"])
        color = _CLASS_COLORS.get(b["class_id"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{b['class_name']} {int(b['confidence'] * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def detect(frame: np.ndarray) -> DetectionResult:
    """Run detection on a single BGR frame and return structured results.

    Only tracked classes (person, cat, dog) are detected. After YOLO's
    per-class NMS, we also run cross-class NMS to suppress overlapping
    boxes of different classes (e.g. the model emitting both 'person'
    and 'cat' on the same object).
    """
    raw_boxes = _run_inference(frame)
    # Hailo path returns all 80 classes; filter early. (Ultralytics path
    # already restricts via the `classes` arg, so this is a no-op there.)
    raw_boxes = [b for b in raw_boxes if b["class_id"] in _TRACKED_CLASSES]
    boxes = _cross_class_nms(raw_boxes)

    # Region-of-interest filter: drop any box whose center is outside the
    # configured polygon (if any). Persons outside the ROI are also dropped
    # — the human-guard only cares about humans in the protected zone.
    h, w = frame.shape[:2]
    polygon = _parse_roi_polygon(w, h)
    if polygon is not None:
        boxes = [b for b in boxes if _box_inside_polygon(b["bbox"], polygon)]

    cat_confidences: list[float] = []
    animal_confidences: list[float] = []  # cat ∪ dog
    dog_count = 0
    human_count = 0

    for b in boxes:
        cls = b["class_id"]
        if cls == _CAT_CLASS:
            cat_confidences.append(b["confidence"])
            animal_confidences.append(b["confidence"])
        elif cls == _DOG_CLASS:
            dog_count += 1
            animal_confidences.append(b["confidence"])
        elif cls == _PERSON_CLASS:
            human_count += 1

    annotated = _draw_boxes(frame, boxes)
    if polygon is not None:
        _draw_polygon(annotated, polygon)

    return DetectionResult(
        cat_count=len(cat_confidences),
        dog_count=dog_count,
        human_count=human_count,
        max_cat_confidence=max(cat_confidences, default=0.0),
        max_animal_confidence=max(animal_confidences, default=0.0),
        annotated_frame=annotated,
        boxes=boxes,
    )


def should_trigger_webhook(result: DetectionResult) -> tuple[bool, str | None]:
    """
    Returns (trigger, skip_reason).
    Trigger when cats (or dogs, as a misclassification safety net) are
    present and no humans are present.
    """
    animal_count = result.cat_count + result.dog_count
    if animal_count == 0:
        return False, "no_cats"
    if result.human_count > 0:
        return False, "human_present"
    return True, None


def save_frame(frame: np.ndarray, frames_dir: str, filename: str) -> Path:
    out_dir = Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    cv2.imwrite(str(path), frame)
    return path
