"""
Hailo-10H inference backend for the YOLOv8x HEF.

Output of the Hailo HEF is post-NMS by class — we get a list of 80 numpy
arrays (one per COCO class), each shape (num_dets, 5) where columns are
[y_min, x_min, y_max, x_max, score] in normalized 0..1 letterboxed coords.

The HEF takes a 640×640×3 UINT8 NHWC input. Frames need a letterbox
(scale + grey-pad) so non-square camera frames don't distort. After
inference the boxes are mapped back to original-frame pixel coords by
reversing the letterbox.
"""

import logging
import threading
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INPUT_SIZE = 640
_PAD_VALUE = 114  # standard YOLO letterbox grey

# Lazy singletons — VDevice + InferModel can take seconds to bring up.
_vdevice = None
_infer_model = None
_configured = None
_output_buffer_size = 0
_lock = threading.Lock()


def _ensure_loaded(hef_path: str) -> None:
    global _vdevice, _infer_model, _configured, _output_buffer_size
    if _configured is not None:
        return
    # Imported lazily so the module is importable on dev machines without
    # the Hailo runtime installed.
    from hailo_platform import VDevice, FormatType  # type: ignore

    _vdevice = VDevice()
    _infer_model = _vdevice.create_infer_model(hef_path)
    _infer_model.set_batch_size(1)
    _infer_model.input().set_format_type(FormatType.UINT8)
    _infer_model.output().set_format_type(FormatType.FLOAT32)
    _output_buffer_size = _infer_model.output().shape[0]
    _configured = _infer_model.configure().__enter__()
    logger.info(
        "Hailo InferModel configured (input=%s, output_size=%d)",
        _infer_model.input().shape, _output_buffer_size,
    )


def _letterbox(frame: np.ndarray, size: int = _INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio, pad with grey to `size×size`.
    Returns (letterboxed, scale, pad_left, pad_top)."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
    )
    return padded, scale, pad_left, pad_top


def _unletterbox_box(
    y_min: float, x_min: float, y_max: float, x_max: float,
    scale: float, pad_left: int, pad_top: int,
    orig_w: int, orig_h: int,
) -> list[float]:
    """Map a normalized 0..1 letterboxed bbox back to original frame pixel coords."""
    # Letterbox space pixels
    y1 = y_min * _INPUT_SIZE - pad_top
    y2 = y_max * _INPUT_SIZE - pad_top
    x1 = x_min * _INPUT_SIZE - pad_left
    x2 = x_max * _INPUT_SIZE - pad_left
    # Original frame pixels
    x1, x2 = x1 / scale, x2 / scale
    y1, y2 = y1 / scale, y2 / scale
    # Clamp to frame. Cast to Python float — Hailo gives np.float32, which
    # JSON serializers don't recognize and which leaks into bbox lists.
    x1 = max(0.0, min(orig_w - 1.0, float(x1)))
    x2 = max(0.0, min(orig_w - 1.0, float(x2)))
    y1 = max(0.0, min(orig_h - 1.0, float(y1)))
    y2 = max(0.0, min(orig_h - 1.0, float(y2)))
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def run_inference(
    frame: np.ndarray,
    hef_path: str,
    tracked_classes: list[int],
    class_names: dict[int, str],
    min_confidence: float,
) -> list[dict[str, Any]]:
    """Run YOLOv8x on the Hailo and return raw box dicts in the same shape as
    the ultralytics path (class_id, class_name, confidence, bbox)."""
    with _lock:
        _ensure_loaded(hef_path)
        assert _configured is not None  # for type checkers

        orig_h, orig_w = frame.shape[:2]
        # YOLO models trained on COCO expect RGB. RTSP frames from OpenCV are BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        letterboxed, scale, pad_left, pad_top = _letterbox(rgb)

        bindings = _configured.create_bindings()
        bindings.input().set_buffer(letterboxed)
        out_buf = np.zeros(_output_buffer_size, dtype=np.float32)
        bindings.output().set_buffer(out_buf)

        _configured.run([bindings], 5000)
        per_class = bindings.output().get_buffer()  # list of 80 arrays

    tracked = set(tracked_classes)
    results: list[dict[str, Any]] = []
    for cls_id, dets in enumerate(per_class):
        if cls_id not in tracked or dets.size == 0:
            continue
        for det in dets:
            y_min, x_min, y_max, x_max, score = det[0], det[1], det[2], det[3], det[4]
            if score < min_confidence:
                continue
            bbox = _unletterbox_box(
                y_min, x_min, y_max, x_max,
                scale, pad_left, pad_top, orig_w, orig_h,
            )
            results.append({
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, str(cls_id)),
                "confidence": round(float(score), 4),
                "bbox": bbox,
            })
    return results
