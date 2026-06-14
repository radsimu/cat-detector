"""
MJPEG live streaming endpoint.

Streams raw RTSP frames from the shared persistent capture, re-encoded
as JPEG. The main camera loop publishes its detection result via
publish_boxes(); streamed frames overlay the cached boxes quickly
(no per-frame YOLO cost on the hot path). Result: near-real-time live
view with slightly stale but visible detection boxes — and a single
inference at a time, which matters on Pi-class CPU.
"""

import asyncio
import base64
import logging
import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from cat_detector.camera import _grab_latest_frame
from cat_detector.detector import _draw_boxes, detect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

# Target frame rate for the stream (frames per second)
_STREAM_FPS = 10
_FRAME_INTERVAL = 1.0 / _STREAM_FPS

# Max width for streamed frames; larger frames get downscaled to reduce
# JPEG encode cost and network bandwidth. Keeps aspect ratio.
_STREAM_MAX_WIDTH = 1280

# Shared cache of latest detection boxes for overlay on streamed frames.
# Populated by the camera loop via publish_boxes() — single source of truth.
_latest_boxes: list = []
_latest_boxes_at: float = 0.0  # unix ts; stale boxes (>3s) are ignored
_latest_boxes_lock = threading.Lock()


def publish_boxes(boxes: list) -> None:
    """Update the shared overlay cache from the camera loop. Called after
    every detection so the live stream and recording buffer can draw boxes
    without running their own YOLO pass."""
    global _latest_boxes, _latest_boxes_at
    with _latest_boxes_lock:
        _latest_boxes = boxes
        _latest_boxes_at = time.time()


def _downscale(frame: np.ndarray) -> np.ndarray:
    """Resize to at most _STREAM_MAX_WIDTH, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= _STREAM_MAX_WIDTH:
        return frame
    scale = _STREAM_MAX_WIDTH / w
    return cv2.resize(frame, (_STREAM_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def _encode_frame(frame: np.ndarray) -> bytes:
    """Encode a BGR frame as JPEG bytes."""
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not success:
        return b""
    return buf.tobytes()


def _placeholder_frame() -> bytes:
    """Generate a dark frame with 'No Camera' text as JPEG."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame, "No Camera Signal", (120, 250),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2,
    )
    return _encode_frame(frame)


async def _generate_frames():
    """Yield MJPEG frames: grab → overlay cached boxes → downscale → encode → yield.
    Detection runs in the main camera loop and publishes here via publish_boxes;
    this hot path only draws the cached boxes (no YOLO cost).
    Uses _grab_latest_frame to drain any buffered frames so the client always gets 'now'."""
    while True:
        frame = await asyncio.get_event_loop().run_in_executor(None, _grab_latest_frame)

        if frame is None:
            placeholder = _placeholder_frame()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
            )
            await asyncio.sleep(1.0)
            continue

        # Overlay cached boxes (ignore if older than 3s — avoids drawing ghost boxes
        # from long-gone subjects)
        with _latest_boxes_lock:
            boxes = _latest_boxes if (time.time() - _latest_boxes_at) < 3.0 else []
        if boxes:
            frame = _draw_boxes(frame, boxes)

        small = _downscale(frame)
        jpeg = _encode_frame(small)
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

        await asyncio.sleep(_FRAME_INTERVAL)


@router.get("/api/stream")
async def video_stream():
    """MJPEG video stream with real-time detection overlays."""
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/api/test-detect")
async def test_detect(file: UploadFile = File(...)) -> dict:
    """Run detection on a single uploaded JPEG and return counts + an
    annotated JPEG. Used by the 'test with my webcam' debug panel to
    confirm the model is actually firing on cats it sees."""
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    result = await asyncio.get_event_loop().run_in_executor(None, detect, frame)
    ok, jpeg = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encode failed")
    return {
        "cat_count": result.cat_count,
        "dog_count": result.dog_count,
        "human_count": result.human_count,
        "max_cat_confidence": result.max_cat_confidence,
        "max_animal_confidence": result.max_animal_confidence,
        "boxes": result.boxes,
        "annotated_jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
    }


@router.get("/api/snapshot")
async def snapshot():
    """Single JPEG of the current camera frame, raw (no detection overlay).
    Used by the ROI polygon editor."""
    frame = await asyncio.get_event_loop().run_in_executor(None, _grab_latest_frame)
    if frame is None:
        raise HTTPException(status_code=503, detail="No camera frame available")
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encode failed")
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")
