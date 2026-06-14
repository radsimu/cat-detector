"""
Video recording with pre-roll buffer.

A background task continuously reads frames from the shared camera capture
and keeps the last N seconds in memory. When trigger_recording() is called,
the buffer is flushed to an MP4 and recording continues for the post-roll
window.
"""

import asyncio
import logging
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cat_detector.camera import _grab_frame
from cat_detector.config import settings
from cat_detector.database import SessionLocal
from cat_detector.detector import _draw_boxes, _draw_polygon, _parse_roi_polygon
from cat_detector.models import Recording

logger = logging.getLogger(__name__)

# Target frame rate for recordings. Low enough to keep buffer memory small,
# high enough for smooth playback.
_TARGET_FPS = 10
_FRAME_INTERVAL = 1.0 / _TARGET_FPS

# Rolling buffer: (timestamp, frame, boxes). Max length = pre_roll * fps + small safety margin.
# Boxes are snapshotted from the stream-overlay cache at frame-add time so each
# recorded frame can be drawn with the boxes that were "current" at its instant.
_buffer: deque[tuple[float, np.ndarray, list]] = deque()
_buffer_lock = threading.Lock()
_buffer_task: asyncio.Task | None = None


def _current_boxes() -> list:
    """Snapshot the latest cached detection boxes from the live-stream
    overlay loop. Returns [] if the cache is stale (>3s old)."""
    # Late import to avoid a circular dependency between recording and stream
    from cat_detector.routers.stream import (
        _latest_boxes, _latest_boxes_at, _latest_boxes_lock,
    )
    with _latest_boxes_lock:
        if (time.time() - _latest_boxes_at) < 3.0:
            return list(_latest_boxes)
    return []


async def _fill_buffer_loop() -> None:
    """Continuously read frames and keep the last pre_roll_seconds in memory."""
    max_frames = int(settings.recording_pre_roll_seconds * _TARGET_FPS) + _TARGET_FPS
    logger.info("Recording buffer started (pre-roll=%ds, fps=%d)",
                settings.recording_pre_roll_seconds, _TARGET_FPS)
    while True:
        try:
            frame = await asyncio.get_event_loop().run_in_executor(None, _grab_frame)
            if frame is not None:
                boxes = _current_boxes()
                with _buffer_lock:
                    _buffer.append((time.time(), frame, boxes))
                    while len(_buffer) > max_frames:
                        _buffer.popleft()
            await asyncio.sleep(_FRAME_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Buffer loop error — continuing")
            await asyncio.sleep(1.0)


def start_buffer() -> None:
    """Start the pre-roll frame buffer task."""
    global _buffer_task
    if _buffer_task is None or _buffer_task.done():
        _buffer_task = asyncio.create_task(_fill_buffer_loop())


def stop_buffer() -> None:
    """Stop the buffer task (called on shutdown)."""
    if _buffer_task is not None and not _buffer_task.done():
        _buffer_task.cancel()


async def trigger_recording(detection_id: int | None = None) -> None:
    """Capture a recording: pre-roll snapshot + post-roll frames pulled from
    the shared buffer.

    The buffer loop is the SINGLE producer of frames; the recording task
    just samples from it. Running a parallel grabber here would add a
    third coroutine fighting for _capture_lock, which dropped real-world
    fps from 10 to ~2.4 during recordings."""
    try:
        pre_roll = settings.recording_pre_roll_seconds
        total = settings.recording_total_seconds
        post_roll = max(total - pre_roll, 1)

        # Snapshot pre-roll
        with _buffer_lock:
            pre_frames = list(_buffer)
        last_ts = pre_frames[-1][0] if pre_frames else 0.0

        # Post-roll: poll the same shared buffer for new frames as the
        # buffer loop appends them. Half the frame interval so we don't
        # miss the boundary between two appends.
        post_frames: list[tuple[float, np.ndarray, list]] = []
        end_time = time.time() + post_roll
        while time.time() < end_time:
            await asyncio.sleep(_FRAME_INTERVAL / 2)
            with _buffer_lock:
                new = [f for f in _buffer if f[0] > last_ts]
            if new:
                post_frames.extend(new)
                last_ts = new[-1][0]

        all_frames = pre_frames + post_frames
        if not all_frames:
            logger.warning("No frames captured for recording")
            return

        # Write MP4
        out_dir = Path(settings.recordings_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_detection{detection_id or 'manual'}.mp4"
        file_path = out_dir / filename

        _, first_frame, _ = all_frames[0]
        height, width = first_frame.shape[:2]
        polygon = _parse_roi_polygon(width, height)
        # Real capture rate is below _TARGET_FPS because the buffer loop
        # shares the camera lock with the detection loop. Tagging the file
        # at _TARGET_FPS made playback slow-motion (e.g. 5 actual fps tagged
        # as 10 → 2× real-time stretch). Compute the actual fps from the
        # frame timestamps so playback matches wall-clock.
        span = max(all_frames[-1][0] - all_frames[0][0], 0.001)
        actual_fps = max(min((len(all_frames) - 1) / span, _TARGET_FPS), 1.0)
        # H.264 (avc1) — required for HTML5 <video> playback. OpenCV's mp4v
        # writes MPEG-4 Part 2 which browsers don't decode. The pip
        # opencv-python wheels on aarch64 ship without x264 due to licensing,
        # so we fall back to mp4v + an ffmpeg transcode pass below.
        used_fallback = False
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(file_path), fourcc, actual_fps, (width, height))
        if not writer.isOpened():
            logger.info("avc1 writer not available — using mp4v + ffmpeg transcode")
            used_fallback = True
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(file_path), fourcc, actual_fps, (width, height))
        try:
            for _, frame, boxes in all_frames:
                annotated = _draw_boxes(frame, boxes) if boxes else frame.copy()
                if polygon is not None:
                    _draw_polygon(annotated, polygon)
                writer.write(annotated)
        finally:
            writer.release()

        if used_fallback:
            await asyncio.get_event_loop().run_in_executor(
                None, _transcode_to_h264, file_path
            )

        file_size = file_path.stat().st_size
        duration = all_frames[-1][0] - all_frames[0][0]

        with SessionLocal() as db:
            recording = Recording(
                detection_id=detection_id,
                file_path=str(file_path),
                duration_seconds=round(duration, 2),
                file_size_bytes=file_size,
            )
            db.add(recording)
            db.commit()

        logger.info("Recording saved: %s (%.1fs, %d bytes)",
                    file_path, duration, file_size)

        # Enforce retention cap
        _enforce_max_recordings()

    except Exception:
        logger.exception("Recording failed")


def _transcode_to_h264(path: Path) -> None:
    """In-place transcode an MPEG-4 Part 2 (mp4v) file to H.264 so HTML5
    <video> can play it. Adds +faststart so the moov atom is at the start
    of the file, letting the browser render before the full file loads."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not on PATH — recording stays as mp4v and won't play in browsers")
        return
    tmp = path.with_name(path.stem + ".transcode.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-movflags", "+faststart",
                str(tmp),
            ],
            check=False, capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                "ffmpeg transcode failed (rc=%d): %s",
                result.returncode, result.stderr.decode(errors="replace")[:500],
            )
            tmp.unlink(missing_ok=True)
            return
        tmp.replace(path)
        logger.info("Transcoded %s to H.264", path.name)
    except Exception:
        logger.exception("ffmpeg transcode error")
        tmp.unlink(missing_ok=True)


def _enforce_max_recordings() -> None:
    """Delete oldest recordings when count exceeds configured cap."""
    with SessionLocal() as db:
        count = db.query(Recording).count()
        if count <= settings.recordings_max_count:
            return
        excess = count - settings.recordings_max_count
        oldest = (
            db.query(Recording)
            .order_by(Recording.created_at.asc())
            .limit(excess)
            .all()
        )
        for rec in oldest:
            try:
                path = Path(rec.file_path)
                if path.exists():
                    path.unlink()
            except Exception:
                logger.exception("Failed to delete file %s", rec.file_path)
            db.delete(rec)
        db.commit()
        logger.info("Deleted %d old recordings to enforce cap of %d",
                    excess, settings.recordings_max_count)
