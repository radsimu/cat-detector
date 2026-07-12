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
    """Continuously read frames and keep the last pre_roll_seconds in memory.

    Trim by AGE not count: a count cap (e.g. pre_roll * fps) blows up when
    the camera/decoder delivers slowly — 110 frames at 2fps span 55s of
    real time, giving an oversized pre-roll. Drop frames older than
    pre_roll_seconds so the window is always faithful regardless of fps.

    Emits a 10-second telemetry line: how many frames captured, max inter-
    frame gap, average gap. Lets us see when the buffer stalls (RTSP, lock
    contention, CPU starvation) instead of guessing from playback artifacts."""
    logger.info("Recording buffer started (pre-roll=%ds, fps=%d, age-based trim)",
                settings.recording_pre_roll_seconds, _TARGET_FPS)
    window_start = time.time()
    last_ts = window_start
    captured = 0
    grab_failed = 0
    max_gap = 0.0
    while True:
        try:
            frame = await asyncio.get_event_loop().run_in_executor(None, _grab_frame)
            now = time.time()
            if frame is not None:
                gap = now - last_ts
                if gap > max_gap:
                    max_gap = gap
                last_ts = now
                captured += 1
                boxes = _current_boxes()
                cutoff = now - settings.recording_pre_roll_seconds
                with _buffer_lock:
                    _buffer.append((now, frame, boxes))
                    while _buffer and _buffer[0][0] < cutoff:
                        _buffer.popleft()
            else:
                grab_failed += 1
            # Telemetry every 10s
            if now - window_start >= 10.0:
                logger.info(
                    "buffer: captured=%d grab_failed=%d max_gap=%.2fs avg_interval=%.0fms over %.1fs",
                    captured, grab_failed, max_gap,
                    1000.0 * (now - window_start) / max(captured, 1),
                    now - window_start,
                )
                window_start = now
                captured = 0
                grab_failed = 0
                max_gap = 0.0
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


def get_buffer_status() -> dict:
    """Snapshot of the pre-roll buffer for /healthz: frame count and
    age (seconds) of the newest frame. Newest-age > expected = camera or
    decoder stuck."""
    with _buffer_lock:
        if not _buffer:
            return {"frames": 0, "newest_age_s": None}
        newest = _buffer[-1][0]
    return {"frames": len(_buffer), "newest_age_s": round(time.time() - newest, 2)}


async def trigger_recording(detection_id: int | None = None) -> None:
    """Capture a recording: pre-roll snapshot + post-roll frames pulled from
    the shared buffer.

    The buffer loop is the SINGLE producer of frames; the recording task
    just samples from it. Running a parallel grabber here would add a
    third coroutine fighting for _capture_lock, which dropped real-world
    fps from 10 to ~2.4 during recordings."""
    # Capture trigger time NOW — used as Recording.created_at so the UI
    # timestamp matches when the event actually happened, not when ffmpeg
    # finished transcoding (which can be 30-60s later).
    event_time = datetime.utcnow()
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
        # Write at constant _TARGET_FPS (10 fps). Buffer loop may have
        # missed some grabs due to camera-lock contention; if we just write
        # captured frames at constant fps, playback compresses gaps and
        # skips past the moments we most want to see (sprinkle, cat
        # escaping). Instead duplicate each frame for as many target slots
        # as the inter-frame timestamp gap covers, so a 500ms gap shows as
        # 5 still frames at 10fps. Playback matches real wall-clock with
        # visible freezes where the buffer dropped frames.
        # H.264 (avc1) — required for HTML5 <video> playback. OpenCV's mp4v
        # writes MPEG-4 Part 2 which browsers don't decode. The pip
        # opencv-python wheels on aarch64 ship without x264 due to licensing,
        # so we fall back to mp4v + an ffmpeg transcode pass below.
        used_fallback = False
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(file_path), fourcc, _TARGET_FPS, (width, height))
        if not writer.isOpened():
            logger.info("avc1 writer not available — using mp4v + ffmpeg transcode")
            used_fallback = True
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(file_path), fourcc, _TARGET_FPS, (width, height))
        # cv2.VideoWriter.write() is a blocking C call; encoding hundreds of
        # frames inline on the event loop stalls the buffer + detection
        # loops. Push the whole annotate-and-write pass to the executor.
        # Stats for logging
        total_slots_written = 0
        max_gap_seconds = 0.0

        def _stamp_pi_clock(img: np.ndarray, ts: float) -> None:
            """Burn Pi-clock timestamp (yellow, top-left) on the frame.
            Authoritative timeline — camera-embedded overlay can drift.
            Includes detection_id so the frame is unmistakable in disputes."""
            label = datetime.fromtimestamp(ts).strftime("Pi %H:%M:%S.") + f"{int((ts % 1) * 1000):03d}"
            if detection_id is not None:
                label += f"  det#{detection_id}"
            else:
                label += "  TEST"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)

        def _write_all() -> None:
            nonlocal total_slots_written, max_gap_seconds
            try:
                for i, (ts, frame, boxes) in enumerate(all_frames):
                    annotated = _draw_boxes(frame, boxes) if boxes else frame.copy()
                    if polygon is not None:
                        _draw_polygon(annotated, polygon)
                    _stamp_pi_clock(annotated, ts)
                    # How many target-fps slots until the next captured
                    # frame? Last frame: 1 slot.
                    if i + 1 < len(all_frames):
                        next_ts = all_frames[i + 1][0]
                        gap = next_ts - ts
                        if gap > max_gap_seconds:
                            max_gap_seconds = gap
                    else:
                        next_ts = ts + _FRAME_INTERVAL
                    slots = max(1, round((next_ts - ts) / _FRAME_INTERVAL))
                    total_slots_written += slots
                    for _ in range(slots):
                        writer.write(annotated)
            finally:
                writer.release()
        await asyncio.get_event_loop().run_in_executor(None, _write_all)

        if used_fallback:
            await asyncio.get_event_loop().run_in_executor(
                None, _transcode_to_h264, file_path
            )

        file_size = file_path.stat().st_size
        duration = all_frames[-1][0] - all_frames[0][0]

        with SessionLocal() as db:
            recording = Recording(
                detection_id=detection_id,
                created_at=event_time,  # event time, not write-finish time
                file_path=str(file_path),
                duration_seconds=round(duration, 2),
                file_size_bytes=file_size,
            )
            db.add(recording)
            db.commit()

        logger.info(
            "Recording stats: captured_frames=%d span=%.1fs slots_written=%d max_gap=%.2fs",
            len(all_frames), duration, total_slots_written, max_gap_seconds,
        )
        logger.info("Recording saved: %s (%.1fs, %d bytes)",
                    file_path, duration, file_size)

        # Enforce retention cap
        _enforce_max_recordings()

    except asyncio.CancelledError:
        # Shutdown / autoheal cancelled mid-recording. Log so we know which
        # detections lost their recording instead of silently dropping.
        logger.warning("Recording for detection=%s CANCELLED mid-flight",
                       detection_id)
        raise
    except Exception:
        logger.exception("Recording failed for detection=%s", detection_id)


def _transcode_to_h264(path: Path) -> None:
    """In-place transcode an MPEG-4 Part 2 (mp4v) file to H.264 so HTML5
    <video> can play it. Adds +faststart so the moov atom is at the start
    of the file, letting the browser render before the full file loads."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not on PATH — recording stays as mp4v and won't play in browsers")
        return
    tmp = path.with_name(path.stem + ".transcode.mp4")
    try:
        # -threads 2 caps libx264 to 2 of 4 cores so systemd-watchdog can
        # still feed /dev/watchdog. nice/ionice deprioritize this CPU- and
        # I/O-heavy job. Without these, the Pi was watchdog-resetting under
        # transcode load. -preset ultrafast trades some compression for
        # 30-50% less CPU than veryfast — fine for short event clips.
        result = subprocess.run(
            [
                "nice", "-n", "10", "ionice", "-c", "2", "-n", "7",
                "ffmpeg", "-y", "-loglevel", "error",
                "-threads", "2",
                "-i", str(path),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-threads", "2",
                "-movflags", "+faststart",
                str(tmp),
            ],
            check=False, capture_output=True, timeout=180,
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
