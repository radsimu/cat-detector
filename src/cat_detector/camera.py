"""
Background service: reads frames from the camera stream on an interval,
runs detection, persists results, and fires the webhook when appropriate.

Frame access is serialized through a threading lock so the camera loop,
MJPEG stream, and report-missed endpoint never call cap.read() concurrently.
"""

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np

from cat_detector.config import settings
from cat_detector.database import SessionLocal
from cat_detector.detector import detect, save_frame, should_trigger_webhook
from cat_detector.models import Detection
from cat_detector import homeassistant
from cat_detector.webhook import call_webhook, call_webhook_off

logger = logging.getLogger(__name__)

_last_webhook_time: datetime | None = None
_consecutive_triggers: int = 0  # frames in a row with a qualifying detection

# Persistent camera capture — kept open to avoid repeated open/close cycles
# which cause the Mac camera indicator to flash and reset auto-exposure.
_capture: cv2.VideoCapture | None = None
_capture_lock = threading.Lock()

# Camera status for UI indicator. Updated by the camera loop.
camera_status: str = "unknown"  # "connected" | "disconnected" | "retrying"


def get_camera_status() -> str:
    return camera_status


def _get_capture() -> cv2.VideoCapture | None:
    """Get or create a persistent camera capture. Caller must hold _capture_lock."""
    global _capture
    if _capture is not None and _capture.isOpened():
        return _capture

    source = settings.camera_url
    is_rtsp = isinstance(source, str) and source.startswith(("rtsp://", "rtsps://"))

    # For RTSP, set FFmpeg low-latency options via env var BEFORE creating capture.
    # These flags flush the demux buffer aggressively so we get the latest frame
    # instead of queued-up old ones.
    if is_rtsp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|reorder_queue_size;0"
        )

    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    _capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG if is_rtsp else cv2.CAP_ANY)
    if not _capture.isOpened():
        logger.error("Cannot open camera source: %s", settings.camera_url)
        _capture = None
        return None

    # Shrink the internal buffer to 1 frame so cap.read() always returns the latest
    _capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.info("Camera opened: %s (rtsp=%s)", settings.camera_url, is_rtsp)
    return _capture


def _release_capture() -> None:
    """Release the persistent camera capture."""
    global _capture
    with _capture_lock:
        if _capture is not None:
            _capture.release()
            _capture = None
            logger.info("Camera released")


def _grab_frame() -> np.ndarray | None:
    """Grab a single frame, serialized through a lock so concurrent
    consumers (camera loop, stream, report-missed) never race on cap.read()."""
    with _capture_lock:
        cap = _get_capture()
        if cap is None:
            return None
        ret, frame = cap.read()
        if not ret:
            _capture_release_inner()
            return None
        return frame


def _grab_latest_frame() -> np.ndarray | None:
    """Like _grab_frame, but drains any buffered frames first so the returned
    frame is as close to 'now' as possible. Used by the live stream to skip
    backlogged frames when encoding falls behind."""
    with _capture_lock:
        cap = _get_capture()
        if cap is None:
            return None
        # Flush up to a few buffered frames (grab is fast — decode skipped)
        for _ in range(5):
            if not cap.grab():
                break
        ret, frame = cap.retrieve()
        if not ret:
            # fall back to regular read
            ret, frame = cap.read()
        if not ret:
            _capture_release_inner()
            return None
        return frame


def _capture_release_inner() -> None:
    """Release capture without acquiring the lock (caller already holds it)."""
    global _capture
    if _capture is not None:
        _capture.release()
        _capture = None
        logger.warning("Camera disconnected — will retry on next grab")


_MAX_DETECTIONS_JSON_BYTES = 32_768  # 32KB — generous cap for ~100 boxes


def _serialize_boxes(boxes: list) -> str | None:
    """Serialize boxes to JSON with a size cap to prevent unbounded storage."""
    if not boxes:
        return None
    data = json.dumps(boxes)
    if len(data) > _MAX_DETECTIONS_JSON_BYTES:
        logger.warning("detections_json too large (%d bytes), truncating to first 50 boxes", len(data))
        data = json.dumps(boxes[:50])
    return data


async def _process_frame(frame: np.ndarray) -> None:
    global _last_webhook_time, _consecutive_triggers

    # YOLOv8x on Pi-class CPU takes seconds — must run off the event loop or
    # the MJPEG stream and HTTP handlers freeze for the duration.
    result = await asyncio.get_event_loop().run_in_executor(None, detect, frame)

    # Publish boxes to the shared overlay cache so the live stream and the
    # recording buffer can draw them without running a second YOLO pass.
    from cat_detector.routers.stream import publish_boxes
    publish_boxes(result.boxes)

    animal_count = result.cat_count + result.dog_count
    if animal_count == 0 and result.human_count == 0:
        # Nothing detected — reset counter and don't store
        _consecutive_triggers = 0
        return
    if animal_count == 0 and not settings.save_human_only_frames:
        # Human-only frame. Human presence also breaks a consecutive run.
        _consecutive_triggers = 0
        return

    trigger, skip_reason = should_trigger_webhook(result)

    # Human-present or no-animal frames reset the counter (mid-sequence reset)
    if not trigger:
        _consecutive_triggers = 0
    elif (
        result.cat_count > 0
        and result.max_cat_confidence >= settings.webhook_instant_fire_confidence
    ):
        # Instant-fire: a single high-confidence CAT detection (no human, by
        # virtue of trigger=True) skips the consecutive-frame gate. Cat-only —
        # the dog safety net is lower-trust by design.
        skip_reason = None
    else:
        # Two-tier confidence gate. Any frame ≥ low_threshold extends the
        # streak (so a cat misclassified as dog mid-sequence doesn't reset
        # it). Fire when either the high-confidence streak hits its minimum,
        # or the lower-confidence streak hits its (longer) minimum.
        conf = result.max_animal_confidence
        high_threshold = settings.webhook_min_confidence
        low_threshold = min(settings.webhook_low_confidence_threshold, high_threshold)
        high_min = settings.webhook_min_consecutive_frames
        low_min = max(settings.webhook_low_confidence_consecutive_frames, high_min)

        if conf < low_threshold:
            trigger = False
            skip_reason = "confidence_too_low"
            _consecutive_triggers = 0
        else:
            _consecutive_triggers += 1
            high_ok = conf >= high_threshold and _consecutive_triggers >= high_min
            low_ok = _consecutive_triggers >= low_min
            if not (high_ok or low_ok):
                trigger = False
                skip_reason = (
                    f"awaiting_consecutive_{_consecutive_triggers}/"
                    f"{high_min if conf >= high_threshold else low_min}"
                )

    # Enforce cooldown (only if still trigger-worthy)
    if trigger and _last_webhook_time is not None:
        elapsed = datetime.utcnow() - _last_webhook_time
        if elapsed < timedelta(seconds=settings.webhook_cooldown_seconds):
            trigger = False
            skip_reason = "cooldown"

    # Persist frame to disk
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_cats{result.cat_count}_humans{result.human_count}.jpg"
    frame_path = save_frame(result.annotated_frame, settings.frames_dir, filename)

    # Persist detection record
    with SessionLocal() as db:
        detection = Detection(
            cat_count=result.cat_count,
            dog_count=result.dog_count,
            human_count=result.human_count,
            max_cat_confidence=result.max_cat_confidence,
            detections_json=_serialize_boxes(result.boxes),
            webhook_triggered=trigger,
            webhook_skipped_reason=skip_reason,
            frame_path=str(frame_path),
        )
        db.add(detection)
        db.commit()
        db.refresh(detection)
        detection_id = detection.id

    if trigger:
        # Start recording immediately so the pre-roll snapshot captures the
        # trigger moment. Valve open runs concurrently — waiting for it first
        # would push the pre-roll window past the wetting moment.
        from cat_detector.recording import trigger_recording
        asyncio.create_task(trigger_recording(detection_id))
        success = await _open_sprinkler_with_retry(detection_id, result)
        if success:
            _last_webhook_time = datetime.utcnow()
            _consecutive_triggers = 0
            asyncio.create_task(_delayed_sprinkler_off(detection_id))


async def _open_sprinkler(detection_id: int, result) -> bool:
    """Fire the sprinkler-open signal via Home Assistant if configured,
    else fall back to the generic webhook URL."""
    if homeassistant.is_configured():
        return await homeassistant.set_switch("on")
    return await call_webhook(
        detection_id, result.cat_count, result.dog_count,
        result.max_cat_confidence,
    )


async def _open_sprinkler_with_retry(detection_id: int, result, attempts: int = 3) -> bool:
    """Symmetric to _close_sprinkler_with_retry. HA's open_valve service
    returns 200 even when the device is offline ('unavailable'), so we
    have to confirm the state actually transitions to 'open'. The
    SmartThings c2c integration also optimistically writes state='open'
    into HA without device confirmation — so 'state==open' alone isn't
    proof. We additionally require `last_reported` to advance past the
    pre-call baseline; that timestamp only moves when the device actually
    pushes back. Backoff between attempts: 2s, 4s."""
    # For the generic-webhook path we can't verify state — just one shot.
    if not homeassistant.is_configured():
        return await _open_sprinkler(detection_id, result)

    baseline_reported = await _device_report_timestamp()
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            await _open_sprinkler(detection_id, result)
        except Exception:
            logger.exception("Valve open attempt %d raised", attempt)

        # Poll up to ~5s for HA to report 'open' AND for the device to
        # actually push back (last_reported advances).
        last_state = None
        last_reported = baseline_reported
        for _ in range(20):
            await asyncio.sleep(0.25)
            info = await homeassistant.get_valve_state_info()
            if info is None:
                continue
            last_state = info.get("state")
            last_reported = info.get("last_reported") or info.get("last_updated")
            if last_state == "open" and last_reported != baseline_reported:
                if attempt > 1:
                    logger.info("Valve open confirmed on attempt %d", attempt)
                return True

        logger.warning(
            "Valve open attempt %d unconfirmed (state=%s, last_reported=%s, baseline=%s)",
            attempt, last_state, last_reported, baseline_reported,
        )
        if attempt < attempts:
            await asyncio.sleep(delay)
            delay *= 2

    msg = (
        f"VALVE FAILED TO OPEN — detection {detection_id}, "
        f"{attempts} attempts. Device may be unavailable."
    )
    logger.error(msg)
    await homeassistant.notify(msg, title="Cat detector: valve open failed")
    return False


async def _device_report_timestamp() -> str | None:
    """Snapshot the valve's last_reported timestamp so we can detect a real
    push from the device vs HA's optimistic state cache."""
    info = await homeassistant.get_valve_state_info()
    if info is None:
        return None
    return info.get("last_reported") or info.get("last_updated")


async def _close_sprinkler(detection_id: int) -> bool:
    """Same priority order, used to close the valve."""
    if homeassistant.is_configured():
        return await homeassistant.set_switch("off")
    return await call_webhook_off(detection_id)


async def _close_sprinkler_with_retry(detection_id: int, attempts: int = 4) -> bool:
    """Try to close the valve AND verify it actually closed. HA's
    close_valve returns 200 even when the device is offline and the c2c
    integration optimistically writes state='closed' regardless of
    device confirmation — so we require both state=='closed' AND a real
    device push (last_reported advanced past baseline).
    Backoff between attempts: 2s, 4s, 8s."""
    baseline_reported = await _device_report_timestamp()
    delay = 2.0
    last_state = None
    last_reported = baseline_reported
    for attempt in range(1, attempts + 1):
        try:
            await _close_sprinkler(detection_id)
        except Exception:
            logger.exception("Valve close attempt %d raised", attempt)

        # Poll up to ~5s for state=='closed' AND device push
        for _ in range(20):
            await asyncio.sleep(0.25)
            info = await homeassistant.get_valve_state_info()
            if info is None:
                continue
            last_state = info.get("state")
            last_reported = info.get("last_reported") or info.get("last_updated")
            if last_state == "closed" and last_reported != baseline_reported:
                if attempt > 1:
                    logger.info("Valve close confirmed on attempt %d", attempt)
                return True

        logger.warning(
            "Valve close attempt %d unconfirmed (state=%s, last_reported=%s, baseline=%s)",
            attempt, last_state, last_reported, baseline_reported,
        )
        if attempt < attempts:
            await asyncio.sleep(delay)
            delay *= 2

    msg = (
        f"VALVE STUCK OPEN — detection {detection_id}, "
        f"close failed after {attempts} attempts. Manual close required."
    )
    logger.error(msg)
    await homeassistant.notify(msg, title="Cat detector: valve stuck")
    return False


async def valve_watchdog_loop() -> None:
    """Independent safety net: poll the valve state every few seconds. If
    HA reports 'open' continuously for longer than valve_max_open_seconds,
    hammer close + send a notification. Catches cases the normal close
    path missed entirely (e.g. cat-detector crashed mid-cycle, or HA
    accepted a close that never reached the device).

    Idempotent — hammering close on an already-closed valve is a no-op."""
    if not homeassistant.is_configured():
        return
    logger.info(
        "Valve watchdog started (max_open=%.1fs, poll=2s)",
        settings.valve_max_open_seconds,
    )
    open_since: float | None = None
    last_notify_at: float = 0.0
    while True:
        try:
            await asyncio.sleep(2.0)
            state = await homeassistant.get_valve_state()
            if state != "open":
                open_since = None
                continue

            now = asyncio.get_event_loop().time()
            if open_since is None:
                open_since = now
                continue

            elapsed = now - open_since
            if elapsed <= settings.valve_max_open_seconds:
                continue

            logger.error(
                "Valve watchdog: open for %.1fs (limit %.1fs) — forcing close",
                elapsed, settings.valve_max_open_seconds,
            )
            await homeassistant.set_switch("off")

            # Rate-limit notifications so we don't spam the user — once every 60s
            # while the alarm condition persists.
            if now - last_notify_at > 60.0:
                await homeassistant.notify(
                    f"Watchdog: valve open for {elapsed:.0f}s — forced close",
                    title="Cat detector: valve watchdog",
                )
                last_notify_at = now
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Valve watchdog loop error — continuing")
            await asyncio.sleep(2.0)


async def fire_test_trigger() -> dict:
    """Run the exact same open/sleep/close path that a real cat detection
    would, just without persisting a detection record or starting a
    recording. Lets the operator verify the full sprinkler chain works
    end-to-end (state-verified open, configured duration, state-verified
    close) from the Settings UI."""
    if not homeassistant.is_configured():
        return {"ok": False, "error": "Home Assistant not configured (url + token + valve_entity all required)"}

    diag = await homeassistant.test_connection()
    if not diag.get("ok"):
        return diag

    open_ok = await _open_sprinkler_with_retry(detection_id=0, result=None)
    if not open_ok:
        return {"ok": False, "stage": "open", "error": "Valve did not reach 'open' state after retries", **diag}

    await asyncio.sleep(settings.sprinkler_duration_seconds)
    close_ok = await _close_sprinkler_with_retry(detection_id=0)
    return {
        "ok": close_ok,
        "open_ok": open_ok,
        "close_ok": close_ok,
        "duration_seconds": settings.sprinkler_duration_seconds,
        **diag,
    }


async def reconcile_valve_on_startup() -> None:
    """Belt-and-braces: send a close on service boot in case a previous
    instance died mid-open (SIGKILL, panic, host reboot). Safe to call when
    the valve is already closed — both eWeLink and SmartThings treat
    redundant close as a no-op."""
    if not (homeassistant.is_configured() or settings.webhook_off_url):
        return
    logger.info("Reconciling valve state on startup — sending close")
    await _close_sprinkler_with_retry(detection_id=0, attempts=2)


async def _delayed_sprinkler_off(detection_id: int) -> None:
    """After sprinkler_duration_seconds, close the valve."""
    try:
        await asyncio.sleep(settings.sprinkler_duration_seconds)
        await _close_sprinkler_with_retry(detection_id)
    except asyncio.CancelledError:
        # Service shutting down — still try to close the valve
        await _close_sprinkler_with_retry(detection_id)
        raise


async def run_camera_loop() -> None:
    """Main loop — runs indefinitely, sampling a frame every interval."""
    logger.info(
        "Camera loop started (source=%s, interval=%ds)",
        settings.camera_url,
        settings.detection_interval_seconds,
    )

    global camera_status
    consecutive_failures = 0

    while True:
        try:
            frame = await asyncio.get_event_loop().run_in_executor(None, _grab_frame)
            if frame is not None:
                camera_status = "connected"
                consecutive_failures = 0
                await _process_frame(frame)
                await asyncio.sleep(settings.detection_interval_seconds)
            else:
                consecutive_failures += 1
                camera_status = "retrying" if consecutive_failures < 5 else "disconnected"
                delay = min(
                    settings.detection_interval_seconds * (2 ** consecutive_failures),
                    settings.camera_reconnect_max_delay,
                )
                logger.warning(
                    "Failed to grab frame (attempt %d) — retrying in %ds",
                    consecutive_failures,
                    delay,
                )
                await asyncio.sleep(delay)
        except Exception:
            logger.exception("Unexpected error in camera loop — continuing")
            await asyncio.sleep(settings.detection_interval_seconds)
