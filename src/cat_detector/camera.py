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
import time
from datetime import datetime, timedelta
from typing import AsyncGenerator

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
_miss_streak: int = 0  # consecutive frames with no animal — used to debounce
                       # the consecutive-frame gate so a flickering detector
                       # doesn't keep resetting the streak to zero.
_MISS_TOLERANCE = 2    # forgive this many "no animal" frames before reset

# Sprinkler-trigger rate-limit alerting. If we fire too many times in a short
# window, something is wrong (loop / sensor / false-positive cluster). Alert
# the user; don't suppress firing (a real cat must still be deterred).
from collections import deque
_TRIGGER_FREQ_COUNT = 5
_TRIGGER_FREQ_WINDOW_S = 1800   # 30 min
_TRIGGER_FREQ_COOLDOWN_S = 3600  # notify at most once per hour
_recent_triggers: deque[float] = deque()
_last_freq_alert_at: float = 0.0

# Camera-offline alerting. consecutive_failures is tracked in run_camera_loop;
# when it crosses this threshold we notify (rate-limited) so a transient blip
# doesn't spam the phone.
_CAMERA_OFFLINE_FAILURES = 5
_CAMERA_OFFLINE_COOLDOWN_S = 600  # 10 min between camera-offline notifications
_last_camera_alert_at: float = 0.0
_camera_was_offline: bool = False  # tracks edge for "back online" notify

# Persistent camera capture — kept open to avoid repeated open/close cycles
# which cause the Mac camera indicator to flash and reset auto-exposure.
_capture: cv2.VideoCapture | None = None
_capture_lock = threading.Lock()

# Camera status for UI indicator. Updated by the camera loop.
camera_status: str = "unknown"  # "connected" | "disconnected" | "retrying"

# Detection-loop heartbeat — updated each time _process_frame completes.
# /healthz reads this to verify the detection coroutine is actually running,
# not just that uvicorn answers HTTP.
_last_process_ts: float = 0.0

# ---- Shared latest-frame slot, filled by the camera reader thread --------
# A single reader thread owns the device and continuously grabs frames at the
# stream's NATIVE rate, so the FFmpeg/RTSP demux buffer never backs up. All
# consumers (detection loop, recording buffer, live stream) read the newest
# decoded frame from this slot instead of calling cap.read() themselves.
#
# Why: consumers that read slower than the camera produces (the 10fps buffer
# loop, the 0.5fps detection loop) let the demux buffer accumulate, so the
# frames they pulled fell progressively behind real time — tens of seconds to
# minutes of latency that grew until the stream reconnected. Draining in one
# always-current reader eliminates that entirely.
_latest_frame: np.ndarray | None = None
_latest_frame_ts: float = 0.0
_frame_slot_lock = threading.Lock()
_reader_thread: threading.Thread | None = None
_reader_stop = threading.Event()

# A slot older than this is treated as "no frame" — the reader has stalled
# (frozen/hung stream) and consumers should back off rather than record or
# detect on a frame that is seconds stale.
_FRAME_STALE_SECONDS = 5.0
# Decode at most this often. grab() still runs every loop at the native rate
# to keep the demux buffer drained (near-zero latency); retrieve() (the
# expensive decode) is throttled to cap CPU while staying above the 10fps the
# recording buffer samples at.
_DECODE_INTERVAL = 0.08


def get_camera_status() -> str:
    return camera_status


def get_last_process_age() -> float | None:
    """Seconds since the detection loop last finished _process_frame.
    None if it has never run (just booted)."""
    if _last_process_ts == 0.0:
        return None
    return time.time() - _last_process_ts


def _redacted_camera_url() -> str:
    """Camera URL safe for logs/notifications: masks the password in an
    rtsp://user:pass@host URL so credentials never land in logs or on the
    phone. Non-URL sources (e.g. a device index) are returned unchanged."""
    url = settings.camera_url
    if not isinstance(url, str) or "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        creds = f"{user}:***"
    return f"{scheme}://{creds}@{host}"


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
        logger.error("Cannot open camera source: %s", _redacted_camera_url())
        _capture = None
        return None

    # Shrink the internal buffer to 1 frame so cap.read() always returns the latest
    _capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.info("Camera opened: %s (rtsp=%s)", _redacted_camera_url(), is_rtsp)
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
    """Return the most recent decoded frame published by the camera reader
    thread, or None if the slot is empty or stale (reader stalled / camera
    down). The reader keeps the slot at real time, so callers never receive a
    backlogged frame. Returns the shared array directly — the reader allocates
    a fresh array per decode and never mutates a published one, and consumers
    copy before drawing, so no defensive copy is needed here."""
    with _frame_slot_lock:
        if _latest_frame is None:
            return None
        if time.time() - _latest_frame_ts > _FRAME_STALE_SECONDS:
            return None
        return _latest_frame


def _grab_latest_frame() -> np.ndarray | None:
    """Alias for _grab_frame. The reader thread already keeps only the newest
    frame in the slot, so 'latest' and 'current' are the same thing now — the
    old per-call drain-5-frames trick is obsolete."""
    return _grab_frame()


def _capture_release_inner() -> None:
    """Release capture without acquiring the lock (caller already holds it)."""
    global _capture
    if _capture is not None:
        _capture.release()
        _capture = None
        logger.warning("Camera disconnected — will retry on next grab")


def _camera_reader_loop() -> None:
    """Single owner of the camera device. Grabs frames at the stream's native
    rate so the FFmpeg/RTSP demux buffer never backs up (the root cause of
    multi-second capture latency), decoding at most every _DECODE_INTERVAL and
    publishing the newest decoded frame to the shared slot. Consumers read the
    slot via _grab_frame; none of them touch the device, so none can fall
    behind real time.

    Owns camera_status. grab()/retrieve() run under _capture_lock so a
    shutdown-time _release_capture can't free the capture mid-call (which can
    segfault the FFmpeg backend)."""
    global camera_status, _latest_frame, _latest_frame_ts
    consecutive_failures = 0
    last_decode = 0.0
    while not _reader_stop.is_set():
        try:
            with _capture_lock:
                cap = _get_capture()
                if cap is None:
                    _cap_fail = True
                else:
                    _cap_fail = not cap.grab()
                    if _cap_fail:
                        _capture_release_inner()
            if _cap_fail:
                consecutive_failures += 1
                camera_status = "retrying" if consecutive_failures < 5 else "disconnected"
                # Exponential backoff on reconnect, capped.
                _reader_stop.wait(min(
                    settings.camera_reconnect_max_delay,
                    0.5 * (2 ** min(consecutive_failures, 6)),
                ))
                continue

            now = time.time()
            if now - last_decode >= _DECODE_INTERVAL:
                with _capture_lock:
                    if _capture is None:
                        continue
                    ret, frame = _capture.retrieve()
                if ret and frame is not None:
                    with _frame_slot_lock:
                        _latest_frame = frame
                        _latest_frame_ts = now
                    last_decode = now

            consecutive_failures = 0
            camera_status = "connected"
        except Exception:
            logger.exception("Camera reader loop error — reopening capture")
            _release_capture()
            camera_status = "retrying"
            _reader_stop.wait(0.5)


def start_camera_reader() -> None:
    """Start the background camera reader thread (idempotent)."""
    global _reader_thread
    if _reader_thread is None or not _reader_thread.is_alive():
        _reader_stop.clear()
        _reader_thread = threading.Thread(
            target=_camera_reader_loop, name="camera-reader", daemon=True
        )
        _reader_thread.start()
        logger.info("Camera reader thread started (native-rate grab, decode<=%.0ffps)",
                    1.0 / _DECODE_INTERVAL)


def stop_camera_reader() -> None:
    """Signal the reader thread to stop and wait briefly for it to exit."""
    _reader_stop.set()
    t = _reader_thread
    if t is not None:
        t.join(timeout=3.0)


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
    global _last_webhook_time, _consecutive_triggers, _last_process_ts

    # YOLOv8x on Pi-class CPU takes seconds — must run off the event loop or
    # the MJPEG stream and HTTP handlers freeze for the duration.
    result = await asyncio.get_event_loop().run_in_executor(None, detect, frame)
    _last_process_ts = time.time()

    # Publish boxes to the shared overlay cache so the live stream and the
    # recording buffer can draw them without running a second YOLO pass.
    from cat_detector.routers.stream import publish_boxes
    publish_boxes(result.boxes)

    global _miss_streak
    animal_count = result.cat_count + result.dog_count
    if animal_count == 0 and result.human_count == 0:
        # No animal AND no human → debounce. A flickering detector loses the
        # cat for 1-2 frames; resetting the streak there means we never reach
        # consecutive_min on marginal subjects.
        _miss_streak += 1
        if _miss_streak > _MISS_TOLERANCE:
            _consecutive_triggers = 0
        return
    if animal_count == 0 and not settings.save_human_only_frames:
        # Human present — hard reset (human guard is non-negotiable, no
        # debounce). Also clears miss streak so any later cat starts fresh.
        _consecutive_triggers = 0
        _miss_streak = 0
        return
    # Animal seen — clear miss debounce.
    _miss_streak = 0

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
        # trigger moment.
        from cat_detector.recording import trigger_recording
        asyncio.create_task(trigger_recording(detection_id))

        # Fire-and-forget open + dwell + close (same pattern as the test
        # path). Awaiting _open_sprinkler_with_retry blocked _process_frame
        # for up to 30s of cloud-confirm retries, by which time the
        # post-roll had ended without ever seeing the sprinkle.
        _last_webhook_time = datetime.utcnow()
        _consecutive_triggers = 0
        asyncio.create_task(_fire_sprinkler_cycle(detection_id, result))
        await _check_trigger_rate()


async def _fire_sprinkler_cycle(detection_id: int, result) -> None:
    """Production sprinkler cycle: send open, dwell, send close, plus three
    safety closes 1s apart. Verification happens in background via the
    valve_watchdog_loop; we don't block here on per-command state polling
    because awaiting the cloud round-trip pushes the sprinkle past the
    recording's post-roll window."""
    try:
        await _open_sprinkler(detection_id, result)
    except Exception:
        logger.exception("Open call raised for detection %d", detection_id)
    await asyncio.sleep(settings.sprinkler_duration_seconds)
    try:
        await _close_sprinkler(detection_id)
    except Exception:
        logger.exception("Close call raised for detection %d", detection_id)
    # Belt-and-braces redundant closes
    for _ in range(3):
        await asyncio.sleep(1.0)
        try:
            await _close_sprinkler(detection_id)
        except Exception:
            pass


async def _check_trigger_rate() -> None:
    """Track recent trigger timestamps. If we fire too often in the rolling
    window, notify the user — likely a false-positive cluster or runaway
    detection. We still fire (don't suppress) so a real cat is deterred."""
    global _last_freq_alert_at
    now = time.time()
    _recent_triggers.append(now)
    while _recent_triggers and now - _recent_triggers[0] > _TRIGGER_FREQ_WINDOW_S:
        _recent_triggers.popleft()
    if (len(_recent_triggers) >= _TRIGGER_FREQ_COUNT
            and now - _last_freq_alert_at > _TRIGGER_FREQ_COOLDOWN_S):
        await homeassistant.notify(
            f"Sprinkler fired {len(_recent_triggers)} times in last "
            f"{_TRIGGER_FREQ_WINDOW_S // 60} min — likely false-positive cluster or bug.",
            title="Cat detector: triggering too often",
        )
        _last_freq_alert_at = now


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
            if homeassistant.is_open_state(last_state) and last_reported != baseline_reported:
                if attempt > 1:
                    logger.info("Valve open confirmed on attempt %d", attempt)
                return True

        logger.warning(
            "Valve open attempt %d unconfirmed (state=%s, last_reported=%s, baseline=%s)",
            attempt, last_state, last_reported, baseline_reported,
        )
        # If HA marked the entity unavailable, the c2c integration likely
        # desynced — force a refresh before the next attempt.
        if last_state == "unavailable":
            await homeassistant.refresh_entity()
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
            if homeassistant.is_closed_state(last_state) and last_reported != baseline_reported:
                if attempt > 1:
                    logger.info("Valve close confirmed on attempt %d", attempt)
                return True

        logger.warning(
            "Valve close attempt %d unconfirmed (state=%s, last_reported=%s, baseline=%s)",
            attempt, last_state, last_reported, baseline_reported,
        )
        if last_state == "unavailable":
            await homeassistant.refresh_entity()
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
    unavailable_since: float | None = None
    last_refresh_at: float = 0.0
    last_unavail_notify_at: float = 0.0
    last_notify_at: float = 0.0
    while True:
        try:
            await asyncio.sleep(2.0)
            state = await homeassistant.get_valve_state()
            now = asyncio.get_event_loop().time()

            # Self-heal: if HA shows entity unavailable for long, the c2c
            # integration likely desynced (entity orphaned, refresh token
            # expired, etc). Force-refresh every 60s while unavailable;
            # if still unavailable after 5min, notify the user once per hour.
            if state == "unavailable":
                if unavailable_since is None:
                    unavailable_since = now
                if now - last_refresh_at > 60.0:
                    await homeassistant.refresh_entity()
                    last_refresh_at = now
                if (now - unavailable_since > 300.0
                        and now - last_unavail_notify_at > 3600.0):
                    await homeassistant.notify(
                        f"Valve entity unavailable for {(now - unavailable_since)/60:.0f} min — "
                        "SmartThings integration may need reauthentication.",
                        title="Cat detector: valve integration desynced",
                    )
                    last_unavail_notify_at = now
            else:
                unavailable_since = None

            if not homeassistant.is_open_state(state):
                open_since = None
                continue

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


async def fire_test_trigger() -> AsyncGenerator[dict, None]:
    """Async generator yielding step events for the test-sprinkler flow.
    The UI consumes this as a server-sent event stream so the operator can
    see exactly what the system is doing and waiting for, and gets a real
    measured sprinkle duration (time between confirmed-open and
    confirmed-close) on success.

    Yields dicts shaped as either:
      {"event": "log", "message": "..."}   — progress line
      {"event": "done", "ok": bool, ...}   — final result, ends the stream
    """
    if not homeassistant.is_configured():
        yield {"event": "done", "ok": False,
               "error": "Home Assistant not configured (url + token + valve_entity all required)"}
        return

    yield {"event": "log", "message": "Checking HA connection..."}
    diag = await homeassistant.test_connection()
    if not diag.get("ok"):
        yield {"event": "done", "ok": False, "stage": "connect",
               "error": diag.get("error", "HA connection failed"), "ha": diag}
        return
    yield {"event": "log",
           "message": f"HA OK — entity {diag.get('entity_id')} state={diag.get('state')}"}

    # If the entity is unavailable, the SmartThings integration may have
    # desynced. Force a refresh before giving up on retries.
    if diag.get("state") == "unavailable":
        yield {"event": "log",
               "message": "Entity unavailable — calling homeassistant.update_entity to nudge integration..."}
        await homeassistant.refresh_entity()
        await asyncio.sleep(2.0)
        info = await homeassistant.get_valve_state_info()
        new_state = info.get("state") if info else None
        yield {"event": "log",
               "message": f"After refresh: state={new_state}"}

    # Mirror the production trigger path: start a recording right now so the
    # pre-roll captures the moment the test starts and the post-roll captures
    # the dwell + close. detection_id=None → filename has "manual" suffix and
    # the recordings table row has a null FK (column is nullable).
    from cat_detector.recording import trigger_recording
    asyncio.create_task(trigger_recording(None))
    yield {"event": "log", "message": "Recording started (will appear under Recordings)"}

    # ---- SEQUENCE: fire-and-forget open + dwell + fire-and-forget close --
    # Waiting for per-command confirmation via the SmartThings cloud adds
    # ~1.5s RTT on each side, blowing up real water-on time. Fire both
    # commands back-to-back so the device-side gap matches the configured
    # dwell, then verify *after* the sequence that the valve actually
    # transitioned (state ended 'closed' with last_changed advanced past
    # baseline). The independent watchdog catches the stuck-open case.
    info0 = await homeassistant.get_valve_state_info() or {}
    baseline_changed = info0.get("last_changed")
    yield {"event": "log",
           "message": f"Baseline state={info0.get('state')}, last_changed={baseline_changed}"}

    # True fire-and-forget: launch the HA service calls as background tasks
    # instead of awaiting them. HA service POSTs block ~3s waiting for the
    # cloud chain to ack — awaiting them would make the actual command gap
    # = dwell + ~6s, not the configured dwell. Letting them run in parallel
    # collapses real water-on time to (open processing latency) + dwell.
    open_send_t = time.monotonic()
    yield {"event": "log", "message": "Sending valve.open_valve (fire-and-forget)..."}
    open_task = asyncio.create_task(_open_sprinkler(0, None))

    dwell_s = settings.sprinkler_duration_seconds
    yield {"event": "log", "message": f"Dwell — {int(dwell_s * 1000)} ms..."}
    await asyncio.sleep(dwell_s)

    close_send_t = time.monotonic()
    yield {"event": "log", "message": "Sending valve.close_valve (fire-and-forget)..."}
    close_task = asyncio.create_task(_close_sprinkler(0))

    # Belt-and-braces: schedule extra close attempts 1s, 2s, 3s after the
    # first one. eWeLink cloud occasionally drops a command; redundant
    # closes are idempotent on the device side but maximize the chance of
    # actually closing the valve. Each is fire-and-forget — none of them
    # block the verification loop.
    async def _safety_closes() -> None:
        for delay in (1.0, 2.0, 3.0):
            await asyncio.sleep(1.0)  # 1s spacing between each
            try:
                await _close_sprinkler(0)
            except Exception:
                logger.warning("safety close fired at +%.0fs raised", delay, exc_info=True)
    safety_task = asyncio.create_task(_safety_closes())

    # Surface any exceptions on the immediate calls, but don't block here —
    # tasks run in parallel; we move on to state verification immediately.
    for t in (open_task, close_task, safety_task):
        t.add_done_callback(lambda f: f.exception() and
                            logger.warning("sprinkler call task error: %s", f.exception()))

    command_gap_ms = int((close_send_t - open_send_t) * 1000)
    yield {"event": "log",
           "message": f"Commands sent {command_gap_ms} ms apart. Verifying cycle (up to 10s)..."}

    # ---- VERIFY -----------------------------------------------------------
    # Require state=='closed' AND last_changed advanced past baseline. The
    # last_changed timestamp only moves on real state transitions, so a
    # stuck-closed entity won't false-confirm.
    deadline = time.monotonic() + 10.0
    last_state: str | None = None
    last_changed = baseline_changed
    last_reported = info0.get("last_reported") or info0.get("last_updated")
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        info = await homeassistant.get_valve_state_info()
        if info is None:
            continue
        last_state = info.get("state")
        last_changed = info.get("last_changed")
        last_reported = info.get("last_reported") or info.get("last_updated")
        if homeassistant.is_closed_state(last_state) and last_changed != baseline_changed:
            yield {"event": "log",
                   "message": f"✓ Cycle confirmed — state=closed, last_changed={last_changed} (advanced)"}
            yield {"event": "done", "ok": True, "sprinkle_ms": command_gap_ms,
                   "duration_seconds": dwell_s, "ha": diag}
            return

    yield {"event": "log",
           "message": (f"✗ No confirmed transition. Final state={last_state}, "
                       f"last_changed={last_changed} (unchanged), last_reported={last_reported}")}
    if last_state == "unavailable":
        await homeassistant.refresh_entity()
        yield {"event": "log",
               "message": "Forced HA refresh — integration may need reauth"}
    yield {"event": "done", "ok": False, "stage": "verify",
           "error": "Cycle did not confirm — no state transition observed during verification window",
           "ha": diag, "last_state": last_state}


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
        _redacted_camera_url(),
        settings.detection_interval_seconds,
    )

    global camera_status, _last_camera_alert_at, _camera_was_offline
    consecutive_failures = 0

    while True:
        try:
            frame = await asyncio.get_event_loop().run_in_executor(None, _grab_frame)
            if frame is not None:
                # camera_status is owned by the reader thread now.
                if _camera_was_offline:
                    _camera_was_offline = False
                    await homeassistant.notify(
                        "Camera back online.",
                        title="Cat detector: camera recovered",
                    )
                consecutive_failures = 0
                await _process_frame(frame)
                await asyncio.sleep(settings.detection_interval_seconds)
            else:
                # No fresh frame in the slot — reader stalled or camera down.
                consecutive_failures += 1
                if consecutive_failures == _CAMERA_OFFLINE_FAILURES:
                    now = time.time()
                    if now - _last_camera_alert_at > _CAMERA_OFFLINE_COOLDOWN_S:
                        _last_camera_alert_at = now
                        _camera_was_offline = True
                        await homeassistant.notify(
                            f"Camera offline — {consecutive_failures} consecutive frame grabs failed. "
                            f"Source: {_redacted_camera_url()}",
                            title="Cat detector: camera offline",
                        )
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
