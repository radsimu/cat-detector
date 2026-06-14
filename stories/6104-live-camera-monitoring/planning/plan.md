# Implementation Plan — US-1 · Live Camera Monitoring (#6104)

## Story

**As a** pet owner, **I can** have the service continuously watch my garden camera feed **so that** I don't have to manually check whether cats are on the sofa.

## Current State

Most of the acceptance criteria are already implemented. The main gap is **AC3: exponential backoff on dropped connections**.

| AC | Status | Notes |
|----|--------|-------|
| AC1: Configurable interval, no leaks | ✅ Done | 5s default, new VideoCapture per frame (no leak) |
| AC2: RTSP + USB support | ✅ Done | `int()` conversion for USB indices |
| AC3: Exponential backoff on reconnect | ❌ Missing | Currently logs error and retries at normal interval |
| AC4: Startup logging | ✅ Done | `run_camera_loop()` logs source + interval |
| AC5: Clean SIGTERM shutdown | ✅ Done | Lifespan cancels task, catches CancelledError |

## Tasks

### Task 1: Add exponential backoff to camera loop (`camera.py`)

When `_grab_frame()` returns `None` (camera unavailable), track consecutive failures and increase the sleep time exponentially:
- Base delay: `detection_interval_seconds` (5s)
- Backoff multiplier: 2x per consecutive failure
- Max delay cap: 60s (new config setting `camera_reconnect_max_delay`)
- Reset to normal interval on successful frame grab
- Log each retry with the current backoff delay

Changes:
- `camera.py` — add backoff state to `run_camera_loop()`, adjust sleep duration on failure
- `config.py` — add `camera_reconnect_max_delay: int = 60`

### Task 2: Add unit tests (`tests/test_camera.py`)

Test the backoff behavior:
- Consecutive failures increase delay exponentially
- Successful grab resets delay to normal
- Delay never exceeds max cap
- Camera loop logs startup info (source + interval)

Mock `_grab_frame` and `_process_frame` to test loop logic without real camera.

### Task 3: Update `.env.example`

Add the new `CAMERA_RECONNECT_MAX_DELAY` setting with a comment.

## Approach

- Single PR — scope is small
- No data model changes, no new dependencies
- Backend-only (no UI)

## Testing Strategy

- Unit tests for backoff logic (mock camera)
- Existing `test_detector.py` unaffected
