# Implementation Plan — US-7 · Environment-Based Configuration (#6110)

## Story

**As a** operator, **I can** configure all service parameters via environment variables or a .env file **so that** I can deploy the service in different environments without code changes.

## Current State

Pydantic-settings integration is already in place (`config.py`). All 12 settings have defaults and are documented in `.env.example`. The remaining work is validation hardening and startup error UX.

## Tasks

### Task 1: Add pydantic field validators to `config.py`

Add `field_validator` checks for:
- `detection_confidence`: must be between 0.0 and 1.0
- `port`: must be between 1 and 65535
- `detection_interval_seconds`: must be > 0
- `webhook_cooldown_seconds`: must be >= 0
- `camera_reconnect_max_delay`: must be > 0

These raise `ValidationError` with clear messages if .env has invalid values.

### Task 2: Add startup validation wrapper in `main.py`

Wrap the `Settings()` instantiation so that if pydantic raises `ValidationError`, the service prints a human-friendly error message listing each invalid field and exits with code 1 — instead of a raw traceback.

Currently `settings = Settings()` is at module level in `config.py`. Move the error handling to the lifespan or add a `try/except` in config.py that catches `ValidationError` and re-raises with a clean message.

### Task 3: Fix `.env.example` defaults alignment

- Change `CAMERA_URL` to use a comment showing the RTSP example but default to `0` (matches config.py default)
- Ensure all values in `.env.example` match `config.py` defaults so copying the file works out of the box

### Task 4: Add unit tests (`tests/test_config.py`)

- Valid defaults load without error
- Invalid confidence (e.g., 1.5) raises clear validation error
- Invalid port (e.g., 99999) raises clear validation error
- Negative interval raises clear validation error
- Environment variables override defaults correctly

## Approach

- Single PR, small scope
- No data model changes, no new dependencies
- Backend-only

## Testing Strategy

- Unit tests for config validation (mock env vars)
- Verify existing tests still pass
