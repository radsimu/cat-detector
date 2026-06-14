"""
Unit tests for camera loop backoff logic.
No real camera or GPU required — heavy dependencies and camera I/O are fully mocked.
"""

import asyncio
import importlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy/native modules that are unavailable in the test environment
# before cat_detector.camera is imported.
# ---------------------------------------------------------------------------

_DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _stub_heavy_modules():
    """Insert lightweight stubs for modules that camera.py imports at module level."""
    stubs = {
        "cv2": MagicMock(),
        "ultralytics": MagicMock(),
    }
    # cat_detector sub-modules that camera.py imports
    for name in (
        "cat_detector.database",
        "cat_detector.detector",
        "cat_detector.models",
        "cat_detector.webhook",
    ):
        stubs[name] = MagicMock()

    for name, stub in stubs.items():
        sys.modules.setdefault(name, stub)


_stub_heavy_modules()

# Now it is safe to import (or re-import) cat_detector.camera
if "cat_detector.camera" in sys.modules:
    del sys.modules["cat_detector.camera"]

import cat_detector.camera as camera_module  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(interval: int = 5, max_delay: int = 60) -> MagicMock:
    s = MagicMock()
    s.camera_url = "0"
    s.detection_interval_seconds = interval
    s.camera_reconnect_max_delay = max_delay
    return s


def _make_fake_run_in_executor(grab_returns):
    """Return a coroutine factory that pops values from grab_returns in order."""
    iterator = iter(grab_returns)

    async def fake_run_in_executor(executor, fn):
        return next(iterator)

    return fake_run_in_executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_failures_increase_delay():
    """Each successive failure should double the sleep duration."""
    settings = _make_settings(interval=5, max_delay=60)
    # Fail twice, then succeed (loop exit triggered by CancelledError on 3rd sleep)
    grab_returns = [None, None, _DUMMY_FRAME]

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        if len(sleep_calls) >= 3:
            raise asyncio.CancelledError

    loop_mock = MagicMock()
    loop_mock.run_in_executor = _make_fake_run_in_executor(grab_returns)

    with (
        patch.object(camera_module, "settings", settings),
        patch.object(camera_module, "_process_frame", new_callable=AsyncMock),
        patch("asyncio.get_event_loop", return_value=loop_mock),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await camera_module.run_camera_loop()

    # failure 1: consecutive_failures=1 → delay = 5 * 2^1 = 10
    assert sleep_calls[0] == 10, f"Expected 10, got {sleep_calls[0]}"
    # failure 2: consecutive_failures=2 → delay = 5 * 2^2 = 20
    assert sleep_calls[1] == 20, f"Expected 20, got {sleep_calls[1]}"


@pytest.mark.asyncio
async def test_backoff_capped_at_max_delay():
    """Backoff delay must never exceed camera_reconnect_max_delay."""
    settings = _make_settings(interval=5, max_delay=30)

    num_failures = 10
    grab_returns = [None] * num_failures + [_DUMMY_FRAME]

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        if len(sleep_calls) >= num_failures + 1:
            raise asyncio.CancelledError

    loop_mock = MagicMock()
    loop_mock.run_in_executor = _make_fake_run_in_executor(grab_returns)

    with (
        patch.object(camera_module, "settings", settings),
        patch.object(camera_module, "_process_frame", new_callable=AsyncMock),
        patch("asyncio.get_event_loop", return_value=loop_mock),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await camera_module.run_camera_loop()

    failure_sleeps = sleep_calls[:num_failures]
    assert all(d <= 30 for d in failure_sleeps), (
        f"Some delays exceeded max_delay=30: {failure_sleeps}"
    )
    # After enough failures the delay should be pinned at max
    assert failure_sleeps[-1] == 30, (
        f"Expected last failure delay to be 30, got {failure_sleeps[-1]}"
    )


@pytest.mark.asyncio
async def test_success_resets_backoff():
    """A successful grab resets the counter so backoff restarts from base on next fail."""
    settings = _make_settings(interval=5, max_delay=60)

    # fail, fail, success, fail → 4 sleep calls before CancelledError
    grab_returns = [None, None, _DUMMY_FRAME, None, _DUMMY_FRAME]

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        if len(sleep_calls) >= 4:
            raise asyncio.CancelledError

    loop_mock = MagicMock()
    loop_mock.run_in_executor = _make_fake_run_in_executor(grab_returns)

    with (
        patch.object(camera_module, "settings", settings),
        patch.object(camera_module, "_process_frame", new_callable=AsyncMock),
        patch("asyncio.get_event_loop", return_value=loop_mock),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await camera_module.run_camera_loop()

    # sleep_calls:
    # [0] = 10  (1st fail, consecutive=1, delay=5*2=10)
    # [1] = 20  (2nd fail, consecutive=2, delay=5*4=20)
    # [2] =  5  (success, reset, normal interval=5)
    # [3] = 10  (1st fail after reset, consecutive=1, delay=5*2=10)
    assert sleep_calls[0] == 10, f"Expected 10, got {sleep_calls[0]}"
    assert sleep_calls[1] == 20, f"Expected 20, got {sleep_calls[1]}"
    assert sleep_calls[2] == 5,  f"Expected 5 (normal interval), got {sleep_calls[2]}"
    assert sleep_calls[3] == 10, f"Expected 10 (reset backoff), got {sleep_calls[3]}"


@pytest.mark.asyncio
async def test_no_failures_uses_normal_interval():
    """When frames are grabbed successfully the loop sleeps the normal interval."""
    settings = _make_settings(interval=5, max_delay=60)
    grab_returns = [_DUMMY_FRAME, _DUMMY_FRAME]

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        if len(sleep_calls) >= 2:
            raise asyncio.CancelledError

    loop_mock = MagicMock()
    loop_mock.run_in_executor = _make_fake_run_in_executor(grab_returns)

    with (
        patch.object(camera_module, "settings", settings),
        patch.object(camera_module, "_process_frame", new_callable=AsyncMock),
        patch("asyncio.get_event_loop", return_value=loop_mock),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await camera_module.run_camera_loop()

    assert all(d == 5 for d in sleep_calls), (
        f"Expected all sleeps to be 5, got {sleep_calls}"
    )
