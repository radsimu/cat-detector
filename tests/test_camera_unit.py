"""
Additional unit tests for camera module — US-1 · Live Camera Monitoring (#6104).
Covers _grab_frame() source parsing and startup logging.
"""

import asyncio
import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub heavy modules before importing camera
_DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

for mod in (
    "cv2",
    "ultralytics",
    "cat_detector.database",
    "cat_detector.detector",
    "cat_detector.models",
    "cat_detector.webhook",
):
    sys.modules.setdefault(mod, MagicMock())

if "cat_detector.camera" in sys.modules:
    del sys.modules["cat_detector.camera"]

import cat_detector.camera as camera_module  # noqa: E402


# ---------------------------------------------------------------------------
# _grab_frame: source parsing
# ---------------------------------------------------------------------------


class TestGrabFrameSourceParsing:
    """_grab_frame converts numeric strings to int for USB, keeps strings for RTSP."""

    def test_usb_index_parsed_as_int(self):
        mock_settings = MagicMock()
        mock_settings.camera_url = "0"

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, _DUMMY_FRAME)

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module.cv2, "VideoCapture", return_value=mock_cap) as vc,
        ):
            frame = camera_module._grab_frame()
            vc.assert_called_once_with(0)  # int, not "0"
            assert frame is not None

    def test_usb_index_2_parsed_as_int(self):
        mock_settings = MagicMock()
        mock_settings.camera_url = "2"

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, _DUMMY_FRAME)

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module.cv2, "VideoCapture", return_value=mock_cap) as vc,
        ):
            camera_module._grab_frame()
            vc.assert_called_once_with(2)

    def test_rtsp_url_kept_as_string(self):
        mock_settings = MagicMock()
        mock_settings.camera_url = "rtsp://192.168.1.100:554/stream"

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, _DUMMY_FRAME)

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module.cv2, "VideoCapture", return_value=mock_cap) as vc,
        ):
            camera_module._grab_frame()
            vc.assert_called_once_with("rtsp://192.168.1.100:554/stream")

    def test_returns_none_when_camera_cannot_open(self):
        mock_settings = MagicMock()
        mock_settings.camera_url = "0"

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module.cv2, "VideoCapture", return_value=mock_cap),
        ):
            frame = camera_module._grab_frame()
            assert frame is None
            mock_cap.release.assert_not_called()

    def test_returns_none_when_read_fails(self):
        mock_settings = MagicMock()
        mock_settings.camera_url = "0"

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module.cv2, "VideoCapture", return_value=mock_cap),
        ):
            frame = camera_module._grab_frame()
            assert frame is None
            mock_cap.release.assert_called_once()

    def test_videocapture_released_on_success(self):
        mock_settings = MagicMock()
        mock_settings.camera_url = "0"

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, _DUMMY_FRAME)

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module.cv2, "VideoCapture", return_value=mock_cap),
        ):
            camera_module._grab_frame()
            mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# run_camera_loop: startup logging
# ---------------------------------------------------------------------------


class TestCameraLoopStartupLogging:
    """AC4: The current sampling interval and camera source are visible in logs."""

    @pytest.mark.asyncio
    async def test_logs_source_and_interval_on_startup(self, caplog):
        mock_settings = MagicMock()
        mock_settings.camera_url = "rtsp://10.0.0.1/stream"
        mock_settings.detection_interval_seconds = 10
        mock_settings.camera_reconnect_max_delay = 60

        call_count = 0

        async def fake_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError

        async def fake_run_in_executor(ex, fn):
            return _DUMMY_FRAME

        loop_mock = MagicMock()
        loop_mock.run_in_executor = fake_run_in_executor

        async def fake_process_frame(frame):
            pass

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module, "_process_frame", side_effect=fake_process_frame),
            patch("asyncio.get_event_loop", return_value=loop_mock),
            patch("asyncio.sleep", side_effect=fake_sleep),
            caplog.at_level(logging.INFO, logger="cat_detector.camera"),
        ):
            with pytest.raises(asyncio.CancelledError):
                await camera_module.run_camera_loop()

        assert "rtsp://10.0.0.1/stream" in caplog.text
        assert "10" in caplog.text


# ---------------------------------------------------------------------------
# run_camera_loop: clean cancellation (SIGTERM)
# ---------------------------------------------------------------------------


class TestCameraLoopCleanShutdown:
    """AC5: The loop can be stopped cleanly via SIGTERM without leaving zombie threads."""

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_cleanly(self):
        """CancelledError during sleep should propagate without being swallowed."""
        mock_settings = MagicMock()
        mock_settings.camera_url = "0"
        mock_settings.detection_interval_seconds = 5
        mock_settings.camera_reconnect_max_delay = 60

        async def fake_run_in_executor(ex, fn):
            return _DUMMY_FRAME

        loop_mock = MagicMock()
        loop_mock.run_in_executor = fake_run_in_executor

        async def fake_process_frame(frame):
            pass

        with (
            patch.object(camera_module, "settings", mock_settings),
            patch.object(camera_module, "_process_frame", side_effect=fake_process_frame),
            patch("asyncio.get_event_loop", return_value=loop_mock),
            patch("asyncio.sleep", side_effect=asyncio.CancelledError),
        ):
            with pytest.raises(asyncio.CancelledError):
                await camera_module.run_camera_loop()
