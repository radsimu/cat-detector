"""
Unit tests for the MJPEG streaming endpoint — US-8 · Live Camera View (#6303).
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub heavy deps
_cv2_stub = types.ModuleType("cv2")
_cv2_stub.VideoCapture = MagicMock()
_cv2_stub.imencode = MagicMock(return_value=(True, np.zeros(100, dtype=np.uint8)))
_cv2_stub.putText = MagicMock()
_cv2_stub.IMWRITE_JPEG_QUALITY = 1
_cv2_stub.FONT_HERSHEY_SIMPLEX = 0
sys.modules.setdefault("cv2", _cv2_stub)

_ultra_stub = types.ModuleType("ultralytics")
_ultra_stub.YOLO = MagicMock()
sys.modules.setdefault("ultralytics", _ultra_stub)

from cat_detector.routers.stream import (  # noqa: E402
    _encode_frame,
    _placeholder_frame,
)


class TestEncodeFrame:
    def test_returns_bytes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(_cv2_stub, "imencode", return_value=(True, np.array([0xFF, 0xD8], dtype=np.uint8))):
            result = _encode_frame(frame)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_empty_on_failure(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(_cv2_stub, "imencode", return_value=(False, None)):
            result = _encode_frame(frame)
        assert result == b""


class TestPlaceholderFrame:
    def test_returns_jpeg_bytes(self):
        with patch.object(_cv2_stub, "imencode", return_value=(True, np.array([0xFF, 0xD8], dtype=np.uint8))):
            result = _placeholder_frame()
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestStreamEndpointRegistered:
    """Verify the /api/stream route exists on the app."""

    def test_route_exists(self):
        # Ensure real submodules for app import
        for mod in ("cat_detector.models", "cat_detector.database", "cat_detector.detector", "cat_detector.webhook"):
            existing = sys.modules.get(mod)
            if existing is not None and isinstance(existing, MagicMock):
                del sys.modules[mod]

        from cat_detector.main import app
        routes = [r.path for r in app.routes]
        assert "/api/stream" in routes
