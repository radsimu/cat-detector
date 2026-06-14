"""
Unit tests for Settings validation logic.
Uses monkeypatch to set environment variables — no .env file needed.
"""

import pytest
from pydantic import ValidationError

from cat_detector.config import Settings


class TestDefaultsLoadSuccessfully:
    """Settings with default values should instantiate without errors."""

    def test_defaults_are_valid(self):
        s = Settings()
        assert s.detection_confidence == 0.35
        assert s.port == 8000
        assert s.detection_interval_seconds == 5
        assert s.webhook_cooldown_seconds == 300
        assert s.camera_reconnect_max_delay == 60


class TestDetectionConfidenceValidation:
    def test_valid_confidence(self):
        s = Settings(detection_confidence=0.75)
        assert s.detection_confidence == 0.75

    def test_confidence_zero_is_valid(self):
        s = Settings(detection_confidence=0.0)
        assert s.detection_confidence == 0.0

    def test_confidence_one_is_valid(self):
        s = Settings(detection_confidence=1.0)
        assert s.detection_confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError, match="detection_confidence"):
            Settings(detection_confidence=1.5)

    def test_confidence_negative_raises(self):
        with pytest.raises(ValidationError, match="detection_confidence"):
            Settings(detection_confidence=-0.1)


class TestPortValidation:
    def test_valid_port(self):
        s = Settings(port=3000)
        assert s.port == 3000

    def test_port_one_is_valid(self):
        s = Settings(port=1)
        assert s.port == 1

    def test_port_65535_is_valid(self):
        s = Settings(port=65535)
        assert s.port == 65535

    def test_port_zero_raises(self):
        with pytest.raises(ValidationError, match="port"):
            Settings(port=0)

    def test_port_too_high_raises(self):
        with pytest.raises(ValidationError, match="port"):
            Settings(port=99999)


class TestDetectionIntervalValidation:
    def test_valid_interval(self):
        s = Settings(detection_interval_seconds=10)
        assert s.detection_interval_seconds == 10

    def test_zero_interval_raises(self):
        with pytest.raises(ValidationError, match="detection_interval_seconds"):
            Settings(detection_interval_seconds=0)

    def test_negative_interval_raises(self):
        with pytest.raises(ValidationError, match="detection_interval_seconds"):
            Settings(detection_interval_seconds=-5)


class TestWebhookCooldownValidation:
    def test_valid_cooldown(self):
        s = Settings(webhook_cooldown_seconds=60)
        assert s.webhook_cooldown_seconds == 60

    def test_zero_cooldown_is_valid(self):
        s = Settings(webhook_cooldown_seconds=0)
        assert s.webhook_cooldown_seconds == 0

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValidationError, match="webhook_cooldown_seconds"):
            Settings(webhook_cooldown_seconds=-1)


class TestCameraReconnectMaxDelayValidation:
    def test_valid_delay(self):
        s = Settings(camera_reconnect_max_delay=30)
        assert s.camera_reconnect_max_delay == 30

    def test_zero_delay_raises(self):
        with pytest.raises(ValidationError, match="camera_reconnect_max_delay"):
            Settings(camera_reconnect_max_delay=0)

    def test_negative_delay_raises(self):
        with pytest.raises(ValidationError, match="camera_reconnect_max_delay"):
            Settings(camera_reconnect_max_delay=-10)


class TestEnvironmentVariableOverrides:
    def test_env_overrides_defaults(self, monkeypatch):
        monkeypatch.setenv("DETECTION_CONFIDENCE", "0.8")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("CAMERA_URL", "rtsp://example.com/stream")
        s = Settings()
        assert s.detection_confidence == 0.8
        assert s.port == 9000
        assert s.camera_url == "rtsp://example.com/stream"

    def test_invalid_env_raises(self, monkeypatch):
        monkeypatch.setenv("DETECTION_CONFIDENCE", "2.0")
        with pytest.raises(ValidationError, match="detection_confidence"):
            Settings()
