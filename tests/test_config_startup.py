"""
Unit tests for config startup error handling — US-7 · Environment-Based Configuration (#6110).
Tests that _load_settings() produces friendly error output and exits on invalid config.
"""

from unittest.mock import patch

import pytest

from cat_detector.config import Settings, _load_settings


class TestLoadSettingsFriendlyErrors:
    """AC2: Missing required variables cause a clear error message, not a runtime crash."""

    def test_valid_config_returns_settings(self):
        s = _load_settings()
        assert isinstance(s, Settings)

    def test_invalid_confidence_prints_field_name_and_exits(self, monkeypatch, capsys):
        monkeypatch.setenv("DETECTION_CONFIDENCE", "5.0")

        with pytest.raises(SystemExit) as exc_info:
            _load_settings()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Invalid configuration" in captured.out
        assert "detection_confidence" in captured.out

    def test_invalid_port_prints_field_name_and_exits(self, monkeypatch, capsys):
        monkeypatch.setenv("PORT", "0")

        with pytest.raises(SystemExit) as exc_info:
            _load_settings()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "port" in captured.out

    def test_multiple_errors_lists_all_fields(self, monkeypatch, capsys):
        monkeypatch.setenv("DETECTION_CONFIDENCE", "2.0")
        monkeypatch.setenv("PORT", "-1")
        monkeypatch.setenv("DETECTION_INTERVAL_SECONDS", "0")

        with pytest.raises(SystemExit) as exc_info:
            _load_settings()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "detection_confidence" in captured.out
        assert "port" in captured.out
        assert "detection_interval_seconds" in captured.out

    def test_env_example_reference_in_error(self, monkeypatch, capsys):
        monkeypatch.setenv("DETECTION_CONFIDENCE", "99")

        with pytest.raises(SystemExit):
            _load_settings()

        captured = capsys.readouterr()
        assert ".env.example" in captured.out
