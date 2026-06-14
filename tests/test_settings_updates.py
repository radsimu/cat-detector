"""
Unit tests for runtime settings updates — US-10 (#6332).
"""

import json
from pathlib import Path

import pytest

from cat_detector import config as cfg_module
from cat_detector.config import (
    MUTABLE_FIELDS,
    READ_ONLY_FIELDS,
    settings,
    update_runtime_settings,
)


@pytest.fixture(autouse=True)
def isolated_overrides_file(tmp_path, monkeypatch):
    """Point the overrides file at a temp path so tests don't pollute the repo."""
    override_path = tmp_path / "settings_overrides.json"
    monkeypatch.setattr(cfg_module, "_OVERRIDES_FILE", override_path)
    yield override_path


@pytest.fixture(autouse=True)
def restore_settings():
    """Snapshot settings before each test, restore after."""
    snapshot = settings.model_dump()
    yield
    for k, v in snapshot.items():
        setattr(settings, k, v)


class TestUpdateRuntimeSettings:
    def test_mutates_live_settings(self):
        update_runtime_settings({"detection_confidence": 0.7})
        assert settings.detection_confidence == 0.7

    def test_persists_to_override_file(self, isolated_overrides_file):
        update_runtime_settings({"detection_interval_seconds": 10})
        data = json.loads(isolated_overrides_file.read_text())
        assert data["detection_interval_seconds"] == 10

    def test_rejects_read_only_fields(self):
        with pytest.raises(ValueError, match="Cannot change at runtime"):
            update_runtime_settings({"camera_url": "rtsp://evil"})

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValueError, match="Unknown setting"):
            update_runtime_settings({"not_a_real_field": 1})

    def test_rejects_invalid_values(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            update_runtime_settings({"detection_confidence": 5.0})

    def test_mutable_and_readonly_disjoint(self):
        assert MUTABLE_FIELDS.isdisjoint(READ_ONLY_FIELDS)

    def test_multiple_fields_at_once(self, isolated_overrides_file):
        update_runtime_settings({
            "detection_confidence": 0.5,
            "webhook_cooldown_seconds": 60,
        })
        assert settings.detection_confidence == 0.5
        assert settings.webhook_cooldown_seconds == 60
        data = json.loads(isolated_overrides_file.read_text())
        assert data["detection_confidence"] == 0.5
        assert data["webhook_cooldown_seconds"] == 60
