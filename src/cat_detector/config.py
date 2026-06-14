import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Runtime overrides are persisted here, loaded after .env on startup,
# and updated whenever the user saves settings from the UI.
_OVERRIDES_FILE = Path("settings_overrides.json")

# Fields that can be mutated at runtime (apply without restart)
MUTABLE_FIELDS = {
    "detection_confidence",
    "detection_interval_seconds",
    "webhook_url",
    "webhook_off_url",
    "webhook_secret",
    "webhook_cooldown_seconds",
    "webhook_min_consecutive_frames",
    "webhook_min_confidence",
    "webhook_low_confidence_threshold",
    "webhook_low_confidence_consecutive_frames",
    "webhook_instant_fire_confidence",
    "roi_polygon",
    "sprinkler_duration_seconds",
    "save_human_only_frames",
    "recording_pre_roll_seconds",
    "recording_total_seconds",
    "recordings_max_count",
    "homeassistant_url",
    "homeassistant_token",
    "homeassistant_valve_entity",
    "homeassistant_notify_service",
    "valve_max_open_seconds",
}

# Fields that are shown read-only (a restart is required to apply changes)
READ_ONLY_FIELDS = {
    "camera_url",
    "camera_reconnect_max_delay",
    "detection_backend",
    "detection_model",
    "detection_hef_path",
    "database_url",
    "frames_dir",
    "recordings_dir",
    "host",
    "port",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Camera
    camera_url: str = "0"
    camera_reconnect_max_delay: int = 60

    # Detection
    detection_backend: str = "ultralytics"  # "ultralytics" (CPU) or "hailo" (Hailo HAT+)
    detection_model: str = "yolov8x.pt"
    # Path to the compiled HEF model when detection_backend == "hailo".
    detection_hef_path: str = ""
    detection_confidence: float = 0.35
    detection_interval_seconds: int = 5
    webhook_cooldown_seconds: int = 300
    webhook_min_consecutive_frames: int = 2  # consecutive trigger-worthy frames required
    webhook_min_confidence: float = 0.5  # min max_cat_confidence to fire webhook
    # Sustained-low-confidence path: if confidence is between
    # webhook_low_confidence_threshold and webhook_min_confidence but holds
    # for `webhook_low_confidence_consecutive_frames` in a row, also fire.
    webhook_low_confidence_threshold: float = 0.40
    webhook_low_confidence_consecutive_frames: int = 5
    # Instant-fire path: a single CAT-class detection at this confidence or
    # higher (with no human present) fires immediately, skipping the
    # consecutive-frame gate. Set to 1.01 to disable.
    webhook_instant_fire_confidence: float = 0.85

    # Region-of-interest mask: JSON-serialized list of normalized [x, y]
    # vertices (each in 0.0..1.0) describing a closed polygon. Detections
    # whose box-center falls outside the polygon are dropped before the
    # trigger logic. Empty string = whole frame.
    roi_polygon: str = ""
    sprinkler_duration_seconds: float = 0.5  # how long the sprinkler runs per trigger
    save_human_only_frames: bool = False  # skip frames where only humans are detected

    # Recording (video clips saved when webhook fires)
    recordings_dir: str = "./data/recordings"
    recording_pre_roll_seconds: int = 5
    recording_total_seconds: int = 15
    recordings_max_count: int = 50  # oldest deleted when exceeded

    # Webhook
    webhook_url: str = ""  # fired when a cat is detected (e.g. IFTTT "open valve")
    webhook_off_url: str = ""  # fired `sprinkler_duration_seconds` after the open (e.g. IFTTT "close valve"). Leave blank if the valve auto-closes.
    webhook_secret: str = ""

    # Home Assistant — HA holds the SmartThings/eWeLink OAuth state and
    # exposes the valve as an entity. We call HA's REST API; HA handles
    # any 24h token rotation internally.
    homeassistant_url: str = ""  # e.g. http://192.168.100.189:8123
    homeassistant_token: str = ""  # Long-Lived Access Token from HA → Profile → Security
    homeassistant_valve_entity: str = ""  # entity_id, e.g. valve.swv
    # HA notify service for stuck-valve alerts. Empty = use persistent_notification.
    # Example: 'notify.mobile_app_radu_iphone' (install HA Companion app + sign in).
    homeassistant_notify_service: str = ""
    # Watchdog: if the valve is reported 'open' continuously for longer than
    # this, the watchdog hammers close + fires a notification. Should be
    # bigger than sprinkler_duration_seconds + HA round-trip (typically 2s).
    valve_max_open_seconds: float = 5.0

    # Storage
    database_url: str = "sqlite:///./cat_detector.db"
    frames_dir: str = "./data/frames"

    # Web UI
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("detection_confidence")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("detection_confidence must be between 0.0 and 1.0")
        return v

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v

    @field_validator("detection_interval_seconds")
    @classmethod
    def _interval_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("detection_interval_seconds must be greater than 0")
        return v

    @field_validator("webhook_cooldown_seconds")
    @classmethod
    def _cooldown_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("webhook_cooldown_seconds must be >= 0")
        return v

    @field_validator("camera_reconnect_max_delay")
    @classmethod
    def _reconnect_delay_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("camera_reconnect_max_delay must be greater than 0")
        return v

    @field_validator("webhook_min_consecutive_frames")
    @classmethod
    def _min_consecutive_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("webhook_min_consecutive_frames must be >= 1")
        return v

    @field_validator("webhook_min_confidence")
    @classmethod
    def _min_confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("webhook_min_confidence must be between 0.0 and 1.0")
        return v

    @field_validator("webhook_low_confidence_threshold")
    @classmethod
    def _low_confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("webhook_low_confidence_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("webhook_low_confidence_consecutive_frames")
    @classmethod
    def _low_consecutive_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("webhook_low_confidence_consecutive_frames must be >= 1")
        return v

    @field_validator("webhook_instant_fire_confidence")
    @classmethod
    def _instant_fire_in_range(cls, v: float) -> float:
        # >1.0 is the "disabled" sentinel; otherwise must be a valid probability
        if not 0.0 <= v <= 1.01:
            raise ValueError("webhook_instant_fire_confidence must be between 0.0 and 1.01")
        return v

    @field_validator("roi_polygon")
    @classmethod
    def _roi_polygon_well_formed(cls, v: str) -> str:
        """Empty string disables the mask. Otherwise must be a JSON list of
        at least 3 [x, y] points with each coordinate in [0.0, 1.0]."""
        if not v:
            return v
        try:
            pts = json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError(f"roi_polygon is not valid JSON: {exc}")
        if not isinstance(pts, list) or len(pts) < 3:
            raise ValueError("roi_polygon must be a list of at least 3 points")
        for p in pts:
            if (
                not isinstance(p, (list, tuple)) or len(p) != 2
                or not all(isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in p)
            ):
                raise ValueError("each polygon point must be [x, y] with 0.0 <= x,y <= 1.0")
        return v

    @field_validator("detection_backend")
    @classmethod
    def _backend_supported(cls, v: str) -> str:
        if v not in ("ultralytics", "hailo"):
            raise ValueError("detection_backend must be 'ultralytics' or 'hailo'")
        return v

    @field_validator("valve_max_open_seconds")
    @classmethod
    def _valve_max_open_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("valve_max_open_seconds must be greater than 0")
        if v > 600:
            raise ValueError("valve_max_open_seconds capped at 600s (10 min) for safety")
        return v

    @field_validator("sprinkler_duration_seconds")
    @classmethod
    def _sprinkler_duration_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sprinkler_duration_seconds must be greater than 0")
        if v > 300:
            raise ValueError("sprinkler_duration_seconds capped at 300s (5 min) for safety")
        return v


def _load_overrides() -> dict[str, Any]:
    """Load the runtime overrides JSON file if present."""
    if not _OVERRIDES_FILE.exists():
        return {}
    try:
        return json.loads(_OVERRIDES_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s — ignoring", _OVERRIDES_FILE, exc)
        return {}


def _load_settings() -> Settings:
    """Load settings from .env + runtime overrides, with friendly errors."""
    try:
        base = Settings()
        overrides = _load_overrides()
        if overrides:
            return base.model_copy(update=overrides)
        return base
    except ValidationError as exc:
        print("ERROR: Invalid configuration. Fix the following issues:\n")
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            print(f"  - {field}: {error['msg']}")
        print("\nSee .env.example for valid defaults.")
        sys.exit(1)


def update_runtime_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Validate updates, mutate the live settings object, persist overrides.

    Returns the dict of values that were actually applied.
    Raises ValidationError if any value fails validation.
    """
    # Reject attempts to change read-only fields
    rejected = set(updates) & READ_ONLY_FIELDS
    if rejected:
        raise ValueError(f"Cannot change at runtime: {sorted(rejected)}")

    unknown = set(updates) - MUTABLE_FIELDS
    if unknown:
        raise ValueError(f"Unknown setting(s): {sorted(unknown)}")

    # Validate by constructing a new Settings with current + updates
    current = settings.model_dump()
    current.update(updates)
    Settings(**current)  # raises ValidationError on bad values

    # Apply in-place on the live settings object so existing references see changes
    for key, value in updates.items():
        setattr(settings, key, value)

    # Persist overrides (merge with existing file contents)
    existing = _load_overrides()
    existing.update(updates)
    _OVERRIDES_FILE.write_text(json.dumps(existing, indent=2))
    logger.info("Runtime settings updated: %s", list(updates.keys()))
    return updates


settings = _load_settings()
