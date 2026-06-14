"""
Runtime settings API — view and edit config without a service restart.
"""

import asyncio
import json
import logging
import os
import signal
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from cat_detector.camera import fire_test_trigger

logger = logging.getLogger(__name__)

from cat_detector.camera import get_camera_status
from cat_detector.config import (
    MUTABLE_FIELDS,
    READ_ONLY_FIELDS,
    settings,
    update_runtime_settings,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Secret fields: masked on GET, preserved on PUT when the client sends the placeholder
_SECRET_FIELDS = {
    "webhook_secret",
    "homeassistant_token",
}
_SECRET_PLACEHOLDER = "***"


@router.get("")
def get_settings() -> dict[str, Any]:
    """Return current runtime configuration, separated into mutable / read-only.
    Secret fields come back as '***' when set (never plaintext)."""
    data = settings.model_dump()
    mutable = {}
    for k, v in data.items():
        if k not in MUTABLE_FIELDS:
            continue
        if k in _SECRET_FIELDS:
            mutable[k] = _SECRET_PLACEHOLDER if v else ""
        else:
            mutable[k] = v
    read_only = {k: v for k, v in data.items() if k in READ_ONLY_FIELDS}
    return {
        "mutable": mutable,
        "read_only": read_only,
        "camera_status": get_camera_status(),
    }


@router.put("")
def update_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Apply partial updates to mutable settings. Persists to override file.
    For secret fields, the '***' placeholder is treated as 'keep current value'."""
    # Drop secret fields whose value is the placeholder (means "unchanged")
    updates = {
        k: v for k, v in updates.items()
        if not (k in _SECRET_FIELDS and v == _SECRET_PLACEHOLDER)
    }
    try:
        applied = update_runtime_settings(updates)
    except ValidationError as exc:
        errors = [
            {"field": ".".join(str(loc) for loc in e["loc"]), "msg": e["msg"]}
            for e in exc.errors()
        ]
        raise HTTPException(status_code=422, detail={"errors": errors})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"applied": applied, "camera_status": get_camera_status()}


@router.post("/test-sprinkler")
async def test_sprinkler() -> StreamingResponse:
    """Stream Server-Sent Events as the test progresses: each step (sending
    open, polling, confirmed, dwell, close, etc) yields a 'log' event; a
    final 'done' event carries ok/error and the actual measured sprinkle
    duration in milliseconds (time between confirmed open and confirmed
    close). Reuses the same code path as a real trigger, so this is a
    faithful proxy for production behavior."""
    async def stream():
        async for event in fire_test_trigger():
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Disable proxy buffering so events flush immediately
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/restart")
async def restart_service() -> dict[str, str]:
    """
    Send SIGTERM to the current process. If supervised (systemd, launchd,
    docker restart:always), the service restarts. Otherwise it just exits.
    """
    logger.warning("Restart requested via API — sending SIGTERM")

    async def _delayed_shutdown() -> None:
        # Let the HTTP response finish before signaling
        await asyncio.sleep(0.3)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(_delayed_shutdown())
    return {"status": "shutting_down"}
