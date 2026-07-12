"""
Home Assistant REST API client. HA holds the SmartThings (or any other)
cloud integration's OAuth refresh token internally and exposes the smart
water valve as a `valve.*` entity. We just call HA's services endpoint —
no 24-hour token rotation on our side.

Auth uses a Long-Lived Access Token (HA → Profile → Security).
"""

import logging
from typing import Any

import httpx

from cat_detector.config import settings

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    return bool(
        settings.homeassistant_url
        and settings.homeassistant_token
        and settings.homeassistant_valve_entity
    )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.homeassistant_token}",
        "Content-Type": "application/json",
    }


def _base() -> str:
    return settings.homeassistant_url.rstrip("/")


async def set_switch(state: str) -> bool:
    """Open ('on') or close ('off') the configured entity via HA. Dispatches
    by entity domain: a `valve.*` entity gets valve.open_valve/close_valve;
    a `switch.*` entity gets switch.turn_on/turn_off. Kept named
    `set_switch` for parity with the old smartthings module so callers in
    camera.py don't need to change."""
    if not is_configured():
        return False
    if state not in ("on", "off"):
        raise ValueError(f"state must be 'on' or 'off', got {state!r}")

    entity = settings.homeassistant_valve_entity
    domain = entity.split(".", 1)[0] if "." in entity else "valve"
    if domain == "valve":
        service = "open_valve" if state == "on" else "close_valve"
        url = f"{_base()}/api/services/valve/{service}"
    elif domain == "switch":
        service = "turn_on" if state == "on" else "turn_off"
        url = f"{_base()}/api/services/switch/{service}"
    else:
        logger.error("HA: unsupported entity domain %r for %s", domain, entity)
        return False
    body = {"entity_id": entity}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=body, headers=_headers())
        resp.raise_for_status()
        logger.info("HA: %s.%s sent to %s", domain, service, entity)
        return True
    except httpx.HTTPError as exc:
        logger.error("HA %s.%s failed: %s", domain, service, exc)
        return False


def is_open_state(s: str | None) -> bool:
    """Normalize valve/switch open semantics: valves report 'open', switches
    report 'on'. Either counts as 'currently delivering water'."""
    return s in ("open", "on")


def is_closed_state(s: str | None) -> bool:
    """Normalize valve/switch closed semantics: 'closed' or 'off'."""
    return s in ("closed", "off")


async def get_valve_state() -> str | None:
    """Return HA's current state for the configured valve entity.
    Possible values: 'open', 'closed', 'opening', 'closing', 'unavailable',
    or None on transport error. 'unavailable' means HA can't reach the
    device — the close call may have been silently lost."""
    info = await get_valve_state_info()
    return info.get("state") if info else None


async def get_valve_state_info() -> dict[str, Any] | None:
    """Return HA's full state record for the valve: state + last_changed +
    last_updated + last_reported timestamps. Needed to distinguish a real
    state transition from a stale cached value — the SmartThings c2c
    integration optimistically reports state='open' even when the physical
    device is offline, so 'open' alone isn't proof of actuation."""
    if not is_configured():
        return None
    url = f"{_base()}/api/states/{settings.homeassistant_valve_entity}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=_headers())
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        logger.warning("HA: get state failed: %s", exc)
        return None


async def refresh_entity() -> bool:
    """Force HA to re-poll the configured valve entity via the
    homeassistant.update_entity service. Use when the entity is stuck at
    'unavailable' or its last_reported is suspiciously stale — the
    SmartThings c2c integration sometimes desyncs and a refresh nudges
    HA to reach out to the cloud again. Best-effort; doesn't raise."""
    if not is_configured():
        return False
    url = f"{_base()}/api/services/homeassistant/update_entity"
    body = {"entity_id": settings.homeassistant_valve_entity}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=body, headers=_headers())
        resp.raise_for_status()
        logger.info("HA: forced refresh of %s", settings.homeassistant_valve_entity)
        return True
    except httpx.HTTPError as exc:
        logger.warning("HA refresh_entity failed: %s", exc)
        return False


async def notify(message: str, title: str = "Cat detector alert") -> bool:
    """Send a notification via HA. Uses settings.homeassistant_notify_service
    if configured (e.g. 'notify.mobile_app_radu_phone'); otherwise falls
    back to persistent_notification.create so the alert at least lands in
    the HA web UI. Best-effort — does not raise."""
    if not is_configured():
        return False
    service_path = settings.homeassistant_notify_service or "persistent_notification/create"
    # Allow user to write either 'notify.mobile_app_foo' or 'notify/mobile_app_foo'
    service_path = service_path.replace(".", "/", 1)
    url = f"{_base()}/api/services/{service_path}"
    body = {"message": message, "title": title}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=body, headers=_headers())
        resp.raise_for_status()
        logger.info("HA notify sent via %s: %s", service_path, message)
        return True
    except httpx.HTTPError as exc:
        logger.error("HA notify via %s failed: %s", service_path, exc)
        return False


async def test_connection() -> dict[str, Any]:
    """Verify the LLAT works and the valve entity exists. Returns a diagnostic."""
    if not is_configured():
        return {
            "ok": False,
            "error": "homeassistant_url / homeassistant_token / homeassistant_valve_entity not all set",
        }
    url = f"{_base()}/api/states/{settings.homeassistant_valve_entity}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        return {
            "ok": True,
            "entity_id": data.get("entity_id"),
            "state": data.get("state"),
            "friendly_name": data.get("attributes", {}).get("friendly_name"),
        }
    except httpx.HTTPStatusError as exc:
        return {"ok": False, "status": exc.response.status_code, "error": exc.response.text}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc)}
