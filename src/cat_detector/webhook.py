import logging
from datetime import datetime

import httpx

from cat_detector.config import settings

logger = logging.getLogger(__name__)


async def _post_webhook(url: str, payload: dict, label: str) -> bool:
    """POST payload to url with optional secret header. Returns True on success."""
    headers = {"Content-Type": "application/json"}
    if settings.webhook_secret:
        headers["X-Webhook-Secret"] = settings.webhook_secret

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info("Webhook %s fired (status %s)", label, response.status_code)
            return True
    except httpx.HTTPError as exc:
        logger.error("Webhook %s failed: %s", label, exc)
        return False


async def call_webhook(
    detection_id: int,
    cat_count: int,
    dog_count: int,
    confidence: float,
) -> bool:
    """Fire the sprinkler-on webhook. Returns True on success."""
    if not settings.webhook_url:
        logger.warning("WEBHOOK_URL not set — skipping webhook call")
        return False

    payload = {
        "event": "cat_detected",
        "detection_id": detection_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cat_count": cat_count,
        "dog_count": dog_count,
        "max_confidence": round(confidence, 4),
    }
    return await _post_webhook(settings.webhook_url, payload, "on")


async def call_webhook_off(detection_id: int) -> bool:
    """Fire the sprinkler-off webhook. No-op if WEBHOOK_OFF_URL isn't set —
    use that to rely on the valve's own auto-close instead."""
    if not settings.webhook_off_url:
        return False

    payload = {
        "event": "cat_clear",
        "detection_id": detection_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return await _post_webhook(settings.webhook_off_url, payload, "off")
