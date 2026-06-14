"""
Unit tests for webhook — US-3 · Webhook Alert on Cat Detection (#6106).
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub ultralytics
_ultra_stub = types.ModuleType("ultralytics")
_ultra_stub.YOLO = MagicMock()
sys.modules.setdefault("ultralytics", _ultra_stub)
sys.modules.setdefault("cv2", MagicMock())

from cat_detector.webhook import call_webhook  # noqa: E402


class TestCallWebhook:
    @pytest.mark.asyncio
    async def test_skips_when_no_url(self):
        with patch("cat_detector.webhook.settings") as mock_settings:
            mock_settings.webhook_url = ""
            result = await call_webhook(1, cat_count=1, dog_count=0, confidence=0.9)
        assert result is False

    @pytest.mark.asyncio
    async def test_sends_post_with_correct_payload(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("cat_detector.webhook.settings") as mock_settings,
            patch("cat_detector.webhook.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.webhook_url = "https://example.com/hook"
            mock_settings.webhook_secret = ""
            result = await call_webhook(42, cat_count=2, dog_count=1, confidence=0.95)

        assert result is True
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["event"] == "cat_detected"
        assert payload["detection_id"] == 42
        assert payload["cat_count"] == 2
        assert payload["dog_count"] == 1
        assert payload["max_confidence"] == 0.95
        assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_includes_secret_header(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("cat_detector.webhook.settings") as mock_settings,
            patch("cat_detector.webhook.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.webhook_url = "https://example.com/hook"
            mock_settings.webhook_secret = "s3cret"
            await call_webhook(1, cat_count=1, dog_count=0, confidence=0.9)

        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["X-Webhook-Secret"] == "s3cret"

    @pytest.mark.asyncio
    async def test_no_secret_header_when_empty(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("cat_detector.webhook.settings") as mock_settings,
            patch("cat_detector.webhook.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.webhook_url = "https://example.com/hook"
            mock_settings.webhook_secret = ""
            await call_webhook(1, cat_count=1, dog_count=0, confidence=0.9)

        headers = mock_client.post.call_args.kwargs["headers"]
        assert "X-Webhook-Secret" not in headers

    @pytest.mark.asyncio
    async def test_returns_false_on_http_error(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("cat_detector.webhook.settings") as mock_settings,
            patch("cat_detector.webhook.httpx.AsyncClient", return_value=mock_client),
        ):
            mock_settings.webhook_url = "https://example.com/hook"
            mock_settings.webhook_secret = ""
            result = await call_webhook(1, cat_count=1, dog_count=0, confidence=0.9)

        assert result is False
