"""OpenAI-compatible chat-completions calls (Story 3.1 plain, Story 3.2 streamed).

No retries, no vendor SDK."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from building_with_rag.settings import get_settings

PROVIDER_NAME = "openai-compatible"
REQUEST_TIMEOUT_SECONDS = 60
TEMPERATURE = 0.0


class ProviderUnavailable(Exception):
    """Message is a safe category (never a URL, key, header or response body)."""


def chat_completion(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    base = (settings.generation_api_base_url or "").strip()
    secret = settings.generation_api_key
    key = secret.get_secret_value().strip() if secret else ""
    missing = [
        name
        for name, value in (("GENERATION_API_BASE_URL", base), ("GENERATION_API_KEY", key))
        if not value
    ]
    if missing:
        raise ProviderUnavailable(f"missing setting: {', '.join(missing)}")
    body: dict[str, Any] = {
        "model": settings.generation_model_name,
        "messages": messages,
        "temperature": TEMPERATURE,
    }
    try:
        response = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException:
        raise ProviderUnavailable("timeout") from None
    except httpx.HTTPError:
        raise ProviderUnavailable("connection error") from None
    except Exception:  # e.g. an invalid URL; never echo it
        raise ProviderUnavailable("request error") from None
    if not 200 <= response.status_code < 300:
        raise ProviderUnavailable(f"http status {response.status_code}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        raise ProviderUnavailable("response missing choices[0].message.content") from None
    if not isinstance(content, str) or not content.strip():
        raise ProviderUnavailable("response missing choices[0].message.content")
    return content


def _settings_or_raise() -> tuple[str, str, str]:
    settings = get_settings()
    base = (settings.generation_api_base_url or "").strip()
    secret = settings.generation_api_key
    key = secret.get_secret_value().strip() if secret else ""
    missing = [
        name
        for name, value in (("GENERATION_API_BASE_URL", base), ("GENERATION_API_KEY", key))
        if not value
    ]
    if missing:
        raise ProviderUnavailable(f"missing setting: {', '.join(missing)}")
    return base, key, settings.generation_model_name


def stream_chat_completion(messages: list[dict[str, str]]) -> Iterator[str]:
    """Yield content pieces of a streamed OpenAI-compatible completion until [DONE].

    Failures raise ProviderUnavailable with a safe category only.
    """
    base, key, model = _settings_or_raise()
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    received = 0
    finished = False
    try:
        with httpx.stream(
            "POST",
            f"{base.rstrip('/')}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            if not 200 <= response.status_code < 300:
                raise ProviderUnavailable(f"http status {response.status_code}")
            for line in response.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = True
                    break
                try:
                    choices = json.loads(data).get("choices")
                    choice = choices[0] if choices else None
                    piece = choice["delta"].get("content") if choice else None
                    if choice and choice.get("finish_reason"):
                        finished = True
                except Exception:
                    raise ProviderUnavailable("malformed stream") from None
                if isinstance(piece, str) and piece:
                    received += 1
                    yield piece
                if finished:
                    break
    except ProviderUnavailable:
        raise
    except httpx.TimeoutException:
        raise ProviderUnavailable("interrupted" if received else "timeout") from None
    except httpx.HTTPError:
        raise ProviderUnavailable("interrupted" if received else "connection error") from None
    except Exception:  # e.g. an invalid URL; never echo it
        raise ProviderUnavailable("interrupted" if received else "request error") from None
    if not received:
        raise ProviderUnavailable("stream had no content")
    if not finished:
        raise ProviderUnavailable("interrupted")
