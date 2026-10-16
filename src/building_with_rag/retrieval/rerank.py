"""Story 4.2: one Voyage-style re-rank call (`POST {base}/rerank`) with httpx.

Request: {"query", "documents": [str], "model", "top_k"}. Response:
{"data": [{"index": <int into documents>, "relevance_score": <number>}, ...]}.
No retries, no vendor SDK, no fallback key. Errors carry a safe category only.
"""

from __future__ import annotations

import math

import httpx

from building_with_rag.settings import get_settings


class RerankUnavailable(Exception):
    """Message is a safe category (never a URL, key, header or response body)."""


def check_limits(settings) -> None:
    candidate, send, ret = (
        settings.rerank_candidate_limit,
        settings.rerank_send_limit,
        settings.rerank_return_limit,
    )
    if not (1 <= ret <= send <= candidate):
        raise RerankUnavailable(
            "invalid setting: need 1 <= RERANK_RETURN_LIMIT <= RERANK_SEND_LIMIT <= "
            "RERANK_CANDIDATE_LIMIT"
        )
    if settings.rerank_request_timeout_seconds <= 0:
        raise RerankUnavailable("invalid setting: RERANK_REQUEST_TIMEOUT_SECONDS")


def rerank(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Score every document; returns `(index, score)` pairs, best first, one per document."""
    settings = get_settings()
    base = (settings.rerank_api_base_url or "").strip()
    secret = settings.rerank_api_key
    key = secret.get_secret_value().strip() if secret else ""
    missing = [
        n for n, v in (("RERANK_API_BASE_URL", base), ("RERANK_API_KEY", key)) if not v
    ]
    if missing:
        raise RerankUnavailable(f"missing setting: {', '.join(missing)}")
    body = {
        "query": query,
        "documents": documents,
        "model": settings.rerank_model_name,
        "top_k": len(documents),  # all scores back, so the cut after re-ranking is inspectable
    }
    try:
        response = httpx.post(
            f"{base.rstrip('/')}/rerank",
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            timeout=settings.rerank_request_timeout_seconds,
        )
    except httpx.TimeoutException:
        raise RerankUnavailable("timeout") from None
    except httpx.HTTPError:
        raise RerankUnavailable("connection error") from None
    except Exception:  # e.g. an invalid URL; never echo it
        raise RerankUnavailable("request error") from None
    if not 200 <= response.status_code < 300:
        raise RerankUnavailable(f"http status {response.status_code}")
    return parse_response(response, len(documents))


def parse_response(response, sent: int) -> list[tuple[int, float]]:
    malformed = RerankUnavailable("malformed re-rank response")
    try:
        data = response.json()["data"]
    except Exception:
        raise malformed from None
    if not isinstance(data, list) or len(data) != sent:
        raise malformed
    seen: set[int] = set()
    pairs: list[tuple[int, float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise malformed
        index, score = item.get("index"), item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < sent:
            raise malformed
        if index in seen:
            raise malformed
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise malformed
        if not math.isfinite(score):
            raise malformed
        seen.add(index)
        pairs.append((index, float(score)))
    return pairs
