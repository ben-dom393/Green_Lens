"""LLM client: Groq API only, with proactive throttle + rate-limit retry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import os
import time
from collections import deque

import httpx

from config import GROQ_MODEL

logger = logging.getLogger(__name__)

# Groq limits (paid/upgraded tier):
# - 500 requests/minute for llama-3.3-70b-versatile
# - Rate limit resets every 60 seconds
_MAX_RETRIES = 5
_BASE_WAIT = 3   # seconds — short initial wait
_MAX_WAIT = 30   # cap at 30s (shorter since higher limit)

# Proactive throttle: stay under 500 req/min by spacing calls
_RATE_LIMIT_WINDOW = 60.0  # seconds
_RATE_LIMIT_MAX_CALLS = 480  # leave 20-call buffer under the 500/min limit


class LLMClient:
    """LLM client: Groq API with proactive throttle + rate-limit retry."""

    def __init__(self):
        self._api_key: str | None = None
        self._total_calls = 0
        self._rate_limited_count = 0
        # Sliding window of recent call timestamps for proactive throttling
        self._call_timestamps: deque[float] = deque()

    def _get_api_key(self) -> str:
        """Get Groq API key from environment."""
        if self._api_key is None:
            self._api_key = os.environ.get("GROQ_API_KEY", "")
        return self._api_key

    def _is_rate_limit_error(self, status_code: int, response_body: str) -> bool:
        """Check if error is a rate limit (429) or server overload (503)."""
        return status_code in (429, 503)

    def _extract_retry_after(self, headers: httpx.Headers) -> float | None:
        """Extract retry-after from response headers if available."""
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        # Groq also uses x-ratelimit-reset-requests / x-ratelimit-reset-tokens
        for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            val = headers.get(header)
            if val:
                try:
                    # These can be like "1m30s" or "30s" or just seconds
                    if "m" in val:
                        parts = val.replace("s", "").split("m")
                        return float(parts[0]) * 60 + float(parts[1] or 0)
                    elif "s" in val:
                        return float(val.replace("s", ""))
                    else:
                        return float(val)
                except (ValueError, IndexError):
                    pass
        return None

    def _throttle(self):
        """Proactive rate limiting: sleep if we're approaching 30 req/min.

        Uses a sliding window of call timestamps. If we've made
        _RATE_LIMIT_MAX_CALLS in the last 60 seconds, sleep until
        the oldest call in the window expires.
        """
        now = time.time()

        # Purge timestamps older than the window
        while self._call_timestamps and (now - self._call_timestamps[0]) > _RATE_LIMIT_WINDOW:
            self._call_timestamps.popleft()

        if len(self._call_timestamps) >= _RATE_LIMIT_MAX_CALLS:
            # Wait until the oldest call in window falls outside the 60s window
            oldest = self._call_timestamps[0]
            sleep_time = _RATE_LIMIT_WINDOW - (now - oldest) + 1.0  # +1s buffer
            if sleep_time > 0:
                logger.info(
                    "Proactive throttle: %d calls in last 60s (limit %d). "
                    "Sleeping %.1fs to avoid rate limit.",
                    len(self._call_timestamps), _RATE_LIMIT_MAX_CALLS, sleep_time,
                )
                time.sleep(sleep_time)

        # Record this call
        self._call_timestamps.append(time.time())

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text using Groq API with proactive throttle + retry.

        Args:
            prompt: The user/query prompt.
            system: Optional system message.

        Returns:
            Generated text string. Returns a placeholder message if
            the API is unavailable after all retries.
        """
        api_key = self._get_api_key()
        if not api_key:
            return "[LLM unavailable: no GROQ_API_KEY set]"

        # Proactive throttle — sleep BEFORE making the call if needed
        self._throttle()

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = ""
        for attempt in range(_MAX_RETRIES):
            try:
                response = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": GROQ_MODEL, "messages": messages},
                    timeout=90.0,
                )

                if response.status_code == 200:
                    self._total_calls += 1
                    return response.json()["choices"][0]["message"]["content"]

                # Rate limited — wait and retry
                if self._is_rate_limit_error(response.status_code, response.text):
                    self._rate_limited_count += 1
                    retry_after = self._extract_retry_after(response.headers)
                    # Cap header value — Groq per-minute limit resets in 60s,
                    # but headers may report daily/token resets (much longer)
                    if retry_after and retry_after > _MAX_WAIT:
                        retry_after = _MAX_WAIT
                    wait_time = retry_after if retry_after else min(
                        _BASE_WAIT * (2 ** attempt), _MAX_WAIT
                    )
                    logger.warning(
                        "Groq rate limited (attempt %d/%d). "
                        "Waiting %.0fs before retry. "
                        "[total calls: %d, rate limits hit: %d]",
                        attempt + 1, _MAX_RETRIES, wait_time,
                        self._total_calls, self._rate_limited_count,
                    )
                    time.sleep(wait_time)
                    continue

                # Other HTTP errors (auth, bad request, etc.) — don't retry
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.error("Groq API error: %s", last_error)
                break

            except httpx.TimeoutException:
                last_error = "Request timed out"
                wait_time = min(_BASE_WAIT * (2 ** attempt), _MAX_WAIT)
                logger.warning(
                    "Groq timeout (attempt %d/%d). Waiting %.0fs.",
                    attempt + 1, _MAX_RETRIES, wait_time,
                )
                time.sleep(wait_time)
                continue

            except Exception as exc:
                last_error = str(exc)
                logger.error("Groq request failed: %s", exc)
                break

        logger.error(
            "Groq API failed after %d attempts. Last error: %s",
            _MAX_RETRIES, last_error,
        )
        return f"[LLM unavailable: Groq failed after {_MAX_RETRIES} retries — {last_error}]"
