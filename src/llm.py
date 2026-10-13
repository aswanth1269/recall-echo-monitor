"""Provider-agnostic LLM client for the Day 3 enrichment step.

Every free LLM tier worth using speaks the OpenAI chat-completions wire
format, so this module points the `openai` client at whichever provider
.env names. Switching provider is three env vars, not a code change --
which matters, because free tiers change without notice: Google stopped
publishing its limits, and Groq moved the Llama models to enterprise
pricing.

Two things here exist specifically because the tier is free:

  * **Disk cache.** Every response is cached by hash of (model, system,
    prompt), so re-running a script costs zero requests. With a daily cap
    in the hundreds, that is the difference between iterating freely and
    burning a day's quota on a typo.
  * **Backoff that honours Retry-After.** On a free tier a 429 is the
    normal case, not an exception.

Usage:
    from src.llm import complete, complete_json, batched

    text = complete("Summarise this defect: ...")
    data = complete_json('Return {"component": "..."} for: ...')

Check it works:
    python -m src.llm --smoke
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
API_KEY = os.getenv("LLM_API_KEY", "")
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ROOT / "data/interim/llm_cache"))

MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "6"))
PLACEHOLDERS = {"", "replace-me", "your-key-here", "sk-ant-replace-me"}

T = TypeVar("T")


class LLMNotConfigured(RuntimeError):
    """Raised when LLM_API_KEY is missing, so the failure is obvious."""


def _client():
    """Build the OpenAI-compatible client, or explain what is missing."""
    if API_KEY.strip() in PLACEHOLDERS:
        raise LLMNotConfigured(
            "LLM_API_KEY is not set in .env.\n"
            "  Get a free key at https://aistudio.google.com/apikey\n"
            "  then set LLM_API_KEY=... in .env (see .env.example)."
        )
    from openai import OpenAI

    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def _cache_key(model: str, system: str | None, prompt: str, **kw: Any) -> str:
    blob = json.dumps([model, system, prompt, sorted(kw.items())], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> str:
    """One chat completion, returned as text. Cached on disk by default."""
    model = model or MODEL
    key = _cache_key(model, system, prompt, max_tokens=max_tokens,
                     temperature=temperature)
    path = CACHE_DIR / f"{key}.json"

    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    text = _call_with_backoff(model, messages, max_tokens, temperature)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": model, "prompt": prompt, "text": text}),
            encoding="utf-8",
        )
    return text


def _call_with_backoff(model: str, messages: list[dict[str, str]],
                       max_tokens: int, temperature: float) -> str:
    """Call the provider, retrying 429s and 5xx. Free tiers 429 routinely."""
    import openai

    client = _client()
    last: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except openai.RateLimitError as exc:
            last = exc
            wait = _retry_after(exc) or min(2 ** attempt + random.random(), 60)
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            last = exc
            wait = min(2 ** attempt + random.random(), 30)
        except openai.APIStatusError as exc:
            if exc.status_code < 500:
                raise  # 400/401/404 will not fix themselves on a retry
            last = exc
            wait = min(2 ** attempt + random.random(), 30)

        if attempt < MAX_ATTEMPTS - 1:
            print(
                f"  [llm] {type(last).__name__}, retry "
                f"{attempt + 1}/{MAX_ATTEMPTS - 1} in {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"LLM call failed after {MAX_ATTEMPTS} attempts: {last}\n"
        "If that was a daily-quota 429, the cap resets on the provider's "
        "clock. Cached results are kept, so re-running resumes."
    ) from last


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


_FENCE = re.compile(r"^\s*`{3}(?:json)?\s*|\s*`{3}\s*$", re.I)


def complete_json(prompt: str, **kw: Any) -> Any:
    """Like complete(), but parse the reply as JSON.

    Small cheap models wrap JSON in markdown fences and add prose even when
    told not to, so strip fences first and fall back to the outermost
    {...} / [...] span.
    """
    raw = complete(prompt, **kw)
    text = _FENCE.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"[\{\[].*[\}\]]", text, re.S)
        if not match:
            raise ValueError(f"No JSON found in reply: {raw[:400]!r}") from None
        return json.loads(match.group(0))


def batched(items: Sequence[T] | Iterable[T], size: int) -> Iterator[list[T]]:
    """Group items into chunks.

    The binding free-tier limit is requests per day, not tokens, so send
    many records per request: 355 recalls at size=20 is 18 requests, not 355.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _smoke() -> int:
    configured = API_KEY.strip() not in PLACEHOLDERS
    print(f"provider : {BASE_URL}")
    print(f"model    : {MODEL}")
    print(f"key      : {f'set ({len(API_KEY)} chars)' if configured else 'NOT SET'}")
    print(f"cache    : {CACHE_DIR}")
    try:
        reply = complete(
            "Reply with exactly: OK",
            system="You answer with the fewest possible words.",
            max_tokens=16,
            use_cache=False,
        )
    except LLMNotConfigured as exc:
        print(f"\nnot configured:\n{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - the smoke test reports anything
        print(f"\ncall failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"\nreply    : {reply!r}")
    print("\nLLM is reachable.")
    return 0


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        raise SystemExit(_smoke())
    print(__doc__)
