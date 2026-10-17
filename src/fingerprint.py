"""A short, stable hash of the text a person or the LLM looked at.

Review decisions and matcher verdicts store it; when the underlying text
changes, the hash changes, so an old decision is known to be stale instead of
being silently applied to text nobody reviewed.
"""

from __future__ import annotations

import hashlib

SEP = "\x1f"  # ASCII unit separator: never appears in recall text


def fingerprint(*parts: object) -> str:
    """16 hex chars of SHA-256 over the parts joined by SEP (None counts as "")."""
    blob = SEP.join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
