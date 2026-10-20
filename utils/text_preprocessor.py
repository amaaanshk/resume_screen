"""Conservative text normalization for job descriptions and resumes."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize layout noise while preserving technical terms and punctuation."""
    if not isinstance(text, str):
        raise TypeError("Text to normalize must be a string.")

    normalized = text.replace("\x00", " ")
    normalized = re.sub(r"[\r\n\t\f\v]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_for_matching(text: str) -> str:
    """Create a lowercase matching form without deleting technical symbols."""
    return normalize_text(text).casefold()
