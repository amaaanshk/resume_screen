"""Configurable, alias-aware skill extraction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from config import SKILL_ALIASES
from .text_preprocessor import normalize_for_matching


def _contains_alias(text: str, alias: str) -> bool:
    """Match a phrase without matching it inside a larger word."""
    # Treat punctuation used in technical names as token characters too.
    pattern = rf"(?<![\w.+#-]){re.escape(alias.casefold())}(?![\w+#-])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def detect_skills(
    text: str,
    skill_aliases: Mapping[str, Sequence[str]] = SKILL_ALIASES,
) -> list[str]:
    """Return canonical skills detected in source order from the vocabulary."""
    normalized = normalize_for_matching(text)
    detected: list[str] = []
    for canonical, aliases in skill_aliases.items():
        if any(_contains_alias(normalized, alias) for alias in aliases):
            detected.append(canonical)
    return detected


def compare_skills(
    required_skills: Sequence[str],
    candidate_skills: Sequence[str],
) -> dict[str, list[str]]:
    """Separate required matches, missing requirements, and extra skills."""
    required = list(dict.fromkeys(required_skills))
    candidate = list(dict.fromkeys(candidate_skills))
    required_set = set(required)
    candidate_set = set(candidate)
    return {
        "matched_required_skills": [skill for skill in required if skill in candidate_set],
        "missing_required_skills": [skill for skill in required if skill not in candidate_set],
        "additional_candidate_skills": [
            skill for skill in candidate if skill not in required_set
        ],
    }


def extract_and_compare_skills(
    job_description: str,
    resume_text: str,
    skill_aliases: Mapping[str, Sequence[str]] = SKILL_ALIASES,
) -> dict[str, list[str]]:
    """Extract skills from both documents and return comparison details."""
    required = detect_skills(job_description, skill_aliases)
    candidate = detect_skills(resume_text, skill_aliases)
    comparison = compare_skills(required, candidate)
    comparison["required_skills"] = required
    comparison["candidate_skills"] = candidate
    return comparison
