"""NLP analysis engine: TF-IDF similarity, Cosine similarity, Experience extraction, and Scoring."""

from __future__ import annotations

import re
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """Compute semantic TF-IDF cosine similarity between two texts (0.0 to 100.0)."""
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
        )
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim) * 100, 1)
    except Exception:
        return 0.0


def extract_candidate_info(text: str) -> dict[str, Any]:
    """Extract candidate contact information (email, phone, linkedin/github links)."""
    # Email
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    email = email_match.group(0) if email_match else "Not detected"

    # Phone number (various common formats)
    phone_match = re.search(
        r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{10,12})",
        text,
    )
    phone = phone_match.group(0).strip() if phone_match else "Not detected"

    # Links
    links = []
    for link in re.findall(r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in/[\w-]+|github\.com/[\w-]+)", text, re.IGNORECASE):
        links.append(link)

    return {
        "email": email,
        "phone": phone,
        "links": list(set(links)),
    }


def estimate_years_of_experience(text: str) -> float:
    """Heuristic extraction of total years of experience from resume text."""
    # Look for explicit patterns like '5+ years of experience', '3 years experience'
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience",
        r"experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        r"total\s+experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    ]
    detected_years = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            try:
                detected_years.append(float(m))
            except ValueError:
                pass

    if detected_years:
        return max(detected_years)

    # Secondary heuristic: count distinct year ranges like 2018 - 2022
    year_ranges = re.findall(r"\b(19\d\d|20\d\d)\s*[-–—to]+\s*(19\d\d|20\d\d|present|current)\b", text, re.IGNORECASE)
    total_est = 0.0
    current_year = 2026
    for start, end in year_ranges:
        try:
            s = int(start)
            e = current_year if end.lower() in ("present", "current") else int(end)
            diff = e - s
            if 0 < diff <= 25:
                total_est += diff
        except Exception:
            continue

    if total_est > 0:
        return min(total_est, 30.0)

    return 0.0


def extract_education(text: str) -> list[str]:
    """Detect degrees and academic credentials."""
    degrees = [
        ("Ph.D / Doctorate", [r"\bph\.?d\b", r"\bdoctorate\b", r"\bdoctor of philosophy\b"]),
        ("Master's Degree", [r"\bm\.?s\b", r"\bm\.?tech\b", r"\bmca\b", r"\bmaster'?s\b", r"\bmba\b"]),
        ("Bachelor's Degree", [r"\bb\.?s\b", r"\bb\.?e\b", r"\bb\.?tech\b", r"\bbca\b", r"\bbachelor'?s\b", r"\bb\.?sc\b"]),
        ("Diploma / Associate", [r"\bdiploma\b", r"\bassociate\b"]),
    ]
    found = []
    for name, patterns in degrees:
        if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
            found.append(name)
    return found


def calculate_overall_fit(
    skill_match_pct: float,
    tfidf_sim_pct: float,
    experience_years: float,
    required_min_experience: float = 0.0,
) -> dict[str, Any]:
    """Calculate a weighted holistic candidate fit score with breakdown."""
    # Experience match score
    if required_min_experience <= 0:
        exp_score = min(100.0, (experience_years / 3.0) * 100.0) if experience_years > 0 else 70.0
    else:
        exp_score = min(100.0, (experience_years / required_min_experience) * 100.0)

    # Weights: Skills (45%), NLP Content Similarity (35%), Experience (20%)
    composite_score = (
        (skill_match_pct * 0.45) +
        (tfidf_sim_pct * 0.35) +
        (exp_score * 0.20)
    )
    composite_score = round(min(100.0, max(0.0, composite_score)), 1)

    if composite_score >= 75.0:
        recommendation = "Strong Candidate (Shortlist)"
        badge_color = "#10b981"
        status_tag = "Shortlist"
    elif composite_score >= 55.0:
        recommendation = "Potential Match (Review Needed)"
        badge_color = "#f59e0b"
        status_tag = "Review"
    else:
        recommendation = "Low Alignment (Not Recommended)"
        badge_color = "#ef4444"
        status_tag = "Reject"

    return {
        "composite_score": composite_score,
        "skill_score": skill_match_pct,
        "tfidf_score": tfidf_sim_pct,
        "exp_score": round(exp_score, 1),
        "recommendation": recommendation,
        "badge_color": badge_color,
        "status_tag": status_tag,
    }
