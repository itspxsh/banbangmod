from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from .fallbacks import template_caption
from .prompts import SUPPORTED_STYLES, TECH_WORDS

RepairCallback = Callable[[str, str, str, dict[str, Any] | None], str]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "while",
    "with",
}


def validate_and_repair_captions(
    captions: Mapping[str, str],
    required_styles: tuple[str, ...],
    neutral_summary: str,
    *,
    checklist: dict[str, Any] | None = None,
    repair_callback: RepairCallback | None = None,
) -> dict[str, str]:
    validated: dict[str, str] = {}

    for style in required_styles:
        if style not in SUPPORTED_STYLES:
            raise ValueError(f"Unsupported style requested: {style}")

        candidate = _normalize_caption(captions.get(style, ""))
        reasons = _validation_reasons(candidate, style, neutral_summary, checklist)
        if reasons and repair_callback is not None:
            repaired = repair_callback(
                neutral_summary,
                candidate or template_caption(neutral_summary, style),
                style,
                checklist,
            )
            candidate = _normalize_caption(repaired)
            reasons = _validation_reasons(candidate, style, neutral_summary, checklist)

        if reasons:
            candidate = _normalize_caption(template_caption(neutral_summary, style))
            reasons = _validation_reasons(candidate, style, neutral_summary, checklist)

        if reasons:
            raise ValueError(f"Unable to validate caption for style '{style}': {', '.join(reasons)}")

        validated[style] = candidate

    return validated


def _validation_reasons(
    caption: str,
    style: str,
    neutral_summary: str,
    checklist: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if not caption:
        reasons.append("empty")
        return reasons

    word_count = len(caption.split())
    if word_count < 10 or word_count > 25:
        reasons.append(f"word_count={word_count}")

    if caption.count(".") + caption.count("!") + caption.count("?") > 1:
        reasons.append("multiple_sentences")

    tokens = set(_content_tokens(caption))
    summary_tokens = set(_content_tokens(neutral_summary))
    overlap = tokens & summary_tokens
    if len(overlap) < 2:
        reasons.append("low_factual_overlap")

    if checklist:
        setting = str(checklist.get("setting", "")).lower()
        if setting:
            setting_tokens = set(_content_tokens(setting))
            if setting_tokens and not (tokens & setting_tokens):
                reasons.append("missing_setting")

        subjects = {
            token
            for subject in checklist.get("subjects", [])
            if isinstance(subject, str)
            for token in _content_tokens(subject)
        }
        if subjects and not (tokens & subjects):
            reasons.append("missing_subject")

    caption_tokens_lower = {token.lower() for token in _word_tokens(caption)}
    if style == "humorous_tech" and not (caption_tokens_lower & TECH_WORDS):
        reasons.append("missing_tech_reference")
    if style == "humorous_non_tech" and caption_tokens_lower & TECH_WORDS:
        reasons.append("contains_tech_terms")

    return reasons


def _normalize_caption(caption: str) -> str:
    normalized = " ".join(str(caption).strip().split())
    if not normalized:
        return ""
    normalized = normalized.strip().rstrip(" .!?")
    return f"{normalized}."


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _content_tokens(text: str) -> list[str]:
    return [token for token in _word_tokens(text) if token not in STOPWORDS and len(token) > 2]
