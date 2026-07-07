from __future__ import annotations


def template_caption(neutral_summary: str, style: str) -> str:
    summary = _clip_words(neutral_summary, 16)
    if style == "formal":
        return _ensure_period(summary)
    if style == "sarcastic":
        return _ensure_period(f"{summary}, apparently handling a very serious assignment")
    if style == "humorous_tech":
        return _ensure_period(f"{summary}, basically running a gentle scene-monitoring script")
    if style == "humorous_non_tech":
        return _ensure_period(f"{summary}, looking like it accidentally became the star of the day")
    raise ValueError(f"Unsupported style: {style}")


def _clip_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    clipped = words[:max_words]
    return " ".join(clipped).rstrip(" ,;:.!?")


def _ensure_period(text: str) -> str:
    stripped = text.strip().rstrip(".!?")
    return f"{stripped}."
