from __future__ import annotations

from typing import Any, Mapping

from .output_schema import ensure_all_styles
from .validator import validate_and_repair_captions


def verify_and_repair_captions(
    captions: Mapping[str, str],
    styles: tuple[str, ...],
    neutral_summary: str,
    *,
    checklist: dict[str, Any] | None = None,
    verifier=None,
) -> dict[str, str]:
    repaired = validate_and_repair_captions(
        captions,
        styles,
        neutral_summary,
        checklist=checklist,
        repair_callback=verifier,
    )
    return ensure_all_styles(repaired, styles)
