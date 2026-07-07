from __future__ import annotations

from typing import Any


class StyleRenderer:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def render_all(
        self,
        neutral_summary: str,
        styles: tuple[str, ...],
        checklist: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        captions: dict[str, str] = {}
        for style in styles:
            captions[style] = self.backend.render_caption(neutral_summary, style, checklist)
        return captions
