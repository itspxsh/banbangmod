from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .prompts import SUPPORTED_STYLES


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    captions: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {"task_id": self.task_id, "captions": dict(self.captions)}


def ensure_all_styles(captions: Mapping[str, str], styles: tuple[str, ...]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for style in styles:
        if style not in SUPPORTED_STYLES:
            raise ValueError(f"Unsupported style requested: {style}")
        caption = str(captions.get(style, "")).strip()
        if not caption:
            raise ValueError(f"Missing caption for requested style: {style}")
        normalized[style] = caption
    return normalized


def validate_result(task_id: str, styles: tuple[str, ...], result: TaskResult) -> None:
    if result.task_id != task_id:
        raise ValueError(f"Result task_id mismatch: expected {task_id}, got {result.task_id}")
    _ = ensure_all_styles(result.captions, styles)
