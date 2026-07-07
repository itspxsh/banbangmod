from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import SUPPORTED_STYLES


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    video_url: str
    styles: tuple[str, ...]


def read_tasks(path: str | Path) -> list[TaskSpec]:
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tasks.json must contain a list of task objects")

    tasks: list[TaskSpec] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Task at index {index} must be an object")

        task_id = _expect_non_empty_string(item.get("task_id"), f"task[{index}].task_id")
        video_url = _expect_non_empty_string(item.get("video_url"), f"task[{index}].video_url")
        styles = _normalize_styles(item.get("styles"), index)
        tasks.append(TaskSpec(task_id=task_id, video_url=video_url, styles=styles))

    return tasks


def write_results(path: str | Path, results: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_styles(raw_styles: Any, index: int) -> tuple[str, ...]:
    if not isinstance(raw_styles, list) or not raw_styles:
        raise ValueError(f"task[{index}].styles must be a non-empty list")

    supported = set(SUPPORTED_STYLES)
    styles: list[str] = []
    seen: set[str] = set()
    for style in raw_styles:
        style_name = _expect_non_empty_string(style, f"task[{index}].styles[]")
        if style_name not in supported:
            raise ValueError(
                f"Unsupported style '{style_name}'. Supported styles: {', '.join(SUPPORTED_STYLES)}"
            )
        if style_name not in seen:
            styles.append(style_name)
            seen.add(style_name)

    return tuple(styles)


def _expect_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
