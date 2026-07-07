from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .fallbacks import template_summary
from .gemma4_client import Gemma4Client, TemplateFallbackBackend
from .io_utils import TaskSpec
from .output_schema import TaskResult, validate_result
from .verify_repair import verify_and_repair_captions
from .video_download import download_video
from .video_sampling import sample_video_frames

LOGGER = logging.getLogger("track2-agent")


@dataclass(frozen=True)
class SamplingConfig:
    uniform_frames: int = 8
    scene_change_frames: int = 4
    max_frame_dimension: int = 768


class CaptionPipeline:
    def __init__(self, client: Gemma4Client, sampling: SamplingConfig) -> None:
        self.client = client
        self.sampling = sampling

    def process_task(self, task: TaskSpec, temp_root: Path) -> TaskResult:
        task_dir = temp_root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        try:
            video_path = download_video(task.video_url, task_dir, task.task_id)
            frames = sample_video_frames(
                video_path,
                num_uniform=self.sampling.uniform_frames,
                num_scene_changes=self.sampling.scene_change_frames,
                max_dimension=self.sampling.max_frame_dimension,
            )
            summary = self.client.describe_video(frames)
            checklist = self.client.extract_checklist(summary)
            raw_captions = self.client.generate_styled_captions(summary, checklist, task.styles)
            captions = verify_and_repair_captions(
                raw_captions,
                task.styles,
                summary,
                checklist=checklist,
                verifier=self.client.verify_and_repair_caption,
            )
        except Exception as exc:
            LOGGER.exception("Task %s failed; using fallback captions: %s", task.task_id, exc)
            fallback = Gemma4Client(TemplateFallbackBackend())
            summary = template_summary()
            checklist = fallback.extract_checklist(summary)
            raw_captions = fallback.generate_styled_captions(summary, checklist, task.styles)
            captions = verify_and_repair_captions(
                raw_captions,
                task.styles,
                summary,
                checklist=checklist,
                verifier=fallback.verify_and_repair_caption,
            )

        result = TaskResult(task_id=task.task_id, captions=captions)
        validate_result(task.task_id, task.styles, result)
        return result
