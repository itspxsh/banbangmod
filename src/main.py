from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frame_sampler import sample_video_frames
from .io_utils import TaskSpec, read_tasks, write_results
from .style_renderer import StyleRenderer
from .validator import validate_and_repair_captions
from .video_download import download_video
from .video_model import EndpointConfig, OpenAICompatibleBackend, TemplateFallbackBackend

LOGGER = logging.getLogger("track2-agent")


@dataclass(frozen=True)
class RuntimeConfig:
    input_path: str
    output_path: str
    uniform_frames: int
    scene_change_frames: int
    max_frame_dimension: int
    backend: str
    vlm_endpoint: EndpointConfig | None
    text_endpoint: EndpointConfig | None


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_runtime_config(args)
    tasks = read_tasks(config.input_path)

    backend = build_backend(config)
    renderer = StyleRenderer(backend)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="track2-agent-") as temp_dir:
        temp_root = Path(temp_dir)
        for task in tasks:
            LOGGER.info("Processing task_id=%s", task.task_id)
            result = process_task(task, temp_root, config, backend, renderer)
            results.append(result)

    write_results(config.output_path, results)
    LOGGER.info("Wrote %s results to %s", len(results), config.output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track 2 video caption agent")
    parser.add_argument("--input", default=os.getenv("INPUT_PATH", "/input/tasks.json"))
    parser.add_argument("--output", default=os.getenv("OUTPUT_PATH", "/output/results.json"))
    return parser.parse_args()


def load_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    backend_name = os.getenv("MODEL_BACKEND", "openai_compatible").strip().lower()
    uniform_frames = int(os.getenv("UNIFORM_FRAME_COUNT", "8"))
    scene_change_frames = int(os.getenv("SCENE_CHANGE_FRAME_COUNT", "4"))
    max_frame_dimension = int(os.getenv("MAX_FRAME_DIMENSION", "768"))

    vlm_endpoint = None
    text_endpoint = None
    if backend_name == "openai_compatible":
        vlm_base_url = os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        vlm_model = os.getenv("VLM_MODEL") or os.getenv("OPENAI_MODEL")
        if not vlm_model:
            raise ValueError("VLM_MODEL or OPENAI_MODEL must be set for MODEL_BACKEND=openai_compatible")
        vlm_api_key = os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        vlm_endpoint = EndpointConfig(
            base_url=vlm_base_url,
            model=vlm_model,
            api_key=vlm_api_key,
            timeout_s=float(os.getenv("VLM_TIMEOUT_SECONDS", "120")),
        )

        text_endpoint = EndpointConfig(
            base_url=os.getenv("TEXT_BASE_URL") or vlm_base_url,
            model=os.getenv("TEXT_MODEL") or vlm_model,
            api_key=os.getenv("TEXT_API_KEY") or vlm_api_key,
            timeout_s=float(os.getenv("TEXT_TIMEOUT_SECONDS", "90")),
        )

    return RuntimeConfig(
        input_path=args.input,
        output_path=args.output,
        uniform_frames=uniform_frames,
        scene_change_frames=scene_change_frames,
        max_frame_dimension=max_frame_dimension,
        backend=backend_name,
        vlm_endpoint=vlm_endpoint,
        text_endpoint=text_endpoint,
    )


def build_backend(config: RuntimeConfig):
    if config.backend == "openai_compatible":
        assert config.vlm_endpoint is not None
        return OpenAICompatibleBackend(config.vlm_endpoint, config.text_endpoint)
    if config.backend == "template_fallback":
        LOGGER.warning("Using template_fallback backend; captions will remain format-safe but low accuracy.")
        return TemplateFallbackBackend()
    raise ValueError(f"Unsupported MODEL_BACKEND: {config.backend}")


def process_task(
    task: TaskSpec,
    temp_root: Path,
    config: RuntimeConfig,
    backend,
    renderer: StyleRenderer,
) -> dict[str, Any]:
    task_dir = temp_root / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = download_video(task.video_url, task_dir, task.task_id)
        frames = sample_video_frames(
            video_path,
            num_uniform=config.uniform_frames,
            num_scene_changes=config.scene_change_frames,
            max_dimension=config.max_frame_dimension,
        )
        neutral_summary = backend.describe_video(frames)
        checklist = backend.extract_checklist(neutral_summary)
        raw_captions = renderer.render_all(neutral_summary, task.styles, checklist)
        captions = validate_and_repair_captions(
            raw_captions,
            task.styles,
            neutral_summary,
            checklist=checklist,
            repair_callback=backend.repair_caption,
        )
    except Exception as exc:
        LOGGER.exception("Task %s failed; using fallback captions: %s", task.task_id, exc)
        neutral_summary = "A video clip shows a visible subject in a real-world scene with clear ongoing action."
        fallback_backend = TemplateFallbackBackend()
        checklist = fallback_backend.extract_checklist(neutral_summary)
        raw_captions = StyleRenderer(fallback_backend).render_all(neutral_summary, task.styles, checklist)
        captions = validate_and_repair_captions(
            raw_captions,
            task.styles,
            neutral_summary,
            checklist=checklist,
            repair_callback=fallback_backend.repair_caption,
        )

    return {
        "task_id": task.task_id,
        "captions": {style: captions[style] for style in task.styles},
    }


if __name__ == "__main__":
    main()
