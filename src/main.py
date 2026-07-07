from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .caption_pipeline import CaptionPipeline, SamplingConfig
from .gemma4_client import build_gemma4_client_from_env
from .io_utils import TaskSpec, read_tasks, write_results

LOGGER = logging.getLogger("track2-agent")


@dataclass(frozen=True)
class RuntimeConfig:
    input_path: str
    output_path: str
    uniform_frames: int
    scene_change_frames: int
    max_frame_dimension: int


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_runtime_config(args)
    tasks = read_tasks(config.input_path)
    pipeline = CaptionPipeline(
        build_gemma4_client_from_env(),
        SamplingConfig(
            uniform_frames=config.uniform_frames,
            scene_change_frames=config.scene_change_frames,
            max_frame_dimension=config.max_frame_dimension,
        ),
    )

    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="track2-agent-") as temp_dir:
        temp_root = Path(temp_dir)
        for task in tasks:
            LOGGER.info("Processing task_id=%s", task.task_id)
            result = process_task(task, temp_root, pipeline)
            results.append(result.as_dict())

    write_results(config.output_path, results)
    LOGGER.info("Wrote %s results to %s", len(results), config.output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 4 Track 2 video caption agent")
    parser.add_argument("--input", default=os.getenv("INPUT_PATH", "/input/tasks.json"))
    parser.add_argument("--output", default=os.getenv("OUTPUT_PATH", "/output/results.json"))
    return parser.parse_args()


def load_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    uniform_frames = int(os.getenv("UNIFORM_FRAME_COUNT", "8"))
    scene_change_frames = int(os.getenv("SCENE_CHANGE_FRAME_COUNT", "4"))
    max_frame_dimension = int(os.getenv("MAX_FRAME_DIMENSION", "768"))

    return RuntimeConfig(
        input_path=args.input,
        output_path=args.output,
        uniform_frames=uniform_frames,
        scene_change_frames=scene_change_frames,
        max_frame_dimension=max_frame_dimension,
    )


def process_task(
    task: TaskSpec,
    temp_root: Path,
    pipeline: CaptionPipeline,
):
    return pipeline.process_task(task, temp_root)


if __name__ == "__main__":
    main()
