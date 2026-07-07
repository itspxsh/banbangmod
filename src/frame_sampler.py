from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampledFrame:
    index: int
    seconds: float
    width: int
    height: int
    jpeg_bytes: bytes

    def to_data_url(self) -> str:
        encoded = base64.b64encode(self.jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def sample_video_frames(
    video_path: str | Path,
    *,
    num_uniform: int = 8,
    num_scene_changes: int = 4,
    max_dimension: int = 768,
) -> list[SampledFrame]:
    cv2 = _import_cv2()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise RuntimeError(f"Unable to determine frame count for video: {video_path}")
        fps = fps if fps > 0 else 1.0

        selected_indices = set(_uniform_indices(frame_count, num_uniform))
        selected_indices.add(0)
        selected_indices.add(frame_count - 1)
        selected_indices.update(_scene_change_indices(capture, frame_count, num_scene_changes))

        sampled: list[SampledFrame] = []
        signatures: list[Any] = []
        for frame_index in sorted(selected_indices):
            frame = _read_frame(capture, frame_index)
            resized = _resize_frame(frame, max_dimension)
            signature = _frame_signature(resized, cv2)
            if any(_is_near_duplicate(signature, previous) for previous in signatures):
                continue

            success, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not success:
                raise RuntimeError(f"Failed to encode frame {frame_index} from {video_path}")

            height, width = resized.shape[:2]
            sampled.append(
                SampledFrame(
                    index=frame_index,
                    seconds=frame_index / fps,
                    width=width,
                    height=height,
                    jpeg_bytes=encoded.tobytes(),
                )
            )
            signatures.append(signature)

        if not sampled:
            raise RuntimeError(f"No frames sampled from {video_path}")

        return sampled
    finally:
        capture.release()


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in runtime environments without cv2
        raise RuntimeError("opencv-python-headless is required to sample video frames") from exc
    return cv2


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in runtime environments without numpy
        raise RuntimeError("numpy is required to sample video frames") from exc
    return np


def _uniform_indices(frame_count: int, num_uniform: int) -> list[int]:
    np = _import_numpy()
    if num_uniform <= 0:
        return []
    count = min(frame_count, num_uniform)
    return [int(round(value)) for value in np.linspace(0, frame_count - 1, num=count)]


def _scene_change_indices(capture, frame_count: int, max_scene_changes: int) -> list[int]:
    if max_scene_changes <= 0 or frame_count < 4:
        return []

    cv2 = _import_cv2()
    np = _import_numpy()
    probe_count = min(max(frame_count, 2), 48)
    probe_indices = [int(round(value)) for value in np.linspace(0, frame_count - 1, num=probe_count)]
    frames = []
    for frame_index in probe_indices:
        frame = _read_frame(capture, frame_index)
        frames.append(_frame_signature(frame, cv2))

    diffs: list[tuple[float, int]] = []
    for current in range(1, len(frames)):
        delta = float(np.mean(np.abs(frames[current] - frames[current - 1])))
        diffs.append((delta, probe_indices[current]))

    diffs.sort(reverse=True)
    chosen: list[int] = []
    minimum_spacing = max(frame_count // 20, 1)
    for _, frame_index in diffs:
        if all(abs(frame_index - existing) >= minimum_spacing for existing in chosen):
            chosen.append(frame_index)
        if len(chosen) >= max_scene_changes:
            break

    return chosen


def _read_frame(capture, frame_index: int):
    cv2 = _import_cv2()
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index}")
    return frame


def _resize_frame(frame: Any, max_dimension: int):
    cv2 = _import_cv2()
    height, width = frame.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= max_dimension:
        return frame

    scale = max_dimension / float(largest_dimension)
    new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _frame_signature(frame: Any, cv2):
    np = _import_numpy()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_AREA)
    return reduced.astype(np.float32)


def _is_near_duplicate(first: Any, second: Any) -> bool:
    np = _import_numpy()
    return float(np.mean(np.abs(first - second))) < 2.0
