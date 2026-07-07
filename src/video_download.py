from __future__ import annotations

import mimetypes
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def download_video(
    video_url: str,
    download_dir: str | Path,
    task_id: str,
    *,
    timeout_s: float = 120.0,
    max_retries: int = 3,
) -> Path:
    target_dir = Path(download_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(video_url)
    if parsed.scheme in ("", "file"):
        source = Path(parsed.path if parsed.scheme == "file" else video_url).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Local video not found: {source}")
        target = target_dir / f"{_safe_name(task_id)}{source.suffix or '.mp4'}"
        shutil.copy2(source, target)
        return target

    target = target_dir / f"{_safe_name(task_id)}{_suffix_from_url(video_url)}"
    temp_target = target_dir / f"{target.name}.part"

    request = Request(
        video_url,
        headers={
            "User-Agent": "track2-video-caption-agent/1.0",
            "Accept": "*/*",
        },
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                content_type = response.headers.get("Content-Type", "")
                suffix = _suffix_from_headers(content_type) or target.suffix
                target = target.with_suffix(suffix or ".mp4")
                temp_target = target_dir / f"{target.name}.part"

                with temp_target.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)

                temp_target.replace(target)
                return target
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if temp_target.exists():
                temp_target.unlink()
            if attempt < max_retries:
                time.sleep(min(2.0 * attempt, 5.0))

    raise RuntimeError(f"Failed to download video from {video_url}") from last_error


def _safe_name(task_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in task_id)
    return cleaned.strip("-") or "task"


def _suffix_from_url(video_url: str) -> str:
    parsed = urlparse(video_url)
    suffix = Path(parsed.path).suffix
    return suffix if suffix else ".mp4"


def _suffix_from_headers(content_type: str) -> str | None:
    if not content_type:
        return None
    mime = content_type.split(";", 1)[0].strip()
    guessed = mimetypes.guess_extension(mime)
    if guessed == ".jpe":
        return ".jpg"
    return guessed
