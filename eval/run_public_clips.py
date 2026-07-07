from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public AMD smoke-test clips through the local agent")
    parser.add_argument("--agent-module", default="src.main")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--output-dir", default="eval/public_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "task_id": "v1",
            "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
            "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
        },
        {
            "task_id": "v2",
            "video_url": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
            "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
        },
        {
            "task_id": "v3",
            "video_url": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
            "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
        },
    ]
    tasks_path = output_dir / "tasks.json"
    results_path = output_dir / "results.json"
    tasks_path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    subprocess.run(
        ["python3", "-m", args.agent_module, "--input", str(tasks_path), "--output", str(results_path)],
        cwd=args.workdir,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
