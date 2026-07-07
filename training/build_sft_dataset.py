from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = "You are a precise video captioning agent."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gemma 4 Track 2 SFT JSONL data")
    parser.add_argument("--input", required=True, help="JSON file containing records with video and captions")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON array")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record {index} must be an object")
            video = str(record.get("video") or record.get("video_url") or "").strip()
            if not video:
                raise ValueError(f"Record {index} must include 'video' or 'video_url'")
            captions = record["captions"]
            if not isinstance(captions, dict) or not captions:
                raise ValueError(f"Record {index} captions must be a non-empty object")
            styles = ", ".join(captions.keys())
            sample = {
                "id": str(record.get("id") or record.get("task_id") or f"sample_{index:06d}"),
                "video": video,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video},
                            {
                                "type": "text",
                                "text": (
                                    f"Generate captions for styles: {styles}. "
                                    "Return JSON only."
                                ),
                            },
                        ],
                    },
                    {"role": "assistant", "content": json.dumps(captions, ensure_ascii=True)},
                ],
            }
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
