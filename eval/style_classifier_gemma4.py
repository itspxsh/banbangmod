from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prompts import TECH_WORDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cheap heuristic style classifier for offline caption triage")
    parser.add_argument("--input", required=True, help="JSON file containing caption records")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = []
    for record in records:
        caption = str(record.get("caption", ""))
        results.append({**record, "predicted_style": classify_style(caption)})
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def classify_style(caption: str) -> str:
    lowered = caption.lower()
    has_tech = any(word in lowered for word in TECH_WORDS)
    if has_tech:
        return "humorous_tech"
    if any(marker in lowered for marker in ("apparently", "obviously", "clearly")):
        return "sarcastic"
    if any(marker in lowered for marker in ("like it owns", "star of the day", "supervisor")):
        return "humorous_non_tech"
    return "formal"


if __name__ == "__main__":
    main()
