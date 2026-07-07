from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate caption evaluation records")
    parser.add_argument("--input", required=True, help="JSON file with evaluation records")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not records:
        raise ValueError("No evaluation records provided")

    summary = {
        "caption_accuracy": _average(records, "factual_accuracy"),
        "style_match": _average(records, "style_match"),
        "hallucination_rate": _average(records, "hallucination"),
        "json_validity_rate": _average(records, "json_validity"),
        "average_caption_length": _average(records, "caption_length"),
        "runtime_per_clip": _average(records, "runtime_seconds"),
        "tokens_per_clip": _average(records, "tokens_used"),
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _average(records: list[dict[str, object]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return round(sum(values) / len(values), 4)


if __name__ == "__main__":
    main()
