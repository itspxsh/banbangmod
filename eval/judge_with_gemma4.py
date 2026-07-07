from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert manual or model judge outputs into normalized scores")
    parser.add_argument("--input", required=True, help="JSON file with factual_accuracy/style_match/fluency/hallucination")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    normalized = []
    for record in records:
        hallucination = float(record.get("hallucination", 0.0))
        normalized.append(
            {
                **record,
                "total_score": round(
                    0.45 * float(record.get("factual_accuracy", 0.0))
                    + 0.30 * float(record.get("style_match", 0.0))
                    + 0.10 * float(record.get("fluency", 0.0))
                    + 0.10 * float(record.get("brevity", 0.0))
                    + 0.05 * float(record.get("json_validity", 1.0))
                    - 0.50 * hallucination,
                    4,
                ),
            }
        )
    Path(args.output).write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
