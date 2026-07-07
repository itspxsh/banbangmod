from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert judged caption candidates into DPO pairs")
    parser.add_argument("--input", required=True, help="JSON file with prompt, candidates, and winner")
    parser.add_argument("--output", required=True, help="Output JSONL for DPO/ORPO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            candidates = record.get("candidates", {})
            winner = str(record.get("winner", "")).strip()
            if winner not in candidates:
                raise ValueError(f"Record {index} winner '{winner}' not found in candidates")
            for loser, loser_caption in candidates.items():
                if loser == winner:
                    continue
                pair = {
                    "prompt": record["prompt"],
                    "chosen": candidates[winner],
                    "rejected": loser_caption,
                }
                handle.write(json.dumps(pair, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
