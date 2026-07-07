from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a DPO training run description")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_name": args.model_name,
        "train_pairs": _count_jsonl(args.train_jsonl),
        "beta": args.beta,
        "trainer": "DPOTrainer",
    }
    (output_dir / "dpo_plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _count_jsonl(path: str) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


if __name__ == "__main__":
    main()
