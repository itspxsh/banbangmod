from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a Gemma adapter merge plan")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "output_dir": args.output_dir,
        "next_step": "Load base model with PEFT and merge the adapter on a ROCm-capable runtime.",
    }
    (output_dir / "merge_plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
