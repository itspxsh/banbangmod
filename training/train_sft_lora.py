from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an SFT training plan for Gemma 4 LoRA")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count = _count_jsonl(args.train_jsonl)
    eval_count = _count_jsonl(args.eval_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "model_name": args.model_name,
        "train_examples": train_count,
        "eval_examples": eval_count,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "bf16": args.bf16,
        "qlora": args.qlora,
        "gradient_checkpointing": args.gradient_checkpointing,
        "next_step": (
            "Wire these settings into your ROCm/Transformers training runner. "
            "This scaffold validates dataset paths and preserves the exact launch config."
        ),
    }
    (output_dir / "training_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def _count_jsonl(path: str) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


if __name__ == "__main__":
    main()
