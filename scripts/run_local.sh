#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${INPUT_PATH:-input/tasks.json}"
OUTPUT_PATH="${OUTPUT_PATH:-output/results.json}"

python3 -m src.main --input "$INPUT_PATH" --output "$OUTPUT_PATH"
