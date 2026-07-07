# Gemma 4 Track 2 Agent

Gemma-4-only Track 2 agent for the AMD Developer Hackathon ACT II video-captioning task. This repo now follows the structure and runtime constraints from [Gemma 4 Fireworks Track 2.md](/Users/posh/Downloads/Gemma%204%20Fireworks%20Track%202.md):

1. Read `/input/tasks.json`.
2. Download each `video_url`.
3. Sample frames once per clip.
4. Use Gemma 4 for neutral summary, checklist extraction, style generation, and verify/repair.
5. Validate requested style keys.
6. Write `/output/results.json`.

## Runtime Backends

### `BACKEND=local_rocm`

Default. Targets a local OpenAI-compatible Gemma 4 server running on AMD ROCm infrastructure.

Required:

- `GEMMA4_MODEL_ID`

Optional:

- `LOCAL_GEMMA4_BASE_URL` default `http://127.0.0.1:8000/v1`
- `LOCAL_GEMMA4_API_KEY`
- `TEXT_BASE_URL`
- `TEXT_MODEL_ID`
- `TEXT_API_KEY`

### `BACKEND=fireworks`

Uses a Fireworks OpenAI-compatible endpoint with a Gemma 4 model ID.

Required:

- `FIREWORKS_API_KEY`
- `GEMMA4_MODEL_ID`

Optional:

- `FIREWORKS_BASE_URL` default `https://api.fireworks.ai/inference/v1`
- `TEXT_BASE_URL`
- `TEXT_MODEL_ID`
- `TEXT_API_KEY`

### `BACKEND=openai_compatible`

Fallback for any OpenAI-compatible endpoint that exposes a Gemma 4 model.

Required:

- `OPENAI_API_KEY`
- `GEMMA4_MODEL_ID`

Optional:

- `OPENAI_BASE_URL`
- `TEXT_BASE_URL`
- `TEXT_MODEL_ID`
- `TEXT_API_KEY`

### `BACKEND=template_fallback`

No model calls. This keeps output shape valid for smoke tests only and will not score competitively.

## Core Layout

- [src/main.py](/Users/posh/Work/AMD/src/main.py): entrypoint
- [src/caption_pipeline.py](/Users/posh/Work/AMD/src/caption_pipeline.py): end-to-end task flow
- [src/gemma4_client.py](/Users/posh/Work/AMD/src/gemma4_client.py): Gemma 4 client and backend selection
- [src/video_sampling.py](/Users/posh/Work/AMD/src/video_sampling.py): sampled frame interface
- [src/verify_repair.py](/Users/posh/Work/AMD/src/verify_repair.py): verify/repair layer
- [src/output_schema.py](/Users/posh/Work/AMD/src/output_schema.py): output contract checks
- [configs](/Users/posh/Work/AMD/configs): inference, SFT, and GRPO config baselines
- [training](/Users/posh/Work/AMD/training): dataset/build-plan scaffolds for SFT, DPO, and GRPO
- [eval](/Users/posh/Work/AMD/eval): public clip runner, lightweight judging, and reporting
- [scripts](/Users/posh/Work/AMD/scripts): build/run/push helpers

## Local Run

```bash
mkdir -p input output
cp examples/tasks.json input/tasks.json

export BACKEND=local_rocm
export GEMMA4_MODEL_ID=google/gemma-4-12B-it
export LOCAL_GEMMA4_BASE_URL=http://127.0.0.1:8000/v1

python3 -m src.main --input input/tasks.json --output output/results.json
```

For Fireworks-hosted inference:

```bash
export BACKEND=fireworks
export FIREWORKS_API_KEY=your_key
export GEMMA4_MODEL_ID=accounts/your-account/models/your-gemma4-model
```

For an offline smoke test:

```bash
BACKEND=template_fallback python3 -m src.main --input input/tasks.json --output output/results.json
```

## Docker

The container now targets an AMD ROCm base image:

```bash
docker buildx build --platform linux/amd64 -t gemma4-track2-agent:latest .
docker run --rm \
  -e BACKEND="$BACKEND" \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e GEMMA4_MODEL_ID="$GEMMA4_MODEL_ID" \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  gemma4-track2-agent:latest
```

## Training and Eval Utilities

The training and eval scripts are scaffolds for the workflow described in the brief. They validate inputs, preserve the exact launch configuration, and write plan artifacts you can carry onto AMD Developer Cloud or Fireworks-managed tuning.

Examples:

```bash
python3 training/build_sft_dataset.py --input data/caption_records.json --output data/sft_train.jsonl
python3 training/train_sft_lora.py --model_name google/gemma-4-12B-it --train_jsonl data/sft_train.jsonl --eval_jsonl data/sft_eval.jsonl --output_dir outputs/gemma4_track2_sft_lora
python3 eval/run_public_clips.py --workdir .
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The current tests cover task parsing, caption validation, and output schema guarantees without making external model calls.
