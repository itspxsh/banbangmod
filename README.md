# Track 2 Video Caption Agent

Containerized Track 2 agent for the AMD Developer Hackathon ACT II video-captioning task. The pipeline follows the spec in [Llava Style Track 2.md](/Users/posh/Work/AMD/Llava%20Style%20Track%202.md):

1. Read `/input/tasks.json`.
2. Download each `video_url`.
3. Sample uniform and scene-change frames once per video.
4. Run one multimodal grounding pass to produce a neutral summary.
5. Generate requested styles from that summary.
6. Validate and repair captions before writing `/output/results.json`.

## Runtime Contract

Input:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/video.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Output:

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

If a task requests only some styles, the agent returns exactly those keys.

## Backend Options

### `MODEL_BACKEND=openai_compatible`

Default. Uses any OpenAI-compatible multimodal `chat/completions` endpoint.

Required:

- `VLM_MODEL` or `OPENAI_MODEL`

Recommended:

- `OPENAI_API_KEY` or `VLM_API_KEY`
- `VLM_BASE_URL` if you are not using `https://api.openai.com/v1`
- `TEXT_MODEL` if you want a separate text-only model for checklist/style generation

Optional:

- `TEXT_BASE_URL`
- `TEXT_API_KEY`
- `UNIFORM_FRAME_COUNT` default `8`
- `SCENE_CHANGE_FRAME_COUNT` default `4`
- `MAX_FRAME_DIMENSION` default `768`

### `MODEL_BACKEND=template_fallback`

No external model calls. This keeps the output format valid but is only a last-resort fallback and will not score well on hidden evaluation videos.

## Local Run

```bash
mkdir -p input output
cp examples/tasks.json input/tasks.json

export OPENAI_API_KEY=your_key
export VLM_MODEL=gpt-4.1-mini

python3 -m src.main --input input/tasks.json --output output/results.json
```

## Docker Run

```bash
docker buildx build --platform linux/amd64 -t track2-agent:latest .
docker run --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e VLM_MODEL="$VLM_MODEL" \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  track2-agent:latest
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests cover input/output contract handling, caption validation, and style guardrails without calling any external model.
