from __future__ import annotations

import json
from typing import Iterable

SUPPORTED_STYLES = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)

STYLE_DEFINITIONS = {
    "formal": (
        "Write a professional, objective, factual video caption. "
        "Avoid jokes, sarcasm, slang, and exaggeration."
    ),
    "sarcastic": (
        "Write a dry, lightly ironic caption. Keep it safe and not mean-spirited. "
        "Do not distort the facts."
    ),
    "humorous_tech": (
        "Write a funny caption using accessible technology or programming references. "
        "Use light references such as debugging, loading, cache, kernel, update, model, "
        "server, or deployment. Do not add false visual details."
    ),
    "humorous_non_tech": (
        "Write a funny everyday caption with no programming or technical jargon. "
        "Use simple humor and do not distort the facts."
    ),
}

TECH_WORDS = {
    "api",
    "app",
    "bug",
    "cache",
    "client",
    "cloud",
    "code",
    "compile",
    "cpu",
    "data",
    "debug",
    "deploy",
    "deployment",
    "download",
    "driver",
    "firmware",
    "gpu",
    "kernel",
    "latency",
    "load",
    "loading",
    "memory",
    "model",
    "network",
    "pipeline",
    "process",
    "script",
    "server",
    "software",
    "stack",
    "status",
    "sync",
    "system",
    "thread",
    "update",
}


def build_neutral_summary_prompt(frame_times: Iterable[float]) -> str:
    ordered_times = ", ".join(f"{timestamp:.1f}s" for timestamp in frame_times)
    return (
        "You are a careful video captioning model.\n\n"
        "Task:\n"
        "Describe the video accurately in 1-2 concise sentences.\n\n"
        "Rules:\n"
        "- Mention only visible content.\n"
        "- Include the main subject, setting, and key actions.\n"
        "- Mention notable objects only if visually clear.\n"
        "- Do not guess identities, brands, locations, or emotions unless visually obvious.\n"
        "- Do not mention camera details unless they matter to the scene.\n"
        "- Do not be humorous yet.\n"
        "- Keep the wording grounded and concise.\n\n"
        "The sampled frames appear in chronological order. Approximate timestamps: "
        f"{ordered_times or 'unknown'}."
    )


def build_checklist_prompt(neutral_summary: str) -> str:
    return (
        "From the video summary below, extract a factual checklist.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "subjects": [],\n'
        '  "setting": "",\n'
        '  "actions": [],\n'
        '  "objects": [],\n'
        '  "mood": "",\n'
        '  "avoid_claims": []\n'
        "}\n\n"
        "Summary:\n"
        f"{neutral_summary}"
    )


def build_style_prompt(
    neutral_summary: str,
    style: str,
    checklist: dict[str, object] | None = None,
) -> str:
    checklist_block = ""
    if checklist:
        checklist_block = (
            "\nFactual checklist:\n"
            f"{json.dumps(checklist, ensure_ascii=True, sort_keys=True)}\n"
        )

    return (
        "You are writing a video caption.\n\n"
        "Use the factual summary as the only source of truth.\n"
        "Preserve the visible facts while matching the requested tone.\n\n"
        f"Factual summary:\n{neutral_summary}\n"
        f"{checklist_block}"
        f"Style definition:\n{STYLE_DEFINITIONS[style]}\n\n"
        "Rules:\n"
        "- One sentence only.\n"
        "- 10 to 25 words.\n"
        "- Preserve the main subject, setting, and action.\n"
        "- Do not add new subjects, actions, objects, locations, or events.\n"
        "- Output the caption only.\n"
    )


def build_batch_style_prompt(
    neutral_summary: str,
    checklist: dict[str, object] | None,
    styles: tuple[str, ...],
) -> str:
    checklist_block = ""
    if checklist:
        checklist_block = (
            "Factual checklist:\n"
            f"{json.dumps(checklist, ensure_ascii=True, sort_keys=True)}\n\n"
        )

    style_block = "\n".join(f"- {style}: {STYLE_DEFINITIONS[style]}" for style in styles)
    json_keys = ", ".join(styles)

    return (
        "You are a video captioning agent.\n\n"
        f"Use this factual summary as the only source of truth:\n{neutral_summary}\n\n"
        f"{checklist_block}"
        "Write one caption for each requested style.\n\n"
        f"Requested styles:\n{style_block}\n\n"
        "Rules:\n"
        "- Each caption must be one sentence.\n"
        "- Each caption must be 10 to 25 words.\n"
        "- Preserve the visible facts.\n"
        "- Do not add new people, objects, actions, locations, or events.\n"
        f"- Return valid JSON only with exactly these keys: {json_keys}.\n"
    )


def build_verifier_prompt(
    neutral_summary: str,
    caption: str,
    style: str,
    checklist: dict[str, object] | None = None,
) -> str:
    checklist_block = ""
    if checklist:
        checklist_block = (
            "Factual checklist:\n"
            f"{json.dumps(checklist, ensure_ascii=True, sort_keys=True)}\n\n"
        )

    return (
        "You are verifying captions against a factual video summary.\n\n"
        f"Factual summary:\n{neutral_summary}\n\n"
        f"{checklist_block}"
        f"Caption:\n{caption}\n\n"
        f"Requested style:\n{style}\n"
        f"Style definition:\n{STYLE_DEFINITIONS[style]}\n\n"
        "Check:\n"
        "1. Does the caption preserve the visible facts?\n"
        "2. Does it add unsupported objects, people, actions, or places?\n"
        "3. Does it match the requested style?\n\n"
        "If valid, return JSON only:\n"
        '{"valid": true, "caption": "...", "reason": "..."}\n\n'
        "If invalid, repair the caption and return JSON only:\n"
        '{"valid": false, "caption": "repaired caption", "reason": "..."}'
    )


def build_repair_prompt(
    neutral_summary: str,
    caption: str,
    style: str,
    checklist: dict[str, object] | None = None,
) -> str:
    return build_verifier_prompt(neutral_summary, caption, style, checklist)
