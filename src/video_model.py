from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .fallbacks import template_caption
from .frame_sampler import SampledFrame
from .prompts import (
    STYLE_DEFINITIONS,
    build_checklist_prompt,
    build_neutral_summary_prompt,
    build_repair_prompt,
    build_style_prompt,
)


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0


class OpenAICompatibleBackend:
    def __init__(self, visual_endpoint: EndpointConfig, text_endpoint: EndpointConfig | None = None) -> None:
        self.visual_endpoint = visual_endpoint
        self.text_endpoint = text_endpoint or visual_endpoint

    def describe_video(self, frames: list[SampledFrame]) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": build_neutral_summary_prompt(frame.seconds for frame in frames),
            }
        ]

        for index, frame in enumerate(frames, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Frame {index} at approximately {frame.seconds:.1f} seconds.",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": frame.to_data_url()},
                }
            )

        response_text = self._chat(
            endpoint=self.visual_endpoint,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise multimodal video understanding assistant.",
                },
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=180,
        )
        return _normalize_text(response_text)

    def extract_checklist(self, neutral_summary: str) -> dict[str, Any]:
        response_text = self._chat(
            endpoint=self.text_endpoint,
            messages=[
                {
                    "role": "system",
                    "content": "You extract concise factual checklists for caption validation.",
                },
                {"role": "user", "content": build_checklist_prompt(neutral_summary)},
            ],
            temperature=0.0,
            max_tokens=220,
        )

        try:
            payload = json.loads(_extract_json_object(response_text))
        except (json.JSONDecodeError, ValueError):
            return _fallback_checklist(neutral_summary)

        return {
            "subjects": _ensure_string_list(payload.get("subjects")),
            "setting": _ensure_string(payload.get("setting")),
            "actions": _ensure_string_list(payload.get("actions")),
            "objects": _ensure_string_list(payload.get("objects")),
            "mood": _ensure_string(payload.get("mood")),
            "avoid_claims": _ensure_string_list(payload.get("avoid_claims")),
        }

    def render_caption(
        self,
        neutral_summary: str,
        style: str,
        checklist: dict[str, Any] | None = None,
    ) -> str:
        temperature = 0.2 if style == "formal" else 0.6
        response_text = self._chat(
            endpoint=self.text_endpoint,
            messages=[
                {
                    "role": "system",
                    "content": "You write short video captions that stay faithful to the provided facts.",
                },
                {
                    "role": "user",
                    "content": build_style_prompt(neutral_summary, style, checklist),
                },
            ],
            temperature=temperature,
            max_tokens=120,
        )
        return _normalize_text(response_text)

    def repair_caption(
        self,
        neutral_summary: str,
        caption: str,
        style: str,
        checklist: dict[str, Any] | None = None,
    ) -> str:
        response_text = self._chat(
            endpoint=self.text_endpoint,
            messages=[
                {
                    "role": "system",
                    "content": "You repair captions by removing unsupported details while preserving style.",
                },
                {
                    "role": "user",
                    "content": build_repair_prompt(neutral_summary, caption, style, checklist),
                },
            ],
            temperature=0.0,
            max_tokens=120,
        )
        return _normalize_text(response_text)

    def _chat(
        self,
        *,
        endpoint: EndpointConfig,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": endpoint.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        request = Request(
            _chat_completions_url(endpoint.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers=_headers(endpoint.api_key),
            method="POST",
        )

        try:
            with urlopen(request, timeout=endpoint.timeout_s) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Caption model request failed with HTTP {exc.code}: {details}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Caption model request failed: {exc}") from exc

        parsed = json.loads(body)
        try:
            message_content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected model response: {body}") from exc

        return _coerce_message_text(message_content)


class TemplateFallbackBackend:
    def describe_video(self, frames: list[SampledFrame]) -> str:  # pragma: no cover - runtime fallback
        del frames
        return "A video clip shows a visible subject in a scene with ongoing action."

    def extract_checklist(self, neutral_summary: str) -> dict[str, Any]:
        return _fallback_checklist(neutral_summary)

    def render_caption(
        self,
        neutral_summary: str,
        style: str,
        checklist: dict[str, Any] | None = None,
    ) -> str:
        del checklist
        return template_caption(neutral_summary, style)

    def repair_caption(
        self,
        neutral_summary: str,
        caption: str,
        style: str,
        checklist: dict[str, Any] | None = None,
    ) -> str:
        del caption, checklist
        return template_caption(neutral_summary, style)


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        if text_parts:
            return "\n".join(text_parts)
    raise RuntimeError(f"Unable to extract text from model response content: {content!r}")


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in response")
    return text[start : end + 1]


def _ensure_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return items


def _fallback_checklist(neutral_summary: str) -> dict[str, Any]:
    fragments = [piece.strip(" ,.") for piece in neutral_summary.split() if len(piece.strip(" ,.")) > 3]
    subjects = fragments[:2]
    setting = ""
    if " in " in neutral_summary:
        setting = neutral_summary.split(" in ", 1)[1].split(".", 1)[0].strip()
    return {
        "subjects": subjects,
        "setting": setting,
        "actions": [],
        "objects": [],
        "mood": "",
        "avoid_claims": [],
    }
