from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .fallbacks import template_caption, template_summary
from .prompts import (
    build_batch_style_prompt,
    build_checklist_prompt,
    build_neutral_summary_prompt,
    build_verifier_prompt,
)
from .video_sampling import SampledFrame


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    model_id: str
    api_key: str | None = None
    timeout_s: float = 120.0


class FireworksGemmaBackend:
    def __init__(self, visual_endpoint: EndpointConfig, text_endpoint: EndpointConfig | None = None) -> None:
        self.visual_endpoint = visual_endpoint
        self.text_endpoint = text_endpoint or visual_endpoint

    def describe_video(self, frames: list[SampledFrame]) -> str:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": build_neutral_summary_prompt(frame.seconds for frame in frames)}
        ]
        for index, frame in enumerate(frames, start=1):
            content.append({"type": "text", "text": f"Frame {index} at approximately {frame.seconds:.1f} seconds."})
            content.append({"type": "image_url", "image_url": {"url": frame.to_data_url()}})

        response = self._chat(
            endpoint=self.visual_endpoint,
            messages=[
                {"role": "system", "content": "You are a careful Gemma 4 video understanding model."},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=180,
        )
        return _normalize_text(response)

    def extract_checklist(self, neutral_summary: str) -> dict[str, Any]:
        response = self._chat(
            endpoint=self.text_endpoint,
            messages=[
                {"role": "system", "content": "You extract concise factual checklists for video captions."},
                {"role": "user", "content": build_checklist_prompt(neutral_summary)},
            ],
            temperature=0.0,
            max_tokens=220,
        )
        return _parse_checklist(response, neutral_summary)

    def generate_styled_captions(
        self,
        neutral_summary: str,
        checklist: Mapping[str, Any] | None,
        styles: tuple[str, ...],
    ) -> dict[str, str]:
        response = self._chat(
            endpoint=self.text_endpoint,
            messages=[
                {"role": "system", "content": "You are a Gemma 4 video captioning agent."},
                {"role": "user", "content": build_batch_style_prompt(neutral_summary, checklist, styles)},
            ],
            temperature=0.35,
            max_tokens=320,
        )
        parsed = _parse_json_object(response)
        captions: dict[str, str] = {}
        for style in styles:
            captions[style] = _normalize_text(str(parsed.get(style, "")).strip())
        return captions

    def verify_and_repair_caption(
        self,
        neutral_summary: str,
        caption: str,
        style: str,
        checklist: dict[str, Any] | None = None,
    ) -> str:
        response = self._chat(
            endpoint=self.text_endpoint,
            messages=[
                {"role": "system", "content": "You verify and repair Gemma 4 video captions."},
                {"role": "user", "content": build_verifier_prompt(neutral_summary, caption, style, checklist)},
            ],
            temperature=0.0,
            max_tokens=180,
        )
        parsed = _parse_json_object(response)
        final_caption = str(parsed.get("caption", "")).strip()
        return _normalize_text(final_caption or caption)

    def _chat(
        self,
        *,
        endpoint: EndpointConfig,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": endpoint.model_id,
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
            raise RuntimeError(f"Gemma 4 request failed with HTTP {exc.code}: {details}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Gemma 4 request failed: {exc}") from exc

        parsed = json.loads(body)
        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemma 4 response: {body}") from exc
        return _coerce_message_text(content)


class TemplateFallbackBackend:
    def describe_video(self, frames: list[SampledFrame]) -> str:
        del frames
        return template_summary()

    def extract_checklist(self, neutral_summary: str) -> dict[str, Any]:
        return _fallback_checklist(neutral_summary)

    def generate_styled_captions(
        self,
        neutral_summary: str,
        checklist: Mapping[str, Any] | None,
        styles: tuple[str, ...],
    ) -> dict[str, str]:
        del checklist
        return {style: template_caption(neutral_summary, style) for style in styles}

    def verify_and_repair_caption(
        self,
        neutral_summary: str,
        caption: str,
        style: str,
        checklist: dict[str, Any] | None = None,
    ) -> str:
        del caption, checklist
        return template_caption(neutral_summary, style)


class Gemma4Client:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def describe_video(self, frames: list[SampledFrame]) -> str:
        return self.backend.describe_video(frames)

    def extract_checklist(self, neutral_summary: str) -> dict[str, Any]:
        return self.backend.extract_checklist(neutral_summary)

    def generate_styled_captions(
        self,
        summary: str,
        checklist: Mapping[str, Any] | None,
        styles: tuple[str, ...],
    ) -> dict[str, str]:
        return self.backend.generate_styled_captions(summary, checklist, styles)

    def verify_and_repair_caption(
        self,
        neutral_summary: str,
        caption: str,
        style: str,
        checklist: dict[str, Any] | None = None,
    ) -> str:
        return self.backend.verify_and_repair_caption(neutral_summary, caption, style, checklist)


def build_gemma4_client_from_env() -> Gemma4Client:
    backend_name = os.getenv("BACKEND", "local_rocm").strip().lower()
    if backend_name == "template_fallback":
        return Gemma4Client(TemplateFallbackBackend())

    model_id = os.getenv("GEMMA4_MODEL_ID", "google/gemma-4-12B-it")
    if backend_name == "local_rocm":
        base_url = os.getenv("LOCAL_GEMMA4_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = os.getenv("LOCAL_GEMMA4_API_KEY")
    elif backend_name == "fireworks":
        base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        api_key = os.getenv("FIREWORKS_API_KEY")
    elif backend_name == "openai_compatible":
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unsupported BACKEND: {backend_name}")

    visual_endpoint = EndpointConfig(
        base_url=base_url,
        model_id=model_id,
        api_key=api_key,
        timeout_s=float(os.getenv("GEMMA4_TIMEOUT_SECONDS", "120")),
    )
    text_endpoint = EndpointConfig(
        base_url=os.getenv("TEXT_BASE_URL", base_url),
        model_id=os.getenv("TEXT_MODEL_ID", model_id),
        api_key=os.getenv("TEXT_API_KEY", api_key),
        timeout_s=float(os.getenv("TEXT_TIMEOUT_SECONDS", "90")),
    )
    return Gemma4Client(FireworksGemmaBackend(visual_endpoint, text_endpoint))


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
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        if parts:
            return "\n".join(parts)
    raise RuntimeError(f"Unable to extract text from model response: {content!r}")


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"No JSON object found in response: {text}")
    return json.loads(text[start : end + 1])


def _parse_checklist(response: str, neutral_summary: str) -> dict[str, Any]:
    try:
        payload = _parse_json_object(response)
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
