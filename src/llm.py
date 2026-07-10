"""LLM backend selection: Anthropic (default) or a self-hosted OpenAI-compatible endpoint.

JobProof's inference can run on Anthropic's API or on your own metal — any
OpenAI-compatible server (vLLM, Ollama, LM Studio, TGI) exposing `/v1/chat/completions`.
A six-node homelab serving local models is a first-class backend, so tailoring and kit
generation can run fully self-hosted at zero marginal cost and with no data leaving the LAN.

The local backend is exposed as a small adapter, `LocalLLM`, that mimics just the slice of
the Anthropic client that `tailor.tailor` and `kit.generate_kit` use —
`client.messages.parse(..., output_format=Model)` returning `.parsed_output` + `.usage`.
Because those call sites already accept an injected `client=`, neither needs to change.

Config (env):
  AUTO_APPLY_LLM_BACKEND   anthropic | local        (default: anthropic)
  AUTO_APPLY_LLM_BASE_URL  e.g. http://cluster.local:8000/v1
  AUTO_APPLY_LLM_MODEL     e.g. llama-3.3-70b-instruct
  AUTO_APPLY_LLM_API_KEY   optional bearer token for the local gateway
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

DEFAULT_LOCAL_MODEL = "local-model"


def _blocks_to_text(system) -> str:
    """Flatten an Anthropic-style system value (str or list of text blocks) to one string."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for b in system:
        if isinstance(b, dict):
            parts.append(b.get("text", ""))
        else:
            parts.append(getattr(b, "text", "") or str(b))
    return "\n\n".join(p for p in parts if p)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply, tolerating ```json fences or surrounding prose."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


class _LocalMessages:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def parse(self, *, output_format, system=None, messages=None, max_tokens=8000, **_ignored):
        """OpenAI-compatible chat call that returns a validated `output_format` instance.

        Mirrors `anthropic.Messages.parse`'s return shape (.parsed_output, .usage, .stop_reason).
        Anthropic-only kwargs (thinking, tools, cache_control) are accepted and ignored.
        """
        import httpx

        schema = output_format.model_json_schema()
        sys_text = _blocks_to_text(system)
        sys_text += (
            "\n\nReturn ONLY a single JSON object — no prose, no code fences — that validates "
            "against this JSON Schema:\n" + json.dumps(schema)
        )
        chat = [{"role": "system", "content": sys_text}]
        for m in messages or []:
            content = m["content"] if isinstance(m["content"], str) else _blocks_to_text(m["content"])
            chat.append({"role": m["role"], "content": content})

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {
            "model": self._model,
            "messages": chat,
            "max_tokens": max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        with httpx.Client(timeout=self._timeout) as c:
            resp = c.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        try:
            parsed = output_format.model_validate(_extract_json(content))
        except Exception as exc:
            raise RuntimeError(f"Local model returned unparseable/invalid JSON: {exc}") from exc

        u = data.get("usage", {}) or {}
        usage = SimpleNamespace(
            input_tokens=int(u.get("prompt_tokens", 0) or 0),
            output_tokens=int(u.get("completion_tokens", 0) or 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return SimpleNamespace(parsed_output=parsed, usage=usage, stop_reason="stop")


class LocalLLM:
    """Adapter over a self-hosted OpenAI-compatible endpoint, shaped like the Anthropic client."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None, timeout: float = 120.0):
        if not base_url:
            raise ValueError("LocalLLM needs a base_url (AUTO_APPLY_LLM_BASE_URL).")
        self.base_url = base_url
        self.model = model or DEFAULT_LOCAL_MODEL
        # Truthy so api._require_key treats a configured local endpoint as 'available'.
        self.api_key = api_key or "local"
        self.auth_token = None
        self.messages = _LocalMessages(base_url, self.model, api_key, timeout)


def backend_name() -> str:
    return os.environ.get("AUTO_APPLY_LLM_BACKEND", "anthropic").strip().lower()


def active_model() -> str:
    """The model id the current backend will use (for display / run records)."""
    if backend_name() == "local":
        return os.environ.get("AUTO_APPLY_LLM_MODEL", DEFAULT_LOCAL_MODEL)
    from . import tailor

    return tailor.MODEL


def make_client():
    """Construct the configured inference client (Anthropic by default, else self-hosted local)."""
    if backend_name() == "local":
        return LocalLLM(
            base_url=os.environ.get("AUTO_APPLY_LLM_BASE_URL", ""),
            model=os.environ.get("AUTO_APPLY_LLM_MODEL", DEFAULT_LOCAL_MODEL),
            api_key=os.environ.get("AUTO_APPLY_LLM_API_KEY") or None,
        )
    import anthropic

    return anthropic.Anthropic()
