"""Tests for the self-hosted (OpenAI-compatible) LLM backend adapter. No network."""

import json

import httpx
import pytest

from src import llm, tailor


def test_backend_selection(monkeypatch):
    monkeypatch.delenv("AUTO_APPLY_LLM_BACKEND", raising=False)
    assert llm.backend_name() == "anthropic"
    assert llm.active_model() == tailor.MODEL
    monkeypatch.setenv("AUTO_APPLY_LLM_BACKEND", "local")
    monkeypatch.setenv("AUTO_APPLY_LLM_MODEL", "llama-3.3-70b")
    assert llm.backend_name() == "local"
    assert llm.active_model() == "llama-3.3-70b"


def test_extract_json_tolerates_fences_and_prose():
    assert llm._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm._extract_json('Sure! Here you go: {"a": 2} hope that helps') == {"a": 2}


def _mock_openai_response(payload: dict):
    """Build a LocalLLM whose httpx client returns one canned chat-completion."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_object"
        assert body["model"] == "test-model"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 321, "completion_tokens": 123},
            },
        )

    transport = httpx.MockTransport(handler)

    client = llm.LocalLLM(base_url="http://cluster.local:8000/v1", model="test-model")

    # patch httpx.Client used inside parse() to use our mock transport
    import src.llm as L

    orig = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs.pop("timeout", None)
        return orig(transport=transport)

    L.httpx = httpx  # ensure module ref
    return client, patched_client, orig


def test_local_tailor_roundtrips_structured_output(monkeypatch):
    payload = {
        "fit_score": 73,
        "matched_keywords": ["Python", "RAG"],
        "missing_keywords": ["Kubernetes"],
        "resume_markdown": "# Robin Vega",
        "cover_letter": "Hello.",
        "application_note": "Apply.",
    }
    client, patched_client, orig = _mock_openai_response(payload)
    monkeypatch.setattr(httpx, "Client", patched_client)
    try:
        result, usage = tailor.tailor("Some AI job posting", master_profile="My experience", client=client)
    finally:
        monkeypatch.setattr(httpx, "Client", orig)
    assert result.fit_score == 73 and result.matched_keywords == ["Python", "RAG"]
    assert usage["input_tokens"] == 321 and usage["output_tokens"] == 123
    # the adapter satisfies api._require_key (configured endpoint == available)
    assert client.api_key


def test_local_llm_requires_base_url():
    with pytest.raises(ValueError):
        llm.LocalLLM(base_url="", model="x")
