"""Shared test fixtures: deterministic fakes so the whole suite runs offline.

No network, no ANTHROPIC_API_KEY: every Anthropic call is faked, and retrieval uses a
deterministic bag-of-words embedding function instead of downloading a model.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

AUTO_APPLY = Path(__file__).resolve().parent.parent
_DIM = 96


def _bow(texts):
    """Deterministic normalized bag-of-words vectors (overlap → cosine similarity)."""
    out = []
    for text in texts:
        v = [0.0] * _DIM
        for tok in re.findall(r"[a-z]+", text.lower()):
            v[hash(tok) % _DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


class FakeEmbeddingFunction:
    """Chroma embedding function with no model download (covers old + new EF APIs)."""

    def name(self):  # some chromadb versions require this for persistence
        return "fake-bow"

    def __call__(self, input):
        return _bow(input)

    def embed_documents(self, input):
        return _bow(input)

    def embed_query(self, input):
        return _bow(input)


def make_tailored(**overrides):
    from src.tailor import TailoredApplication

    base = dict(
        fit_score=80,
        matched_keywords=["Python", "RAG"],
        missing_keywords=["Kubernetes"],
        resume_markdown="# Robin Vega\nApplied AI Engineer. Python, RAG, FastAPI, Docker.",
        cover_letter="Dear hiring team,",
        application_note="Strong fit; apply.",
    )
    base.update(overrides)
    return TailoredApplication(**base)


def make_kit(**overrides):
    from src.kit import ApplicationKit, EvidenceItem

    base = dict(
        resume_markdown="# Robin Vega",
        cover_letter="Dear team,",
        why_me=["a", "b", "c", "d", "e"],
        recruiter_dm="Hi — I build RAG systems.",
        interview_stories=["s1", "s2", "s3"],
        evidence_map=[EvidenceItem(claim="Built RAG", snippet="RAG over Chroma", source="00-master.md")],
    )
    base.update(overrides)
    return ApplicationKit(**base)


class _Usage(SimpleNamespace):
    pass


def _usage():
    return _Usage(
        input_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=200
    )


class FakeMessages:
    def __init__(self, parsed=None, tool_input=None, kit=None):
        self._parsed = parsed or make_tailored()
        self._kit = kit or make_kit()
        self._tool_input = tool_input or {
            "title": "Applied AI Engineer",
            "company": "Northwind Labs",
            "location": "Remote",
            "remote": True,
            "salary": "$150k",
            "required_skills": ["Python", "RAG"],
            "description": "Build LLM apps.",
        }

    def parse(self, **kw):  # used by tailor.tailor and kit.generate_kit
        out = (
            self._kit
            if getattr(kw.get("output_format"), "__name__", "") == "ApplicationKit"
            else self._parsed
        )
        return SimpleNamespace(parsed_output=out, usage=_usage(), stop_reason="end_turn")

    def create(self, **kw):  # used by ingest.parse_posting (tool use)
        block = SimpleNamespace(type="tool_use", name="record_job_posting", input=self._tool_input)
        return SimpleNamespace(content=[block], usage=_usage(), stop_reason="tool_use")


class FakeAnthropic:
    def __init__(self, parsed=None, tool_input=None, kit=None):
        self.api_key = "test-key"  # satisfies api._require_key
        self.auth_token = None
        self.messages = FakeMessages(parsed=parsed, tool_input=tool_input, kit=kit)


@pytest.fixture
def fake_client():
    return FakeAnthropic()


@pytest.fixture
def fake_ef():
    return FakeEmbeddingFunction()


@pytest.fixture
def sample_corpus_dir():
    return AUTO_APPLY / "sample-corpus"


@pytest.fixture
def rag_index(tmp_path, fake_ef, sample_corpus_dir):
    """Build a hermetic Chroma index from the sample corpus; returns (index_path, ef)."""
    from src import rag

    index_path = str(tmp_path / "chroma")
    rag.build_index(sample_corpus_dir, index_path=index_path, embedding_function=fake_ef)
    return index_path, fake_ef
