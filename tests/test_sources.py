"""Tests for the ATS / board ingestion sources (Greenhouse, Lever, Ashby, HN).

Each source's normalize() is a pure function over the provider's JSON shape, so we
test the mapping directly with sample payloads — no network.
"""

from src import ingest


def test_greenhouse_normalize_maps_and_unescapes():
    data = {
        "jobs": [
            {
                "id": 123,
                "title": "Senior AI Engineer",
                "location": {"name": "Remote - US"},
                "content": "&lt;p&gt;Build &lt;b&gt;RAG&lt;/b&gt; systems in Python.&lt;/p&gt;",
                "absolute_url": "https://boards.greenhouse.io/acme/jobs/123",
                "updated_at": "2026-06-01T00:00:00Z",
                "departments": [{"name": "Engineering"}],
            },
            {"id": 9, "title": "Office Manager", "location": {"name": "NYC"}, "content": "admin"},
        ]
    }
    out = ingest.GreenhouseSource.normalize(data, board="acme", query="ai engineer", limit=10)
    assert len(out) == 1
    j = out[0]
    assert j["source"] == "greenhouse" and j["source_id"] == "123"
    assert j["company"] == "Acme" and j["remote"] is True
    assert j["source_confidence"] == "direct_ats"
    assert "<p>" in j["description_html"] and "&lt;" not in j["description_html"]  # unescaped
    assert j["url"].endswith("/jobs/123")


def test_lever_normalize():
    data = [
        {
            "id": "abc",
            "text": "ML Platform Engineer",
            "categories": {"location": "Remote", "team": "ML", "commitment": "Full-time"},
            "descriptionPlain": "Own the ML platform. Python, Kubernetes.",
            "description": "<p>Own the ML platform.</p>",
            "hostedUrl": "https://jobs.lever.co/acme/abc",
            "createdAt": 1717200000000,
        }
    ]
    out = ingest.LeverSource.normalize(data, board="acme", query="ml", limit=10)
    assert len(out) == 1 and out[0]["source"] == "lever"
    assert out[0]["title"] == "ML Platform Engineer" and out[0]["remote"] is True
    assert "ML" in out[0]["tags"] and out[0]["source_confidence"] == "direct_ats"


def test_ashby_normalize_remote_flag_and_filter():
    data = {
        "jobs": [
            {
                "id": "j1",
                "title": "Applied AI Engineer",
                "location": "San Francisco",
                "isRemote": True,
                "descriptionHtml": "<p>LLMs and evals.</p>",
                "descriptionPlain": "LLMs and evals.",
                "jobUrl": "https://jobs.ashbyhq.com/acme/j1",
                "department": "AI",
                "publishedAt": "2026-06-02",
            }
        ]
    }
    out = ingest.AshbySource.normalize(data, board="acme", query="applied ai", limit=10)
    assert len(out) == 1
    assert out[0]["remote"] is True  # from isRemote even though location is SF
    assert out[0]["url"].endswith("/j1") and out[0]["source"] == "ashby"
    # a non-matching query drops it
    assert ingest.AshbySource.normalize(data, board="acme", query="rust embedded", limit=10) == []


def test_hn_normalize_extracts_company_and_filters():
    data = {
        "hits": [
            {
                "objectID": "111",
                "comment_text": "Northwind Labs | Senior AI Engineer | REMOTE | We build LLM tooling.",
                "created_at": "2026-06-01T00:00:00Z",
            },
            {"objectID": "222", "comment_text": "Some non-job chatter about the weather."},
        ]
    }
    out = ingest.HackerNewsSource.normalize(data, board="40000000", query="ai engineer", limit=10)
    assert len(out) == 1
    assert out[0]["company"] == "Northwind Labs" and out[0]["remote"] is True
    assert out[0]["url"] == "https://news.ycombinator.com/item?id=111"


def test_board_sources_require_board():
    import pytest

    for name in ("greenhouse", "lever", "ashby", "hn"):
        with pytest.raises(SystemExit):
            ingest.get_source(name)  # no board → clear error
    # registry exposes them
    assert {"greenhouse", "lever", "ashby", "hn"} <= set(ingest.SOURCES)
