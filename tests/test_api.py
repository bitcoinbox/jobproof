from fastapi.testclient import TestClient

from src import api


def test_healthz_needs_no_key():
    with TestClient(api.app) as c:
        h = c.get("/healthz").json()
    assert h["status"] == "ok"
    assert h["model"] == "claude-opus-4-8"
    assert set(["rag_ready", "allow_url", "has_master"]) <= set(h)


def test_tailor_full_mode(fake_client):
    with TestClient(api.app) as c:
        api.app.state.client = fake_client
        api.app.state.master = "MASTER PROFILE TEXT"
        r = c.post("/api/tailor", json={"job_text": "Python + RAG + FastAPI", "mode": "full"})
    assert r.status_code == 200
    d = r.json()
    assert d["fit_score"] == 80 and d["mode"] == "full"
    assert "Python" in d["matched_keywords"]


def test_validation_rejects_both_inputs():
    with TestClient(api.app) as c:
        r = c.post("/api/tailor", json={"job_text": "x", "job_url": "https://y.com"})
    assert r.status_code == 422


def test_url_disabled_by_default():
    with TestClient(api.app) as c:
        api.app.state.allow_url = False
        r = c.post("/api/tailor", json={"job_url": "https://example.com"})
    assert r.status_code == 400


def test_index_page_served():
    with TestClient(api.app) as c:
        assert c.get("/").status_code == 200
