import sqlite3
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from src import api, store


def _job(**kw):
    base = dict(
        source="pr-leads",
        source_id="a",
        url="http://x",
        title="Engineer",
        company="Evertec",
        location="San Juan, Puerto Rico",
        remote=False,
        salary=None,
        required_skills=[],
        description="Build.",
        posted_at=None,
        fetched_at=None,
    )
    base.update(kw)
    return NS(**base)


def test_dashboard_rows_and_stats(tmp_path):
    conn = store.connect(str(tmp_path / "d.db"))
    store.upsert_job(
        conn, _job(source_id="a", company="Evertec", location="San Juan, Puerto Rico", remote=False)
    )
    store.upsert_job(conn, _job(source_id="b", company="OpenRouter", location="Remote (US)", remote=True))
    rows = store.dashboard_rows(conn)
    assert len(rows) == 2
    pr = next(r for r in rows if r["company"] == "Evertec")
    rem = next(r for r in rows if r["company"] == "OpenRouter")
    assert pr["puerto_rico"] is True and pr["remote"] is False
    assert rem["puerto_rico"] is False and rem["remote"] is True
    s = store.stats(conn)
    assert s["total"] == 2 and s["puerto_rico"] == 1 and s["remote"] == 1
    assert s["by_status"]["new"] == 2 and s["avg_fit"] is None


def test_set_job_status(tmp_path):
    conn = store.connect(str(tmp_path / "d.db"))
    jid = store.upsert_job(conn, _job(source_id="a"))
    store.set_job_status(conn, jid, "applied")
    assert store.dashboard_rows(conn)[0]["status"] == "applied"
    with pytest.raises(ValueError):
        store.set_job_status(conn, jid, "bogus")


def test_best_fit_surfaces_in_dashboard(tmp_path):
    conn = store.connect(str(tmp_path / "d.db"))
    jid = store.upsert_job(conn, _job(source_id="a"))
    result = NS(fit_score=82, matched_keywords=["Python"], missing_keywords=[], application_note="x")
    store.record_application(conn, jid, result, mode="rag")
    row = store.dashboard_rows(conn)[0]
    assert row["best_fit"] == 82
    assert store.stats(conn)["avg_fit"] == 82.0


def test_migration_adds_columns(tmp_path):
    """An older DB created before status/notes existed gets migrated on connect()."""
    p = str(tmp_path / "old.db")
    c = sqlite3.connect(p)
    c.executescript(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, source TEXT, source_id TEXT, url TEXT,"
        " title TEXT, company TEXT, location TEXT, remote INTEGER, salary TEXT,"
        " required_skills TEXT, description TEXT, posted_at TEXT, fetched_at TEXT,"
        " UNIQUE(source, source_id));"
    )
    c.commit()
    c.close()
    conn = store.connect(p)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert "status" in cols and "notes" in cols


def test_api_jobs_and_status(tmp_path):
    db = str(tmp_path / "api.db")
    conn = store.connect(db)
    store.upsert_job(conn, _job(source_id="a", company="Evertec"))
    conn.close()
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        d = c.get("/api/jobs").json()
        assert d["stats"]["total"] == 1 and "new" in d["statuses"]
        jid = d["jobs"][0]["id"]
        assert c.post(f"/api/jobs/{jid}/status", json={"status": "applied"}).status_code == 200
        assert c.post(f"/api/jobs/{jid}/status", json={"status": "nope"}).status_code == 400
        assert c.get("/").status_code == 200  # dashboard page
        assert c.get("/tailor").status_code == 200  # tailoring page


def test_dashboard_exposes_cockpit_workflow_controls():
    html = (api._STATIC / "dashboard.html").read_text(encoding="utf-8")
    assert 'id="capture-modal"' in html
    assert 'id="email-modal"' in html
    assert 'id="task-add"' in html
    assert "/api/capture-job" in html
    assert "/api/email-events/parse" in html


def test_notes_endpoint_and_description(tmp_path):
    db = str(tmp_path / "n.db")
    conn = store.connect(db)
    store.upsert_job(conn, _job(source_id="a", description="Build LLM apps with Python."))
    conn.close()
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        jid = c.get("/api/jobs").json()["jobs"][0]["id"]
        assert c.post(f"/api/jobs/{jid}/notes", json={"notes": "called recruiter"}).status_code == 200
        row = next(j for j in c.get("/api/jobs").json()["jobs"] if j["id"] == jid)
        assert row["notes"] == "called recruiter"
        assert "Python" in row["description"]
