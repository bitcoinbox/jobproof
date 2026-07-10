"""Tests for the job-search-cockpit upgrade: diligence fields, import targets,
fit-explanation fields, and application-kit generation."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from src import api, ingest, kit, store
from tests.conftest import AUTO_APPLY, make_tailored

FIX = AUTO_APPLY / "fixtures"


# --- migrations -------------------------------------------------------------


def test_migration_adds_diligence_and_fit_columns(tmp_path):
    """An old DB (pre-cockpit schema) gains all new columns on connect()."""
    p = str(tmp_path / "old.db")
    c = sqlite3.connect(p)
    c.executescript(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, source TEXT, source_id TEXT, url TEXT,"
        " title TEXT, company TEXT, location TEXT, remote INTEGER, salary TEXT,"
        " required_skills TEXT, description TEXT, posted_at TEXT, fetched_at TEXT,"
        " UNIQUE(source, source_id));"
        "CREATE TABLE applications (id INTEGER PRIMARY KEY, job_id INTEGER, status TEXT,"
        " fit_score INTEGER, matched_keywords TEXT, missing_keywords TEXT, application_note TEXT,"
        " mode TEXT, out_dir TEXT, created_at TEXT, updated_at TEXT);"
    )
    c.commit()
    c.close()
    conn = store.connect(p)
    jobcols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    appcols = {r["name"] for r in conn.execute("PRAGMA table_info(applications)")}
    for col in (
        "legitimacy_status",
        "diligence_notes",
        "review_signal",
        "source_confidence",
        "priority",
        "apply_recommendation",
        "next_action",
        "applied_at",
        "follow_up_at",
    ):
        assert col in jobcols, col
    assert {"kits", "generation_runs"} <= {
        r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    for col in (
        "why_match",
        "why_not",
        "missing_proof",
        "keywords_to_mirror",
        "recruiter_angle",
        "evidence_map",
    ):
        assert col in appcols, col


# --- diligence --------------------------------------------------------------


def _job(**kw):
    from types import SimpleNamespace as NS

    base = dict(
        source="wellfound",
        source_id="x",
        url=None,
        title="Founding AI Engineer",
        company="Ruli",
        location="Remote / unverified",
        remote=True,
        salary=None,
        required_skills=[],
        description="d",
        posted_at=None,
        fetched_at=None,
    )
    base.update(kw)
    return NS(**base)


def test_update_diligence_valid_and_invalid(tmp_path):
    conn = store.connect(str(tmp_path / "d.db"))
    jid = store.upsert_job(conn, _job())
    store.update_diligence(
        conn, jid, {"legitimacy_status": "verified", "apply_recommendation": "apply_today", "priority": 1}
    )
    row = next(r for r in store.dashboard_rows(conn) if r["id"] == jid)
    assert (
        row["legitimacy_status"] == "verified"
        and row["apply_recommendation"] == "apply_today"
        and row["priority"] == 1
    )
    with pytest.raises(ValueError):
        store.update_diligence(conn, jid, {"legitimacy_status": "bogus"})
    with pytest.raises(ValueError):
        store.update_diligence(conn, jid, {"priority": 9})


def test_dashboard_rows_have_diligence_defaults(tmp_path):
    conn = store.connect(str(tmp_path / "d.db"))
    store.upsert_job(conn, _job())
    r = store.dashboard_rows(conn)[0]
    assert r["legitimacy_status"] == "needs_diligence"
    assert r["apply_recommendation"] == "consider"
    assert r["review_signal"] == "unknown" and r["source_confidence"] == "unverified" and r["priority"] == 3
    assert "why" in r


# --- import targets (idempotent) -------------------------------------------


def test_import_targets_idempotent_with_diligence(tmp_path):
    conn = store.connect(str(tmp_path / "t.db"))
    first = ingest.ingest(
        "fixture", "", limit=100, conn=conn, use_llm=False, fixtures_dir=FIX, provider="startup-targets"
    )
    again = ingest.ingest(
        "fixture", "", limit=100, conn=conn, use_llm=False, fixtures_dir=FIX, provider="startup-targets"
    )
    assert len(first) == 6 and again == []
    rows = {r["company"]: r for r in store.dashboard_rows(conn)}
    assert rows["Arize AI"]["legitimacy_status"] == "verified"  # verified live posting
    assert rows["HomeTeams"]["legitimacy_status"] == "needs_diligence"  # unverifiable
    assert rows["Arize AI"]["url"] and rows["HomeTeams"]["url"] is None  # no fabricated link


# --- fit-explanation fields -------------------------------------------------


def test_fit_explanation_fields_serialize_and_persist(tmp_path):
    ta = make_tailored(
        why_match="strong RAG fit",
        why_not="no k8s",
        missing_proof="no prod k8s",
        keywords_to_mirror=["RAG", "FastAPI"],
        recruiter_angle="I ship RAG",
    )
    dumped = ta.model_dump()
    assert dumped["why_match"] == "strong RAG fit" and dumped["keywords_to_mirror"] == ["RAG", "FastAPI"]

    conn = store.connect(str(tmp_path / "f.db"))
    jid = store.upsert_job(conn, _job())
    store.record_application(
        conn, jid, ta, mode="rag", evidence_map=[{"claim": "c", "snippet": "s", "source": "src"}]
    )
    app = store.job_detail(conn, jid)["application"]
    assert app["why_match"] == "strong RAG fit"
    assert app["keywords_to_mirror"] == ["RAG", "FastAPI"]
    assert app["evidence_map"] == [{"claim": "c", "snippet": "s", "source": "src"}]


def test_tailored_application_backwards_compatible():
    """Old-style construction (no decision fields) still validates with defaults."""
    ta = make_tailored()
    assert ta.why_match == "" and ta.keywords_to_mirror == [] and ta.recruiter_angle == ""


# --- API --------------------------------------------------------------------


def _seed_db(tmp_path):
    db = str(tmp_path / "api.db")
    conn = store.connect(db)
    ingest.ingest(
        "fixture", "", limit=100, conn=conn, use_llm=False, fixtures_dir=FIX, provider="startup-targets"
    )
    conn.close()
    return db


def test_api_jobs_returns_diligence_and_options(tmp_path):
    db = _seed_db(tmp_path)
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        d = c.get("/api/jobs").json()
        assert d["stats"]["needs_diligence"] >= 1
        assert "apply_today" in d["stats"]
        assert "legitimacy_status" in d["options"] and "apply_recommendation" in d["options"]
        job = d["jobs"][0]
        for f in ("legitimacy_status", "apply_recommendation", "source_confidence", "priority", "why"):
            assert f in job


def test_api_job_detail_and_diligence_update(tmp_path):
    db = _seed_db(tmp_path)
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        jid = c.get("/api/jobs").json()["jobs"][0]["id"]
        detail = c.get(f"/api/jobs/{jid}").json()
        assert detail["job"]["id"] == jid and "application" in detail
        assert c.get("/api/jobs/999999").status_code == 404
        r = c.post(f"/api/jobs/{jid}/diligence", json={"apply_recommendation": "apply_today", "priority": 1})
        assert r.status_code == 200
        assert c.post(f"/api/jobs/{jid}/diligence", json={"legitimacy_status": "nope"}).status_code == 400
        updated = next(j for j in c.get("/api/jobs").json()["jobs"] if j["id"] == jid)
        assert updated["apply_recommendation"] == "apply_today" and updated["priority"] == 1


def test_api_tailor_job_backfills_fit(tmp_path, fake_client):
    db = _seed_db(tmp_path)
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        api.app.state.client = fake_client
        api.app.state.master = "MASTER PROFILE"
        api.app.state.index_path = str(tmp_path / "no-index")  # rag not ready -> full-profile path
        jid = c.get("/api/jobs").json()["jobs"][0]["id"]
        r = c.post(f"/api/jobs/{jid}/tailor")
        assert r.status_code == 200 and r.json()["fit_score"] == 80 and r.json()["mode"] == "full-profile"
        # fit now backfilled on the dashboard + detail
        row = next(j for j in c.get("/api/jobs").json()["jobs"] if j["id"] == jid)
        assert row["best_fit"] == 80
        assert c.get(f"/api/jobs/{jid}").json()["application"]["fit_score"] == 80
        assert c.get("/api/jobs").json()["stats"]["generation_runs"] == 1


def test_tracker_status_sets_followup_and_next_action(tmp_path):
    conn = store.connect(str(tmp_path / "tracker.db"))
    jid = store.upsert_job(conn, _job())
    store.set_job_status(conn, jid, "applied")
    row = store.dashboard_rows(conn)[0]
    assert row["next_action"] == "follow_up"
    assert row["applied_at"] and row["follow_up_at"]
    store.update_tracker(conn, jid, {"next_action": "message_recruiter", "notes": "sent app"})
    row = store.dashboard_rows(conn)[0]
    assert row["next_action"] == "message_recruiter" and row["notes"] == "sent app"
    with pytest.raises(ValueError):
        store.update_tracker(conn, jid, {"next_action": "panic"})


# --- application kit --------------------------------------------------------


def test_generate_and_write_kit(tmp_path, fake_client):
    k, usage = kit.generate_kit("Some AI engineer JD", "EXPERIENCE CONTEXT", client=fake_client)
    assert len(k.why_me) == 5 and k.evidence_map and k.recruiter_dm
    dest = tmp_path / "out" / "kits" / "acme-1"
    written = kit.write_kit(dest, k, job_meta={"company": "Acme", "job_id": 1})
    names = {p.name for p in written}
    assert {
        "resume.md",
        "cover-letter.md",
        "why-me.md",
        "recruiter-dm.md",
        "interview-stories.md",
        "evidence-map.md",
        "kit.json",
    } <= names
    assert (dest / "evidence-map.md").read_text().count("###") >= 1  # evidence rendered


def test_api_kit_persists_history_and_evidence(tmp_path, fake_client, monkeypatch):
    db = _seed_db(tmp_path)
    from types import SimpleNamespace as NS

    monkeypatch.setattr(
        api.rag, "retrieve", lambda *a, **k: [NS(source="resume.md", heading="RAG", distance=0.1)]
    )
    monkeypatch.setattr(api.rag, "format_context", lambda chunks: "[from resume.md]\nRAG over Chroma")
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        api.app.state.client = fake_client
        api.app.state.index_path = str(tmp_path)
        jid = c.get("/api/jobs").json()["jobs"][0]["id"]
        r = c.post(f"/api/jobs/{jid}/kit")
        assert r.status_code == 200 and r.json()["kit_id"]
        detail = c.get(f"/api/jobs/{jid}").json()
        assert detail["kits"] and detail["kits"][0]["evidence_map"]
        assert detail["runs"][0]["kind"] == "kit" and detail["runs"][0]["status"] == "done"
        latest = c.get(f"/api/jobs/{jid}/kit/latest").json()
        assert latest["recruiter_dm"]
        row = next(j for j in c.get("/api/jobs").json()["jobs"] if j["id"] == jid)
        assert row["kit_count"] == 1 and row["latest_kit_at"]


def test_kit_output_dir_is_gitignored():
    """Generated kits live under out/ which the repo ignores (no packets committed)."""
    gitignore = (AUTO_APPLY / ".gitignore").read_text()
    assert "out/" in gitignore


# --- job-ops product features ----------------------------------------------


def test_tasks_board_calendar_and_search_runs(tmp_path):
    conn = store.connect(str(tmp_path / "ops.db"))
    jid = store.upsert_job(conn, _job(company="Acme", title="AI Engineer"))
    store.set_job_status(conn, jid, "applied")
    tid = store.create_task(conn, "Ping recruiter", job_id=jid, due_at="2026-06-15T12:00:00+00:00")
    assert store.board(conn)["applied"][0]["id"] == jid
    cal = store.calendar_items(conn)
    assert any(i["kind"] == "follow_up" for i in cal)
    assert any(i.get("task_id") == tid for i in cal)
    rid = store.record_search_run(
        conn, query="ai engineer", sources=["wellfound"], found_count=12, deduped_count=3
    )
    assert store.search_runs(conn)[0]["id"] == rid
    store.complete_task(conn, tid)
    assert store.tasks(conn) == []


def test_api_capture_analysis_email_and_autofill(tmp_path):
    db = str(tmp_path / "ops_api.db")
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        api.app.state.master = "# Robin\nPython RAG FastAPI Docker Chroma LLM evals."
        payload = {
            "title": "Applied AI Engineer",
            "company": "Acme AI",
            "location": "Remote",
            "url": "https://example.com/job",
            "description": "Build RAG agents with Python, FastAPI, Docker, evals, and vector databases.",
            "required_skills": ["Python", "RAG"],
        }
        r = c.post("/api/capture-job", json=payload)
        assert r.status_code == 200
        jid = r.json()["job_id"]
        assert c.get("/api/board").json()["columns"]["new"][0]["id"] == jid

        task = c.post("/api/tasks", json={"job_id": jid, "title": "Find hiring manager", "priority": 2})
        assert task.status_code == 200
        assert c.get("/api/tasks").json()["tasks"][0]["job_id"] == jid

        ats = c.post(f"/api/jobs/{jid}/ats").json()
        assert ats["score"] > 0 and "python" in ats["matched"]
        diff = c.post(
            f"/api/jobs/{jid}/diff",
            json={"text": "# Robin\nPython RAG FastAPI Docker Chroma LLM evals.\nAgents."},
        ).json()
        assert diff["changed_lines"] > 0
        prep = c.post(f"/api/jobs/{jid}/interview-prep").json()
        assert prep["likely_questions"] and prep["report_id"]
        plan = c.post(f"/api/jobs/{jid}/autofill-plan").json()
        assert plan["dry_run"] is True and plan["submit"] is False

        email = c.post(
            "/api/email-events/parse",
            json={"job_id": jid, "text": "Interview request\nCan you share availability for a call?"},
        ).json()
        assert email["inferred_status"] == "interview"
        assert c.get(f"/api/jobs/{jid}").json()["job"]["status"] == "interview"


def test_api_search_runs(tmp_path):
    db = _seed_db(tmp_path)
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        r = c.post(
            "/api/search-runs",
            json={"query": "ai engineer remote", "sources": ["github", "wellfound"], "found_count": 20},
        )
        assert r.status_code == 200
        listed = c.get("/api/search-runs").json()["search_runs"][0]
        assert listed["query"] == "ai engineer remote" and listed["sources"] == ["github", "wellfound"]
