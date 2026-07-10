"""Integration: capture -> store.record_capture -> dashboard, and the /api/capture route."""

from pathlib import Path

from fastapi.testclient import TestClient

from src import api, capture, scoring, store

FIX = Path(__file__).resolve().parent / "fixtures" / "captures"


def _html(name):
    return (FIX / name).read_text(encoding="utf-8")


def test_record_capture_persists_and_seeds_from_breakdown(tmp_path):
    conn = store.connect(str(tmp_path / "c.db"))
    job = capture.parse_job(_html("clearancejobs-booz-allen.html"))
    fb = scoring.score_job(job)
    jid = store.record_capture(conn, job, fb)

    row = next(r for r in store.dashboard_rows(conn) if r["id"] == jid)
    assert row["clearance"] == "TS/SCI"
    assert row["work_mode"] == "remote"
    assert row["recommended_variant"] == "Cleared Defense AI"
    assert row["quick_fit"] == fb.overall
    assert row["apply_recommendation"] in ("apply_today", "consider")  # apply / ask_recruiter
    assert row["priority"] <= 2  # high overall -> top priority
    assert row["source_confidence"] == "direct_ats"

    detail = store.job_detail(conn, jid)
    assert detail["fit_breakdown"]["resume_variant"] == "Cleared Defense AI"
    assert detail["fit_breakdown"]["skill_match"]["score"] >= 90
    # captured event logged + raw preserved
    assert any(e["kind"] == "captured" for e in store.job_events(conn, job_id=jid))
    raw = conn.execute("SELECT raw_capture FROM jobs WHERE id = ?", (jid,)).fetchone()["raw_capture"]
    assert "STIG" in raw


def test_record_capture_dedups(tmp_path):
    conn = store.connect(str(tmp_path / "c.db"))
    job = capture.parse_job(_html("dice-devsecops.html"))
    fb = scoring.score_job(job)
    a = store.record_capture(conn, job, fb)
    b = store.record_capture(conn, job, fb)
    assert a == b and store.stats(conn)["total"] == 1


def test_helpdesk_capture_maps_to_skip(tmp_path):
    conn = store.connect(str(tmp_path / "c.db"))
    job = capture.parse_job(_html("linkedin-helpdesk.html"))
    jid = store.record_capture(conn, job, scoring.score_job(job))
    row = next(r for r in store.dashboard_rows(conn) if r["id"] == jid)
    assert row["apply_recommendation"] == "skip" and row["priority"] >= 4


def test_api_capture_quick_score_does_not_save(tmp_path):
    db = str(tmp_path / "c.db")
    with TestClient(api.app) as c:
        c.app.state.db_path = db
        r = c.post("/api/capture", json={"content": _html("dice-devsecops.html"), "save": False})
        assert r.status_code == 200
        d = r.json()
        assert d["saved"] is False and "job_id" not in d
        assert d["fields"]["company"] == "Northwind Systems"
        assert d["breakdown"]["resume_variant"] in ("AI Software", "Cleared Defense AI", "AI Automation")
        assert "raw" not in d["fields"]  # raw not shipped to the client
        assert c.get("/api/jobs").json()["stats"]["total"] == 0  # nothing persisted


def test_api_capture_save_creates_tracked_job(tmp_path):
    db = str(tmp_path / "c.db")
    with TestClient(api.app) as c:
        c.app.state.db_path = db
        r = c.post(
            "/api/capture",
            json={
                "content": _html("clearancejobs-booz-allen.html"),
                "save": True,
                "source_hint": "clearancejobs",
            },
        )
        assert r.status_code == 200 and r.json()["saved"] is True
        jid = r.json()["job_id"]
        detail = c.get(f"/api/jobs/{jid}").json()
        assert detail["job"]["clearance"] == "TS/SCI"
        assert detail["fit_breakdown"]["recommendation"] == "apply"


def test_api_capture_rejects_empty():
    with TestClient(api.app) as c:
        assert c.post("/api/capture", json={"content": "   "}).status_code == 422
