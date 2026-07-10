"""Tests for the weekly ops digest (longitudinal report from events + runs)."""

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from src import api, ingest, store

NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


def _job(conn, n, source="remoteok"):
    return store.upsert_job(
        conn,
        ingest.JobPosting(
            source=source,
            source_id=f"d-{n}",
            url=f"https://x.test/{n}",
            title=f"Role {n}",
            company=f"Co{n}",
            description="Build AI things.",
            fetched_at=NOW.isoformat(),
        ),
        now=NOW,
    )


def test_digest_counts_week_activity(tmp_path):
    conn = store.connect(str(tmp_path / "d.db"))
    a = _job(conn, 1, source="greenhouse")
    _job(conn, 2, source="remoteok")
    store.set_job_status(conn, a, "applied", now=NOW)  # logs status_change + sets applied_at

    d = store.digest(conn, days=7, now=NOW + timedelta(days=1))
    assert d["totals"]["total"] == 2
    assert len(d["applied_this_week"]) == 1 and d["applied_this_week"][0]["company"] == "Co1"
    assert d["events"].get("status_change", 0) >= 1
    assert d["by_source"]["greenhouse"]["applied"] == 1
    assert d["by_source"]["remoteok"]["tracked"] == 1

    md = store.format_digest(d)
    assert "weekly digest" in md and "Applied this week" in md and "Co1" in md


def test_digest_window_excludes_old(tmp_path):
    conn = store.connect(str(tmp_path / "d.db"))
    a = _job(conn, 1)
    store.set_job_status(conn, a, "applied", now=NOW - timedelta(days=30))  # old
    d = store.digest(conn, days=7, now=NOW)
    assert d["applied_this_week"] == []  # the 30-day-old application is outside the window


def test_digest_api(tmp_path):
    db = str(tmp_path / "d.db")
    conn = store.connect(db)
    _job(conn, 1)
    conn.close()
    with TestClient(api.app) as c:
        c.app.state.db_path = db
        r = c.get("/api/digest?days=7")
        assert r.status_code == 200
        body = r.json()
        assert "markdown" in body and body["window_days"] == 7
