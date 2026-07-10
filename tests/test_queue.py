"""Tests for the action queue, follow-up flow, contacts, and automatic activity events."""

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from src import api, ingest, store

NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)


def _job(conn, n, **kw):
    posting = ingest.JobPosting(
        source="test",
        source_id=f"q-{n}",
        title=kw.pop("title", f"Role {n}"),
        company=kw.pop("company", f"Co{n}"),
        description="Build AI things with Python and RAG.",
        fetched_at=kw.pop("fetched_at", NOW.isoformat()),
        **kw,
    )
    return store.upsert_job(conn, posting, now=NOW)


def test_contacts_roundtrip(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    jid = _job(conn, 1)
    store.update_tracker(
        conn,
        jid,
        {"contact_name": "Sam Rivera", "contact_email": "sam@co1.com", "contact_url": "https://x.test/sam"},
    )
    row = next(r for r in store.dashboard_rows(conn) if r["id"] == jid)
    assert row["contact_name"] == "Sam Rivera"
    assert row["contact_email"] == "sam@co1.com"
    assert row["contact_url"] == "https://x.test/sam"


def test_status_change_logs_activity_event(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    jid = _job(conn, 1)
    store.set_job_status(conn, jid, "applied", now=NOW)
    events = store.job_events(conn, job_id=jid)
    assert any(e["kind"] == "status_change" and "applied" in e["subject"] for e in events)
    # Setting the same status again is not an event.
    n = len(events)
    store.set_job_status(conn, jid, "applied", now=NOW)
    assert len(store.job_events(conn, job_id=jid)) == n


def test_complete_follow_up_logs_and_reschedules(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    jid = _job(conn, 1)
    store.set_job_status(conn, jid, "applied", now=NOW)  # schedules follow-up at +7d
    next_at = store.complete_follow_up(conn, jid, now=NOW + timedelta(days=8))
    assert next_at is not None and next_at[:10] == (NOW + timedelta(days=15)).date().isoformat()
    assert any(e["kind"] == "followed_up" for e in store.job_events(conn, job_id=jid))
    # A closed job gets no next cycle.
    store.set_job_status(conn, jid, "rejected", now=NOW)
    assert store.complete_follow_up(conn, jid, now=NOW) is None


def test_snooze_follow_up(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    jid = _job(conn, 1)
    store.set_job_status(conn, jid, "applied", now=NOW)
    next_at = store.snooze_follow_up(conn, jid, days=3, now=NOW + timedelta(days=10))
    assert next_at[:10] == (NOW + timedelta(days=13)).date().isoformat()


def test_action_queue_tiers_and_order(tmp_path):
    conn = store.connect(str(tmp_path / "q.db"))
    later = NOW + timedelta(days=20)

    overdue = _job(conn, 1, company="OverdueCo")
    store.set_job_status(conn, overdue, "applied", now=NOW)  # follow-up at NOW+7 → overdue at `later`

    _job(conn, 2, company="ApplyCo", apply_recommendation="apply_today")

    interview = _job(conn, 3, company="InterviewCo")
    store.set_job_status(conn, interview, "interview", now=NOW)

    stale = _job(conn, 4, company="StaleCo", fetched_at=NOW.isoformat())
    store.set_job_status(conn, stale, "applied", now=NOW)
    store.update_tracker(conn, stale, {"follow_up_at": None})  # no follow-up scheduled → pure stale

    store.create_task(
        conn, "Email portfolio", priority=1, due_at=(later - timedelta(days=2)).isoformat(), now=NOW
    )

    items = store.action_queue(conn, now=later)
    kinds = {i["kind"] for i in items}
    assert {"follow_up", "apply", "interview", "stale", "task"} <= kinds

    fu = next(i for i in items if i["kind"] == "follow_up")
    assert fu["company"] == "OverdueCo" and fu["urgency"] == 0 and fu["days_overdue"] >= 12
    stales = [i for i in items if i["kind"] == "stale"]
    assert any(i["company"] == "StaleCo" for i in stales)
    assert all(i["days_overdue"] >= store.STALE_DAYS for i in stales)
    # Overdue tier sorts before attention/stale tiers.
    assert items[0]["urgency"] == 0
    assert [i["urgency"] for i in items] == sorted(i["urgency"] for i in items)
    # The interview job has a future-or-overdue follow-up too, but never double-reports stale.
    assert not any(i["kind"] == "stale" and i["company"] == "OverdueCo" for i in items)


def test_queue_and_follow_up_api(tmp_path):
    db = str(tmp_path / "q.db")
    conn = store.connect(db)
    jid = _job(conn, 1, company="ApiCo")
    real_past = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    store.set_job_status(conn, jid, "applied")
    store.update_tracker(conn, jid, {"follow_up_at": real_past})
    conn.close()

    with TestClient(api.app) as c:
        c.app.state.db_path = db
        items = c.get("/api/queue").json()["items"]
        assert any(i["kind"] == "follow_up" and i["job_id"] == jid for i in items)

        r = c.post(f"/api/jobs/{jid}/follow-up", json={"action": "snooze", "days": 5})
        assert r.status_code == 200
        assert r.json()["follow_up_at"][:10] == (datetime.now(UTC) + timedelta(days=5)).date().isoformat()

        r = c.post(f"/api/jobs/{jid}/follow-up", json={"action": "done"})
        assert r.status_code == 200 and r.json()["follow_up_at"] is not None

        # Contacts save through the tracker route and come back on the dashboard.
        r = c.post(f"/api/jobs/{jid}/tracker", json={"contact_name": "Dana", "contact_email": "d@a.co"})
        assert r.status_code == 200
        job = next(j for j in c.get("/api/jobs").json()["jobs"] if j["id"] == jid)
        assert job["contact_name"] == "Dana" and job["contact_email"] == "d@a.co"

        # Stats expose the queue-level counts.
        stats = c.get("/api/jobs").json()["stats"]
        assert {"interviews", "stale", "follow_up_overdue"} <= set(stats)
