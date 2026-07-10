"""Tests for the link-checker: classification, store flagging, queue exclusion. No network."""

from fastapi.testclient import TestClient

from src import api, ingest, links, store


def _job(conn, n, url):
    return store.upsert_job(
        conn,
        ingest.JobPosting(
            source="test",
            source_id=f"l-{n}",
            url=url,
            title=f"Role {n}",
            company=f"Co{n}",
            description="Build AI things.",
            fetched_at="2026-06-10T00:00:00+00:00",
            apply_recommendation="apply_today",
        ),
    )


def test_classify():
    assert links.classify(200) == "alive"
    assert links.classify(301) == "alive"
    assert links.classify(404) == "dead"
    assert links.classify(410) == "dead"
    assert links.classify(403) == "unknown"
    assert links.classify(500) == "unknown"
    assert links.classify(None) == "unknown"


def test_check_url_never_raises_on_fetcher_error():
    def boom(url):
        raise links.LinkError("DNS failure")

    r = links.check_url("https://x.test/job", fetcher=boom)
    assert r["link_status"] == "unknown" and "DNS" in r["note"]
    # empty url is handled
    assert links.check_url(None)["link_status"] == "unknown"


def test_check_jobs_flags_dead_and_logs_event(tmp_path):
    conn = store.connect(str(tmp_path / "l.db"))
    alive = _job(conn, 1, "https://ok.test/job")
    dead = _job(conn, 2, "https://gone.test/job")
    _job(conn, 3, None)  # no url → skipped

    codes = {"https://ok.test/job": 200, "https://gone.test/job": 404}
    summary = links.check_jobs(conn, fetcher=lambda u: codes[u])
    assert summary["checked"] == 2 and summary["alive"] == 1 and summary["dead"] == 1

    rows = {r["id"]: r for r in store.dashboard_rows(conn)}
    assert rows[alive]["link_status"] == "alive"
    assert rows[dead]["link_status"] == "dead"
    # a dead link writes a timeline event
    assert any(e["kind"] == "dead_link" for e in store.job_events(conn, job_id=dead))


def test_dead_link_drops_from_apply_queue(tmp_path):
    conn = store.connect(str(tmp_path / "l.db"))
    good = _job(conn, 1, "https://ok.test/job")
    bad = _job(conn, 2, "https://gone.test/job")
    # both are apply_today → both would be in the queue's apply tier
    q0 = {i["job_id"] for i in store.action_queue(conn) if i["kind"] == "apply"}
    assert {good, bad} <= q0

    store.set_link_status(conn, bad, "dead")
    q1 = {i["job_id"] for i in store.action_queue(conn) if i["kind"] == "apply"}
    assert good in q1 and bad not in q1  # dead one dropped


def test_check_link_api(tmp_path):
    db = str(tmp_path / "l.db")
    conn = store.connect(db)
    jid = _job(conn, 1, "https://gone.test/job")
    conn.close()
    with TestClient(api.app) as c:
        c.app.state.db_path = db
        # monkeypatch the module fetcher so no real network call happens
        import src.links as L

        orig = L._default_fetcher
        L._default_fetcher = lambda url, **kw: 404
        try:
            r = c.post(f"/api/jobs/{jid}/check-link")
            assert r.status_code == 200 and r.json()["link_status"] == "dead"
            stats = c.get("/api/jobs").json()["stats"]
            assert stats["dead_links"] == 1
        finally:
            L._default_fetcher = orig
