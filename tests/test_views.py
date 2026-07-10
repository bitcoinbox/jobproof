"""Tests for the offline demo seed and the board / calendar / tasks views."""

from fastapi.testclient import TestClient

from src import api, cli, store


def _seeded(tmp_path):
    db = str(tmp_path / "v.db")
    cli._cmd_demo(["--db", db])  # offline: ingests committed fixtures, no key/network
    return db


def test_demo_seeds_jobs_offline(tmp_path):
    db = _seeded(tmp_path)
    conn = store.connect(db)
    s = store.stats(conn)
    assert s["total"] >= 8  # remoteok + remotive + startup-targets fixtures
    # re-running is idempotent (dedup on source/source_id)
    cli._cmd_demo(["--db", db])
    assert store.stats(store.connect(db))["total"] == s["total"]


def test_board_calendar_tasks(tmp_path):
    db = _seeded(tmp_path)
    with TestClient(api.app) as c:
        api.app.state.db_path = db
        board = c.get("/api/board").json()
        assert "columns" in board and "statuses" in board
        assert "items" in c.get("/api/calendar").json()

        assert c.get("/api/tasks").json()["tasks"] == []
        r = c.post("/api/tasks", json={"title": "Apply to Arize", "priority": 1})
        assert r.status_code == 200
        tid = r.json()["task_id"]
        titles = [t["title"] for t in c.get("/api/tasks").json()["tasks"]]
        assert "Apply to Arize" in titles
        assert c.post(f"/api/tasks/{tid}/complete").status_code == 200
        active = [t["title"] for t in c.get("/api/tasks").json()["tasks"]]
        assert "Apply to Arize" not in active  # completed tasks hidden by default
