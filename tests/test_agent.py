"""Tests for the tool-using agent loop. Fully offline: a scripted fake Anthropic client
drives the reason→act→observe cycle, so we test the loop + tool dispatch without network."""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src import agent, api, capture, scoring, store

FIX = Path(__file__).resolve().parent / "fixtures" / "captures"


def _seed(db):
    conn = store.connect(db)
    for name in ("clearancejobs-booz-allen.html", "linkedin-field-tech.html"):
        job = capture.parse_job((FIX / name).read_text())
        store.record_capture(conn, job, scoring.score_job(job))
    conn.close()


def _tool_use(name, inp, tid="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=tid)


def _text(t):
    return SimpleNamespace(type="text", text=t)


class _ScriptedMessages:
    """Returns canned responses in order; records the messages it was called with."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        stop, content = self.script.pop(0)
        return SimpleNamespace(stop_reason=stop, content=content)


class _ScriptedClient:
    def __init__(self, script):
        self.api_key = "test-key"
        self.auth_token = None
        self.messages = _ScriptedMessages(script)


# --------------------------------------------------------------------------- tool dispatch (pure)


def test_tools_run_over_real_backend(tmp_path):
    db = str(tmp_path / "a.db")
    _seed(db)
    conn = store.connect(db)
    sj = agent._run_tool("search_jobs", {"recommendation": "apply_today"}, conn=conn, index_path="x")
    assert sj["count"] >= 1 and any(j["company"] == "Booz Allen Hamilton" for j in sj["jobs"])
    booz = next(j for j in sj["jobs"] if j["company"] == "Booz Allen Hamilton")
    gj = agent._run_tool("get_job", {"job_id": booz["id"]}, conn=conn, index_path="x")
    assert gj["clearance"] == "TS/SCI" and gj["fit"]["recommendation"] in ("apply", "ask_recruiter")
    st = agent._run_tool(
        "score_job_text", {"text": "Help Desk Technician I. Onsite. $40k."}, conn=conn, index_path="x"
    )
    assert st["recommendation"] == "skip"
    assert agent._run_tool("get_job", {"job_id": 9999}, conn=conn, index_path="x")["error"]
    conn.close()


# --------------------------------------------------------------------------- the loop


def test_agent_loops_tool_then_answers(tmp_path):
    db = str(tmp_path / "a.db")
    _seed(db)
    script = [
        ("tool_use", [_tool_use("search_jobs", {"recommendation": "apply_today"})]),
        ("end_turn", [_text("Prioritize Booz Allen Hamilton (#1) — remote, TS/SCI, strong pay.")]),
    ]
    client = _ScriptedClient(script)
    out = agent.run_agent("What should I apply to today?", client=client, db_path=db, index_path="x")
    assert out["truncated"] is False
    assert out["steps"] == 1
    assert out["trace"][0]["tool"] == "search_jobs"
    assert "Booz Allen" in out["trace"][0]["output"]["jobs"][0]["company"]
    assert "Booz Allen" in out["answer"]
    # the loop fed the tool result back: 2nd create call has a tool_result block in messages
    second = client.messages.calls[1]["messages"]
    blocks = [b for m in second if isinstance(m["content"], list) for b in m["content"]]
    assert any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)


def test_agent_respects_step_cap(tmp_path):
    db = str(tmp_path / "a.db")
    _seed(db)
    # always asks for a tool → never answers → must stop at max_steps
    script = [("tool_use", [_tool_use("search_jobs", {})]) for _ in range(10)]
    client = _ScriptedClient(script)
    out = agent.run_agent("loop forever", client=client, db_path=db, index_path="x", max_steps=3)
    assert out["truncated"] is True and out["steps"] == 3


def test_agent_api(tmp_path):
    db = str(tmp_path / "a.db")
    _seed(db)
    script = [
        ("tool_use", [_tool_use("search_jobs", {})]),
        ("end_turn", [_text("Here are your top roles.")]),
    ]
    with TestClient(api.app) as c:
        c.app.state.db_path = db
        c.app.state.client = _ScriptedClient(script)
        c.app.state.backend = "anthropic"
        c.app.state.index_path = "x"
        r = c.post("/api/agent", json={"question": "top roles?"})
        assert r.status_code == 200
        d = r.json()
        assert d["answer"] == "Here are your top roles." and d["trace"][0]["tool"] == "search_jobs"
        assert c.post("/api/agent", json={"question": "  "}).status_code == 422
