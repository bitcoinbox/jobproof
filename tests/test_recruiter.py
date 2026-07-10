"""Tests for the recruiter message parser + reply generator. Offline, deterministic."""

from pathlib import Path

from fastapi.testclient import TestClient

from src import api, recruiter, store

FIX = Path(__file__).resolve().parent / "fixtures" / "captures"
SAMPLE = (FIX / "recruiter-sample.txt").read_text(encoding="utf-8")


def test_parse_extracts_recruiter_company_contact_role():
    p = recruiter.parse_message(SAMPLE)
    assert p.recruiter_name == "Jordan Lee"
    assert p.company and "Example Defense" in p.company
    assert p.email == "recruiter@example.test"
    assert p.phone and "555-0142" in p.phone
    assert p.inferred_role and "Systems Engineer" in p.inferred_role


def test_generate_concise_late_reply_with_phone():
    r = recruiter.generate_reply(SAMPLE, intent="interested", tone="concise", phone="555-867-5309", late=True)
    reply = r.suggested_reply
    assert reply.startswith("Hi Jordan,")
    assert "Apologies for the late reply" in reply
    assert "555-867-5309" in reply
    assert "remote" in reply.lower()
    # sounds like me, not corporate: no buzzwords
    low = reply.lower()
    assert not any(b in low for b in ("synergy", "leverage", "thrilled", "passionate", "rockstar"))
    assert r.follow_up_task and "Jordan" in r.follow_up_task


def test_intents_and_tones():
    for intent in recruiter.INTENTS:
        for tone in recruiter.TONES:
            r = recruiter.generate_reply(SAMPLE, intent=intent, tone=tone, phone="555-867-5309")
            assert r.suggested_reply and r.intent == intent and r.tone == tone
    # decline never pushes a phone number
    d = recruiter.generate_reply(SAMPLE, intent="decline", phone="555-867-5309")
    assert "555-867-5309" not in d.suggested_reply
    # ask_salary actually asks about compensation
    s = recruiter.generate_reply(SAMPLE, intent="ask_salary")
    assert "salary" in s.suggested_reply.lower() or "range" in s.suggested_reply.lower()


def test_availability_and_no_late():
    r = recruiter.generate_reply(SAMPLE, intent="interested", availability="weekday afternoons", late=False)
    assert "weekday afternoons" in r.suggested_reply
    assert "Apologies for the late reply" not in r.suggested_reply


def test_bad_intent_raises():
    import pytest

    with pytest.raises(ValueError):
        recruiter.generate_reply(SAMPLE, intent="nope")


def test_recruiter_api_reply_logs_to_job(tmp_path):
    db = str(tmp_path / "r.db")
    conn = store.connect(db)
    from src import capture, scoring

    job = capture.parse_job((FIX / "clearancejobs-booz-allen.html").read_text())
    jid = store.record_capture(conn, job, scoring.score_job(job))
    conn.close()
    with TestClient(api.app) as c:
        c.app.state.db_path = db
        r = c.post("/api/recruiter/reply", json={"message": SAMPLE, "job_id": jid, "phone": "555-867-5309"})
        assert r.status_code == 200
        d = r.json()
        assert d["parsed"]["recruiter_name"] == "Jordan Lee"
        assert "555-867-5309" in d["suggested_reply"]
        assert "message_id" in d
        # logged to the job + a follow-up task created
        detail = c.get(f"/api/jobs/{jid}").json()
        assert (
            detail["recruiter_messages"]
            and detail["recruiter_messages"][0]["recruiter_name"] == "Jordan Lee"
        )
        assert any("Follow up" in t["title"] or "Jordan" in t["title"] for t in detail["tasks"])


def test_recruiter_parse_endpoint():
    with TestClient(api.app) as c:
        d = c.post("/api/recruiter/parse", json={"text": SAMPLE}).json()
        assert d["email"] == "recruiter@example.test"


def test_decision_updates_state(tmp_path):
    db = str(tmp_path / "d.db")
    conn = store.connect(db)
    from src import capture, scoring

    job = capture.parse_job((FIX / "clearancejobs-booz-allen.html").read_text())
    jid = store.record_capture(conn, job, scoring.score_job(job))
    conn.close()
    with TestClient(api.app) as c:
        c.app.state.db_path = db
        assert c.post(f"/api/jobs/{jid}/decision", json={"decision": "ask_recruiter"}).status_code == 200
        detail = c.get(f"/api/jobs/{jid}").json()
        assert detail["job"]["status"] == "interested"
        assert any("recruiter" in t["title"].lower() for t in detail["tasks"])
        assert c.post(f"/api/jobs/{jid}/decision", json={"decision": "skip"}).status_code == 200
        assert next(j for j in c.get("/api/jobs").json()["jobs"] if j["id"] == jid)["status"] == "skipped"
        assert c.post(f"/api/jobs/{jid}/decision", json={"decision": "bogus"}).status_code == 400
