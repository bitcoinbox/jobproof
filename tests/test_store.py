from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

from src import store

NOW = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)


def test_absolutize_variants():
    assert store.absolutize(1700000000).startswith("2023-11-14")
    assert store.absolutize("3 days ago", now=NOW) == "2026-06-05T12:00:00+00:00"
    assert store.absolutize("2026-05-01 09:30:00") == "2026-05-01T09:30:00+00:00"
    assert store.absolutize("2026-05-01T09:30:00Z") == "2026-05-01T09:30:00+00:00"
    assert store.absolutize("whenever") is None
    assert store.absolutize(None) is None


def _job(**kw):
    base = dict(
        source="fixture",
        source_id="abc",
        url="http://x/1",
        title="AI Engineer",
        company="Northwind Labs",
        location="Remote",
        remote=True,
        salary="$150k",
        required_skills=["Python"],
        description="Build.",
        posted_at="2 days ago",
        fetched_at=None,
    )
    base.update(kw)
    return NS(**base)


def test_upsert_dedup_and_seen():
    conn = store.connect(":memory:")
    a = store.upsert_job(conn, _job(), now=NOW)
    b = store.upsert_job(conn, _job(), now=NOW)
    assert a == b  # dedup on (source, source_id)
    assert store.seen(conn, "fixture", "abc")
    assert not store.seen(conn, "fixture", "nope")


def test_record_application_and_reports():
    conn = store.connect(":memory:")
    jid = store.upsert_job(conn, _job(), now=NOW)
    result = NS(
        fit_score=88,
        matched_keywords=["Python", "RAG"],
        missing_keywords=["Kubernetes"],
        application_note="Apply.",
    )
    retrieved = [
        NS(source="00-master-resume.md", heading="Summary", distance=0.7),
        NS(source="projects/01.md", heading="RAG", distance=0.9),
    ]
    aid = store.record_application(
        conn, jid, result, mode="rag", out_dir="out/x", retrieved=retrieved, now=NOW
    )
    assert (
        conn.execute("SELECT COUNT(*) FROM retrieved_sources WHERE application_id=?", (aid,)).fetchone()[0]
        == 2
    )
    store.set_status(conn, aid, "applied", now=NOW)
    md = store.report_markdown(conn)
    assert "Northwind Labs" in md and "88/100" in md and "applied" in md
    csv = store.report_csv(conn)
    assert csv.splitlines()[0].startswith("app_id,fit_score")
    assert "Northwind Labs" in csv


def test_set_status_rejects_unknown():
    conn = store.connect(":memory:")
    jid = store.upsert_job(conn, _job(), now=NOW)
    result = NS(fit_score=50, matched_keywords=[], missing_keywords=[], application_note="x")
    aid = store.record_application(conn, jid, result, mode="full-profile", now=NOW)
    with pytest.raises(ValueError):
        store.set_status(conn, aid, "bogus")
