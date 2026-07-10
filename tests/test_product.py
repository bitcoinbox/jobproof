"""Unit tests for the offline product tools (src/product.py) — pure, no network, no key."""

from src import product


def test_extract_keywords_finds_skills_and_caps():
    kws = product.extract_keywords(
        "We need Python, FastAPI, RAG, and a vector database. Docker a plus.", limit=10
    )
    low = [k.lower() for k in kws]
    assert "python" in low and "fastapi" in low and "rag" in low
    assert len(kws) <= 10


def test_ats_report_scores_coverage():
    job = "Senior AI engineer: Python, RAG, vector database, FastAPI, Docker, evals."
    resume = "Built Python + FastAPI services with RAG over a vector database and evals."
    r = product.ats_report(job, resume)
    assert 0 <= r["score"] <= 100
    assert "python" in [m.lower() for m in r["matched"]]
    assert isinstance(r["missing"], list) and isinstance(r["warnings"], list)
    # docker is in the JD but not the resume → should be missing
    assert any("docker" == m.lower() for m in r["missing"])


def test_diff_report_counts_changes():
    master = "Line A\nLine B\nLine C"
    tailored = "Line A\nLine B changed\nLine C\nLine D"
    d = product.diff_report(master, tailored)
    assert d["added_lines"] >= 1 and d["removed_lines"] >= 1
    assert d["changed_lines"] == d["added_lines"] + d["removed_lines"]
    assert "patch" in d


def test_interview_prep_shape():
    job = {"title": "Applied AI Engineer", "company": "Northwind", "description": "Python, RAG, evals"}
    app = {
        "missing_keywords": ["Kubernetes"],
        "matched_keywords": ["Python", "RAG"],
        "why_match": "ships LLM apps",
    }
    prep = product.interview_prep(job, app, {"interview_stories": ["s1"]})
    assert prep["role"].startswith("Applied AI Engineer")
    assert prep["opening_pitch"] and len(prep["likely_questions"]) >= 3
    assert any("Kubernetes" in g for g in prep["gap_drills"])


def test_parse_email_event_infers_status():
    assert (
        product.parse_email_event("Unfortunately we are moving forward with other candidates")[
            "inferred_status"
        ]
        == "rejected"
    )
    assert (
        product.parse_email_event("Can we schedule a call / interview next week?")["inferred_status"]
        == "interview"
    )
    assert (
        product.parse_email_event("We'd like to extend an offer; start date TBD")["inferred_status"]
        == "offer"
    )
    assert (
        product.parse_email_event("Thanks for applying — we received your application")["inferred_status"]
        == "applied"
    )
    assert product.parse_email_event("hello there")["inferred_status"] == "message"


def test_autofill_plan_is_dry_run():
    plan = product.autofill_plan(
        {"url": "https://example.com/job"}, {"resume_markdown": "# R", "cover_letter": "x"}
    )
    assert plan["dry_run"] is True and plan["submit"] is False
    assert plan["target_url"] == "https://example.com/job"
    assert any(f["field"] == "resume" and f["available"] for f in plan["fields"])
