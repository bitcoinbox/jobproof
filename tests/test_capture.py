"""Tests for universal job capture + multi-dimension scoring + resume recommendation.

Fully offline: every fixture is a saved HTML page or pasted text. No network, no API key.
"""

from pathlib import Path

from src import capture, scoring

FIX = Path(__file__).resolve().parent / "fixtures" / "captures"


def _load(name):
    return (FIX / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- extraction


def test_clearancejobs_booz_allen_extraction():
    job = capture.parse_job(_load("clearancejobs-booz-allen.html"), source_hint="clearancejobs")
    assert job.parser == "jsonld"
    assert job.title == "Systems Engineer"
    assert job.company == "Booz Allen Hamilton"
    assert job.source == "clearancejobs"
    assert job.work_mode == "remote"  # TELECOMMUTE
    assert job.clearance == "TS/SCI"
    assert job.salary_min == 86800 and job.salary_max == 198000
    assert job.job_id == "7654321"
    assert job.posted_date == "2026-06-01"
    assert job.degree is None or isinstance(job.degree, str)
    assert "Security+" in job.certifications
    skills = " ".join(job.required_skills + job.preferred_skills).lower()
    for kw in ("network", "vpn", "firewall", "python", "bash", "puppet", "katello"):
        assert kw in skills, kw
    assert "STIG" in job.raw  # raw preserved for audit
    assert job.confidence >= 0.8 and job.needs_review is False


def test_dice_devsecops():
    job = capture.parse_job(_load("dice-devsecops.html"))
    assert job.company == "Northwind Systems"
    assert job.work_mode == "remote"
    assert job.salary_max == 175000
    assert job.source == "dice"
    assert job.years_experience == 7
    assert job.degree == "Bachelor's"


def test_linkedin_field_tech_extraction():
    job = capture.parse_job(_load("linkedin-field-tech.html"))
    assert job.title == "IT Field Technician"
    assert job.company == "Globex Corp"
    assert job.work_mode == "field"  # field-based / travel to client sites
    assert job.source == "linkedin"
    assert job.salary_max == 58000


def test_generic_company_ats():
    job = capture.parse_job(_load("company-ats-applied-ai.html"), url="https://tidepool.example/careers/123")
    assert job.title == "Applied AI Engineer"
    assert job.company == "Tide Pool Systems"
    assert job.work_mode == "remote"
    assert job.source == "company_site"


def test_pasted_text_heuristic_fallback():
    job = capture.parse_job(_load("clearancejobs-paste.txt"), source_hint="clearancejobs")
    assert job.parser == "heuristic"
    assert job.title == "Network Engineer"
    assert job.company == "Leidos"
    assert job.clearance == "TS/SCI w/ Poly"
    assert job.polygraph is True
    assert job.work_mode == "hybrid"
    assert job.salary_min == 135000 and job.salary_max == 165000
    assert job.job_id == "R-00098765"
    assert job.location == "Aurora, CO"


def test_capture_never_raises_on_junk():
    job = capture.parse_job("")
    assert job.title == "" and job.raw == ""
    job2 = capture.parse_job("<html><body><p>just some text</p></body></html>")
    assert job2.parser in ("heuristic", "meta")


# --------------------------------------------------------------------------- scoring


def test_booz_allen_scores_apply_and_cleared_variant():
    job = capture.parse_job(_load("clearancejobs-booz-allen.html"))
    fb = scoring.score_job(job)
    assert fb.recommendation in ("apply", "ask_recruiter")
    assert fb.overall >= 70
    assert fb.clearance_match.score == 100
    assert fb.resume_variant == "Cleared Defense AI"
    assert "Booz Allen" in fb.resume_reason or "defense" in fb.resume_reason.lower()
    assert "network" in " ".join(fb.matched_requirements).lower() or fb.matched_requirements


def test_field_tech_is_beneath_me_skip():
    job = capture.parse_job(_load("linkedin-field-tech.html"))
    fb = scoring.score_job(job)
    assert fb.recommendation == "skip"
    assert fb.overall <= 45
    assert fb.risk_score >= 40
    assert any("field-based" in f for f in fb.risk_flags)
    assert any("low-leverage" in f or "help-desk" in f for f in fb.risk_flags)
    assert fb.seniority_fit.score < 40
    assert fb.why_caution


def test_remote_applied_ai_variant():
    job = capture.parse_job(_load("company-ats-applied-ai.html"))
    fb = scoring.score_job(job)
    assert fb.resume_variant == "Applied AI"
    assert fb.remote_fit.score == 100
    assert fb.salary_fit.score == 100  # $200k → very strong
    assert fb.recommendation == "apply"


def test_dice_devsecops_strong_remote():
    job = capture.parse_job(_load("dice-devsecops.html"))
    fb = scoring.score_job(job)
    assert fb.recommendation in ("apply", "maybe")
    assert fb.resume_variant in ("Cleared Defense AI", "AI Software", "AI Automation")
    assert fb.remote_fit.score == 100


def test_unknown_salary_asks_recruiter():
    job = capture.parse_job(
        "Senior Network Engineer at Globex\nRemote. TS/SCI. Python, Bash, VPN, firewalls, networking.\n"
        "Strong systems engineering team.",
        source_hint="company_site",
    )
    assert job.salary_min is None and job.salary_max is None
    fb = scoring.score_job(job)
    assert fb.recommendation == "ask_recruiter"
    assert any("salary not listed" in f for f in fb.risk_flags)


def test_polygraph_above_level_is_skip():
    job = capture.parse_job(
        "Staff Engineer at MegaCorp\nClearance: TS/SCI with Full Scope Polygraph required\n"
        "Remote. Python, AI, backend.\nSalary: $180,000 - $220,000",
        source_hint="company_site",
    )
    assert job.polygraph is True
    fb = scoring.score_job(job)
    assert fb.clearance_match.score < 50
    assert fb.recommendation == "skip"
    assert any("polygraph" in f for f in fb.risk_flags)


def test_breakdown_complete_and_serializable():
    job = capture.parse_job(_load("dice-devsecops.html"))
    d = scoring.score_job(job).model_dump()
    for k in ("skill_match", "clearance_match", "salary_fit", "remote_fit", "seniority_fit", "passion_fit"):
        assert 0 <= d[k]["score"] <= 100 and d[k]["reason"]
    assert d["recommendation"] in ("apply", "ask_recruiter", "maybe", "skip")
    assert d["resume_variant"] in scoring.RESUME_VARIANTS
    assert 0 <= d["risk_score"] <= 100


def test_profile_is_editable_config():
    # Salary floors live in profile.py, not in scoring logic.
    from src import profile

    assert profile.PROFILE["min_salary"] == 120000
    assert any("booz allen" == c for c in profile.DEFENSE_CONTRACTORS)
