"""Deterministic, offline fit scoring — a breakdown, not a single black-box number.

Splits "should I apply?" into seven readable dimensions (skill / clearance / salary / remote /
seniority / passion / risk) plus matched/missing requirements, an apply recommendation
(apply / ask_recruiter / maybe / skip), an explicit why-apply vs why-caution, and a resume
recommendation. All preferences live in `profile.py` so they're editable without touching logic.
No model, no network — this is the fast triage that runs on every capture and powers the
dashboard breakdown. The LLM tailor still does the deep per-claim work.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from . import profile as P
from .capture import CapturedJob

# Backwards-compatible alias (older callers / tests import scoring.DEFAULT_PROFILE).
DEFAULT_PROFILE = P.PROFILE

_CLEAR_RANK = {
    None: 0,
    "Public Trust": 1,
    "Secret": 2,
    "Top Secret": 3,
    "TS/SCI": 4,
    "TS/SCI w/ Poly": 5,
}

_SENIOR = re.compile(P.SENIOR_TERMS, re.I)
_JUNIOR = re.compile(P.JUNIOR_TERMS, re.I)

RESUME_VARIANTS = ("Cleared Defense AI", "AI Software", "Applied AI", "AI Automation", "ATS Plain")


class Dimension(BaseModel):
    score: int = Field(description="0-100 sub-score for this dimension.")
    reason: str = Field(description="Short factual explanation.")


class FitBreakdown(BaseModel):
    overall: int
    recommendation: str  # apply | ask_recruiter | maybe | skip
    resume_variant: str
    resume_reason: str = ""
    alternate_variant: str = ""
    resume_warnings: list[str] = Field(default_factory=list)
    skill_match: Dimension
    clearance_match: Dimension
    salary_fit: Dimension
    remote_fit: Dimension
    seniority_fit: Dimension
    passion_fit: Dimension
    risk_score: int = 0
    risk_flags: list[str] = Field(default_factory=list)
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    why_apply: str = ""
    why_caution: str = ""
    why_beneath: str = ""  # alias of why_caution, kept for back-compat


def _hits(text: str, terms) -> list[str]:
    low = text.lower()
    return [t for t in terms if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low)]


def _known_skill(skill: str, prof: list[str]) -> bool:
    s = skill.lower()
    return any(p == s or p in s or s in p for p in prof)


def _skill_match(job: CapturedJob, profile: dict) -> Dimension:
    hay = job.description + " " + " ".join(job.required_skills) + " " + job.title
    matched = _hits(hay, profile["skills"])
    n = len(set(matched))
    score = min(100, 25 + n * 9)
    top = ", ".join(sorted(set(matched))[:6]) or "few recognizable skills"
    return Dimension(score=score, reason=f"Matches {n} profile skills ({top}).")


def _clearance_match(job: CapturedJob, profile: dict) -> Dimension:
    need = _CLEAR_RANK.get(job.clearance, 0)
    have = _CLEAR_RANK.get(profile.get("clearance"), 0)
    if job.polygraph and not profile.get("has_polygraph"):
        # A poly requirement you don't hold is the disqualifier even if the base level matches.
        return Dimension(
            score=25, reason=f"Requires a polygraph ({job.clearance or 'clearance'}); you don't hold one."
        )
    if need == 0:
        return Dimension(score=75, reason="No clearance required; yours is a bonus.")
    if have >= need:
        return Dimension(score=100, reason=f"Requires {job.clearance}; you hold {profile['clearance']}.")
    gap = need - have
    return Dimension(
        score=max(0, 40 - gap * 15),
        reason=f"Requires {job.clearance}, above your {profile.get('clearance') or 'none'} — likely disqualifying.",
    )


def _salary_fit(job: CapturedJob, profile: dict) -> Dimension:
    top = job.salary_max or job.salary_min
    if not top:
        return Dimension(score=P.SALARY_UNKNOWN_SCORE, reason="No salary listed — ask the recruiter.")
    for floor, score, label in P.SALARY_TIERS:
        if top >= floor:
            return Dimension(score=score, reason=f"Top of band ${top:,}: {label}.")
    return Dimension(score=25, reason=f"Top of band ${top:,}.")


def _remote_fit(job: CapturedJob, preset: dict) -> Dimension:
    base = P.REMOTE_SCORES.get(job.work_mode, 55)
    # The default preset doesn't punish onsite; the cleared preset does.
    if job.work_mode == "onsite" and preset.get("onsite_ok", True):
        base = max(base, 60)
    labels = {
        "remote": "Remote.",
        "hybrid": "Hybrid.",
        "onsite": "Onsite.",
        "field": "Field-based (driving to sites).",
        "unknown": "Work mode not stated.",
    }
    return Dimension(score=base, reason=labels.get(job.work_mode, "Work mode not stated."))


def _seniority_fit(job: CapturedJob) -> Dimension:
    if any(b in (job.title or "").lower() for b in P.BENEATH_TERMS):
        return Dimension(score=15, reason="Help-desk / field-tech title — beneath a senior profile.")
    if _JUNIOR.search(job.title):
        return Dimension(score=20, reason="Junior/entry-level title.")
    if _SENIOR.search(job.title):
        return Dimension(score=95, reason="Senior-level title.")
    return Dimension(score=65, reason="Mid-level / unspecified seniority.")


def _passion_fit(job: CapturedJob, profile: dict, preset: dict) -> Dimension:
    hay = job.title + " " + job.description
    likes = _hits(hay, profile["passions"])
    dislikes = _hits(hay, preset.get("penalize", []))
    score = max(0, min(100, 50 + len(set(likes)) * 10 - len(set(dislikes)) * 25))
    bits = []
    if likes:
        bits.append("builder-track: " + ", ".join(sorted(set(likes))[:4]))
    if dislikes:
        bits.append("grunt-work signals: " + ", ".join(sorted(set(dislikes))[:3]))
    return Dimension(score=score, reason="; ".join(bits) or "neutral content.")


def _risk_flags(job: CapturedJob, profile: dict, preset: dict) -> list[str]:
    flags = []
    hay = (job.title + " " + job.description).lower()
    if job.polygraph and not profile.get("has_polygraph"):
        flags.append("polygraph required (you don't hold one)")
    elif _CLEAR_RANK.get(job.clearance, 0) > _CLEAR_RANK.get(profile.get("clearance"), 0):
        flags.append(f"requires {job.clearance} (above your level)")
    if job.salary_max and job.salary_max < profile["floor_salary"]:
        flags.append(f"salary below ${profile['floor_salary']:,}")
    elif job.salary_max and job.salary_max < profile["min_salary"]:
        flags.append(f"salary below ${profile['min_salary']:,}")
    if not (job.salary_min or job.salary_max):
        flags.append("salary not listed")
    if re.search(
        r"(\d{1,3})\s*%\s*travel|travel up to|travel\s*(?:required|expected|extensive)", hay
    ) and not preset.get("travel_ok", True):
        flags.append("travel required")
    if job.work_mode == "field":
        flags.append("field-based role")
    elif job.work_mode == "onsite" and not preset.get("onsite_ok", True):
        flags.append("fully onsite")
    if _hits(hay, preset.get("penalize", [])) or any(b in (job.title or "").lower() for b in P.BENEATH_TERMS):
        flags.append("low-leverage role (help-desk / field-tech signals)")
    if _JUNIOR.search(job.title):
        flags.append("junior/entry title")
    if re.search(r"\b(contract|contract-to-hire|c2c|w2 only|temporary|6[\s-]month)\b", hay):
        flags.append("contract/temp")
    return flags


def recommend_resume(job: CapturedJob) -> tuple[str, str, str, list[str]]:
    """Return (variant, reason, alternate_variant, warnings)."""
    text = (
        job.title + " " + job.description + " " + " ".join(job.required_skills + job.preferred_skills)
    ).lower()
    company = (job.company or "").lower()
    matches: list[tuple[str, str]] = []

    contractor = next((c for c in P.DEFENSE_CONTRACTORS if c in company), None)
    if contractor:
        matches.append(("Cleared Defense AI", f"defense contractor ({contractor.title()})"))
    elif job.clearance:
        matches.append(("Cleared Defense AI", f"clearance required ({job.clearance})"))

    for variant in P.RESUME_ORDER:
        if variant == "ATS Plain" or any(v == variant for v, _ in matches):
            continue
        pat = P.RESUME_RULES.get(variant)
        if pat and re.search(pat, text):
            matches.append((variant, f"matches {variant} keywords"))

    if not matches:
        matches.append(("ATS Plain", "no strong signal — plain, ATS-safe format"))

    primary, reason = matches[0]
    alternate = matches[1][0] if len(matches) > 1 else ""
    warnings = []
    if primary == "Cleared Defense AI" and not job.clearance and not contractor:
        warnings.append(
            "Routed to Cleared by network/infra keywords but no clearance stated — confirm it's a cleared role."
        )
    if job.clearance and primary != "Cleared Defense AI":
        warnings.append("Posting mentions a clearance — Cleared Defense AI may land better.")
    if primary == "Applied AI" and not re.search(r"\b(llm|rag|machine learning|ai engineer|genai)\b", text):
        warnings.append("Light on AI specifics — don't overstate LLM/RAG depth.")
    return primary, reason, alternate, warnings


def score_job(job: CapturedJob, *, profile: dict | None = None, preset: str = "cleared") -> FitBreakdown:
    """Compute the full breakdown. `preset` defaults to the cleared/defense profile."""
    profile = profile or P.PROFILE
    pre = P.PRESETS.get(preset, P.PRESETS["default"])

    skill = _skill_match(job, profile)
    clearance = _clearance_match(job, profile)
    salary = _salary_fit(job, profile)
    remote = _remote_fit(job, pre)
    seniority = _seniority_fit(job)
    passion = _passion_fit(job, profile, pre)
    risks = _risk_flags(job, profile, pre)

    w = {"skill": 0.28, "clearance": 0.14, "salary": 0.16, "remote": 0.14, "seniority": 0.12, "passion": 0.16}
    overall = round(
        skill.score * w["skill"]
        + clearance.score * w["clearance"]
        + salary.score * w["salary"]
        + remote.score * w["remote"]
        + seniority.score * w["seniority"]
        + passion.score * w["passion"]
    )
    critical = any(
        f.startswith("requires ")
        or f.startswith("polygraph")
        or f.startswith("low-leverage")
        or f == "junior/entry title"
        or f == "field-based role"
        for f in risks
    )
    if critical:
        overall = min(overall, 45)

    salary_unknown = not (job.salary_min or job.salary_max)
    if critical or overall < 45:
        rec = "skip"
    elif salary_unknown and overall >= 55:
        rec = "ask_recruiter"  # decent role, comp not stated — open with a question
    elif overall >= 70:
        rec = "apply"
    elif overall >= 52:
        rec = "maybe"
    else:
        rec = "skip"

    risk_score = min(100, 18 * len(risks) + (35 if critical else 0))

    prof_skills = [p.lower() for p in profile["skills"]]
    req = job.required_skills or []
    matched_req = [s for s in req if _known_skill(s, prof_skills)]
    missing_req = [s for s in req if not _known_skill(s, prof_skills)]

    variant, r_reason, alternate, r_warnings = recommend_resume(job)

    dims = {
        "skill": skill,
        "clearance": clearance,
        "salary": salary,
        "remote": remote,
        "seniority": seniority,
        "passion": passion,
    }
    strong = sorted(dims.values(), key=lambda d: d.score, reverse=True)[:3]
    why_apply = (
        "; ".join(d.reason for d in strong if d.score >= 60) or "Nothing stands out as a strong match."
    )
    why_caution = (
        "; ".join(risks)
        if risks
        else "; ".join(d.reason for d in sorted(dims.values(), key=lambda d: d.score)[:2] if d.score < 55)
    )
    why_caution = why_caution or "No major risks or downgrades detected."

    return FitBreakdown(
        overall=overall,
        recommendation=rec,
        resume_variant=variant,
        resume_reason=r_reason,
        alternate_variant=alternate,
        resume_warnings=r_warnings,
        skill_match=skill,
        clearance_match=clearance,
        salary_fit=salary,
        remote_fit=remote,
        seniority_fit=seniority,
        passion_fit=passion,
        risk_score=risk_score,
        risk_flags=risks,
        matched_requirements=matched_req,
        missing_requirements=missing_req,
        why_apply=why_apply,
        why_caution=why_caution,
        why_beneath=why_caution,
    )
