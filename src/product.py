"""Offline product features for the job-search cockpit.

These helpers keep high-value workflow features available without an API key:
ATS-style keyword coverage, resume diffs, interview prep, email/status parsing,
and a safe autofill dry-run plan.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

_STOP = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "build",
    "can",
    "for",
    "from",
    "have",
    "into",
    "our",
    "that",
    "the",
    "their",
    "this",
    "with",
    "will",
    "work",
    "you",
    "your",
}
_SKILL_PATTERNS = [
    "agentic ai",
    "applied ai",
    "machine learning",
    "natural language processing",
    "prompt engineering",
    "retrieval augmented generation",
    "vector database",
    "python",
    "typescript",
    "javascript",
    "fastapi",
    "react",
    "next.js",
    "postgres",
    "sqlite",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "rag",
    "llm",
    "evals",
    "agents",
    "langchain",
    "llamaindex",
    "openai",
    "anthropic",
    "pydantic",
    "chromadb",
    # infrastructure / networking / cleared-defense skills
    "vpn",
    "firewall",
    "vlan",
    "cisco",
    "routing",
    "switching",
    "bash",
    "linux",
    "networking",
    "network engineering",
    "systems engineering",
    "stig",
    "rmf",
    "devsecops",
    "secops",
    "security+",
    "puppet",
    "katello",
    "ansible",
    "vmware",
    "multi-enclave",
    "automation",
    "ci/cd",
    "terraform",
]


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9.+#-]{1,}", (text or "").lower()) if t not in _STOP]


def extract_keywords(text: str, *, limit: int = 24) -> list[str]:
    """Return likely ATS keywords from known skills plus high-signal posting terms."""
    low = (text or "").lower()
    found = [p for p in _SKILL_PATTERNS if re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", low)]
    counts = Counter(t for t in _tokens(text) if len(t) > 3)
    for term, _ in counts.most_common(limit * 2):
        if term not in found and not term.isdigit():
            found.append(term)
        if len(found) >= limit:
            break
    return found[:limit]


def ats_report(job_text: str, resume_text: str) -> dict:
    """Simple coverage report: what the posting asks for vs. what the resume says."""
    keywords = extract_keywords(job_text)
    resume_low = (resume_text or "").lower()
    matched = [k for k in keywords if k.lower() in resume_low]
    missing = [k for k in keywords if k not in matched]
    score = round((len(matched) / len(keywords)) * 100) if keywords else 0
    warnings = []
    if len(_tokens(job_text)) < 80:
        warnings.append("Posting text is thin; score may be noisy.")
    if len(_tokens(resume_text)) < 120:
        warnings.append("Resume text is short for ATS coverage checks.")
    if missing[:5]:
        warnings.append("Mirror proof for: " + ", ".join(missing[:5]))
    return {
        "score": score,
        "keywords": keywords,
        "matched": matched,
        "missing": missing,
        "warnings": warnings,
    }


def diff_report(master_text: str, tailored_text: str) -> dict:
    """Line-level resume diff with compact counts and a preview patch."""
    master_lines = (master_text or "").splitlines()
    tailored_lines = (tailored_text or "").splitlines()
    diff = list(
        difflib.unified_diff(
            master_lines,
            tailored_lines,
            fromfile="master",
            tofile="tailored",
            lineterm="",
        )
    )
    added = [ln[1:] for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln[1:] for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    return {
        "added_lines": len(added),
        "removed_lines": len(removed),
        "changed_lines": len(added) + len(removed),
        "added_preview": added[:12],
        "removed_preview": removed[:12],
        "patch": "\n".join(diff[:120]),
    }


def interview_prep(job: dict, application: dict | None = None, kit: dict | None = None) -> dict:
    """Generate a practical interview prep sheet from stored app evidence."""
    missing = (application or {}).get("missing_keywords") or extract_keywords(job.get("description", ""))[:5]
    matched = (application or {}).get("matched_keywords") or []
    stories = (kit or {}).get("interview_stories") or []
    why = (application or {}).get("why_match") or (job.get("why") or "")
    return {
        "role": f"{job.get('title')} at {job.get('company')}",
        "opening_pitch": why or "Lead with shipped AI systems, product judgment, and fast iteration.",
        "likely_questions": [
            f"How would you build the first 30 days of AI engineering work for {job.get('company')}?",
            "Walk me through a production AI system you shipped and how you evaluated it.",
            "What tradeoffs do you make between speed, quality, and model cost?",
            "How do you debug hallucinations, retrieval misses, or bad tool calls?",
        ],
        "gap_drills": [f"Prepare a concrete proof point for {kw}." for kw in missing[:6]],
        "keywords_to_echo": matched[:8],
        "stories": stories[:5],
        "closing_questions": [
            "What would make this hire a clear win after 60 days?",
            "Where is the current AI workflow most brittle?",
            "How do engineering, product, and GTM decide which AI bets matter?",
        ],
    }


def parse_email_event(text: str) -> dict:
    """Infer job-search event status from pasted recruiter/company email text."""
    low = (text or "").lower()
    status = "message"
    if any(
        x in low for x in ("unfortunately", "not moving forward", "decided to pursue", "other candidates")
    ):
        status = "rejected"
    elif any(x in low for x in ("interview", "schedule a call", "meet with", "calendly", "availability")):
        status = "interview"
    elif any(x in low for x in ("offer", "compensation", "start date")):
        status = "offer"
    elif any(
        x in low for x in ("received your application", "thanks for applying", "application was submitted")
    ):
        status = "applied"
    subject = next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")[:160]
    return {
        "inferred_status": status,
        "subject": subject,
        "body_excerpt": re.sub(r"\s+", " ", text or "").strip()[:500],
    }


def autofill_plan(job: dict, kit: dict | None = None) -> dict:
    """Return a submit-nothing browser automation checklist for a posting."""
    resume = (kit or {}).get("resume_markdown") or ""
    return {
        "dry_run": True,
        "submit": False,
        "target_url": job.get("url"),
        "fields": [
            {"field": "resume", "source": "latest kit resume", "available": bool(resume)},
            {
                "field": "cover_letter",
                "source": "latest kit cover letter",
                "available": bool((kit or {}).get("cover_letter")),
            },
            {"field": "linkedin", "source": "profile vault/manual", "available": False},
            {"field": "salary_expectation", "source": "manual review", "available": False},
            {"field": "work_authorization", "source": "manual review", "available": False},
        ],
        "checks": [
            "Open the posting URL.",
            "Detect required fields and compare them against the field plan.",
            "Populate only fields marked available after human review.",
            "Stop before submit and show a final review screen.",
        ],
    }
