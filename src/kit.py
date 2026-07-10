"""Application-kit generation.

Given a job posting and the candidate's relevant experience, produce a complete,
ready-to-review application kit: a tailored resume + cover letter, a 5-bullet
"why me", a recruiter DM, interview-story prompts, and — critically — an
**evidence map** that ties every claim back to a snippet from the experience
corpus, so you can verify there's no hallucination before you send anything.

Everything is grounded in the experience passed in (the RAG-retrieved chunks,
each labelled with its source). Kits are written under out/kits/ which is
gitignored, so generated packets never get committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import anthropic

MODEL = "claude-opus-4-8"

KIT_INSTRUCTIONS = """\
You are a career strategist assembling ONE candidate's application kit for ONE
job. Use ONLY facts present in the RELEVANT EXPERIENCE below — never invent
employers, titles, metrics, tools, or claims. If the job needs something the
candidate lacks, do not fake it.

Produce:
- resume_markdown: a tailored one-page resume (Markdown), facts only from the experience.
- cover_letter: under 200 words, names something specific about the company, no cliché opener.
- why_me: exactly 5 short, concrete bullets — each a distinct, specific reason this candidate fits.
- recruiter_dm: a concise (<120 word) message to a recruiter; specific hook, no fluff.
- interview_stories: 3-5 prompts the candidate can prep, each pointing at a real project/experience
  they can tell a STAR-style story about for this role.
- evidence_map: for EVERY non-trivial claim you make in the resume bullets and why_me, add an entry
  with: the `claim` (short), the `snippet` (a short verbatim-ish quote from the experience that
  supports it), and the `source` (the [from ...] label the snippet came from). If a claim has no
  supporting snippet, do not make the claim.

Write like a person. Minimal em-dashes. No buzzwords (results-driven, spearheaded, leveraged,
proven track record, passionate, synergy, robust, seamless).
"""


class EvidenceItem(BaseModel):
    claim: str = Field(description="A short claim made in the resume/why-me.")
    snippet: str = Field(description="A supporting quote from the experience corpus.")
    source: str = Field(description="The source label the snippet came from.")


class ApplicationKit(BaseModel):
    """A complete, verifiable application kit for one job."""

    resume_markdown: str
    cover_letter: str
    why_me: list[str] = Field(description="Exactly 5 'why me' bullets.")
    recruiter_dm: str
    interview_stories: list[str] = Field(description="3-5 interview story prompts.")
    evidence_map: list[EvidenceItem] = Field(
        default_factory=list, description="Claim -> supporting snippet + source, for every claim."
    )


def generate_kit(
    job_text: str, experience: str, *, client: anthropic.Anthropic | None = None
) -> tuple[ApplicationKit, dict]:
    """Generate an application kit grounded in `experience`. Returns (kit, usage)."""
    import anthropic

    client = client or anthropic.Anthropic()
    system = [
        {
            "type": "text",
            "text": KIT_INSTRUCTIONS
            + "\n\n=== RELEVANT EXPERIENCE (source of truth) ===\n\n"
            + experience.strip(),
        }
    ]
    response = client.messages.parse(
        model=MODEL,
        max_tokens=12000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[
            {
                "role": "user",
                "content": "Build my application kit for the job posting below.\n\n"
                "=== JOB POSTING ===\n\n" + job_text.strip(),
            }
        ],
        output_format=ApplicationKit,
    )
    kit = response.parsed_output
    if kit is None:
        reason = getattr(response, "stop_reason", "unknown")
        raise RuntimeError(f"Model did not return a parseable kit (stop_reason={reason}).")
    u = response.usage
    usage = {
        "input_tokens": u.input_tokens,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
        "output_tokens": u.output_tokens,
    }
    return kit, usage


def write_kit(
    dest: Path, kit: ApplicationKit, *, job_meta: dict | None = None, make_pdf: bool = False
) -> list[Path]:
    """Write a kit to `dest` as separate review-ready files. Returns the files written.

    With `make_pdf=True`, also render the tailored resume to an upload-ready one-page
    `resume.pdf` (best-effort: skipped with a note if headless Chrome isn't installed).
    """
    dest.mkdir(parents=True, exist_ok=True)
    files = {
        "resume.md": kit.resume_markdown,
        "cover-letter.md": kit.cover_letter,
        "why-me.md": "# Why me\n\n" + "\n".join(f"- {b}" for b in kit.why_me),
        "recruiter-dm.md": "# Recruiter DM\n\n" + kit.recruiter_dm,
        "interview-stories.md": "# Interview story prompts\n\n"
        + "\n".join(f"- {s}" for s in kit.interview_stories),
        "evidence-map.md": _evidence_md(kit),
        "kit.json": json.dumps({**(job_meta or {}), **kit.model_dump()}, indent=2, ensure_ascii=False),
    }
    written = []
    for name, content in files.items():
        path = dest / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    if make_pdf:
        from . import pdfgen

        company = (job_meta or {}).get("company", "Resume")
        pdf = pdfgen.resume_pdf(kit.resume_markdown, dest / "resume.pdf", title=f"Resume — {company}")
        if pdf:
            written.append(pdf)
    return written


def _evidence_md(kit: ApplicationKit) -> str:
    lines = [
        "# Evidence map",
        "",
        "Every claim below is backed by a snippet from your experience corpus.",
        "",
    ]
    if not kit.evidence_map:
        lines.append("_No evidence entries returned._")
    for e in kit.evidence_map:
        lines += [f"### {e.claim}", f"> {e.snippet}", f"— `{e.source}`", ""]
    return "\n".join(lines)
