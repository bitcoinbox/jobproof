"""Core resume-tailoring logic.

Takes a master profile (source of truth) + a job posting and returns a tailored
one-page resume, a short cover letter, an honest fit score, and the keyword gap.

Uses an LLM with:
  - adaptive thinking (the model decides how much to reason per job)
  - prompt caching on the master profile (the stable prefix), so tailoring to
    many jobs in a row reuses that context at ~10% of the input cost
  - structured outputs (Pydantic) so the score + keywords come back validated,
    no string parsing
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-4-8"

# Stable instructions. Kept BEFORE the master profile so the whole system block
# is one cacheable prefix (see build_system).
SYSTEM_INSTRUCTIONS = """\
You are an expert technical resume writer and career strategist tailoring ONE
candidate's application to ONE job posting.

Hard rules:
- Use ONLY facts present in the candidate's experience below (the MASTER PROFILE
  or RELEVANT EXPERIENCE section — whichever is provided). Never invent
  employers, titles, dates, metrics, tools, or experience. If the job wants
  something the candidate doesn't have, leave it out and report it in
  missing_keywords. (Under RELEVANT EXPERIENCE you're seeing only the chunks
  retrieved for this job, not the full resume — tailor from what's here.)
- Mirror the job posting's exact vocabulary where the candidate genuinely has
  the skill (if they say "GenAI", don't only say "LLM"). Do not keyword-stuff.
- Write like a person, not a generator. Specific over impressive. Vary sentence
  length. Minimal em-dashes. BANNED words: results-driven, spearheaded,
  leveraged, proven track record, passionate, dynamic, rare, robust, seamless,
  synergy, realm, intricate, showcasing, pivotal, force multiplier.
- The resume is ONE page (~450-550 words). Reorder so the most relevant
  experience for THIS job leads. If the candidate holds a security clearance
  the posting requires, surface it near the top — never imply one they lack.
- The cover letter is under 200 words, names something specific about the
  company, and never uses a cliché opener.

Scoring:
- fit_score (0-100): an honest estimate of how well the candidate matches the
  role's *hard* requirements. Do not inflate. A cleared role the candidate is
  qualified for scores high; a senior-ML role needing a PhD scores low.
- matched_keywords: job requirements the master profile genuinely supports.
- missing_keywords: job requirements NOT supported by the master profile.
- application_note: 2-3 sentences — should they apply, what to emphasize, and
  how to handle the biggest gap.

Decision fields (be concrete and honest — these drive an apply/skip decision):
- why_match: 1-2 sentences on the strongest reasons this candidate fits THIS role.
- why_not: 1-2 sentences on the real risks/gaps that could screen them out.
- missing_proof: what evidence the candidate lacks to back a key requirement
  (e.g. "no production Kubernetes shown"). Empty string if none.
- keywords_to_mirror: exact phrases from the posting to echo in the resume,
  but ONLY ones the candidate genuinely supports.
- recruiter_angle: one sentence the candidate could open a recruiter message
  with — the single most compelling, specific hook for this role.
"""


class TailoredApplication(BaseModel):
    """Validated structured output for one tailored application."""

    fit_score: int = Field(description="Honest 0-100 match against the role's core/hard requirements.")
    matched_keywords: list[str] = Field(
        description="Job-posting keywords genuinely supported by the master profile."
    )
    missing_keywords: list[str] = Field(
        description="Job-required skills/tools NOT found in the master profile."
    )
    resume_markdown: str = Field(
        description="Tailored one-page resume in Markdown. Facts only from the master profile."
    )
    cover_letter: str = Field(
        description="A cover letter under 200 words, specific to this company and role."
    )
    application_note: str = Field(
        description="2-3 sentence strategy note: apply or not, what to emphasize, biggest gap."
    )
    # Decision fields. Defaulted for backwards compatibility — older callers/tests
    # that build TailoredApplication without them still validate.
    why_match: str = Field(default="", description="Strongest reasons the candidate fits this role.")
    why_not: str = Field(default="", description="Real risks/gaps that could screen them out.")
    missing_proof: str = Field(default="", description="Evidence the candidate lacks for a key requirement.")
    keywords_to_mirror: list[str] = Field(
        default_factory=list, description="Exact posting phrases to echo (only genuinely-supported ones)."
    )
    recruiter_angle: str = Field(default="", description="One-sentence hook to open a recruiter message.")


def build_system(
    profile_text: str, *, cache: bool = True, label: str = "MASTER PROFILE (source of truth)"
) -> list[dict]:
    """One system block = stable instructions + the candidate's facts.

    With the full master profile (`cache=True`), the cache_control marker makes
    everything up to here a reusable prefix: the only thing that changes between
    jobs is the user message (the posting), so every job after the first reads
    this prefix from cache.

    With RAG (`cache=False`), the facts are chunks retrieved for THIS job, so the
    block differs per posting and there's nothing to reuse across jobs — we drop
    the marker rather than pay cache-write cost for a prefix that's never reread.
    """
    block = {
        "type": "text",
        "text": (SYSTEM_INSTRUCTIONS + f"\n\n=== {label} ===\n\n" + profile_text.strip()),
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def tailor(
    job_text: str,
    master_profile: str | None = None,
    *,
    experience: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> tuple[TailoredApplication, dict]:
    """Tailor one application. Returns (result, usage).

    Pass `master_profile` for the cached full-profile path (the MVP), or
    `experience` for the RAG path (chunks retrieved for this job, uncached).
    """
    if (master_profile is None) == (experience is None):
        raise ValueError("Pass exactly one of master_profile or experience.")
    client = client or anthropic.Anthropic()

    if experience is not None:
        system = build_system(
            experience,
            cache=False,
            label="RELEVANT EXPERIENCE (retrieved for this job — source of truth)",
        )
    else:
        system = build_system(master_profile)

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=system,
        messages=[
            {
                "role": "user",
                "content": (
                    "Tailor my application to the job posting below. Return the "
                    "structured fields.\n\n=== JOB POSTING ===\n\n" + job_text.strip()
                ),
            }
        ],
        output_format=TailoredApplication,
    )

    result = response.parsed_output
    if result is None:
        reason = getattr(response, "stop_reason", "unknown")
        raise RuntimeError(
            f"Model did not return a parseable application (stop_reason={reason}). "
            "If this was a refusal, check the job text; otherwise raise max_tokens."
        )

    u = response.usage
    usage = {
        "input_tokens": u.input_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
        "output_tokens": u.output_tokens,
    }
    return result, usage
