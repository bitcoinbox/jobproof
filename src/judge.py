"""LLM-as-judge: does the generated résumé only claim what the evidence supports?

Faithfulness (a.k.a. groundedness) is the eval that backs JobProof's "never fabricate"
promise. We hand the model the EVIDENCE (the candidate's real experience) and the
generated TAILORED RÉSUMÉ, and ask it to decompose the résumé into atomic factual claims
and verdict each one against the evidence. Forced **structured output** via
``messages.parse`` (mirrors ``tailor.py``) — we never parse free text.

    faithfulness = supported_claims / total_claims     (1.0 = fully grounded)

The scorer (``faithfulness_score``) is a pure function — unit-testable with no API call.
The judge (``judge_faithfulness``) takes the same ``client=`` seam the rest of the code
uses, so tests inject a ``FakeAnthropic`` and never hit the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from . import tailor

if TYPE_CHECKING:
    import anthropic

MODEL = tailor.MODEL


class ClaimVerdict(BaseModel):
    """One atomic factual claim the résumé makes, judged against the evidence."""

    claim: str = Field(description="A single factual claim: a skill, tool, employer, role, metric, or accomplishment.")
    supported: bool = Field(
        description="True ONLY if the EVIDENCE directly supports this claim. If the evidence is "
        "silent on it or contradicts it, false — even if the claim is plausible."
    )
    evidence: str = Field(description="Short quote/paraphrase from the EVIDENCE supporting it, or why it is unsupported.")


class FaithfulnessReport(BaseModel):
    """The judge's per-claim verdicts for one generated résumé."""

    verdicts: list[ClaimVerdict]


def faithfulness_score(report: FaithfulnessReport) -> float:
    """Pure: fraction of claims the evidence supports (1.0 when there are no claims)."""
    if not report.verdicts:
        return 1.0
    supported = sum(1 for v in report.verdicts if v.supported)
    return round(supported / len(report.verdicts), 3)


def unsupported_claims(report: FaithfulnessReport) -> list[str]:
    """The claims the evidence does NOT back — the fabrication risks worth surfacing."""
    return [v.claim for v in report.verdicts if not v.supported]


JUDGE_SYSTEM = (
    "You are a strict grounding judge. You are given EVIDENCE (a candidate's real, "
    "verified experience) and a TAILORED RÉSUMÉ generated from it. Decompose the résumé "
    "into atomic factual claims — specific skills, tools, employers, roles, metrics, and "
    "accomplishments. For each claim, decide whether the EVIDENCE directly supports it. "
    "Be skeptical: a claim the evidence does not mention is UNSUPPORTED, even if it sounds "
    "plausible or industry-standard. Ignore generic filler, soft phrasing, and formatting — "
    "judge only substantive factual claims. Return a verdict for every claim."
)


def judge_faithfulness(
    resume_markdown: str,
    evidence: str,
    *,
    client: anthropic.Anthropic | None = None,
) -> tuple[float, FaithfulnessReport]:
    """Judge whether the résumé is grounded in the evidence.

    Returns ``(faithfulness_score, report)``. Raises if the model returns nothing
    parseable (same failure contract as ``tailor.tailor``).
    """
    import anthropic

    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    "=== EVIDENCE (source of truth) ===\n\n"
                    + evidence.strip()
                    + "\n\n=== TAILORED RÉSUMÉ (judge this) ===\n\n"
                    + resume_markdown.strip()
                ),
            }
        ],
        output_format=FaithfulnessReport,
    )
    report = response.parsed_output
    if report is None:
        reason = getattr(response, "stop_reason", "unknown")
        raise RuntimeError(f"Judge returned no parseable verdict (stop_reason={reason}).")
    return faithfulness_score(report), report
