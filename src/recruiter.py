"""Recruiter message tool: parse an inbound message, draft a reply that sounds like me.

Deterministic and offline — extracts the recruiter's name / company / email / phone and the
role being pitched, then composes a short, human reply for a chosen intent and tone. The voice
is deliberately plain: confident, casual, not corporate, not over-AI, not desperate.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

INTENTS = ("interested", "ask_salary", "ask_remote", "ask_timeline", "decline", "follow_up")
TONES = ("concise", "warm", "direct")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
# Name words are separated by spaces/tabs only (never newlines, so a sign-off name doesn't
# swallow the next line). The lead-in keywords are case-insensitive via inline (?i:...);
# the name's capitalization stays strict.
_NAME = [
    re.compile(r"(?i:i'?m|this is|my name is)\s+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2})"),
    re.compile(r"\b([A-Z][a-z]+[ \t]+[A-Z][a-z]+)[ \t]+here\b"),
    re.compile(
        r"(?i:thanks|best|regards|cheers|sincerely|best regards)[,!]?[ \t]*\n+[ \t]*([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2})"
    ),
]
_COMPANY = [
    re.compile(r"(?i:with|at|from|representing)\s+([A-Z][A-Za-z&.\-]+(?:[ \t]+[A-Z][A-Za-z&.\-]+){0,3})"),
]
# Capitalized phrase immediately before role/position/opening/opportunity.
_ROLE = re.compile(r"\b([A-Z][A-Za-z][A-Za-z/ \-]{2,40}?)\s+(?:role|position|opening|opportunity|req)\b")
_COMMON_FIRST_WORDS = {"The", "We", "Our", "I", "Hi", "Hello", "Hey", "Thanks", "Just", "This"}


class ParsedRecruiter(BaseModel):
    recruiter_name: str | None = None
    company: str | None = None
    email: str | None = None
    phone: str | None = None
    inferred_role: str | None = None


def parse_message(text: str) -> ParsedRecruiter:
    """Best-effort extraction of recruiter/company/contact + the role being pitched."""
    text = text or ""
    email = _EMAIL.search(text)
    phone = _PHONE.search(text)

    name = None
    for pat in _NAME:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip()
            if cand.split()[0] not in _COMMON_FIRST_WORDS:
                name = cand
                break

    company = None
    for pat in _COMPANY:
        for m in pat.finditer(text):
            cand = m.group(1).strip().rstrip(".")
            if cand.split()[0] not in _COMMON_FIRST_WORDS and len(cand) > 2:
                company = cand
                break
        if company:
            break
    # If a name was captured and accidentally swept into company, prefer the name.
    if company and name and name in company:
        company = company.replace(name, "").strip() or None

    role = None
    m = _ROLE.search(text)
    if m:
        role = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".,").title()

    return ParsedRecruiter(
        recruiter_name=name,
        company=company,
        email=(email.group(0) if email else None),
        phone=(phone.group(0) if phone else None),
        inferred_role=role,
    )


class RecruiterReply(BaseModel):
    suggested_reply: str
    follow_up_task: str
    parsed: ParsedRecruiter = Field(default_factory=ParsedRecruiter)
    intent: str = "interested"
    tone: str = "concise"


def _greeting(name: str | None, tone: str) -> str:
    who = name.split()[0] if name else "there"
    if tone == "warm":
        return f"Hi {who}, thanks for reaching out!"
    if tone == "direct":
        return f"Hi {who} —"
    return f"Hi {who},"


def _late_opener() -> str:
    return "Apologies for the late reply, I didn't see this until now."


def generate_reply(
    text: str,
    *,
    intent: str = "interested",
    tone: str = "concise",
    phone: str | None = None,
    availability: str | None = None,
    late: bool = True,
) -> RecruiterReply:
    """Draft a reply in the user's voice. `text` is the recruiter's message."""
    if intent not in INTENTS:
        raise ValueError(f"Unknown intent {intent!r}. Expected one of {', '.join(INTENTS)}.")
    if tone not in TONES:
        raise ValueError(f"Unknown tone {tone!r}. Expected one of {', '.join(TONES)}.")

    parsed = parse_message(text)
    role = parsed.inferred_role
    role_phrase = f" the {role} role" if role else " the role"

    lines: list[str] = [_greeting(parsed.recruiter_name, tone)]
    if late:
        lines.append(_late_opener())

    if intent == "interested":
        lines.append(
            "I recently updated my ClearanceJobs profile to see what's out there. "
            "I'm mainly looking for remote, or hybrid if it's the right fit, and I'd be "
            f"interested in hearing more about{role_phrase}."
        )
    elif intent == "ask_salary":
        lines.append(
            f"{role_phrase.strip().capitalize()} sounds interesting. Before we go further, "
            "could you share the salary range and whether it's remote or hybrid?"
        )
    elif intent == "ask_remote":
        lines.append(
            f"Thanks for thinking of me for{role_phrase}. Is this remote, or hybrid? "
            "Remote is my strong preference; I'd consider hybrid for the right role."
        )
    elif intent == "ask_timeline":
        lines.append(
            f"Interested in{role_phrase}. What does the timeline look like, and what are the next steps?"
        )
    elif intent == "decline":
        lines.append(
            f"Appreciate you reaching out about{role_phrase}, but it's not the right fit for me right now. "
            "Feel free to keep me in mind for remote senior systems/network or AI roles down the line."
        )
    elif intent == "follow_up":
        lines.append(
            f"Just following up on{role_phrase} — still interested and happy to talk. "
            "Let me know if you need anything else from me."
        )

    if availability:
        lines.append(f"I'm generally free {availability}.")
    if phone and intent != "decline":
        lines.append(f"Feel free to call me anytime at {phone}.")

    reply = " ".join(lines)

    who = parsed.recruiter_name or (parsed.company or "the recruiter")
    if intent == "decline":
        task = f"Declined {who} — no follow-up needed."
    elif intent in ("ask_salary", "ask_remote", "ask_timeline"):
        task = f"Await {who}'s answer; follow up in 3 days if no reply."
    else:
        task = f"Follow up with {who} in 3 days if no reply."

    return RecruiterReply(suggested_reply=reply, follow_up_task=task, parsed=parsed, intent=intent, tone=tone)
