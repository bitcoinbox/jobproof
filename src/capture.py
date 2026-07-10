"""Universal job capture: turn pasted text or a saved HTML page into a clean record.

Login-gated boards (ClearanceJobs, Dice, LinkedIn) can't be scraped reliably, so JobProof
is user-driven: paste the posting text, or save the page (Cmd-S) and import the HTML. Either
way we extract the same structured fields and **preserve the raw input** for audit.

Extraction is layered, most-robust first — no brittle DOM/selector scraping:
  1. schema.org `JobPosting` JSON-LD — embedded by ClearanceJobs/Dice/LinkedIn and most ATS
     pages for SEO. Stable, structured, and exactly the fields we want.
  2. OpenGraph / `<meta>` tags + `<title>` — when JSON-LD is absent.
  3. Heuristic text parser — for pasted plain text: first line = title, "at <Company>",
     and regexes for clearance, work mode, salary, and job id.

Everything is stdlib (html.parser, json, re) — no new dependencies, fully offline-testable.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

from pydantic import BaseModel, Field

from .ingest import strip_html

# --------------------------------------------------------------------------- model

SOURCES = ("clearancejobs", "dice", "linkedin", "company_site", "recruiter_email", "manual", "unknown")
WORK_MODES = ("remote", "hybrid", "onsite", "field", "unknown")

# Clearance ladder, highest first — detection returns the highest level mentioned.
_CLEARANCE = [
    (
        "TS/SCI w/ Poly",
        r"\b(ts/?sci|top secret/?sci)\b[^.]{0,40}\b(poly|polygraph)\b|full[\s-]?scope poly|\bci poly\b|\bfs poly\b",
    ),
    ("TS/SCI", r"\bts/?sci\b|\btop secret/?sci\b|\bsci\b"),
    ("Top Secret", r"\btop secret\b|\bts\b(?!\s*/?\s*sci)"),
    ("Secret", r"\bsecret\b"),
    ("Public Trust", r"\bpublic trust\b"),
]


class CapturedJob(BaseModel):
    """A posting parsed from text/HTML. `raw` is preserved verbatim for audit."""

    title: str = ""
    company: str = ""
    location: str | None = None
    work_mode: str = "unknown"  # remote | hybrid | onsite | unknown
    clearance: str | None = None
    salary: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    polygraph: bool = False
    job_id: str | None = None
    url: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    years_experience: int | None = None
    degree: str | None = None  # e.g. "Bachelor's", "Master's", or None
    certifications: list[str] = Field(default_factory=list)
    posted_date: str | None = None
    description: str = ""
    source: str = "manual"
    parser: str = "heuristic"  # which path produced this (jsonld | meta | heuristic)
    confidence: float = 1.0  # 0-1; <0.6 => fields likely need a human review/edit
    needs_review: bool = False
    raw: str = ""


# --------------------------------------------------------------------------- HTML scrape (structured only)


class _Extractor(HTMLParser):
    """Pull JSON-LD blocks, <meta> og/name tags, and <title> — not DOM content."""

    def __init__(self) -> None:
        super().__init__()
        self.ld: list[str] = []
        self.meta: dict[str, str] = {}
        self.title = ""
        self._in_ld = False
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._in_ld = True
            self.ld.append("")
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"])

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_ld = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_ld and self.ld:
            self.ld[-1] += data
        elif self._in_title:
            self.title += data


def _iter_jsonld_objects(blocks: list[str]):
    """Yield every dict in the JSON-LD blocks, flattening @graph and arrays."""
    for block in blocks:
        try:
            data = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                yield node
                if "@graph" in node:
                    stack.append(node["@graph"])


def _is_jobposting(node: dict) -> bool:
    t = node.get("@type")
    types = t if isinstance(t, list) else [t]
    return any(str(x).lower() == "jobposting" for x in types)


def _from_jsonld(node: dict) -> dict:
    """Map a schema.org JobPosting node to our fields."""
    org = node.get("hiringOrganization")
    company = org.get("name") if isinstance(org, dict) else (org if isinstance(org, str) else "")

    loc = node.get("jobLocation")
    location = None
    if isinstance(loc, list):
        loc = loc[0] if loc else None
    if isinstance(loc, dict):
        addr = loc.get("address", loc)
        if isinstance(addr, dict):
            location = (
                ", ".join(
                    str(addr[k])
                    for k in ("addressLocality", "addressRegion", "addressCountry")
                    if addr.get(k)
                )
                or None
            )
    if node.get("jobLocationType") == "TELECOMMUTE" or node.get("applicantLocationRequirements"):
        location = location or "Remote"

    salary = salary_min = salary_max = None
    bs = node.get("baseSalary")
    if isinstance(bs, dict):
        val = bs.get("value", {})
        if isinstance(val, dict):
            lo, hi = val.get("minValue"), val.get("maxValue") or val.get("value")
            unit = (val.get("unitText") or "YEAR").upper()
            mult = {"HOUR": 2080, "DAY": 260, "WEEK": 52, "MONTH": 12, "YEAR": 1}.get(unit, 1)
            try:
                salary_min = int(float(lo) * mult) if lo else None
                salary_max = int(float(hi) * mult) if hi else None
            except (TypeError, ValueError):
                pass
            if salary_min or salary_max:
                cur = bs.get("currency", "USD")
                salary = f"{cur} {salary_min or '?'}–{salary_max or '?'}/yr"

    return {
        "title": (node.get("title") or "").strip(),
        "company": (company or "").strip(),
        "location": location,
        "salary": salary,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "job_id": str(node.get("identifier", {}).get("value"))
        if isinstance(node.get("identifier"), dict)
        else (str(node["identifier"]) if node.get("identifier") else None),
        "url": node.get("url"),
        "posted_date": (node.get("datePosted") or "")[:10] or None,
        "description": strip_html(node.get("description") or ""),
    }


# --------------------------------------------------------------------------- field heuristics

_SALARY_RANGE = re.compile(
    r"\$?\s?(\d{2,3}(?:,\d{3})?|\d{2,3}\s?[kK])\s*(?:-|–|to)\s*\$?\s?(\d{2,3}(?:,\d{3})?|\d{2,3}\s?[kK])",
)
_SALARY_SINGLE = re.compile(r"\$\s?(\d{2,3}(?:,\d{3})?|\d{2,3}\s?[kK])\b")
_JOB_ID = re.compile(
    r"(?:job\s*(?:id|number|code|req(?:uisition)?(?:\s*id)?)|requisition|posting\s*id)\s*[:#]?\s*([A-Za-z]?[\w-]{3,})",
    re.I,
)


def detect_work_mode(text: str) -> str:
    low = text.lower()
    # Field-based work (driving to client sites) is distinct from a fixed onsite office.
    if re.search(
        r"field[\s-]?(based|technician|service|engineer)|travel to (client|customer|site)|multiple sites", low
    ):
        return "field"
    has_hybrid = "hybrid" in low
    has_remote = bool(
        re.search(r"\bremote\b|work from home|\bwfh\b|telework|telecommut", low)
    ) and not re.search(r"\bnot?\s+remote\b|no remote", low)
    has_onsite = bool(re.search(r"on-?site|in-office|in[\s-]person|on premises?|on-?prem", low))
    if has_hybrid:
        return "hybrid"
    if has_remote:
        return "remote"
    if has_onsite:
        return "onsite"
    return "unknown"


def detect_clearance(text: str) -> str | None:
    low = text.lower()
    for label, pat in _CLEARANCE:
        if re.search(pat, low):
            return label
    return None


def detect_polygraph(text: str) -> bool:
    return bool(re.search(r"\b(poly(graph)?|ci poly|fs poly|full[\s-]?scope)\b", text, re.I))


_DEGREE = [
    ("PhD", r"\bph\.?d\b|doctorate"),
    ("Master's", r"\bmaster'?s?\b|\bm\.?s\.?\b|\bmba\b"),
    ("Bachelor's", r"\bbachelor'?s?\b|\bb\.?s\.?\b|\bb\.?a\.?\b|four[\s-]year degree|undergraduate degree"),
    ("Associate's", r"\bassociate'?s?\b|\ba\.?a\.?s?\b"),
]
_CERTS = [
    "Security+",
    "Network+",
    "A+",
    "CISSP",
    "CISM",
    "CEH",
    "CCNA",
    "CCNP",
    "CCIE",
    "AWS Certified",
    "Azure",
    "PMP",
    "ITIL",
    "Linux+",
    "CASP",
    "GSEC",
    "OSCP",
    "Sec+",
]


def detect_degree(text: str) -> str | None:
    low = text.lower()
    for label, pat in _DEGREE:
        if re.search(pat, low):
            return label
    return None


def detect_certifications(text: str) -> list[str]:
    found = []
    for c in _CERTS:
        if re.search(rf"(?<![a-z0-9]){re.escape(c.lower())}(?![a-z0-9])", text.lower()):
            found.append("Security+" if c == "Sec+" else c)
    return sorted(set(found))


def detect_years_experience(text: str) -> int | None:
    # "8+ years", "5-7 years", "minimum of 10 years"
    m = re.search(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)\b", text, re.I)
    return int(m.group(1)) if m else None


def detect_posted_date(text: str) -> str | None:
    m = re.search(
        r"posted\s*(?:on|:)?\s*([A-Z][a-z]{2,8}\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})",
        text,
    )
    if m:
        return m.group(1)
    m = re.search(r"\b(\d+)\s+days?\s+ago\b", text, re.I)
    return f"{m.group(1)} days ago" if m else None


def split_skills(text: str, skills: list[str]) -> tuple[list[str], list[str]]:
    """Partition skills into required vs preferred using nearby 'preferred/nice-to-have' cues."""
    low = text.lower()
    # crude segmentation: everything after a 'preferred/nice to have/bonus/plus' header is preferred
    m = re.search(r"(preferred|nice[\s-]to[\s-]have|bonus|a plus|desired|pluses)\b", low)
    pref_zone = low[m.start() :] if m else ""
    required, preferred = [], []
    for s in skills:
        (preferred if s.lower() in pref_zone else required).append(s)
    return required, preferred


def _to_annual(num_str: str) -> int | None:
    s = num_str.lower().replace(",", "").replace("$", "").strip()
    try:
        if s.endswith("k"):
            return int(float(s[:-1]) * 1000)
        n = int(float(s))
    except ValueError:
        return None
    # bare 2-3 digit numbers in a salary context are thousands ("120 - 160")
    return n * 1000 if n < 1000 else n


def detect_salary(text: str) -> tuple[str | None, int | None, int | None]:
    m = _SALARY_RANGE.search(text)
    if m:
        lo, hi = _to_annual(m.group(1)), _to_annual(m.group(2))
        if lo and hi and lo <= hi and hi >= 30000:
            return (f"${lo:,}–${hi:,}", lo, hi)
    m = _SALARY_SINGLE.search(text)
    if m:
        v = _to_annual(m.group(1))
        if v and v >= 30000:
            return (f"${v:,}", v, v)
    return (None, None, None)


def detect_job_id(text: str, url: str | None) -> str | None:
    m = _JOB_ID.search(text)
    if m:
        return m.group(1)
    if url:
        m = re.search(r"/(?:jobs?|posting|view)/(?:[\w-]*?-)?(\d{5,})", url)
        if m:
            return m.group(1)
    return None


def _heuristic_fields(text: str) -> dict:
    """Title/company from the top of a pasted posting."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0] if lines else ""
    company = ""
    # "<Title> at <Company>" or "<Company> · <Title>" or "<Company> hiring <Title>"
    m = re.match(r"(.+?)\s+at\s+(.+)", title)
    if m:
        title, company = m.group(1).strip(), m.group(2).strip()
    else:
        m = re.search(r"(.+?)\s+hiring\s+(.+?)\s+in\s+", title, re.I)
        if m:
            company, title = m.group(1).strip(), m.group(2).strip()
        elif len(lines) > 1 and len(lines[1]) < 60:
            company = lines[1]
    # trim a trailing " | Company | Location"-style title
    title = re.split(r"\s+[|·]\s+", title)[0].strip()
    return {"title": title[:160], "company": company[:120]}


# --------------------------------------------------------------------------- source detection

_HOST_SOURCE = {
    "clearancejobs.com": "clearancejobs",
    "dice.com": "dice",
    "linkedin.com": "linkedin",
}


def detect_source(text: str, url: str | None, hint: str | None) -> str:
    if hint in SOURCES:
        return hint
    hay = f"{url or ''} {text[:400]}".lower()
    for host, src in _HOST_SOURCE.items():
        if host in hay:
            return src
    if url:
        return "company_site"
    return "manual"


# --------------------------------------------------------------------------- main entry


def parse_job(content: str, *, url: str | None = None, source_hint: str | None = None) -> CapturedJob:
    """Parse pasted text OR saved HTML into a CapturedJob. Never raises; falls back to heuristics."""
    content = content or ""
    is_html = (
        "<" in content and re.search(r"<\s*(html|div|script|meta|body|head)\b", content, re.I) is not None
    )

    fields: dict = {}
    parser = "heuristic"
    plain = content

    if is_html:
        ex = _Extractor()
        try:
            ex.feed(content)
        except Exception:
            pass
        job_node = next((n for n in _iter_jsonld_objects(ex.ld) if _is_jobposting(n)), None)
        if job_node:
            fields = {k: v for k, v in _from_jsonld(job_node).items() if v}
            parser = "jsonld"
            url = url or fields.get("url")
        if not fields.get("title") and (ex.meta.get("og:title") or ex.title):
            parser = parser if parser == "jsonld" else "meta"
            fields.setdefault("title", (ex.meta.get("og:title") or ex.title).strip())
        if not fields.get("description") and ex.meta.get("og:description"):
            fields.setdefault("description", ex.meta["og:description"].strip())
        url = url or ex.meta.get("og:url")
        # text used for clearance/salary/skill regexes = structured description + stripped page
        plain = (fields.get("description") or "") + "\n" + strip_html(content)

    # Heuristic title/company only when structured extraction didn't supply them.
    heur = _heuristic_fields(plain if not is_html else (fields.get("title", "") + "\n" + plain))
    title = fields.get("title") or heur["title"]
    company = fields.get("company") or heur["company"]
    description = fields.get("description") or (plain.strip() if not is_html else strip_html(content))

    scan = f"{title}\n{description}"
    salary, smin, smax = (fields.get("salary"), fields.get("salary_min"), fields.get("salary_max"))
    if not salary:
        salary, smin, smax = detect_salary(scan)

    from . import product

    skills = product.extract_keywords(scan, limit=18)
    required, preferred = split_skills(description, skills)
    work_mode = detect_work_mode(scan)

    # Confidence: structured (jsonld) is trusted; thin/heuristic captures flag for review.
    confidence = 0.95 if parser == "jsonld" else 0.75 if parser == "meta" else 0.55
    if len(description) < 200:
        confidence -= 0.2
    if not company or not title:
        confidence -= 0.2
    confidence = round(max(0.1, min(1.0, confidence)), 2)

    return CapturedJob(
        title=title,
        company=company,
        location=fields.get("location") or _heuristic_location(scan),
        work_mode=work_mode,
        clearance=detect_clearance(scan),
        polygraph=detect_polygraph(scan),
        salary=salary,
        salary_min=smin,
        salary_max=smax,
        job_id=fields.get("job_id") or detect_job_id(scan, url),
        url=url,
        required_skills=required,
        preferred_skills=preferred,
        years_experience=detect_years_experience(scan),
        degree=detect_degree(scan),
        certifications=detect_certifications(scan),
        posted_date=fields.get("posted_date") or detect_posted_date(plain),
        description=description.strip(),
        source=detect_source(content, url, source_hint),
        parser=parser,
        confidence=confidence,
        needs_review=confidence < 0.6,
        raw=content[:200_000],
    )


_LOC = re.compile(
    r"\b([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+)?),\s*([A-Z]{2})\b"  # City, ST
)


def _heuristic_location(text: str) -> str | None:
    if re.search(r"\bremote\b", text, re.I):
        return "Remote"
    m = _LOC.search(text)
    return f"{m.group(1)}, {m.group(2)}" if m else None
