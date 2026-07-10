"""Job ingestion: fetch remote postings and normalize each into a structured record.

Two stages:
  1. FETCH — a pluggable Source pulls raw postings. RemoteOK and Remotive are public
     JSON APIs (no auth); FixtureSource reads committed sample files so the whole
     pipeline runs offline (tests, demo, no API key).
  2. PARSE — each raw posting is normalized into a JobPosting. The quality path sends
     the messy posting (HTML description, loose tags) to the LLM and extracts a clean,
     typed record via a forced tool call (function calling). The offline `--no-parse`
     path does the same mapping heuristically with stdlib HTML stripping — no model,
     no key — so CI and the demo never need network.

A JobPosting's `.description` is exactly the `job_text` that tailor.tailor() consumes,
so ingestion plugs straight into the existing core.
"""

from __future__ import annotations

import html as _html
import json
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import anthropic

MODEL = "claude-opus-4-8"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
USER_AGENT = "jobproof/1.0 (+https://github.com/bitcoinbox/jobproof)"
REQUEST_DELAY_S = 1.0  # be polite to public APIs


class JobPosting(BaseModel):
    """One normalized posting. `description` is the job_text for tailor.tailor()."""

    source: str
    source_id: str
    url: str | None = None
    title: str
    company: str
    location: str | None = None
    remote: bool = True
    salary: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    description: str
    posted_at: str | None = None
    fetched_at: str
    # Diligence fields — usually defaulted; imported targets (fixtures) can set them.
    legitimacy_status: str = "needs_diligence"
    diligence_notes: str | None = None
    review_signal: str = "unknown"
    source_confidence: str = "unverified"
    priority: int = 3
    apply_recommendation: str = "consider"


# --------------------------------------------------------------------------- HTML


class _Stripper(HTMLParser):
    """Minimal HTML → text: block tags become newlines, the rest is collapsed."""

    _BLOCK = {"p", "br", "li", "div", "h1", "h2", "h3", "h4", "ul", "ol", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln).strip()


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    if "<" not in html:
        return html.strip()
    p = _Stripper()
    p.feed(html)
    return p.text()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------------------ sources


class Source(Protocol):
    name: str

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        """Return raw postings in the common shape (see RemoteOKSource.fetch)."""
        ...


def _matches(query: str, *fields) -> bool:
    if not query:
        return True
    hay = " ".join(str(f or "") for f in fields).lower()
    return all(term in hay for term in query.lower().split())


def _http_get_json(url: str, params: dict | None = None) -> object:
    import httpx  # lazy: the offline path never imports httpx

    with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as c:
        for attempt in range(3):
            resp = c.get(url, params=params)
            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 2**attempt))
                time.sleep(min(wait, 10))
                continue
            resp.raise_for_status()
            return resp.json()
    resp.raise_for_status()
    return resp.json()


class RemoteOKSource:
    """https://remoteok.com/api — JSON array; element [0] is legal metadata (skipped)."""

    name = "remoteok"

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        data = _http_get_json("https://remoteok.com/api")
        items = [d for d in data if isinstance(d, dict) and d.get("id")]
        out = []
        for d in items:
            if not _matches(
                query,
                d.get("position"),
                d.get("company"),
                " ".join(d.get("tags") or []),
                d.get("description"),
            ):
                continue
            sal = None
            if d.get("salary_min") or d.get("salary_max"):
                sal = f"${d.get('salary_min', '?')}–${d.get('salary_max', '?')}"
            out.append(
                {
                    "source": self.name,
                    "source_id": str(d["id"]),
                    "url": d.get("url"),
                    "title": d.get("position") or "",
                    "company": d.get("company") or "",
                    "location": d.get("location") or "Remote",
                    "remote": True,
                    "salary": sal,
                    "tags": d.get("tags") or [],
                    "description_html": d.get("description") or "",
                    "posted_at": d.get("date") or d.get("epoch"),
                }
            )
            if len(out) >= limit:
                break
        time.sleep(REQUEST_DELAY_S)
        return out


class RemotiveSource:
    """https://remotive.com/api/remote-jobs?search=<query> — {"jobs": [...]}."""

    name = "remotive"

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        data = _http_get_json(
            "https://remotive.com/api/remote-jobs", params={"search": query} if query else None
        )
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        out = []
        for d in jobs[:limit]:
            out.append(
                {
                    "source": self.name,
                    "source_id": str(d.get("id")),
                    "url": d.get("url"),
                    "title": d.get("title") or "",
                    "company": d.get("company_name") or "",
                    "location": d.get("candidate_required_location") or "Remote",
                    "remote": True,
                    "salary": d.get("salary") or None,
                    "tags": d.get("tags") or [],
                    "description_html": d.get("description") or "",
                    "posted_at": d.get("publication_date"),
                }
            )
        time.sleep(REQUEST_DELAY_S)
        return out


def _looks_remote(*fields) -> bool:
    hay = " ".join(str(f or "") for f in fields).lower()
    return "remote" in hay or "anywhere" in hay or "distributed" in hay


class GreenhouseSource:
    """A company's Greenhouse board: boards-api.greenhouse.io/v1/boards/<board>/jobs?content=true.

    `board` is the company's Greenhouse token (e.g. 'stripe'); `query` keyword-filters the board.
    """

    name = "greenhouse"

    def __init__(self, board: str) -> None:
        if not board:
            raise SystemExit("greenhouse source needs --board <company-token> (e.g. --board stripe).")
        self.board = board

    @staticmethod
    def normalize(data: dict, *, board: str, query: str, limit: int) -> list[dict]:
        company = board.replace("-", " ").title()
        out = []
        for j in (data or {}).get("jobs", []):
            loc = ((j.get("location") or {}).get("name")) or "—"
            content = _html.unescape(j.get("content") or "")
            if not _matches(query, j.get("title"), company, loc, content):
                continue
            out.append(
                {
                    "source": "greenhouse",
                    "source_id": str(j.get("id")),
                    "url": j.get("absolute_url"),
                    "title": j.get("title") or "",
                    "company": company,
                    "location": loc,
                    "remote": _looks_remote(loc, j.get("title")),
                    "salary": None,
                    "tags": [d.get("name") for d in (j.get("departments") or []) if d.get("name")],
                    "description_html": content,
                    "posted_at": j.get("updated_at"),
                    "source_confidence": "direct_ats",
                }
            )
            if len(out) >= limit:
                break
        return out

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        data = _http_get_json(f"https://boards-api.greenhouse.io/v1/boards/{self.board}/jobs?content=true")
        out = self.normalize(data, board=self.board, query=query, limit=limit)
        time.sleep(REQUEST_DELAY_S)
        return out


class LeverSource:
    """A company's Lever board: api.lever.co/v0/postings/<board>?mode=json."""

    name = "lever"

    def __init__(self, board: str) -> None:
        if not board:
            raise SystemExit("lever source needs --board <company> (e.g. --board netflix).")
        self.board = board

    @staticmethod
    def normalize(data: list, *, board: str, query: str, limit: int) -> list[dict]:
        company = board.replace("-", " ").title()
        out = []
        for j in data or []:
            cats = j.get("categories") or {}
            loc = cats.get("location") or "—"
            desc = j.get("descriptionPlain") or j.get("description") or ""
            if not _matches(query, j.get("text"), company, loc, cats.get("team"), desc):
                continue
            tags = [v for v in (cats.get("team"), cats.get("commitment"), cats.get("department")) if v]
            out.append(
                {
                    "source": "lever",
                    "source_id": str(j.get("id")),
                    "url": j.get("hostedUrl") or j.get("applyUrl"),
                    "title": j.get("text") or "",
                    "company": company,
                    "location": loc,
                    "remote": _looks_remote(loc, cats.get("commitment"), j.get("workplaceType")),
                    "salary": None,
                    "tags": tags,
                    "description_html": j.get("description") or desc,
                    "posted_at": j.get("createdAt"),
                    "source_confidence": "direct_ats",
                }
            )
            if len(out) >= limit:
                break
        return out

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        data = _http_get_json(f"https://api.lever.co/v0/postings/{self.board}?mode=json")
        out = self.normalize(data, board=self.board, query=query, limit=limit)
        time.sleep(REQUEST_DELAY_S)
        return out


class AshbySource:
    """A company's Ashby board: api.ashbyhq.com/posting-api/job-board/<board>."""

    name = "ashby"

    def __init__(self, board: str) -> None:
        if not board:
            raise SystemExit("ashby source needs --board <company> (e.g. --board ramp).")
        self.board = board

    @staticmethod
    def normalize(data: dict, *, board: str, query: str, limit: int) -> list[dict]:
        company = board.replace("-", " ").title()
        out = []
        for j in (data or {}).get("jobs", []):
            loc = j.get("location") or "—"
            desc_html = j.get("descriptionHtml") or ""
            desc_plain = j.get("descriptionPlain") or ""
            if not _matches(query, j.get("title"), company, loc, j.get("department"), desc_plain):
                continue
            out.append(
                {
                    "source": "ashby",
                    "source_id": str(j.get("id")),
                    "url": j.get("jobUrl") or j.get("applyUrl"),
                    "title": j.get("title") or "",
                    "company": company,
                    "location": loc,
                    "remote": bool(j.get("isRemote")) or _looks_remote(loc),
                    "salary": None,
                    "tags": [t for t in (j.get("department"), j.get("team"), j.get("employmentType")) if t],
                    "description_html": desc_html or desc_plain,
                    "posted_at": j.get("publishedAt") or j.get("updatedAt"),
                    "source_confidence": "direct_ats",
                }
            )
            if len(out) >= limit:
                break
        return out

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        data = _http_get_json(
            f"https://api.ashbyhq.com/posting-api/job-board/{self.board}?includeCompensation=true"
        )
        out = self.normalize(data, board=self.board, query=query, limit=limit)
        time.sleep(REQUEST_DELAY_S)
        return out


class HackerNewsSource:
    """HN 'Who is hiring?' comments via the Algolia API.

    `board` is the monthly thread's story id (the big 'Ask HN: Who is hiring?' post);
    `query` keyword-filters the top-level comments, each of which is one job ad.
    """

    name = "hn"

    def __init__(self, board: str) -> None:
        if not board:
            raise SystemExit("hn source needs --board <story-id> (the monthly 'Who is hiring?' thread id).")
        self.board = board

    @staticmethod
    def _company(text: str) -> str:
        # HN ads conventionally start "Company | Role | Location | ...".
        first = (text or "").split("\n", 1)[0]
        head = first.split("|", 1)[0].strip()
        return (head[:60] or "HN post") if head else "HN post"

    @staticmethod
    def normalize(data: dict, *, board: str, query: str, limit: int) -> list[dict]:
        out = []
        for c in (data or {}).get("hits", []):
            text = _html.unescape(c.get("comment_text") or "")
            if not text or not _matches(query, text):
                continue
            out.append(
                {
                    "source": "hn",
                    "source_id": str(c.get("objectID")),
                    "url": f"https://news.ycombinator.com/item?id={c.get('objectID')}",
                    "title": "Who is hiring — see post",
                    "company": HackerNewsSource._company(text),
                    "location": "—",
                    "remote": _looks_remote(text),
                    "salary": None,
                    "tags": [],
                    "description_html": text,
                    "posted_at": c.get("created_at"),
                    "source_confidence": "job_board",
                }
            )
            if len(out) >= limit:
                break
        return out

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        data = _http_get_json(
            "https://hn.algolia.com/api/v1/search",
            params={"tags": f"comment,story_{self.board}", "hitsPerPage": 200},
        )
        out = self.normalize(data, board=self.board, query=query, limit=limit)
        time.sleep(REQUEST_DELAY_S)
        return out


class FixtureSource:
    """Offline source: reads fixtures/<name>.sample.json (committed, fictional)."""

    name = "fixture"

    def __init__(self, fixtures_dir: Path | None = None, provider: str = "remoteok") -> None:
        self._dir = fixtures_dir or FIXTURES_DIR
        self._provider = provider

    def fetch(self, query: str, *, limit: int) -> list[dict]:
        path = self._dir / f"{self._provider}.sample.json"
        if not path.exists():
            raise SystemExit(f"Fixture not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        out = [
            d
            for d in raw
            if _matches(
                query,
                d.get("title"),
                d.get("company"),
                " ".join(d.get("tags") or []),
                d.get("description_html"),
            )
        ]
        return out[:limit]


SOURCES = {
    "remoteok": RemoteOKSource,
    "remotive": RemotiveSource,
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
    "ashby": AshbySource,
    "hn": HackerNewsSource,
    "fixture": FixtureSource,
}
# Sources that target one company's board / thread instead of keyword search.
BOARD_SOURCES = {"greenhouse", "lever", "ashby", "hn"}


def get_source(
    name: str,
    *,
    fixtures_dir: Path | None = None,
    provider: str = "remoteok",
    board: str | None = None,
) -> Source:
    if name not in SOURCES:
        raise SystemExit(f"Unknown source {name!r}. Choose from: {', '.join(SOURCES)}")
    if name == "fixture":
        return FixtureSource(fixtures_dir=fixtures_dir, provider=provider)
    if name in BOARD_SOURCES:
        return SOURCES[name](board)
    return SOURCES[name]()


# ------------------------------------------------------------------------- parse

PARSE_TOOL = {
    "name": "record_job_posting",
    "description": "Record one job posting as a normalized, structured record.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The role title."},
            "company": {"type": "string", "description": "Hiring company."},
            "location": {"type": "string", "description": "Location, or 'Remote'."},
            "remote": {"type": "boolean", "description": "True if the role is remote-eligible."},
            "salary": {"type": "string", "description": "Verbatim/normalized comp, or empty if none stated."},
            "required_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete skills, tools, or clearances the posting names as required.",
            },
            "description": {
                "type": "string",
                "description": "The posting as clean plain text — HTML stripped, boilerplate trimmed.",
            },
        },
        "required": ["title", "company", "remote", "required_skills", "description"],
    },
}

_PARSE_PROMPT = (
    "Normalize this job posting into a structured record by calling record_job_posting. "
    "Clean the description to plain text (strip HTML and boilerplate). List only skills/tools/"
    "clearances the posting actually requires. Do not invent details.\n\nRAW POSTING:\n"
)


def parse_posting(
    raw: dict, *, client: anthropic.Anthropic | None = None, use_llm: bool = True
) -> JobPosting:
    """Normalize one raw posting into a JobPosting (LLM tool-call, or heuristic if use_llm=False)."""
    base = {
        "source": raw["source"],
        "source_id": str(raw["source_id"]),
        "url": raw.get("url"),
        "posted_at": (str(raw["posted_at"]) if raw.get("posted_at") is not None else None),
        "fetched_at": _now_iso(),
    }
    # Carry diligence fields through when a fixture/import provides them (defaults otherwise).
    for f in (
        "legitimacy_status",
        "diligence_notes",
        "review_signal",
        "source_confidence",
        "priority",
        "apply_recommendation",
    ):
        if raw.get(f) is not None:
            base[f] = raw[f]

    if not use_llm:
        return JobPosting(
            **base,
            title=raw.get("title", ""),
            company=raw.get("company", ""),
            location=raw.get("location"),
            remote=bool(raw.get("remote", True)),
            salary=raw.get("salary"),
            required_skills=list(raw.get("tags") or []),
            description=strip_html(raw.get("description_html")),
        )

    import anthropic

    client = client or anthropic.Anthropic()
    payload = json.dumps(
        {k: raw.get(k) for k in ("title", "company", "location", "salary", "tags", "description_html")},
        indent=2,
    )
    # Forced tool use is incompatible with thinking, so disable thinking for this
    # single-shot extraction (it needs none) and force exactly one tool call.
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "disabled"},
        tools=[PARSE_TOOL],
        tool_choice={"type": "tool", "name": "record_job_posting"},
        messages=[{"role": "user", "content": _PARSE_PROMPT + payload}],
    )
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if block is None:
        raise RuntimeError(f"Parse did not return a tool call (stop_reason={resp.stop_reason}).")
    d = block.input
    return JobPosting(
        **base,
        title=d.get("title") or raw.get("title", ""),
        company=d.get("company") or raw.get("company", ""),
        location=d.get("location") or raw.get("location"),
        remote=bool(d.get("remote", raw.get("remote", True))),
        salary=(d.get("salary") or None) or raw.get("salary"),
        required_skills=list(d.get("required_skills") or raw.get("tags") or []),
        description=d.get("description") or strip_html(raw.get("description_html")),
    )


def ingest(
    source: str,
    query: str,
    *,
    limit: int = 10,
    client: anthropic.Anthropic | None = None,
    conn=None,
    use_llm: bool = True,
    fixtures_dir: Path | None = None,
    provider: str = "remoteok",
    board: str | None = None,
) -> list[JobPosting]:
    """Fetch + parse postings, skipping ones already in the store (dedup before parse)."""
    from . import store as _store

    src = get_source(source, fixtures_dir=fixtures_dir, provider=provider, board=board)
    raw_items = src.fetch(query, limit=limit)
    postings: list[JobPosting] = []
    for raw in raw_items:
        sid = str(raw["source_id"])
        if conn is not None and _store.seen(conn, raw["source"], sid):
            continue
        posting = parse_posting(raw, client=client, use_llm=use_llm)
        if conn is not None:
            _store.upsert_job(conn, posting)
        postings.append(posting)
    return postings
