"""SQLite persistence for the job-search pipeline.

Turns one-off tailoring runs into a tracked pipeline: every fetched posting, every
tailored application (with its fit score and the RAG sources that produced it), and
the application's status as it moves through the funnel. Plain stdlib ``sqlite3`` —
no ORM, no extra dependency.

Three tables:
  jobs               one row per posting (deduped on source + source_id)
  applications       one row per tailoring of a job (fit, keywords, mode, status)
  retrieved_sources  the RAG chunks that fed a given application (provenance)

All timestamps are stored as absolute ISO-8601 UTC. Providers report dates every
which way (epoch ints, naive local strings, "3 days ago"); ``absolutize`` normalizes
them at write time so the tracker never holds a relative date.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid runtime import cycles / heavy imports
    from .ingest import JobPosting
    from .rag import Retrieved
    from .tailor import TailoredApplication

DEFAULT_DB = "jobsearch.db"

# Application-level statuses (a tailoring run).
STATUSES = ("tailored", "to_apply", "applied", "interview", "rejected", "offer", "skipped")
# Job-level pipeline statuses (the unit the dashboard tracks).
JOB_STATUSES = ("new", "interested", "to_apply", "applied", "interview", "offer", "rejected", "skipped")
NEXT_ACTIONS = (
    "review",
    "verify_posting",
    "tailor_resume",
    "generate_kit",
    "apply",
    "message_recruiter",
    "follow_up",
    "prep_interview",
    "skip",
)
RUN_STATUSES = ("queued", "running", "done", "failed")
LINK_STATUSES = ("alive", "dead", "unknown", "unchecked")

# Diligence vocabularies (validated by update_diligence).
LEGITIMACY = ("verified", "likely_legit", "needs_diligence", "skip")
REVIEW_SIGNAL = ("positive", "mixed", "sparse", "negative", "unknown")
SOURCE_CONFIDENCE = ("direct_ats", "company_site", "wellfound", "job_board", "unverified")
APPLY_RECOMMENDATION = ("apply_today", "consider", "wait", "skip")
# Maps a diligence field name -> its allowed values (priority is the int 1-5 exception).
DILIGENCE_ENUMS = {
    "legitimacy_status": LEGITIMACY,
    "review_signal": REVIEW_SIGNAL,
    "source_confidence": SOURCE_CONFIDENCE,
    "apply_recommendation": APPLY_RECOMMENDATION,
}
DILIGENCE_FIELDS = (*DILIGENCE_ENUMS, "diligence_notes", "priority")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id             INTEGER PRIMARY KEY,
    source         TEXT NOT NULL,
    source_id      TEXT NOT NULL,
    url            TEXT,
    title          TEXT NOT NULL,
    company        TEXT NOT NULL,
    location       TEXT,
    remote         INTEGER NOT NULL DEFAULT 1,
    salary         TEXT,
    required_skills TEXT,                 -- JSON array
    description    TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'new',
    notes          TEXT,
    next_action    TEXT NOT NULL DEFAULT 'review',
    applied_at     TEXT,
    follow_up_at   TEXT,
    legitimacy_status   TEXT NOT NULL DEFAULT 'needs_diligence',
    diligence_notes     TEXT,
    review_signal       TEXT NOT NULL DEFAULT 'unknown',
    source_confidence   TEXT NOT NULL DEFAULT 'unverified',
    priority            INTEGER NOT NULL DEFAULT 3,
    apply_recommendation TEXT NOT NULL DEFAULT 'consider',
    contact_name   TEXT,
    contact_email  TEXT,
    contact_url    TEXT,
    link_status    TEXT NOT NULL DEFAULT 'unchecked',  -- alive | dead | unknown | unchecked
    link_checked_at TEXT,
    clearance      TEXT,                  -- captured clearance requirement
    work_mode      TEXT,                  -- remote | hybrid | onsite | unknown
    raw_capture    TEXT,                  -- preserved raw text/HTML for audit
    fit_breakdown  TEXT,                  -- JSON: deterministic 7-dimension breakdown
    recommended_variant TEXT,            -- which resume to upload
    capture_confidence REAL,             -- 0-1 extraction confidence
    needs_review   INTEGER NOT NULL DEFAULT 0,
    posted_at      TEXT,                  -- absolute ISO-8601
    fetched_at     TEXT NOT NULL,         -- absolute ISO-8601
    UNIQUE(source, source_id)
);
CREATE TABLE IF NOT EXISTS applications (
    id               INTEGER PRIMARY KEY,
    job_id           INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    status           TEXT NOT NULL DEFAULT 'tailored',
    fit_score        INTEGER,
    matched_keywords TEXT,                -- JSON array
    missing_keywords TEXT,                -- JSON array
    application_note TEXT,
    why_match        TEXT,
    why_not          TEXT,
    missing_proof    TEXT,
    keywords_to_mirror TEXT,              -- JSON array
    recruiter_angle  TEXT,
    evidence_map     TEXT,                -- JSON array of {claim, snippet, source}
    mode             TEXT,                -- 'rag' | 'full-profile'
    out_dir          TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrieved_sources (
    id             INTEGER PRIMARY KEY,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    source         TEXT NOT NULL,
    heading        TEXT,
    distance       REAL,
    rank           INTEGER
);
CREATE TABLE IF NOT EXISTS kits (
    id             INTEGER PRIMARY KEY,
    job_id         INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    application_id INTEGER REFERENCES applications(id) ON DELETE SET NULL,
    out_dir        TEXT NOT NULL,
    resume_markdown TEXT,
    cover_letter   TEXT,
    why_me         TEXT,
    recruiter_dm   TEXT,
    interview_stories TEXT,
    evidence_map   TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generation_runs (
    id             INTEGER PRIMARY KEY,
    job_id         INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    kind           TEXT NOT NULL,          -- tailor | kit | manual
    status         TEXT NOT NULL DEFAULT 'queued',
    model          TEXT,
    mode           TEXT,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms     INTEGER,
    error          TEXT,
    started_at     TEXT NOT NULL,
    finished_at    TEXT
);
"""

_REL_RE = re.compile(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago", re.I)
_REL_UNIT = {
    "second": "seconds",
    "minute": "minutes",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
    "month": "days",
    "year": "days",
}
_REL_SCALE = {"month": 30, "year": 365}


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


def absolutize(value, *, now: datetime | None = None) -> str | None:
    """Normalize a provider date (epoch / ISO / naive / 'N days ago') to absolute ISO-8601 UTC.

    ``now`` is injectable so tests are deterministic. Returns None if the value is
    empty or can't be parsed.
    """
    if value is None or value == "":
        return None
    now = _now(now)

    # Epoch seconds (int, float, or numeric string).
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    text = str(value).strip()
    if re.fullmatch(r"\d{9,11}", text):  # plausible epoch seconds
        return datetime.fromtimestamp(int(text), tz=UTC).isoformat()

    # "N <unit> ago"
    m = _REL_RE.search(text)
    if m:
        n = int(m.group(1)) * _REL_SCALE.get(m.group(2).lower(), 1)
        delta = timedelta(**{_REL_UNIT[m.group(2).lower()]: n})
        return (now - delta).isoformat()

    # ISO-8601 (tolerate trailing Z) or "YYYY-MM-DD HH:MM:SS".
    iso = text.replace("Z", "+00:00")
    for candidate in (iso, iso.replace(" ", "T", 1)):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC).isoformat()
        except ValueError:
            continue
    return None


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Open the DB (creating the schema if absent) with FK enforcement and Row access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _add_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Idempotently ADD COLUMN for any of `columns` not already on `table`."""
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a DB was first created (idempotent)."""
    _add_columns(
        conn,
        "jobs",
        {
            "status": "TEXT NOT NULL DEFAULT 'new'",
            "notes": "TEXT",
            "next_action": "TEXT NOT NULL DEFAULT 'review'",
            "applied_at": "TEXT",
            "follow_up_at": "TEXT",
            "legitimacy_status": "TEXT NOT NULL DEFAULT 'needs_diligence'",
            "diligence_notes": "TEXT",
            "review_signal": "TEXT NOT NULL DEFAULT 'unknown'",
            "source_confidence": "TEXT NOT NULL DEFAULT 'unverified'",
            "priority": "INTEGER NOT NULL DEFAULT 3",
            "apply_recommendation": "TEXT NOT NULL DEFAULT 'consider'",
            "contact_name": "TEXT",
            "contact_email": "TEXT",
            "contact_url": "TEXT",
            "link_status": "TEXT NOT NULL DEFAULT 'unchecked'",
            "link_checked_at": "TEXT",
            "clearance": "TEXT",
            "work_mode": "TEXT",
            "raw_capture": "TEXT",
            "fit_breakdown": "TEXT",
            "recommended_variant": "TEXT",
            "capture_confidence": "REAL",
            "needs_review": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    _add_columns(
        conn,
        "applications",
        {
            "why_match": "TEXT",
            "why_not": "TEXT",
            "missing_proof": "TEXT",
            "keywords_to_mirror": "TEXT",
            "recruiter_angle": "TEXT",
            "evidence_map": "TEXT",
        },
    )
    _ensure_support_tables(conn)


def _ensure_support_tables(conn: sqlite3.Connection) -> None:
    """Create post-MVP support tables/indexes when migrating old DBs."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kits (
            id             INTEGER PRIMARY KEY,
            job_id         INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            application_id INTEGER REFERENCES applications(id) ON DELETE SET NULL,
            out_dir        TEXT NOT NULL,
            resume_markdown TEXT,
            cover_letter   TEXT,
            why_me         TEXT,
            recruiter_dm   TEXT,
            interview_stories TEXT,
            evidence_map   TEXT,
            created_at     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS generation_runs (
            id             INTEGER PRIMARY KEY,
            job_id         INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
            kind           TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'queued',
            model          TEXT,
            mode           TEXT,
            input_tokens   INTEGER NOT NULL DEFAULT 0,
            cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens  INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            latency_ms     INTEGER,
            error          TEXT,
            started_at     TEXT NOT NULL,
            finished_at    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_apply_priority ON jobs(apply_recommendation, priority);
        CREATE INDEX IF NOT EXISTS idx_jobs_follow_up ON jobs(follow_up_at);
        CREATE INDEX IF NOT EXISTS idx_applications_job_fit ON applications(job_id, fit_score DESC);
        CREATE INDEX IF NOT EXISTS idx_kits_job_created ON kits(job_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_generation_runs_job_started ON generation_runs(job_id, started_at DESC);
        CREATE TABLE IF NOT EXISTS tasks (
            id             INTEGER PRIMARY KEY,
            job_id         INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
            title          TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'open',
            priority       INTEGER NOT NULL DEFAULT 3,
            due_at         TEXT,
            notes          TEXT,
            created_at     TEXT NOT NULL,
            completed_at   TEXT
        );
        CREATE TABLE IF NOT EXISTS resume_versions (
            id             INTEGER PRIMARY KEY,
            job_id         INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
            application_id INTEGER REFERENCES applications(id) ON DELETE SET NULL,
            kit_id         INTEGER REFERENCES kits(id) ON DELETE SET NULL,
            label          TEXT NOT NULL,
            content        TEXT NOT NULL,
            diff_summary   TEXT,
            created_at     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS search_runs (
            id             INTEGER PRIMARY KEY,
            query          TEXT NOT NULL,
            sources        TEXT,
            location       TEXT,
            status         TEXT NOT NULL DEFAULT 'planned',
            found_count    INTEGER NOT NULL DEFAULT 0,
            deduped_count  INTEGER NOT NULL DEFAULT 0,
            notes          TEXT,
            created_at     TEXT NOT NULL,
            finished_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS job_events (
            id             INTEGER PRIMARY KEY,
            job_id         INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
            kind           TEXT NOT NULL,
            source         TEXT,
            subject        TEXT,
            body_excerpt   TEXT,
            inferred_status TEXT,
            created_at     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analysis_reports (
            id             INTEGER PRIMARY KEY,
            job_id         INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            kind           TEXT NOT NULL,
            payload        TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recruiter_messages (
            id             INTEGER PRIMARY KEY,
            job_id         INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
            raw            TEXT NOT NULL,
            recruiter_name TEXT,
            company        TEXT,
            email          TEXT,
            phone          TEXT,
            inferred_role  TEXT,
            intent         TEXT,
            tone           TEXT,
            reply          TEXT,
            created_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_recruiter_msgs_job ON recruiter_messages(job_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(status, due_at);
        CREATE INDEX IF NOT EXISTS idx_tasks_job ON tasks(job_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_search_runs_created ON search_runs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analysis_reports_job ON analysis_reports(job_id, kind, created_at DESC);
        """
    )


def seen(conn: sqlite3.Connection, source: str, source_id: str) -> bool:
    """True if this posting is already stored (dedup gate — check before spending parse tokens)."""
    row = conn.execute(
        "SELECT 1 FROM jobs WHERE source = ? AND source_id = ?", (source, source_id)
    ).fetchone()
    return row is not None


def upsert_job(conn: sqlite3.Connection, posting: JobPosting, *, now: datetime | None = None) -> int:
    """Insert a posting (or return the existing row's id on the source/source_id dedup key)."""
    row = conn.execute(
        "SELECT id FROM jobs WHERE source = ? AND source_id = ?",
        (posting.source, posting.source_id),
    ).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        """INSERT INTO jobs (source, source_id, url, title, company, location, remote,
                             salary, required_skills, description, posted_at, fetched_at,
                             legitimacy_status, diligence_notes, review_signal,
                             source_confidence, priority, apply_recommendation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            posting.source,
            posting.source_id,
            getattr(posting, "url", None),
            posting.title,
            posting.company,
            getattr(posting, "location", None),
            1 if getattr(posting, "remote", True) else 0,
            getattr(posting, "salary", None),
            json.dumps(list(getattr(posting, "required_skills", []) or [])),
            posting.description,
            absolutize(getattr(posting, "posted_at", None), now=now),
            absolutize(getattr(posting, "fetched_at", None), now=now) or _now(now).isoformat(),
            getattr(posting, "legitimacy_status", None) or "needs_diligence",
            getattr(posting, "diligence_notes", None),
            getattr(posting, "review_signal", None) or "unknown",
            getattr(posting, "source_confidence", None) or "unverified",
            int(getattr(posting, "priority", None) or 3),
            getattr(posting, "apply_recommendation", None) or "consider",
        ),
    )
    conn.commit()
    return cur.lastrowid


_REC_TO_APPLY = {"apply": "apply_today", "maybe": "consider", "skip": "skip"}


def _priority_from(overall: int) -> int:
    return 1 if overall >= 80 else 2 if overall >= 70 else 3 if overall >= 55 else 4 if overall >= 45 else 5


def record_capture(conn: sqlite3.Connection, captured, breakdown=None, *, now: datetime | None = None) -> int:
    """Persist a captured posting (+ optional deterministic breakdown) and return its job id.

    Upserts the job on (source, source_id), preserves the raw capture for audit, and — when a
    breakdown is given — seeds apply_recommendation / priority / recommended_variant from it.
    """
    import hashlib

    from .ingest import JobPosting

    sid = (
        captured.job_id
        or hashlib.sha1(
            (captured.url or f"{captured.company}:{captured.title}:{captured.description[:120]}").encode(
                "utf-8"
            )
        ).hexdigest()[:16]
    )
    direct = captured.source in ("clearancejobs", "dice", "greenhouse", "lever", "ashby")
    posting = JobPosting(
        source=captured.source or "manual",
        source_id=str(sid),
        url=captured.url,
        title=captured.title or "Untitled role",
        company=captured.company or "Unknown",
        location=captured.location,
        remote=(captured.work_mode == "remote"),
        salary=captured.salary,
        required_skills=captured.required_skills,
        description=captured.description or captured.raw[:4000] or "(no description captured)",
        fetched_at=_now(now).isoformat(),
        source_confidence="direct_ats" if direct else "unverified",
    )
    job_id = upsert_job(conn, posting, now=now)

    sets = {
        "clearance": captured.clearance,
        "work_mode": captured.work_mode,
        "raw_capture": captured.raw,
        "capture_confidence": getattr(captured, "confidence", None),
        "needs_review": 1 if getattr(captured, "needs_review", False) else 0,
    }
    if breakdown is not None:
        payload = breakdown.model_dump() if hasattr(breakdown, "model_dump") else dict(breakdown)
        sets["fit_breakdown"] = json.dumps(payload)
        sets["recommended_variant"] = payload.get("resume_variant")
        sets["apply_recommendation"] = _REC_TO_APPLY.get(payload.get("recommendation"), "consider")
        sets["priority"] = _priority_from(int(payload.get("overall", 50)))
    cols = ", ".join(f"{k} = ?" for k in sets)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", [*sets.values(), job_id])
    _log_event(conn, job_id, "captured", f"Captured from {captured.source} ({captured.parser})", now=now)
    conn.commit()
    return job_id


def record_recruiter_message(
    conn: sqlite3.Connection, reply, *, job_id: int | None = None, raw: str = "", now: datetime | None = None
) -> int:
    """Persist a recruiter message + the drafted reply. `reply` is a recruiter.RecruiterReply."""
    p = reply.parsed
    cur = conn.execute(
        """INSERT INTO recruiter_messages (job_id, raw, recruiter_name, company, email, phone,
                                           inferred_role, intent, tone, reply, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id,
            raw or "",
            p.recruiter_name,
            p.company,
            p.email,
            p.phone,
            p.inferred_role,
            reply.intent,
            reply.tone,
            reply.suggested_reply,
            _now(now).isoformat(),
        ),
    )
    if job_id:
        _log_event(conn, job_id, "recruiter_msg", f"Drafted {reply.intent} reply", now=now)
    conn.commit()
    return cur.lastrowid


def recruiter_messages(conn: sqlite3.Connection, job_id: int, *, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT id, job_id, recruiter_name, company, email, phone, inferred_role, intent, tone,
                  reply, created_at FROM recruiter_messages WHERE job_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (job_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def update_diligence(conn: sqlite3.Connection, job_id: int, fields: dict) -> None:
    """Update one or more diligence fields on a job (validated)."""
    sets, params = [], []
    for key, val in fields.items():
        if key not in DILIGENCE_FIELDS:
            raise ValueError(f"Unknown diligence field {key!r}. Allowed: {', '.join(DILIGENCE_FIELDS)}")
        if key in DILIGENCE_ENUMS and val not in DILIGENCE_ENUMS[key]:
            raise ValueError(f"Invalid {key}={val!r}. Allowed: {', '.join(DILIGENCE_ENUMS[key])}")
        if key == "priority":
            val = int(val)
            if not 1 <= val <= 5:
                raise ValueError("priority must be 1-5")
        sets.append(f"{key} = ?")
        params.append(val)
    if not sets:
        return
    params.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()


def _estimate_cost(usage: dict | None) -> float:
    """Rough Opus-class cost estimate; exact pricing can move, so keep this informational."""
    if not usage:
        return 0.0
    uncached_in = int(usage.get("input_tokens", 0) or 0)
    cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    # Conservative Opus-class placeholders: $15/M input, $18.75/M cache write,
    # $1.50/M cache read, $75/M output.
    return round(
        uncached_in * 0.000015 + cache_create * 0.00001875 + cache_read * 0.0000015 + out * 0.000075,
        6,
    )


def start_generation_run(
    conn: sqlite3.Connection,
    *,
    kind: str,
    job_id: int | None = None,
    model: str | None = None,
    mode: str | None = None,
    now: datetime | None = None,
) -> int:
    """Create a generation-run row. Synchronous endpoints still get an auditable run record."""
    ts = _now(now).isoformat()
    cur = conn.execute(
        """INSERT INTO generation_runs (job_id, kind, status, model, mode, started_at)
           VALUES (?, ?, 'running', ?, ?, ?)""",
        (job_id, kind, model, mode, ts),
    )
    conn.commit()
    return cur.lastrowid


def finish_generation_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str = "done",
    usage: dict | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
    now: datetime | None = None,
) -> None:
    """Mark a generation run complete/failed and persist token/latency metrics."""
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status {status!r}. Expected one of {', '.join(RUN_STATUSES)}.")
    usage = usage or {}
    conn.execute(
        """UPDATE generation_runs
           SET status = ?, input_tokens = ?, cache_creation_input_tokens = ?,
               cache_read_input_tokens = ?, output_tokens = ?, estimated_cost_usd = ?,
               latency_ms = ?, error = ?, finished_at = ?
           WHERE id = ?""",
        (
            status,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("cache_creation_input_tokens", 0) or 0),
            int(usage.get("cache_read_input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            _estimate_cost(usage),
            latency_ms,
            error,
            _now(now).isoformat(),
            run_id,
        ),
    )
    conn.commit()


def record_application(
    conn: sqlite3.Connection,
    job_id: int,
    result: TailoredApplication,
    *,
    mode: str,
    out_dir: str | None = None,
    retrieved: list[Retrieved] | None = None,
    evidence_map: list[dict] | None = None,
    status: str = "tailored",
    now: datetime | None = None,
) -> int:
    """Persist one tailoring (application row + its RAG provenance) in a single transaction."""
    ts = _now(now).isoformat()
    try:
        cur = conn.execute(
            """INSERT INTO applications (job_id, status, fit_score, matched_keywords,
                                        missing_keywords, application_note, why_match, why_not,
                                        missing_proof, keywords_to_mirror, recruiter_angle,
                                        evidence_map, mode, out_dir, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                status,
                result.fit_score,
                json.dumps(list(result.matched_keywords)),
                json.dumps(list(result.missing_keywords)),
                result.application_note,
                getattr(result, "why_match", None) or None,
                getattr(result, "why_not", None) or None,
                getattr(result, "missing_proof", None) or None,
                json.dumps(list(getattr(result, "keywords_to_mirror", []) or [])),
                getattr(result, "recruiter_angle", None) or None,
                json.dumps(evidence_map) if evidence_map else None,
                mode,
                out_dir,
                ts,
                ts,
            ),
        )
        app_id = cur.lastrowid
        for rank, r in enumerate(retrieved or []):
            conn.execute(
                """INSERT INTO retrieved_sources (application_id, source, heading, distance, rank)
                   VALUES (?, ?, ?, ?, ?)""",
                (app_id, r.source, r.heading, r.distance, rank),
            )
        _log_event(conn, job_id, "tailored", f"Tailored ({mode}) · fit {result.fit_score}/100", now=now)
        conn.commit()
        return app_id
    except Exception:
        conn.rollback()
        raise


def record_kit(
    conn: sqlite3.Connection,
    job_id: int,
    kit,
    *,
    out_dir: str,
    application_id: int | None = None,
    now: datetime | None = None,
) -> int:
    """Persist a generated application kit and its evidence map for later review."""
    ts = _now(now).isoformat()
    cur = conn.execute(
        """INSERT INTO kits (job_id, application_id, out_dir, resume_markdown, cover_letter,
                             why_me, recruiter_dm, interview_stories, evidence_map, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id,
            application_id,
            out_dir,
            kit.resume_markdown,
            kit.cover_letter,
            json.dumps(list(kit.why_me)),
            kit.recruiter_dm,
            json.dumps(list(kit.interview_stories)),
            json.dumps([e.model_dump() if hasattr(e, "model_dump") else e for e in kit.evidence_map]),
            ts,
        ),
    )
    # If the current best application has no evidence/out_dir, backfill it so the
    # drawer's evidence inspector survives refresh.
    app = conn.execute(
        """SELECT id FROM applications WHERE job_id = ?
           ORDER BY fit_score DESC, created_at DESC LIMIT 1""",
        (job_id,),
    ).fetchone()
    if app:
        conn.execute(
            "UPDATE applications SET evidence_map = ?, out_dir = ?, updated_at = ? WHERE id = ?",
            (
                json.dumps([e.model_dump() if hasattr(e, "model_dump") else e for e in kit.evidence_map]),
                out_dir,
                ts,
                app["id"],
            ),
        )
    _log_event(conn, job_id, "kit_generated", f"Application kit saved to {out_dir}", now=now)
    conn.commit()
    return cur.lastrowid


def set_status(
    conn: sqlite3.Connection, application_id: int, status: str, *, now: datetime | None = None
) -> None:
    """Advance an application through the funnel."""
    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r}. Expected one of {', '.join(STATUSES)}.")
    conn.execute(
        "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
        (status, _now(now).isoformat(), application_id),
    )
    conn.commit()


def _rows_for_report(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT a.id AS app_id, a.status, a.fit_score, a.mode, a.missing_keywords,
                  a.created_at, j.company, j.title, j.url, j.location
           FROM applications a JOIN jobs j ON j.id = a.job_id
           ORDER BY a.fit_score IS NULL, a.fit_score DESC, a.created_at DESC"""
    ).fetchall()


def report_markdown(conn: sqlite3.Connection) -> str:
    """A tracker table of all applications, ranked by fit."""
    rows = _rows_for_report(conn)
    out = [
        "# Application Tracker",
        "",
        f"_{len(rows)} application(s). Generated by JobProof._",
        "",
        "| # | Fit | Company | Role | Status | Mode | Top missing | Date | Link |",
        "|---|-----|---------|------|--------|------|-------------|------|------|",
    ]
    for r in rows:
        missing = ", ".join(json.loads(r["missing_keywords"] or "[]")[:4]) or "—"
        date = (r["created_at"] or "")[:10]
        link = f"[link]({r['url']})" if r["url"] else "—"
        fit = "—" if r["fit_score"] is None else f"{r['fit_score']}/100"
        out.append(
            f"| {r['app_id']} | {fit} | {r['company']} | {r['title']} | "
            f"{r['status']} | {r['mode'] or '—'} | {missing} | {date} | {link} |"
        )
    return "\n".join(out) + "\n"


def report_csv(conn: sqlite3.Connection) -> str:
    """The same tracker as CSV (spreadsheet-friendly)."""
    rows = _rows_for_report(conn)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        ["app_id", "fit_score", "company", "role", "status", "mode", "missing_keywords", "created_at", "url"]
    )
    for r in rows:
        w.writerow(
            [
                r["app_id"],
                r["fit_score"],
                r["company"],
                r["title"],
                r["status"],
                r["mode"],
                "; ".join(json.loads(r["missing_keywords"] or "[]")),
                r["created_at"],
                r["url"] or "",
            ]
        )
    return buf.getvalue()


# --------------------------------------------------------------------------- dashboard


def _log_event(
    conn: sqlite3.Connection,
    job_id: int | None,
    kind: str,
    subject: str,
    *,
    source: str = "app",
    now: datetime | None = None,
) -> None:
    """Append to the job's activity timeline (no commit — caller owns the transaction)."""
    conn.execute(
        "INSERT INTO job_events (job_id, kind, source, subject, created_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, kind, source, subject, _now(now).isoformat()),
    )


def set_job_status(
    conn: sqlite3.Connection, job_id: int, status: str, *, now: datetime | None = None
) -> None:
    """Set a job's pipeline status (the unit the dashboard tracks). Logs the transition."""
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown status {status!r}. Expected one of {', '.join(JOB_STATUSES)}.")
    ts = _now(now)
    prev = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    applied_at = ts.isoformat() if status == "applied" else None
    follow_up_at = (ts + timedelta(days=7)).isoformat() if status == "applied" else None
    next_action = {
        "new": "review",
        "interested": "tailor_resume",
        "to_apply": "apply",
        "applied": "follow_up",
        "interview": "prep_interview",
        "offer": "review",
        "rejected": "skip",
        "skipped": "skip",
    }[status]
    if status == "applied":
        conn.execute(
            """UPDATE jobs SET status = ?, next_action = ?, applied_at = COALESCE(applied_at, ?),
               follow_up_at = COALESCE(follow_up_at, ?) WHERE id = ?""",
            (status, next_action, applied_at, follow_up_at, job_id),
        )
    else:
        conn.execute(
            "UPDATE jobs SET status = ?, next_action = ? WHERE id = ?", (status, next_action, job_id)
        )
    if prev and prev["status"] != status:
        _log_event(conn, job_id, "status_change", f"{prev['status']} → {status}", now=now)
    conn.commit()


DECISIONS = ("apply", "ask_recruiter", "save", "skip", "follow_up")


def decide(conn: sqlite3.Connection, job_id: int, decision: str, *, now: datetime | None = None) -> dict:
    """Apply a one-click decision from the dashboard; updates status / next action / tasks."""
    if decision not in DECISIONS:
        raise ValueError(f"Unknown decision {decision!r}. Expected one of {', '.join(DECISIONS)}.")
    ts = _now(now)
    created_task = None
    if decision == "apply":
        set_job_status(conn, job_id, "to_apply", now=now)
        update_tracker(conn, job_id, {"next_action": "apply"})
    elif decision == "ask_recruiter":
        set_job_status(conn, job_id, "interested", now=now)
        update_tracker(conn, job_id, {"next_action": "message_recruiter"})
        created_task = create_task(
            conn, "Ask recruiter: salary + remote/hybrid", job_id=job_id, priority=2, now=now
        )
    elif decision == "save":
        set_job_status(conn, job_id, "interested", now=now)
        update_tracker(conn, job_id, {"next_action": "review"})
    elif decision == "skip":
        set_job_status(conn, job_id, "skipped", now=now)
    elif decision == "follow_up":
        due = (ts + timedelta(days=3)).isoformat()
        update_tracker(conn, job_id, {"next_action": "follow_up", "follow_up_at": due})
        created_task = create_task(
            conn, "Follow up with recruiter", job_id=job_id, priority=2, due_at=due, now=now
        )
    _log_event(conn, job_id, "decision", f"Decision: {decision}", now=now)
    conn.commit()
    return {"job_id": job_id, "decision": decision, "task_id": created_task}


def set_job_notes(conn: sqlite3.Connection, job_id: int, notes: str) -> None:
    conn.execute("UPDATE jobs SET notes = ? WHERE id = ?", (notes, job_id))
    conn.commit()


def set_link_status(
    conn: sqlite3.Connection, job_id: int, status: str, *, now: datetime | None = None
) -> None:
    """Record the result of a link check. A newly-dead link is logged to the timeline."""
    if status not in LINK_STATUSES:
        raise ValueError(f"Unknown link_status {status!r}. Expected one of {', '.join(LINK_STATUSES)}.")
    prev = conn.execute("SELECT link_status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.execute(
        "UPDATE jobs SET link_status = ?, link_checked_at = ? WHERE id = ?",
        (status, _now(now).isoformat(), job_id),
    )
    if status == "dead" and (not prev or prev["link_status"] != "dead"):
        _log_event(conn, job_id, "dead_link", "Posting link returned dead (404/410)", now=now)
    conn.commit()


def update_tracker(conn: sqlite3.Connection, job_id: int, fields: dict) -> None:
    """Update application-tracker fields on a job."""
    allowed = {
        "next_action",
        "applied_at",
        "follow_up_at",
        "notes",
        "contact_name",
        "contact_email",
        "contact_url",
    }
    sets, params = [], []
    for key, val in fields.items():
        if key not in allowed:
            raise ValueError(f"Unknown tracker field {key!r}. Allowed: {', '.join(sorted(allowed))}")
        if key == "next_action" and val not in NEXT_ACTIONS:
            raise ValueError(f"Invalid next_action={val!r}. Allowed: {', '.join(NEXT_ACTIONS)}")
        sets.append(f"{key} = ?")
        params.append(val)
    if not sets:
        return
    params.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()


FOLLOW_UP_CYCLE_DAYS = 7  # after sending a follow-up, schedule the next check-in this far out
STALE_DAYS = 10  # an applied/interview job with no activity for this long is going cold


def complete_follow_up(conn: sqlite3.Connection, job_id: int, *, now: datetime | None = None) -> str | None:
    """Mark a follow-up as sent: log it and schedule the next cycle (or clear if job is closed).

    Returns the next follow_up_at (ISO) or None.
    """
    ts = _now(now)
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"No job with id {job_id}")
    next_at = (
        (ts + timedelta(days=FOLLOW_UP_CYCLE_DAYS)).isoformat()
        if row["status"] in ("applied", "interview")
        else None
    )
    conn.execute("UPDATE jobs SET follow_up_at = ? WHERE id = ?", (next_at, job_id))
    _log_event(conn, job_id, "followed_up", "Follow-up sent", now=now)
    conn.commit()
    return next_at


def snooze_follow_up(
    conn: sqlite3.Connection, job_id: int, *, days: int = 3, now: datetime | None = None
) -> str:
    """Push a job's follow-up date out by `days` from now (not from the overdue date)."""
    days = int(days)
    if not 1 <= days <= 90:
        raise ValueError("days must be 1-90")
    if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
        raise ValueError(f"No job with id {job_id}")
    next_at = (_now(now) + timedelta(days=days)).isoformat()
    conn.execute("UPDATE jobs SET follow_up_at = ? WHERE id = ?", (next_at, job_id))
    conn.commit()
    return next_at


def _days_between(earlier: str | None, later: datetime) -> int | None:
    if not earlier:
        return None
    try:
        dt = datetime.fromisoformat(earlier.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (later - dt).days


def action_queue(
    conn: sqlite3.Connection, *, now: datetime | None = None, stale_days: int = STALE_DAYS
) -> list[dict]:
    """The prioritized 'do this now' list the dashboard leads with.

    Urgency tiers: 0 = overdue, 1 = due today, 2 = needs attention (interviews to
    prep, apply-today roles not yet applied), 3 = going stale. Within a tier,
    most-overdue first, then priority.
    """
    ts = _now(now)
    today = ts.date().isoformat()
    items: list[dict] = []
    rows = dashboard_rows(conn, now=ts)

    for r in rows:
        base = {
            "job_id": r["id"],
            "company": r["company"],
            "title": r["title"],
            "priority": r["priority"],
            "contact_name": r["contact_name"],
        }
        has_follow_up_item = False
        fu = r["follow_up_at"]
        if fu and r["status"] in ("applied", "interview"):
            fu_day = fu[:10]
            has_follow_up_item = fu_day <= today
            if fu_day < today:
                overdue = _days_between(fu, ts) or 0
                items.append(
                    {
                        **base,
                        "kind": "follow_up",
                        "urgency": 0,
                        "date": fu,
                        "days_overdue": overdue,
                        "label": f"Follow up with {r['company']} — {overdue}d overdue",
                    }
                )
            elif fu_day == today:
                items.append(
                    {
                        **base,
                        "kind": "follow_up",
                        "urgency": 1,
                        "date": fu,
                        "days_overdue": 0,
                        "label": f"Follow up with {r['company']} — due today",
                    }
                )
        if r["status"] == "interview":
            items.append(
                {
                    **base,
                    "kind": "interview",
                    "urgency": 2,
                    "date": None,
                    "days_overdue": 0,
                    "label": f"Prep interview — {r['company']}",
                }
            )
        if (
            r["apply_recommendation"] == "apply_today"
            and r["status"] in ("new", "interested", "to_apply")
            and r["link_status"] != "dead"  # never push a known-dead posting to "apply"
        ):
            items.append(
                {
                    **base,
                    "kind": "apply",
                    "urgency": 2,
                    "date": None,
                    "days_overdue": 0,
                    "label": f"Apply — {r['company']} · {r['title']}",
                }
            )
        if r["stale"] and not has_follow_up_item:
            items.append(
                {
                    **base,
                    "kind": "stale",
                    "urgency": 3,
                    "date": r["last_activity_at"],
                    "days_overdue": r["days_since_activity"] or 0,
                    "label": f"{r['company']} going cold — {r['days_since_activity']}d since activity",
                }
            )

    for t in tasks(conn):
        if not t["due_at"]:
            continue
        due_day = t["due_at"][:10]
        if due_day > today:
            continue
        overdue = max(_days_between(t["due_at"], ts) or 0, 0)
        items.append(
            {
                "kind": "task",
                "urgency": 0 if due_day < today else 1,
                "task_id": t["id"],
                "job_id": t["job_id"],
                "company": t["company"],
                "title": t["job_title"],
                "priority": t["priority"],
                "contact_name": None,
                "date": t["due_at"],
                "days_overdue": overdue,
                "label": t["title"] + (f" — {overdue}d overdue" if due_day < today else " — due today"),
            }
        )

    items.sort(key=lambda i: (i["urgency"], -i["days_overdue"], i["priority"]))
    return items


def dashboard_rows(conn: sqlite3.Connection, *, now: datetime | None = None) -> list[dict]:
    """Every job + its best tailored fit (if any), shaped for the dashboard."""
    ts = _now(now)
    today = ts.date().isoformat()
    rows = conn.execute(
        """SELECT j.id, j.company, j.title, j.location, j.remote, j.salary, j.url,
                  j.status, j.notes, j.next_action, j.applied_at, j.follow_up_at,
                  j.contact_name, j.contact_email, j.contact_url,
                  j.link_status, j.link_checked_at,
                  j.clearance, j.work_mode, j.recommended_variant, j.fit_breakdown,
                  j.capture_confidence, j.needs_review,
                  j.source, j.posted_at, j.fetched_at, j.description,
                  j.legitimacy_status, j.diligence_notes, j.review_signal,
                  j.source_confidence, j.priority, j.apply_recommendation,
                  (SELECT MAX(fit_score) FROM applications a WHERE a.job_id = j.id) AS best_fit,
                  (SELECT a2.out_dir FROM applications a2 WHERE a2.job_id = j.id
                     ORDER BY a2.fit_score DESC LIMIT 1) AS out_dir,
                  (SELECT a3.why_match FROM applications a3 WHERE a3.job_id = j.id
                     ORDER BY a3.fit_score DESC LIMIT 1) AS why_match,
                  (SELECT COUNT(*) FROM kits k WHERE k.job_id = j.id) AS kit_count,
                  (SELECT k2.created_at FROM kits k2 WHERE k2.job_id = j.id
                     ORDER BY k2.created_at DESC LIMIT 1) AS latest_kit_at,
                  (SELECT MAX(e.created_at) FROM job_events e WHERE e.job_id = j.id) AS last_event_at
           FROM jobs j
           ORDER BY j.priority ASC, best_fit IS NULL, best_fit DESC, j.fetched_at DESC"""
    ).fetchall()
    out = []
    for r in rows:
        loc = r["location"] or ""
        is_pr = "puerto rico" in loc.lower() or " pr" in f" {loc.lower()}" or loc.lower().endswith(", pr")
        # One-line reason for the row: prefer the tailored why_match, else diligence notes.
        why = r["why_match"] or r["diligence_notes"] or ""
        # Last activity = the newest of: timeline event, applied date, fetch date.
        last_activity = max(
            filter(None, (r["last_event_at"], r["applied_at"], r["fetched_at"])), default=None
        )
        days_since_activity = _days_between(last_activity, ts)
        follow_up_overdue = bool(
            r["follow_up_at"] and r["follow_up_at"][:10] < today and r["status"] in ("applied", "interview")
        )
        stale = bool(
            r["status"] in ("applied", "interview")
            and days_since_activity is not None
            and days_since_activity >= STALE_DAYS
        )
        out.append(
            {
                "id": r["id"],
                "company": r["company"],
                "title": r["title"],
                "location": loc,
                "remote": bool(r["remote"]),
                "puerto_rico": is_pr,
                "salary": r["salary"],
                "url": r["url"],
                "status": r["status"],
                "notes": r["notes"],
                "next_action": r["next_action"],
                "applied_at": r["applied_at"],
                "follow_up_at": r["follow_up_at"],
                "contact_name": r["contact_name"],
                "contact_email": r["contact_email"],
                "contact_url": r["contact_url"],
                "link_status": r["link_status"],
                "link_checked_at": r["link_checked_at"],
                "clearance": r["clearance"],
                "work_mode": r["work_mode"],
                "recommended_variant": r["recommended_variant"],
                "quick_fit": (json.loads(r["fit_breakdown"]).get("overall") if r["fit_breakdown"] else None),
                "capture_confidence": r["capture_confidence"],
                "needs_review": bool(r["needs_review"]),
                "source": r["source"],
                "best_fit": r["best_fit"],
                "out_dir": r["out_dir"],
                "kit_count": r["kit_count"],
                "latest_kit_at": r["latest_kit_at"],
                "posted_at": r["posted_at"],
                "description": r["description"],
                "legitimacy_status": r["legitimacy_status"],
                "diligence_notes": r["diligence_notes"],
                "review_signal": r["review_signal"],
                "source_confidence": r["source_confidence"],
                "priority": r["priority"],
                "apply_recommendation": r["apply_recommendation"],
                "why": why,
                "last_activity_at": last_activity,
                "days_since_activity": days_since_activity,
                "days_since_applied": _days_between(r["applied_at"], ts),
                "follow_up_overdue": follow_up_overdue,
                "stale": stale,
            }
        )
    return out


def job_detail(conn: sqlite3.Connection, job_id: int) -> dict | None:
    """A single job plus its latest tailored application (fit explanation + evidence)."""
    rows = [r for r in dashboard_rows(conn) if r["id"] == job_id]
    if not rows:
        return None
    job = rows[0]
    app = conn.execute(
        """SELECT id, fit_score, matched_keywords, missing_keywords, application_note,
                  why_match, why_not, missing_proof, keywords_to_mirror, recruiter_angle,
                  evidence_map, mode, out_dir, created_at
           FROM applications WHERE job_id = ? ORDER BY fit_score DESC, created_at DESC LIMIT 1""",
        (job_id,),
    ).fetchone()
    application = None
    if app:
        application = {
            "fit_score": app["fit_score"],
            "matched_keywords": json.loads(app["matched_keywords"] or "[]"),
            "missing_keywords": json.loads(app["missing_keywords"] or "[]"),
            "application_note": app["application_note"],
            "why_match": app["why_match"],
            "why_not": app["why_not"],
            "missing_proof": app["missing_proof"],
            "keywords_to_mirror": json.loads(app["keywords_to_mirror"] or "[]"),
            "recruiter_angle": app["recruiter_angle"],
            "evidence_map": json.loads(app["evidence_map"] or "[]"),
            "mode": app["mode"],
            "out_dir": app["out_dir"],
            "created_at": app["created_at"],
            "retrieved_sources": [
                {"source": s["source"], "heading": s["heading"], "distance": s["distance"]}
                for s in conn.execute(
                    "SELECT source, heading, distance FROM retrieved_sources "
                    "WHERE application_id = ? ORDER BY rank",
                    (app["id"],),
                ).fetchall()
            ],
        }
    fb_row = conn.execute("SELECT fit_breakdown FROM jobs WHERE id = ?", (job_id,)).fetchone()
    fit_breakdown = json.loads(fb_row["fit_breakdown"]) if fb_row and fb_row["fit_breakdown"] else None

    return {
        "job": job,
        "application": application,
        "fit_breakdown": fit_breakdown,
        "recruiter_messages": recruiter_messages(conn, job_id),
        "kits": kit_rows(conn, job_id),
        "runs": generation_runs(conn, job_id=job_id, limit=8),
        "tasks": tasks(conn, job_id=job_id),
        "events": job_events(conn, job_id=job_id, limit=8),
        "analysis": analysis_reports(conn, job_id, limit=8),
        "resume_versions": resume_versions(conn, job_id, limit=8),
    }


def _kit_from_row(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "job_id": r["job_id"],
        "application_id": r["application_id"],
        "out_dir": r["out_dir"],
        "resume_markdown": r["resume_markdown"],
        "cover_letter": r["cover_letter"],
        "why_me": json.loads(r["why_me"] or "[]"),
        "recruiter_dm": r["recruiter_dm"],
        "interview_stories": json.loads(r["interview_stories"] or "[]"),
        "evidence_map": json.loads(r["evidence_map"] or "[]"),
        "created_at": r["created_at"],
    }


def kit_rows(conn: sqlite3.Connection, job_id: int) -> list[dict]:
    """All kits for a job, newest first."""
    rows = conn.execute(
        """SELECT id, job_id, application_id, out_dir, resume_markdown, cover_letter,
                  why_me, recruiter_dm, interview_stories, evidence_map, created_at
           FROM kits WHERE job_id = ? ORDER BY created_at DESC""",
        (job_id,),
    ).fetchall()
    return [_kit_from_row(r) for r in rows]


def latest_kit(conn: sqlite3.Connection, job_id: int) -> dict | None:
    row = conn.execute(
        """SELECT id, job_id, application_id, out_dir, resume_markdown, cover_letter,
                  why_me, recruiter_dm, interview_stories, evidence_map, created_at
           FROM kits WHERE job_id = ? ORDER BY created_at DESC LIMIT 1""",
        (job_id,),
    ).fetchone()
    return _kit_from_row(row) if row else None


def generation_runs(conn: sqlite3.Connection, *, job_id: int | None = None, limit: int = 20) -> list[dict]:
    """Recent generation runs, optionally scoped to one job."""
    where = "WHERE job_id = ?" if job_id is not None else ""
    params = [job_id] if job_id is not None else []
    params.append(limit)
    rows = conn.execute(
        f"""SELECT id, job_id, kind, status, model, mode, input_tokens,
                   cache_creation_input_tokens, cache_read_input_tokens, output_tokens,
                   estimated_cost_usd, latency_ms, error, started_at, finished_at
            FROM generation_runs {where}
            ORDER BY started_at DESC LIMIT ?""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def board(conn: sqlite3.Connection) -> dict:
    """Jobs grouped into Kanban columns by pipeline status."""
    grouped = {s: [] for s in JOB_STATUSES}
    for row in dashboard_rows(conn):
        grouped.setdefault(row["status"], []).append(row)
    return grouped


def create_task(
    conn: sqlite3.Connection,
    title: str,
    *,
    job_id: int | None = None,
    priority: int = 3,
    due_at: str | None = None,
    notes: str | None = None,
    now: datetime | None = None,
) -> int:
    """Create a tracker task, optionally attached to a job."""
    priority = int(priority)
    if not 1 <= priority <= 5:
        raise ValueError("priority must be 1-5")
    cur = conn.execute(
        """INSERT INTO tasks (job_id, title, priority, due_at, notes, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (job_id, title, priority, due_at, notes, _now(now).isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def complete_task(conn: sqlite3.Connection, task_id: int, *, now: datetime | None = None) -> None:
    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
        (_now(now).isoformat(), task_id),
    )
    conn.commit()


def tasks(conn: sqlite3.Connection, *, job_id: int | None = None, include_done: bool = False) -> list[dict]:
    where, params = [], []
    if job_id is not None:
        where.append("t.job_id = ?")
        params.append(job_id)
    if not include_done:
        where.append("t.status != 'done'")
    sql_where = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""SELECT t.id, t.job_id, t.title, t.status, t.priority, t.due_at, t.notes,
                   t.created_at, t.completed_at, j.company, j.title AS job_title
            FROM tasks t LEFT JOIN jobs j ON j.id = t.job_id
            {sql_where}
            ORDER BY t.status = 'done', t.due_at IS NULL, t.due_at ASC, t.priority ASC, t.created_at DESC""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def calendar_items(conn: sqlite3.Connection) -> list[dict]:
    """Follow-ups and open tasks with dates, sorted for a lightweight calendar view."""
    items = []
    for r in dashboard_rows(conn):
        if r["follow_up_at"]:
            items.append(
                {
                    "kind": "follow_up",
                    "job_id": r["id"],
                    "company": r["company"],
                    "title": r["title"],
                    "date": r["follow_up_at"],
                    "label": f"Follow up with {r['company']}",
                }
            )
        if r["applied_at"]:
            items.append(
                {
                    "kind": "applied",
                    "job_id": r["id"],
                    "company": r["company"],
                    "title": r["title"],
                    "date": r["applied_at"],
                    "label": f"Applied to {r['company']}",
                }
            )
    for t in tasks(conn):
        if t["due_at"]:
            items.append(
                {
                    "kind": "task",
                    "task_id": t["id"],
                    "job_id": t["job_id"],
                    "company": t["company"],
                    "title": t["job_title"],
                    "date": t["due_at"],
                    "label": t["title"],
                }
            )
    return sorted(items, key=lambda x: x["date"] or "")


def record_search_run(
    conn: sqlite3.Connection,
    *,
    query: str,
    sources: list[str] | None = None,
    location: str | None = None,
    status: str = "planned",
    found_count: int = 0,
    deduped_count: int = 0,
    notes: str | None = None,
    finished_at: str | None = None,
    now: datetime | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO search_runs (query, sources, location, status, found_count, deduped_count,
                                    notes, created_at, finished_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            query,
            json.dumps(sources or []),
            location,
            status,
            int(found_count or 0),
            int(deduped_count or 0),
            notes,
            _now(now).isoformat(),
            finished_at,
        ),
    )
    conn.commit()
    return cur.lastrowid


def search_runs(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT id, query, sources, location, status, found_count, deduped_count,
                  notes, created_at, finished_at
           FROM search_runs ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"] or "[]")
        out.append(d)
    return out


def record_analysis(
    conn: sqlite3.Connection,
    job_id: int,
    kind: str,
    payload: dict,
    *,
    now: datetime | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO analysis_reports (job_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
        (job_id, kind, json.dumps(payload), _now(now).isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def analysis_reports(conn: sqlite3.Connection, job_id: int, *, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT id, job_id, kind, payload, created_at
           FROM analysis_reports WHERE job_id = ? ORDER BY created_at DESC LIMIT ?""",
        (job_id, limit),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"] or "{}")
        out.append(d)
    return out


def record_resume_version(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    label: str,
    content: str,
    application_id: int | None = None,
    kit_id: int | None = None,
    diff_summary: dict | None = None,
    now: datetime | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO resume_versions (job_id, application_id, kit_id, label, content,
                                        diff_summary, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id,
            application_id,
            kit_id,
            label,
            content,
            json.dumps(diff_summary or {}),
            _now(now).isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def resume_versions(conn: sqlite3.Connection, job_id: int, *, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT id, job_id, application_id, kit_id, label, content, diff_summary, created_at
           FROM resume_versions WHERE job_id = ? ORDER BY created_at DESC LIMIT ?""",
        (job_id, limit),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["diff_summary"] = json.loads(d["diff_summary"] or "{}")
        out.append(d)
    return out


def record_job_event(
    conn: sqlite3.Connection,
    *,
    job_id: int | None,
    kind: str,
    source: str | None = None,
    subject: str | None = None,
    body_excerpt: str | None = None,
    inferred_status: str | None = None,
    now: datetime | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO job_events (job_id, kind, source, subject, body_excerpt, inferred_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (job_id, kind, source, subject, body_excerpt, inferred_status, _now(now).isoformat()),
    )
    if job_id and inferred_status in JOB_STATUSES:
        set_job_status(conn, job_id, inferred_status, now=now)
    conn.commit()
    return cur.lastrowid


def job_events(conn: sqlite3.Connection, *, job_id: int | None = None, limit: int = 20) -> list[dict]:
    where = "WHERE e.job_id = ?" if job_id is not None else ""
    params = [job_id] if job_id is not None else []
    params.append(limit)
    rows = conn.execute(
        f"""SELECT e.id, e.job_id, e.kind, e.source, e.subject, e.body_excerpt,
                   e.inferred_status, e.created_at, j.company, j.title AS job_title
            FROM job_events e LEFT JOIN jobs j ON j.id = e.job_id
            {where}
            ORDER BY e.created_at DESC LIMIT ?""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def digest(conn: sqlite3.Connection, *, days: int = 7, now: datetime | None = None) -> dict:
    """Aggregate the last `days` of pipeline activity into a structured weekly report.

    Reads the job_events timeline + generation_runs so the numbers are longitudinal
    (what moved this week), not just a snapshot. Pure data; format_digest renders it.
    """
    ts = _now(now)
    since = (ts - timedelta(days=days)).isoformat()
    rows = dashboard_rows(conn, now=ts)

    ev = conn.execute(
        "SELECT kind, COUNT(*) AS n FROM job_events WHERE created_at >= ? GROUP BY kind",
        (since,),
    ).fetchall()
    events = {r["kind"]: r["n"] for r in ev}

    applied_rows = conn.execute(
        """SELECT j.company, j.title, j.applied_at FROM jobs j
           WHERE j.applied_at IS NOT NULL AND j.applied_at >= ? ORDER BY j.applied_at DESC""",
        (since,),
    ).fetchall()

    # Per-source funnel: how many tracked, how many reached 'applied'.
    by_source: dict[str, dict] = {}
    for r in rows:
        s = by_source.setdefault(r["source"] or "?", {"tracked": 0, "applied": 0})
        s["tracked"] += 1
        if r["status"] in ("applied", "interview", "offer"):
            s["applied"] += 1

    runs = conn.execute(
        """SELECT COUNT(*) AS n, COALESCE(SUM(estimated_cost_usd), 0) AS cost,
                  COALESCE(SUM(input_tokens + cache_creation_input_tokens + cache_read_input_tokens
                               + output_tokens), 0) AS tokens
           FROM generation_runs WHERE started_at >= ? AND status = 'done'""",
        (since,),
    ).fetchone()

    overdue = [
        {"company": r["company"], "title": r["title"], "days": _days_between(r["follow_up_at"], ts)}
        for r in rows
        if r["follow_up_overdue"]
    ]
    stale = [
        {"company": r["company"], "title": r["title"], "days": r["days_since_activity"]}
        for r in rows
        if r["stale"]
    ]
    queue = action_queue(conn, now=ts)

    return {
        "generated_at": ts.isoformat(),
        "window_days": days,
        "totals": stats(conn),
        "events": events,
        "applied_this_week": [
            {"company": r["company"], "title": r["title"], "applied_at": r["applied_at"]}
            for r in applied_rows
        ],
        "by_source": by_source,
        "generation": {
            "runs": int(runs["n"] or 0),
            "tokens": int(runs["tokens"] or 0),
            "cost_usd": round(float(runs["cost"] or 0), 4),
        },
        "overdue_followups": overdue,
        "going_stale": stale,
        "queue_size": len(queue),
    }


def format_digest(d: dict) -> str:
    """Render digest() as a readable markdown report."""
    t = d["totals"]
    lines = [
        f"# JobProof — weekly digest ({d['window_days']}d)",
        "",
        f"_Generated {d['generated_at'][:16].replace('T', ' ')} · {t['total']} roles tracked._",
        "",
        "## Pipeline",
        f"- **{t['applied']}** applied · **{t['interviews']}** interviewing · "
        f"**{t.get('offers', t['by_status'].get('offer', 0))}** offers",
        f"- **{d['queue_size']}** items in the action queue · **{len(d['overdue_followups'])}** overdue follow-ups · "
        f"**{len(d['going_stale'])}** going stale · **{t.get('dead_links', 0)}** dead links",
        "",
        f"## This week ({d['window_days']}d)",
        f"- Applied to **{len(d['applied_this_week'])}** role(s)",
        "- Activity: "
        + (", ".join(f"{k.replace('_', ' ')} ×{v}" for k, v in sorted(d["events"].items())) or "—"),
        f"- Generation: {d['generation']['runs']} run(s), {d['generation']['tokens']:,} tokens, ${d['generation']['cost_usd']}",
    ]
    if d["applied_this_week"]:
        lines += ["", "### Applied this week"]
        lines += [
            f"- {a['company']} — {a['title']} ({(a['applied_at'] or '')[:10]})"
            for a in d["applied_this_week"]
        ]
    if d["overdue_followups"]:
        lines += ["", "### ⚠ Overdue follow-ups (do these)"]
        lines += [f"- {o['company']} — {o['title']} ({o['days']}d overdue)" for o in d["overdue_followups"]]
    if d["going_stale"]:
        lines += ["", "### Going stale"]
        lines += [f"- {s['company']} — {s['title']} ({s['days']}d quiet)" for s in d["going_stale"]]
    if d["by_source"]:
        lines += ["", "### By source", "| Source | Tracked | In funnel |", "|---|---|---|"]
        lines += [f"| {s} | {v['tracked']} | {v['applied']} |" for s, v in sorted(d["by_source"].items())]
    return "\n".join(lines) + "\n"


def stats(conn: sqlite3.Connection) -> dict:
    """Summary counts for the dashboard header."""
    rows = dashboard_rows(conn)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    fits = [r["best_fit"] for r in rows if r["best_fit"] is not None]
    now_iso = datetime.now(UTC).isoformat()
    run = conn.execute(
        """SELECT COUNT(*) AS runs,
                  COALESCE(SUM(input_tokens + cache_creation_input_tokens + cache_read_input_tokens + output_tokens), 0) AS tokens,
                  COALESCE(SUM(estimated_cost_usd), 0) AS cost,
                  AVG(latency_ms) AS avg_latency
           FROM generation_runs WHERE status = 'done'"""
    ).fetchone()
    support = conn.execute(
        """SELECT
              (SELECT COUNT(*) FROM tasks WHERE status != 'done') AS open_tasks,
              (SELECT COUNT(*) FROM search_runs) AS search_runs,
              (SELECT COUNT(*) FROM job_events) AS events,
              (SELECT COUNT(*) FROM analysis_reports) AS analyses"""
    ).fetchone()
    return {
        "total": len(rows),
        "by_status": by_status,
        "puerto_rico": sum(1 for r in rows if r["puerto_rico"]),
        "remote": sum(1 for r in rows if r["remote"]),
        "tailored": len(fits),
        "kits": sum(int(r["kit_count"] or 0) for r in rows),
        "avg_fit": round(sum(fits) / len(fits), 1) if fits else None,
        "apply_today": sum(1 for r in rows if r["apply_recommendation"] == "apply_today"),
        "needs_diligence": sum(1 for r in rows if r["legitimacy_status"] == "needs_diligence"),
        "applied": sum(1 for r in rows if r["status"] == "applied"),
        "interviews": sum(1 for r in rows if r["status"] == "interview"),
        "stale": sum(1 for r in rows if r["stale"]),
        "dead_links": sum(1 for r in rows if r["link_status"] == "dead"),
        "follow_up_overdue": sum(1 for r in rows if r["follow_up_overdue"]),
        "follow_up_due": sum(
            1 for r in rows if r["follow_up_at"] and r["follow_up_at"] <= now_iso and r["status"] == "applied"
        ),
        "tokens": int(run["tokens"] or 0),
        "estimated_cost_usd": round(float(run["cost"] or 0), 4),
        "avg_latency_ms": round(float(run["avg_latency"] or 0), 1) if run["avg_latency"] else None,
        "generation_runs": int(run["runs"] or 0),
        "open_tasks": int(support["open_tasks"] or 0),
        "search_runs": int(support["search_runs"] or 0),
        "events": int(support["events"] or 0),
        "analyses": int(support["analyses"] or 0),
    }
