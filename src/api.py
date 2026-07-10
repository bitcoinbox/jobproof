"""FastAPI web UI / API over the tailoring core.

Paste a job posting (or, if explicitly enabled, a URL) and get back the tailored
resume, cover letter, fit score, keyword gap, and (in RAG mode) the retrieved
sources — reusing tailor.tailor() and rag.* unchanged. A single static page drives it.

Designed to degrade gracefully: /healthz needs no API key, and tailoring returns a
clean 503 (not a stack trace) when ANTHROPIC_API_KEY is missing, so a public demo
deploys and loads even before a key is set.

Routes are sync (`def`) — FastAPI runs them in a threadpool, so the blocking Anthropic
call doesn't stall the event loop and tailor.py needs no async rewrite.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import secrets
import socket
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import agent, capture, ingest, links, llm, product, rag, recruiter, scoring, store, tailor
from .tailor import TailoredApplication

_HERE = Path(__file__).resolve().parent.parent
_STATIC = _HERE / "app" / "static"
_MASTER_FALLBACKS = [
    _HERE / "master-resume.md",
    _HERE.parent / "resume-master.md",
    _HERE / "sample-corpus" / "00-master-resume.md",
]
_MAX_URL_BYTES = 1_000_000


def _resolve_master() -> str | None:
    for c in _MASTER_FALLBACKS:
        if c.exists():
            return c.read_text(encoding="utf-8")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Anthropic (default) or a self-hosted OpenAI-compatible endpoint (AUTO_APPLY_LLM_BACKEND=local).
    app.state.client = llm.make_client()  # construction is lazy; no key needed here
    app.state.backend = llm.backend_name()
    app.state.model = llm.active_model()
    app.state.master = _resolve_master()
    app.state.index_path = os.environ.get("AUTO_APPLY_INDEX", rag.DEFAULT_INDEX_PATH)
    app.state.allow_url = os.environ.get("AUTO_APPLY_ALLOW_URL", "").lower() in ("1", "true", "yes")
    app.state.db_path = os.environ.get("AUTO_APPLY_DB", store.DEFAULT_DB)
    yield


app = FastAPI(
    title="JobProof",
    version="1.0",
    lifespan=lifespan,
    # auto-docs disclosed the full API surface to anyone — turn them off in prod
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


# --- HTTP Basic Auth over the whole app -------------------------------------
# Single-user tool: the browser prompts once, then sends creds on every request
# (the HTML pages AND the dashboard's fetch() calls to /api/*), so no frontend
# change is needed. /healthz + /static stay open so Railway's health check and the
# page assets still load. Fails CLOSED — if JOBPROOF_PASSWORD is unset, deny all.
_AUTH_USER = os.environ.get("JOBPROOF_USER", "admin")
_AUTH_PASS = os.environ.get("JOBPROOF_PASSWORD", "")


def _unauthorized(detail: str = "Unauthorized") -> JSONResponse:
    return JSONResponse(
        {"detail": detail},
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="JobProof"'},
    )


@app.middleware("http")
async def _require_basic_auth(request: Request, call_next):
    path = request.url.path
    if path == "/healthz" or path.startswith("/static/"):
        return await call_next(request)
    if not _AUTH_PASS:
        return _unauthorized("server auth not configured")
    header = request.headers.get("authorization", "")
    if header.startswith("Basic "):
        try:
            user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            if secrets.compare_digest(user, _AUTH_USER) and secrets.compare_digest(pw, _AUTH_PASS):
                return await call_next(request)
        except Exception:
            pass
    return _unauthorized()


class TailorRequest(BaseModel):
    job_text: str | None = None
    job_url: str | None = None
    mode: str = Field("full", pattern="^(full|rag)$")
    top_k: int = rag.DEFAULT_K

    @model_validator(mode="after")
    def _exactly_one(self):
        if bool(self.job_text) == bool(self.job_url):
            raise ValueError("Provide exactly one of job_text or job_url.")
        return self


class TailorResponse(BaseModel):
    fit_score: int
    matched_keywords: list[str]
    missing_keywords: list[str]
    resume_markdown: str
    cover_letter: str
    application_note: str
    why_match: str = ""
    why_not: str = ""
    missing_proof: str = ""
    keywords_to_mirror: list[str] = []
    recruiter_angle: str = ""
    retrieved_sources: list[str] = []
    mode: str
    usage: dict


def _rag_ready(index_path: str) -> bool:
    return Path(index_path).exists()


def _require_key(client) -> None:
    """503 cleanly when no Anthropic credentials are configured (instead of a 500)."""
    if not (getattr(client, "api_key", None) or getattr(client, "auth_token", None)):
        raise HTTPException(503, "Tailoring/kit unavailable: set ANTHROPIC_API_KEY on the server.")


def _safe_fetch(url: str) -> str:
    """Fetch a URL with an SSRF guard (https only, no private/loopback/link-local IPs)."""
    import httpx

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise HTTPException(400, "Only https:// URLs are allowed.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise HTTPException(400, "Could not resolve host.")
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(400, "URL resolves to a disallowed (internal) address.")
    with httpx.Client(timeout=15.0, follow_redirects=False, headers={"User-Agent": ingest.USER_AGENT}) as c:
        resp = c.get(url)
        resp.raise_for_status()
        return ingest.strip_html(resp.text[:_MAX_URL_BYTES])


@app.get("/healthz")
def healthz(request: Request):
    st = request.app.state
    client = st.client
    # Public (Railway health check). Only the booleans the UI needs — no model/backend
    # /allow_url disclosure (that was an unauthenticated info leak).
    return {
        "status": "ok",
        # whether AI generation (tailor/kit/agent) is configured — boolean only, never the key
        "ai_ready": bool(getattr(client, "api_key", None) or getattr(client, "auth_token", None)),
        "rag_ready": _rag_ready(st.index_path),
        "has_master": st.master is not None,
    }


@app.get("/")
def dashboard_page():
    page = _STATIC / "dashboard.html"
    if not page.exists():
        return {"detail": "Dashboard UI not found; use GET /api/jobs."}
    return FileResponse(page)


@app.get("/tailor")
def tailor_page():
    page = _STATIC / "index.html"
    if not page.exists():
        return {"detail": "UI not found; use POST /api/tailor."}
    return FileResponse(page)


@app.get("/capture")
def capture_page():
    page = _STATIC / "capture.html"
    if not page.exists():
        return {"detail": "Capture UI not found; use POST /api/capture-job."}
    return FileResponse(page)


@app.post("/api/tailor", response_model=TailorResponse)
def tailor_route(req: TailorRequest, request: Request):
    st = request.app.state

    if req.job_url:
        if not st.allow_url:
            raise HTTPException(
                400, "URL input is disabled. Set AUTO_APPLY_ALLOW_URL=1 to enable, or paste text."
            )
        job_text = _safe_fetch(req.job_url)
    else:
        job_text = req.job_text

    _require_key(st.client)
    retrieved_sources: list[str] = []
    try:
        if req.mode == "rag":
            if not _rag_ready(st.index_path):
                raise HTTPException(400, "RAG index not built. Run: python -m src.cli index sample-corpus/")
            chunks = rag.retrieve(job_text, index_path=st.index_path, k=req.top_k)
            retrieved_sources = [f"{c.source}{(' › ' + c.heading) if c.heading else ''}" for c in chunks]
            result, usage = tailor.tailor(job_text, experience=rag.format_context(chunks), client=st.client)
        else:
            if st.master is None:
                raise HTTPException(
                    400, "No master profile available. Add sample-corpus/ or master-resume.md."
                )
            result, usage = tailor.tailor(job_text, master_profile=st.master, client=st.client)
    except anthropic.AuthenticationError:
        raise HTTPException(503, "Tailoring unavailable: set ANTHROPIC_API_KEY on the server.")
    except RuntimeError as exc:
        raise HTTPException(502, f"Model did not return a usable result: {exc}")

    assert isinstance(result, TailoredApplication)
    return TailorResponse(
        **result.model_dump(), retrieved_sources=retrieved_sources, mode=req.mode, usage=usage
    )


# --------------------------------------------------------------------------- dashboard


class JobStatusUpdate(BaseModel):
    status: str


class JobNotesUpdate(BaseModel):
    notes: str = ""


class TrackerUpdate(BaseModel):
    next_action: str | None = None
    applied_at: str | None = None
    follow_up_at: str | None = None
    notes: str | None = None
    contact_name: str | None = None
    contact_email: str | None = None
    contact_url: str | None = None


class FollowUpAction(BaseModel):
    action: str = Field("done", pattern="^(done|snooze)$")
    days: int = 3


class DiligenceUpdate(BaseModel):
    # All optional — send only the fields you're changing.
    legitimacy_status: str | None = None
    review_signal: str | None = None
    source_confidence: str | None = None
    apply_recommendation: str | None = None
    priority: int | None = None
    diligence_notes: str | None = None


class TaskCreate(BaseModel):
    title: str
    job_id: int | None = None
    priority: int = 3
    due_at: str | None = None
    notes: str | None = None


class SearchRunCreate(BaseModel):
    query: str
    sources: list[str] = []
    location: str | None = None
    status: str = "planned"
    found_count: int = 0
    deduped_count: int = 0
    notes: str | None = None
    finished_at: str | None = None


class CaptureJobRequest(BaseModel):
    title: str
    company: str
    description: str
    url: str | None = None
    location: str | None = None
    remote: bool = True
    salary: str | None = None
    required_skills: list[str] = []
    source: str = "manual"
    source_id: str | None = None


class CaptureRequest(BaseModel):
    # Provide the posting as pasted text OR saved-page HTML (both land in `content`).
    content: str
    url: str | None = None
    source_hint: str | None = None
    preset: str = "cleared"
    save: bool = True  # False = "quick score only" (parse + score, don't persist)

    @model_validator(mode="after")
    def _has_content(self):
        if not (self.content or "").strip():
            raise ValueError("content is required (paste the job text or the saved page HTML).")
        return self


class RecruiterReplyRequest(BaseModel):
    message: str
    job_id: int | None = None
    intent: str = "interested"
    tone: str = "concise"
    phone: str | None = None
    availability: str | None = None
    late: bool = True
    save: bool = True

    @model_validator(mode="after")
    def _has_message(self):
        if not (self.message or "").strip():
            raise ValueError("message is required (paste the recruiter's message).")
        return self


class EmailParseRequest(BaseModel):
    text: str
    job_id: int | None = None
    source: str = "email"


class TextRequest(BaseModel):
    text: str | None = None


@app.get("/api/jobs")
def list_jobs(request: Request):
    """All tracked jobs + summary stats + the vocabularies the dashboard uses."""
    conn = store.connect(request.app.state.db_path)
    try:
        return {
            "jobs": store.dashboard_rows(conn),
            "stats": store.stats(conn),
            "statuses": list(store.JOB_STATUSES),
            "next_actions": list(store.NEXT_ACTIONS),
            "options": {
                "legitimacy_status": list(store.LEGITIMACY),
                "review_signal": list(store.REVIEW_SIGNAL),
                "source_confidence": list(store.SOURCE_CONFIDENCE),
                "apply_recommendation": list(store.APPLY_RECOMMENDATION),
            },
        }
    finally:
        conn.close()


@app.get("/api/queue")
def get_queue(request: Request):
    """The prioritized action queue: overdue/due follow-ups, interviews, apply-today, stale leads."""
    conn = store.connect(request.app.state.db_path)
    try:
        return {"items": store.action_queue(conn)}
    finally:
        conn.close()


@app.post("/api/jobs/{job_id}/follow-up")
def follow_up_action(job_id: int, body: FollowUpAction, request: Request):
    """One-click follow-up handling: 'done' logs + schedules the next cycle, 'snooze' pushes it out."""
    conn = store.connect(request.app.state.db_path)
    try:
        if body.action == "done":
            next_at = store.complete_follow_up(conn, job_id)
        else:
            next_at = store.snooze_follow_up(conn, job_id, days=body.days)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id, "action": body.action, "follow_up_at": next_at}


@app.post("/api/check-links")
def check_links(request: Request, only_unchecked: bool = False):
    """Verify every stored posting's URL; dead (404/410) links are flagged and drop from the queue."""
    conn = store.connect(request.app.state.db_path)
    try:
        return links.check_jobs(conn, only_unchecked=only_unchecked)
    finally:
        conn.close()


@app.post("/api/jobs/{job_id}/check-link")
def check_one_link(job_id: int, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        detail = store.job_detail(conn, job_id)
        if detail is None:
            raise HTTPException(404, "Job not found")
        result = links.check_url(detail["job"]["url"])
        store.set_link_status(conn, job_id, result["link_status"])
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id, **result}


@app.get("/api/board")
def get_board(request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        return {"columns": store.board(conn), "statuses": list(store.JOB_STATUSES)}
    finally:
        conn.close()


@app.get("/api/calendar")
def get_calendar(request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        return {"items": store.calendar_items(conn)}
    finally:
        conn.close()


@app.get("/api/tasks")
def list_tasks(request: Request, include_done: bool = False):
    conn = store.connect(request.app.state.db_path)
    try:
        return {"tasks": store.tasks(conn, include_done=include_done)}
    finally:
        conn.close()


@app.post("/api/tasks")
def create_task(body: TaskCreate, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        task_id = store.create_task(
            conn,
            body.title,
            job_id=body.job_id,
            priority=body.priority,
            due_at=body.due_at,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        conn.close()
    return {"ok": True, "task_id": task_id}


@app.post("/api/tasks/{task_id}/complete")
def complete_task(task_id: int, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        store.complete_task(conn, task_id)
    finally:
        conn.close()
    return {"ok": True, "task_id": task_id}


@app.get("/api/search-runs")
def list_search_runs(request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        return {"search_runs": store.search_runs(conn)}
    finally:
        conn.close()


@app.post("/api/search-runs")
def create_search_run(body: SearchRunCreate, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        run_id = store.record_search_run(conn, **body.model_dump())
    finally:
        conn.close()
    return {"ok": True, "search_run_id": run_id}


@app.post("/api/capture")
def capture_posting(body: CaptureRequest, request: Request):
    """Parse pasted text / saved HTML, score it (deterministic, offline), and optionally save.

    Powers Paste job / Import saved HTML / Quick score only. No API key needed for any of it.
    """
    parsed = capture.parse_job(body.content, url=body.url, source_hint=body.source_hint)
    breakdown = scoring.score_job(parsed, preset=body.preset)
    result = {
        "fields": parsed.model_dump(exclude={"raw"}),
        "breakdown": breakdown.model_dump(),
        "recommendation": breakdown.recommendation,
        "resume_variant": breakdown.resume_variant,
        "saved": False,
    }
    if body.save:
        conn = store.connect(request.app.state.db_path)
        try:
            result["job_id"] = store.record_capture(conn, parsed, breakdown)
            result["saved"] = True
        finally:
            conn.close()
    return result


@app.post("/api/recruiter/parse")
def recruiter_parse(body: TextRequest):
    """Extract recruiter name / company / email / phone / role from a pasted message."""
    return recruiter.parse_message((body.text if body else "") or "").model_dump()


@app.post("/api/recruiter/reply")
def recruiter_reply(body: RecruiterReplyRequest, request: Request):
    """Draft a reply in the user's voice for the chosen intent + tone; optionally log it to a job."""
    try:
        reply = recruiter.generate_reply(
            body.message,
            intent=body.intent,
            tone=body.tone,
            phone=body.phone,
            availability=body.availability,
            late=body.late,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    out = reply.model_dump()
    if body.save:
        conn = store.connect(request.app.state.db_path)
        try:
            out["message_id"] = store.record_recruiter_message(
                conn, reply, job_id=body.job_id, raw=body.message
            )
            if body.job_id and body.intent in (
                "interested",
                "follow_up",
                "ask_salary",
                "ask_remote",
                "ask_timeline",
            ):
                store.create_task(conn, reply.follow_up_task, job_id=body.job_id, priority=2)
        finally:
            conn.close()
    return out


@app.post("/api/capture-job")
def capture_job(body: CaptureJobRequest, request: Request):
    source_id = body.source_id
    if not source_id:
        seed = body.url or f"{body.company}:{body.title}:{body.description[:160]}"
        source_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    posting = ingest.JobPosting(
        source=body.source,
        source_id=source_id,
        url=body.url,
        title=body.title,
        company=body.company,
        location=body.location,
        remote=body.remote,
        salary=body.salary,
        required_skills=body.required_skills,
        description=body.description,
        fetched_at=datetime.now(UTC).isoformat(),
    )
    conn = store.connect(request.app.state.db_path)
    try:
        job_id = store.upsert_job(conn, posting)
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id, "source_id": source_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int, request: Request):
    """One job + its latest tailored application (fit explanation + evidence) for the drawer."""
    conn = store.connect(request.app.state.db_path)
    try:
        detail = store.job_detail(conn, job_id)
    finally:
        conn.close()
    if detail is None:
        raise HTTPException(404, "Job not found")
    return detail


@app.get("/api/jobs/{job_id}/analysis")
def list_job_analysis(job_id: int, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        if store.job_detail(conn, job_id) is None:
            raise HTTPException(404, "Job not found")
        return {
            "analysis": store.analysis_reports(conn, job_id),
            "events": store.job_events(conn, job_id=job_id),
        }
    finally:
        conn.close()


@app.post("/api/jobs/{job_id}/ats")
def ats_job(job_id: int, request: Request, body: TextRequest | None = None):
    conn = store.connect(request.app.state.db_path)
    try:
        detail = store.job_detail(conn, job_id)
        if detail is None:
            raise HTTPException(404, "Job not found")
        latest = store.latest_kit(conn, job_id)
        resume_text = (
            (body.text if body else None) or (latest or {}).get("resume_markdown") or request.app.state.master
        )
        if not resume_text:
            raise HTTPException(400, "No resume text available. Generate a kit or send {text}.")
        report = product.ats_report(detail["job"]["description"], resume_text)
        report["report_id"] = store.record_analysis(conn, job_id, "ats", report)
    finally:
        conn.close()
    return report


@app.post("/api/jobs/{job_id}/diff")
def diff_job_resume(job_id: int, request: Request, body: TextRequest | None = None):
    conn = store.connect(request.app.state.db_path)
    try:
        detail = store.job_detail(conn, job_id)
        if detail is None:
            raise HTTPException(404, "Job not found")
        latest = store.latest_kit(conn, job_id)
        tailored_text = (body.text if body else None) or (latest or {}).get("resume_markdown")
        if not request.app.state.master:
            raise HTTPException(400, "No master resume available for diff.")
        if not tailored_text:
            raise HTTPException(400, "No tailored resume available. Generate a kit or send {text}.")
        report = product.diff_report(request.app.state.master, tailored_text)
        report["report_id"] = store.record_analysis(conn, job_id, "resume_diff", report)
        store.record_resume_version(
            conn,
            job_id,
            label="latest tailored resume",
            content=tailored_text,
            kit_id=(latest or {}).get("id"),
            diff_summary={k: report[k] for k in ("added_lines", "removed_lines", "changed_lines")},
        )
    finally:
        conn.close()
    return report


@app.post("/api/jobs/{job_id}/interview-prep")
def interview_prep(job_id: int, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        detail = store.job_detail(conn, job_id)
        if detail is None:
            raise HTTPException(404, "Job not found")
        report = product.interview_prep(
            detail["job"], application=detail.get("application"), kit=store.latest_kit(conn, job_id)
        )
        report["report_id"] = store.record_analysis(conn, job_id, "interview_prep", report)
        store.create_task(
            conn,
            f"Prep interview for {detail['job']['company']}",
            job_id=job_id,
            priority=2,
            notes="Generated from interview prep report.",
        )
    finally:
        conn.close()
    return report


@app.post("/api/jobs/{job_id}/autofill-plan")
def autofill_plan(job_id: int, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        detail = store.job_detail(conn, job_id)
        if detail is None:
            raise HTTPException(404, "Job not found")
        report = product.autofill_plan(detail["job"], kit=store.latest_kit(conn, job_id))
        report["report_id"] = store.record_analysis(conn, job_id, "autofill_plan", report)
    finally:
        conn.close()
    return report


@app.post("/api/email-events/parse")
def parse_email_event(body: EmailParseRequest, request: Request):
    parsed = product.parse_email_event(body.text)
    conn = store.connect(request.app.state.db_path)
    try:
        event_id = store.record_job_event(
            conn,
            job_id=body.job_id,
            kind="email",
            source=body.source,
            subject=parsed["subject"],
            body_excerpt=parsed["body_excerpt"],
            inferred_status=parsed["inferred_status"],
        )
    finally:
        conn.close()
    return {"ok": True, "event_id": event_id, **parsed}


@app.post("/api/jobs/{job_id}/status")
def update_job_status(job_id: int, body: JobStatusUpdate, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        store.set_job_status(conn, job_id, body.status)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id, "status": body.status}


class DecisionRequest(BaseModel):
    decision: str  # apply | ask_recruiter | save | skip | follow_up


@app.post("/api/jobs/{job_id}/decision")
def job_decision(job_id: int, body: DecisionRequest, request: Request):
    """One-click next action from the dashboard — updates status, next action, and tasks."""
    conn = store.connect(request.app.state.db_path)
    try:
        if store.job_detail(conn, job_id) is None:
            raise HTTPException(404, "Job not found")
        return {"ok": True, **store.decide(conn, job_id, body.decision)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        conn.close()


@app.post("/api/jobs/{job_id}/notes")
def update_job_notes(job_id: int, body: JobNotesUpdate, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        store.set_job_notes(conn, job_id, body.notes)
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id}


@app.post("/api/jobs/{job_id}/tracker")
def update_job_tracker(job_id: int, body: TrackerUpdate, request: Request):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    conn = store.connect(request.app.state.db_path)
    try:
        store.update_tracker(conn, job_id, fields)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id, "updated": list(fields)}


@app.post("/api/jobs/{job_id}/diligence")
def update_job_diligence(job_id: int, body: DiligenceUpdate, request: Request):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    conn = store.connect(request.app.state.db_path)
    try:
        store.update_diligence(conn, job_id, fields)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id, "updated": list(fields)}


@app.post("/api/jobs/{job_id}/tailor")
def tailor_job(job_id: int, request: Request):
    """Tailor + score a stored job and persist the application (backfills fit on the dashboard)."""
    st = request.app.state
    _require_key(st.client)
    conn = store.connect(st.db_path)
    run_id = store.start_generation_run(
        conn, kind="tailor", job_id=job_id, model=getattr(st, "model", tailor.MODEL)
    )
    t0 = time.perf_counter()
    try:
        detail = store.job_detail(conn, job_id)
        if detail is None:
            raise HTTPException(404, "Job not found")
        job = detail["job"]
        retrieved = None
        try:
            if _rag_ready(st.index_path):
                mode = "rag"
                retrieved = rag.retrieve(job["description"], index_path=st.index_path, k=rag.DEFAULT_K)
                result, usage = tailor.tailor(
                    job["description"], experience=rag.format_context(retrieved), client=st.client
                )
            elif st.master is not None:
                mode = "full-profile"
                result, usage = tailor.tailor(job["description"], master_profile=st.master, client=st.client)
            else:
                raise HTTPException(400, "No RAG index and no master profile. Build an index or add one.")
        except anthropic.AuthenticationError:
            store.finish_generation_run(
                conn,
                run_id,
                status="failed",
                error="authentication",
                latency_ms=round((time.perf_counter() - t0) * 1000),
            )
            raise HTTPException(503, "Tailoring unavailable: set ANTHROPIC_API_KEY on the server.")
        except RuntimeError as exc:
            store.finish_generation_run(
                conn,
                run_id,
                status="failed",
                error=str(exc),
                latency_ms=round((time.perf_counter() - t0) * 1000),
            )
            raise HTTPException(502, f"Model did not return a usable result: {exc}")
        store.record_application(conn, job_id, result, mode=mode, retrieved=retrieved)
        store.finish_generation_run(
            conn,
            run_id,
            status="done",
            usage=usage,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
    finally:
        conn.close()
    return {"ok": True, "job_id": job_id, "fit_score": result.fit_score, "mode": mode}


@app.post("/api/jobs/{job_id}/kit")
def generate_kit_route(job_id: int, request: Request):
    """Generate an application kit for a job, written to out/kits/<slug>-<id>/ (gitignored)."""
    from . import kit as kitmod
    from .pipeline import slug

    st = request.app.state
    _require_key(st.client)
    conn = store.connect(st.db_path)
    run_id = store.start_generation_run(
        conn, kind="kit", job_id=job_id, model=getattr(st, "model", kitmod.MODEL), mode="rag"
    )
    t0 = time.perf_counter()
    try:
        detail = store.job_detail(conn, job_id)
        if detail is None:
            store.finish_generation_run(
                conn,
                run_id,
                status="failed",
                error="Job not found",
                latency_ms=round((time.perf_counter() - t0) * 1000),
            )
            raise HTTPException(404, "Job not found")
        job = detail["job"]
        if not _rag_ready(st.index_path):
            store.finish_generation_run(
                conn,
                run_id,
                status="failed",
                error="RAG index not built",
                latency_ms=round((time.perf_counter() - t0) * 1000),
            )
            raise HTTPException(400, "RAG index not built. Run: python -m src.cli index sample-corpus/")
        chunks = rag.retrieve(job["description"], index_path=st.index_path, k=rag.DEFAULT_K)
        kit, usage = kitmod.generate_kit(job["description"], rag.format_context(chunks), client=st.client)
        dest = _HERE / "out" / "kits" / f"{slug(job['company'])}-{job_id}"
        kitmod.write_kit(dest, kit, job_meta={"company": job["company"], "job_id": job_id})
        kit_id = store.record_kit(conn, job_id, kit, out_dir=str(dest))
        store.finish_generation_run(
            conn,
            run_id,
            status="done",
            usage=usage,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
    except anthropic.AuthenticationError:
        store.finish_generation_run(
            conn,
            run_id,
            status="failed",
            error="authentication",
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        raise HTTPException(503, "Kit generation unavailable: set ANTHROPIC_API_KEY on the server.")
    except RuntimeError as exc:
        store.finish_generation_run(
            conn,
            run_id,
            status="failed",
            error=str(exc),
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        raise HTTPException(502, f"Model did not return a usable kit: {exc}")
    finally:
        conn.close()
    return {
        "ok": True,
        "job_id": job_id,
        "kit_id": kit_id,
        "out_dir": str(dest),
        "usage": usage,
        "kit": kit.model_dump(),
    }


@app.get("/api/jobs/{job_id}/kits")
def list_job_kits(job_id: int, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        return {"kits": store.kit_rows(conn, job_id)}
    finally:
        conn.close()


@app.get("/api/jobs/{job_id}/kit/latest")
def latest_job_kit(job_id: int, request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        k = store.latest_kit(conn, job_id)
    finally:
        conn.close()
    if k is None:
        raise HTTPException(404, "No kit for this job")
    return k


@app.get("/api/digest")
def get_digest(request: Request, days: int = 7):
    """Weekly ops digest: pipeline movement, applications, follow-ups, source funnel, gen cost."""
    conn = store.connect(request.app.state.db_path)
    try:
        data = store.digest(conn, days=days)
        return {**data, "markdown": store.format_digest(data)}
    finally:
        conn.close()


class AgentAsk(BaseModel):
    question: str

    @model_validator(mode="after")
    def _q(self):
        if not (self.question or "").strip():
            raise ValueError("question is required.")
        return self


@app.post("/api/agent")
def agent_ask(body: AgentAsk, request: Request):
    """JobProof Copilot: a tool-using agent answers over your pipeline (Anthropic-backed)."""
    st = request.app.state
    _require_key(st.client)
    if getattr(st, "backend", "anthropic") != "anthropic":
        raise HTTPException(
            400, "The agent needs the Anthropic backend (tool use); local backend is text-only."
        )
    try:
        return agent.run_agent(body.question, client=st.client, db_path=st.db_path, index_path=st.index_path)
    except anthropic.AuthenticationError:
        raise HTTPException(503, "Agent unavailable: set ANTHROPIC_API_KEY on the server.")


@app.get("/api/generation-runs")
def list_generation_runs(request: Request):
    conn = store.connect(request.app.state.db_path)
    try:
        return {"runs": store.generation_runs(conn)}
    finally:
        conn.close()
