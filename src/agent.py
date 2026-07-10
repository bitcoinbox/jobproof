"""JobProof Copilot — a tool-using agent over your pipeline.

This is the canonical agent loop, not a single model call:

    user question
        │
        ▼
   ┌─► the LLM decides: answer, or call a tool? ──(answer)──► done
   │        │ (tool_use)
   │        ▼
   │   run the tool (search jobs / score a posting / search experience)
   │        │
   └────────┘  feed the result back, loop (bounded by max_steps)

It exposes real functions over the existing store / scoring / RAG modules as tools, lets
the model choose which to call and when, executes them, feeds results back, and repeats
until the model answers or we hit the step cap. Every tool call is recorded in a `trace`
so you can see exactly how the agent reasoned — the inspectable transcript is the point.

Anthropic tool use (mirrors the proven pattern in ingest.py): `messages.create` with a
`tools` schema; when `stop_reason == "tool_use"`, run each tool and return a `tool_result`
block referencing its `tool_use_id`; thinking is disabled so the message history stays
simple (no thinking-block preservation needed for a tool-routing agent).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

from . import capture, rag, scoring, store, tailor

MODEL = tailor.MODEL
MAX_STEPS = 6
MAX_TOOL_CHARS = 6000  # cap tool output fed back to the model

SYSTEM = """\
You are JobProof Copilot, a sharp assistant for one cleared senior AI/network engineer's job
search. Answer the user's question by calling the available tools to get real data from their
pipeline — never invent jobs, scores, or facts. Prefer remote/hybrid, senior, well-paid, cleared
or AI/software roles; flag help-desk/field-tech/low-salary roles as beneath them. Be concise and
specific, cite job ids and companies, and end with a clear recommended next action.
"""

TOOLS = [
    {
        "name": "search_jobs",
        "description": "Search the user's tracked jobs. Filter by free-text query (company/title), "
        "status (new/interested/to_apply/applied/interview/offer/rejected/skipped), and/or apply "
        "recommendation (apply_today/consider/wait/skip). Returns compact rows sorted by fit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Substring match on company or title."},
                "status": {"type": "string"},
                "recommendation": {"type": "string"},
                "limit": {"type": "integer", "description": "Max rows (default 8)."},
            },
        },
    },
    {
        "name": "get_job",
        "description": "Full detail for one job id: fields, clearance, salary, and the fit breakdown.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "score_job_text",
        "description": "Parse and score an arbitrary pasted job posting (not yet tracked). Returns the "
        "extracted fields + fit breakdown + recommendation + which resume to use.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "search_experience",
        "description": "Hybrid search over the candidate's own experience corpus (their background), "
        "to ground claims about what they've actually done. Returns relevant snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "Number of snippets (default 5)."},
            },
            "required": ["query"],
        },
    },
]


def _run_tool(name: str, args: dict, *, conn, index_path: str) -> dict:
    """Dispatch a tool call to the real backend. Always returns a JSON-able dict."""
    if name == "search_jobs":
        rows = store.dashboard_rows(conn)
        q = (args.get("query") or "").lower()
        st = args.get("status")
        rec = args.get("recommendation")
        out = []
        for r in rows:
            if q and q not in f"{r['company']} {r['title']}".lower():
                continue
            if st and r["status"] != st:
                continue
            if rec and r["apply_recommendation"] != rec:
                continue
            out.append(
                {
                    "id": r["id"],
                    "company": r["company"],
                    "title": r["title"],
                    "status": r["status"],
                    "work_mode": r.get("work_mode"),
                    "clearance": r.get("clearance"),
                    "salary": r.get("salary"),
                    "quick_fit": r.get("quick_fit"),
                    "apply_recommendation": r["apply_recommendation"],
                    "recommended_variant": r.get("recommended_variant"),
                }
            )
        out.sort(key=lambda x: (x["quick_fit"] is None, -(x["quick_fit"] or 0)))
        return {"count": len(out), "jobs": out[: int(args.get("limit", 8))]}

    if name == "get_job":
        detail = store.job_detail(conn, int(args["job_id"]))
        if not detail:
            return {"error": f"no job with id {args.get('job_id')}"}
        j, fb = detail["job"], detail.get("fit_breakdown")
        return {
            "id": j["id"],
            "company": j["company"],
            "title": j["title"],
            "status": j["status"],
            "clearance": j.get("clearance"),
            "work_mode": j.get("work_mode"),
            "salary": j.get("salary"),
            "fit": (
                {
                    "overall": fb["overall"],
                    "recommendation": fb["recommendation"],
                    "resume_variant": fb["resume_variant"],
                    "why_apply": fb["why_apply"],
                    "why_caution": fb.get("why_caution"),
                    "risk_flags": fb.get("risk_flags", []),
                }
                if fb
                else None
            ),
        }

    if name == "score_job_text":
        job = capture.parse_job(args.get("text", ""))
        fb = scoring.score_job(job)
        return {
            "title": job.title,
            "company": job.company,
            "clearance": job.clearance,
            "work_mode": job.work_mode,
            "salary": job.salary,
            "overall": fb.overall,
            "recommendation": fb.recommendation,
            "resume_variant": fb.resume_variant,
            "why_apply": fb.why_apply,
            "why_caution": fb.why_caution,
            "risk_flags": fb.risk_flags,
        }

    if name == "search_experience":
        try:
            chunks = rag.retrieve(args["query"], index_path=index_path, k=int(args.get("k", 5)))
        except SystemExit as exc:
            return {"error": str(exc)}
        return {
            "snippets": [{"source": c.source, "heading": c.heading, "text": c.text[:400]} for c in chunks]
        }

    return {"error": f"unknown tool {name}"}


def run_agent(
    question: str,
    *,
    client: anthropic.Anthropic,
    db_path: str = store.DEFAULT_DB,
    index_path: str = rag.DEFAULT_INDEX_PATH,
    max_steps: int = MAX_STEPS,
) -> dict:
    """Run the tool-using loop. Returns {answer, trace, steps, truncated}."""
    conn = store.connect(db_path)
    messages: list[dict] = [{"role": "user", "content": question}]
    trace: list[dict] = []
    try:
        for _ in range(max_steps):
            resp = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                thinking={"type": "disabled"},
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    output = _run_tool(block.name, dict(block.input), conn=conn, index_path=index_path)
                    trace.append({"tool": block.name, "input": dict(block.input), "output": output})
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(output)[:MAX_TOOL_CHARS],
                        }
                    )
                messages.append({"role": "user", "content": results})
                continue
            answer = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
            return {"answer": answer.strip(), "trace": trace, "steps": len(trace), "truncated": False}
        return {
            "answer": "I hit the step limit before finishing — try a narrower question.",
            "trace": trace,
            "steps": len(trace),
            "truncated": True,
        }
    finally:
        conn.close()
