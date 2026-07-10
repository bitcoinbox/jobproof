"""End-to-end orchestration: plan → ingest → (retrieve) → tailor → score → persist → report.

This is a deterministic workflow, not an open-ended agent: each stage is a plain
function call and the only model-driven decision is the tool-call parse inside
ingest. That's the right altitude for the task — the control flow is known, so it
lives in code, and the LLM is used where judgement is actually needed (parsing and
tailoring). Everything composes the existing pieces: ingest.ingest, rag.retrieve,
tailor.tailor, store.*.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import ingest, rag, store, tailor

if TYPE_CHECKING:
    import anthropic

    from .tailor import TailoredApplication


def slug(name: str) -> str:
    """File-safe slug for an output directory name (shared by cli + pipeline)."""
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[\s_-]+", "-", name) or "job"


def _write_outputs(
    dest: Path,
    result: TailoredApplication,
    *,
    mode: str,
    usage: dict,
    retrieved_sources: list[str] | None = None,
    job_file: str | None = None,
) -> dict:
    """Write resume.md, cover-letter.md, report.json for one tailoring; return the report dict.

    Single source of truth for output artifacts so the `tailor` and `run` commands
    produce identical files.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "resume.md").write_text(result.resume_markdown, encoding="utf-8")
    (dest / "cover-letter.md").write_text(result.cover_letter, encoding="utf-8")
    report = {
        "job_file": job_file,
        "mode": mode,
        "fit_score": result.fit_score,
        "matched_keywords": result.matched_keywords,
        "missing_keywords": result.missing_keywords,
        "application_note": result.application_note,
        "retrieved_sources": retrieved_sources or [],
        "usage": usage,
    }
    (dest / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _source_labels(retrieved: list[rag.Retrieved] | None) -> list[str]:
    return [f"{r.source}{(' › ' + r.heading) if r.heading else ''}" for r in (retrieved or [])]


def run(
    *,
    source: str = "fixture",
    query: str = "",
    limit: int = 10,
    use_rag: bool = False,
    index_path: str = rag.DEFAULT_INDEX_PATH,
    k: int = rag.DEFAULT_K,
    db_path: str = store.DEFAULT_DB,
    out: Path = Path("out"),
    client: anthropic.Anthropic | None = None,
    master_text: str | None = None,
    use_llm_parse: bool = True,
    fixtures_dir: Path | None = None,
    provider: str = "remoteok",
    board: str | None = None,
    write_report: bool = True,
) -> dict:
    """Run the full pipeline. Returns a summary dict."""
    import anthropic

    client = client or anthropic.Anthropic()
    conn = store.connect(db_path)
    mode = "rag" if use_rag else "full-profile"

    print(
        f"Plan: source={source} query={query!r} limit={limit} mode={mode} model={tailor.MODEL} db={db_path}"
    )

    postings = ingest.ingest(
        source,
        query,
        limit=limit,
        client=client,
        conn=conn,
        use_llm=use_llm_parse,
        fixtures_dir=fixtures_dir,
        provider=provider,
        board=board,
    )
    print(f"Ingested {len(postings)} new posting(s).")

    reports = []
    for posting in postings:
        job_id = store.upsert_job(conn, posting)
        retrieved = None
        if use_rag:
            retrieved = rag.retrieve(posting.description, index_path=index_path, k=k)
            result, usage = tailor.tailor(
                posting.description, experience=rag.format_context(retrieved), client=client
            )
        else:
            result, usage = tailor.tailor(posting.description, master_profile=master_text, client=client)

        dest = out / slug(posting.company or posting.title)
        report = _write_outputs(
            dest,
            result,
            mode=mode,
            usage=usage,
            retrieved_sources=_source_labels(retrieved),
            job_file=posting.url,
        )
        store.record_application(conn, job_id, result, mode=mode, out_dir=str(dest), retrieved=retrieved)
        reports.append({**report, "company": posting.company, "title": posting.title})
        print(f"  [{posting.company}] {posting.title}  fit {result.fit_score}/100 -> {dest}/")

    if write_report and reports:
        Path("tracker.md").write_text(store.report_markdown(conn), encoding="utf-8")
        Path("tracker.csv").write_text(store.report_csv(conn), encoding="utf-8")
        print("Wrote tracker.md / tracker.csv")

    if reports:
        print("\n=== ranked by fit ===")
        for r in sorted(reports, key=lambda r: r["fit_score"], reverse=True):
            print(f"  {r['fit_score']:>3}/100  {r['company']} — {r['title']}")

    conn.close()
    return {"ingested": len(postings), "applications": len(reports), "mode": mode}
