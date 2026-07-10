"""CLI for JobProof (role-fit scoring + evidence-grounded tailoring).

Subcommands:
    index           build the RAG experience index from a corpus
    tailor          tailor your resume to a job file/folder (default if first arg is a path)
    ingest          fetch + parse remote postings into the SQLite store
    import-targets  import startup target roles (Wellfound) from a fixture (idempotent)
    demo            seed the cockpit with sample jobs (offline, no API key)
    kit             generate a full application kit (resume, cover, why-me, DM, stories, evidence map)
    run             full pipeline: ingest -> (RAG) -> tailor -> persist -> tracker
    eval            measure RAG vs full-profile (keyword-match + fit) on a labeled set
    report          regenerate the tracker, or advance an application's status
    capture         capture a job from pasted text / saved HTML: extract, score, track
    check-links     verify posting URLs and flag dead ones (drops them from the queue)
    serve           run the FastAPI web UI / API

Examples:
    python -m src.cli index sample-corpus/
    python -m src.cli jobs/acme.txt --rag
    python -m src.cli ingest --source fixture --query "ai engineer" --no-parse
    python -m src.cli run --source remotive --query "llm" --limit 5 --rag
    python -m src.cli eval --dataset evals/labeled.jsonl -k 12
    python -m src.cli report --set 3=applied
    python -m src.cli serve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from . import eval as evalmod
from . import ingest, links, llm, pipeline, rag, store
from . import kit as kitmod
from .pipeline import _write_outputs, slug
from .tailor import tailor

JOB_SUFFIXES = {".txt", ".md"}
_HERE = Path(__file__).resolve().parent.parent  # project root

# Master-profile fallback chain: your real resume first (gitignored), then the
# committed fictional persona so a fresh public clone runs out of the box.
_MASTER_FALLBACKS = [
    _HERE / "master-resume.md",
    _HERE.parent / "resume-master.md",
    _HERE / "sample-corpus" / "00-master-resume.md",
]


def _resolve_master(path: Path) -> Path:
    if path.exists():
        return path
    for candidate in _MASTER_FALLBACKS:
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"Master profile not found at {path} or any fallback. "
        "Add master-resume.md (your real resume, gitignored) or use the committed sample-corpus/."
    )


def _collect_jobs(target: Path) -> list[Path]:
    if target.is_dir():
        jobs = sorted(p for p in target.iterdir() if p.suffix.lower() in JOB_SUFFIXES)
        if not jobs:
            raise SystemExit(f"No .txt/.md job files found in {target}")
        return jobs
    if not target.exists():
        raise SystemExit(f"Job file not found: {target}")
    return [target]


def _process(
    job_path,
    outdir,
    client,
    *,
    master=None,
    use_rag=False,
    index_path=rag.DEFAULT_INDEX_PATH,
    k=rag.DEFAULT_K,
):
    job_text = job_path.read_text(encoding="utf-8")
    retrieved_sources: list[str] = []
    if use_rag:
        chunks = rag.retrieve(job_text, index_path=index_path, k=k)
        retrieved_sources = [f"{c.source}{(' › ' + c.heading) if c.heading else ''}" for c in chunks]
        result, usage = tailor(job_text, experience=rag.format_context(chunks), client=client)
    else:
        result, usage = tailor(job_text, master, client=client)

    dest = outdir / slug(job_path.stem)
    report = _write_outputs(
        dest,
        result,
        mode="rag" if use_rag else "full-profile",
        usage=usage,
        retrieved_sources=retrieved_sources,
        job_file=str(job_path),
    )
    cached = usage["cache_read_input_tokens"]
    print(f"\n[{job_path.name}]  fit {result.fit_score}/100  ->  {dest}/")
    print(f"  matched : {', '.join(result.matched_keywords[:8]) or '—'}")
    print(f"  missing : {', '.join(result.missing_keywords[:8]) or '—'}")
    print(f"  note    : {result.application_note}")
    if use_rag:
        print(f"  retrieved: {', '.join(retrieved_sources[:5]) or '—'}")
    print(
        f"  tokens  : in {usage['input_tokens']} | cache_read {cached} | out {usage['output_tokens']}"
        + ("  (cache hit)" if cached else "")
    )
    return report


# ------------------------------------------------------------------ subcommands


def _cmd_index(argv):
    p = argparse.ArgumentParser(prog="src.cli index", description="Build the RAG experience index.")
    p.add_argument(
        "corpus",
        nargs="?",
        default="sample-corpus",
        help="A .md/.txt file or folder (default: sample-corpus/).",
    )
    p.add_argument("--index-path", default=rag.DEFAULT_INDEX_PATH)
    a = p.parse_args(argv)
    print(f"Indexing {a.corpus} -> {a.index_path} ...")
    files, chunks = rag.build_index(Path(a.corpus), index_path=a.index_path)
    print(f"Indexed {chunks} chunks from {files} file(s). Tailor with: --rag")
    return 0


def _cmd_tailor(argv):
    p = argparse.ArgumentParser(prog="src.cli", description="Tailor a resume to a job posting.")
    p.add_argument("job", help="A job .txt/.md file, or a folder of them.")
    p.add_argument("--master", default="master-resume.md")
    p.add_argument("--out", default="out")
    p.add_argument("--rag", action="store_true")
    p.add_argument("--index-path", default=rag.DEFAULT_INDEX_PATH)
    p.add_argument("-k", "--top-k", type=int, default=rag.DEFAULT_K)
    a = p.parse_args(argv)

    master_text = None if a.rag else _resolve_master(Path(a.master)).read_text(encoding="utf-8")
    jobs = _collect_jobs(Path(a.job))
    outdir = Path(a.out)
    client = llm.make_client()
    mode = "RAG (retrieved experience)" if a.rag else "cached full profile"
    print(f"Tailoring {len(jobs)} job(s) with {llm.active_model()} [{mode}] ...")
    reports = []
    for jp in jobs:
        try:
            reports.append(
                _process(
                    jp, outdir, client, master=master_text, use_rag=a.rag, index_path=a.index_path, k=a.top_k
                )
            )
        except Exception as exc:
            print(f"\n[{jp.name}]  ERROR: {exc}", file=sys.stderr)
    if reports:
        reports.sort(key=lambda r: r["fit_score"], reverse=True)
        print("\n=== ranked by fit ===")
        for r in reports:
            print(f"  {r['fit_score']:>3}/100  {Path(r['job_file']).name}")
    return 0


def _cmd_ingest(argv):
    p = argparse.ArgumentParser(
        prog="src.cli ingest", description="Fetch + parse remote postings into the store."
    )
    p.add_argument("--source", default="fixture", choices=list(ingest.SOURCES))
    p.add_argument("--query", default="")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--no-parse", action="store_true", help="Heuristic parse (no LLM, no key).")
    p.add_argument(
        "--provider", default="remoteok", help="Which fixture provider file (fixture source only)."
    )
    p.add_argument(
        "--board",
        help="Company board token/thread id for greenhouse/lever/ashby/hn (e.g. --board stripe).",
    )
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--write-jobs", help="Also write each posting's text to this folder as <id>.txt.")
    a = p.parse_args(argv)

    conn = store.connect(a.db)
    client = None if a.no_parse else anthropic.Anthropic()
    postings = ingest.ingest(
        a.source,
        a.query,
        limit=a.limit,
        client=client,
        conn=conn,
        use_llm=not a.no_parse,
        provider=a.provider,
        board=a.board,
    )
    print(f"Ingested {len(postings)} new posting(s) into {a.db}.")
    for pst in postings:
        print(f"  [{pst.source_id}] {pst.title} @ {pst.company}")
    if a.write_jobs and postings:
        d = Path(a.write_jobs)
        d.mkdir(parents=True, exist_ok=True)
        for pst in postings:
            (d / f"{slug(pst.company)}-{pst.source_id}.txt").write_text(pst.description, encoding="utf-8")
        print(f"Wrote {len(postings)} job text file(s) to {d}/")
    return 0


def _cmd_run(argv):
    p = argparse.ArgumentParser(
        prog="src.cli run", description="Full pipeline: ingest -> tailor -> persist -> tracker."
    )
    p.add_argument("--source", default="fixture", choices=list(ingest.SOURCES))
    p.add_argument("--query", default="")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--rag", action="store_true")
    p.add_argument("--index-path", default=rag.DEFAULT_INDEX_PATH)
    p.add_argument("-k", "--top-k", type=int, default=rag.DEFAULT_K)
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--out", default="out")
    p.add_argument("--master", default="master-resume.md")
    p.add_argument("--no-parse", action="store_true")
    p.add_argument("--provider", default="remoteok")
    p.add_argument("--board", help="Company board token/thread id for greenhouse/lever/ashby/hn.")
    a = p.parse_args(argv)

    master_text = None if a.rag else _resolve_master(Path(a.master)).read_text(encoding="utf-8")
    summary = pipeline.run(
        source=a.source,
        query=a.query,
        limit=a.limit,
        use_rag=a.rag,
        index_path=a.index_path,
        k=a.top_k,
        db_path=a.db,
        out=Path(a.out),
        master_text=master_text,
        use_llm_parse=not a.no_parse,
        provider=a.provider,
        board=a.board,
    )
    print(f"\nDone: {summary}")
    return 0


def _cmd_eval(argv):
    p = argparse.ArgumentParser(
        prog="src.cli eval",
        description="Evaluate the pipeline: retrieval quality (no key) + RAG-vs-full generation.",
    )
    p.add_argument("--dataset", default="evals/labeled.jsonl")
    p.add_argument("--master", default="master-resume.md")
    p.add_argument("--index-path", default=rag.DEFAULT_INDEX_PATH)
    p.add_argument("-k", "--top-k", type=int, default=rag.DEFAULT_K)
    p.add_argument("--out", default="evals/report.json")
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only score retrieval quality (deterministic, no API key) — ideal for CI.",
    )
    p.add_argument(
        "--faithfulness",
        action="store_true",
        help="Also judge whether the RAG résumé is grounded in the evidence (extra model call/job).",
    )
    p.add_argument(
        "--gate",
        nargs="?",
        const="evals/thresholds.json",
        default=None,
        metavar="THRESHOLDS_JSON",
        help="Fail (exit 1) if a mean metric drops below its floor. Default floors: evals/thresholds.json.",
    )
    a = p.parse_args(argv)

    # Retrieval eval first: deterministic, needs only the index — no API key.
    retr = evalmod.evaluate_retrieval(a.dataset, index_path=a.index_path, k=a.top_k)
    print("Retrieval quality (vs labeled relevant_sources):\n")
    print(evalmod.format_retrieval_report(retr))

    def _run_gate(metrics: dict) -> int:
        """Check metrics against the thresholds file; print + return process exit code."""
        if not a.gate:
            return 0
        thresholds = {
            k: v for k, v in json.loads(Path(a.gate).read_text()).items() if not k.startswith("_")
        }
        ok, failures = evalmod.gate(metrics, thresholds)
        if ok:
            print(f"\n✅ gate PASSED — all {len(thresholds)} floors met ({a.gate})")
            return 0
        print(f"\n❌ gate FAILED ({a.gate}):")
        for f in failures:
            print(f"   - {f}")
        return 1

    if a.retrieval_only:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"retrieval": retr}, indent=2), encoding="utf-8")
        print(f"\nWrote {a.out}")
        return _run_gate(retr["mean"])

    master_text = _resolve_master(Path(a.master)).read_text(encoding="utf-8")
    results = evalmod.evaluate(
        a.dataset,
        master_text=master_text,
        index_path=a.index_path,
        k=a.top_k,
        score_faithfulness=a.faithfulness,
    )
    results["retrieval"] = retr
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + evalmod.format_report(results))
    print(f"\nWrote {a.out}")
    # Gate on retrieval means + faithfulness (if it was measured).
    gate_metrics = dict(retr["mean"])
    if "mean_faithfulness" in results["summary"].get("rag", {}):
        gate_metrics["mean_faithfulness"] = results["summary"]["rag"]["mean_faithfulness"]
    return _run_gate(gate_metrics)


def _cmd_report(argv):
    p = argparse.ArgumentParser(
        prog="src.cli report", description="Regenerate the tracker, or set an application status."
    )
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--set", help="Advance status, e.g. --set 3=applied")
    p.add_argument("--digest", action="store_true", help="Print a weekly ops digest instead of the tracker.")
    p.add_argument("--days", type=int, default=7, help="Digest window in days (default 7).")
    p.add_argument("--out", help="Write the digest/tracker markdown to this path too.")
    a = p.parse_args(argv)

    conn = store.connect(a.db)
    if a.set:
        app_id, _, status = a.set.partition("=")
        store.set_status(conn, int(app_id), status.strip())
        print(f"Application {app_id} -> {status.strip()}")
    if a.digest:
        md = store.format_digest(store.digest(conn, days=a.days))
        if a.out:
            Path(a.out).write_text(md, encoding="utf-8")
        print(md)
        if a.out:
            print(f"Wrote {a.out}")
        return 0
    Path("tracker.md").write_text(store.report_markdown(conn), encoding="utf-8")
    Path("tracker.csv").write_text(store.report_csv(conn), encoding="utf-8")
    print(store.report_markdown(conn))
    print("Wrote tracker.md / tracker.csv")
    return 0


def _cmd_import_targets(argv):
    p = argparse.ArgumentParser(
        prog="src.cli import-targets",
        description="Import startup target roles (e.g. Wellfound) from a fixture into the store. Idempotent.",
    )
    p.add_argument("--provider", default="startup-targets", help="Fixture name (default: startup-targets).")
    p.add_argument("--db", default=store.DEFAULT_DB)
    a = p.parse_args(argv)
    conn = store.connect(a.db)
    postings = ingest.ingest(
        "fixture", "", limit=1000, client=None, conn=conn, use_llm=False, provider=a.provider
    )
    print(f"Imported {len(postings)} new target(s) into {a.db}.")
    for pst in postings:
        print(f"  [{pst.legitimacy_status}] {pst.title} @ {pst.company}")
    return 0


def _cmd_demo(argv):
    p = argparse.ArgumentParser(
        prog="src.cli demo",
        description="Seed the cockpit with sample jobs (offline, no API key) so the dashboard is populated.",
    )
    p.add_argument("--db", default=store.DEFAULT_DB)
    a = p.parse_args(argv)
    conn = store.connect(a.db)
    total = 0
    for provider in ("remoteok", "remotive", "startup-targets"):
        total += len(
            ingest.ingest("fixture", "", limit=100, client=None, conn=conn, use_llm=False, provider=provider)
        )
    s = store.stats(conn)
    print(f"Seeded {total} new sample job(s) → {a.db} ({s['total']} total, {s['apply_today']} apply-today).")
    print("Optional RAG index (one-time, small local model):  python -m src.cli index sample-corpus/")
    print("Open the cockpit:  python -m src.cli serve   →  http://127.0.0.1:8000")
    return 0


def _cmd_kit(argv):
    p = argparse.ArgumentParser(
        prog="src.cli kit",
        description="Generate an application kit (resume, cover, why-me, DM, stories, evidence map).",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--job-id", type=int, help="A job id from the store.")
    g.add_argument("--job", help="A job posting .txt/.md file.")
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--index-path", default=rag.DEFAULT_INDEX_PATH)
    p.add_argument("-k", "--top-k", type=int, default=rag.DEFAULT_K)
    p.add_argument("--out", default="out/kits")
    p.add_argument(
        "--pdf", action="store_true", help="Also render an upload-ready resume.pdf (needs Chrome)."
    )
    a = p.parse_args(argv)

    if a.job_id:
        conn = store.connect(a.db)
        detail = store.job_detail(conn, a.job_id)
        if not detail:
            raise SystemExit(f"Job {a.job_id} not found in {a.db}")
        job = detail["job"]
        job_text, company, jid = job["description"], job["company"], job["id"]
    else:
        job_text = Path(a.job).read_text(encoding="utf-8")
        company, jid = Path(a.job).stem, "file"

    chunks = rag.retrieve(job_text, index_path=a.index_path, k=a.top_k)
    experience = rag.format_context(chunks)
    kit, usage = kitmod.generate_kit(job_text, experience, client=llm.make_client())
    dest = Path(a.out) / f"{slug(company)}-{jid}"
    written = kitmod.write_kit(dest, kit, job_meta={"company": company, "job_id": jid}, make_pdf=a.pdf)
    print(f"Wrote application kit to {dest}/ ({len(written)} files; {usage['output_tokens']} out tokens):")
    for w in written:
        print(f"  - {w.name}")
    if a.pdf and not any(w.name == "resume.pdf" for w in written):
        print("  (resume.pdf skipped — no headless Chrome/Chromium found on PATH)")
    return 0


def _cmd_capture(argv):
    p = argparse.ArgumentParser(
        prog="src.cli capture",
        description="Capture a job from pasted text or a saved HTML file: extract, score, track.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="A saved .html page or a .txt/.md paste of the posting.")
    g.add_argument("--text", help="The posting text inline.")
    p.add_argument("--url", help="Source URL (helps source detection + job id).")
    p.add_argument("--source", help="Override source: clearancejobs|dice|linkedin|company_site|manual.")
    p.add_argument("--preset", default="cleared", help="Scoring preset (default: cleared).")
    p.add_argument("--score-only", action="store_true", help="Print extraction + score; don't save.")
    p.add_argument("--db", default=store.DEFAULT_DB)
    a = p.parse_args(argv)

    from . import capture as capmod
    from . import scoring

    content = Path(a.file).read_text(encoding="utf-8", errors="replace") if a.file else a.text
    job = capmod.parse_job(content, url=a.url, source_hint=a.source)
    fb = scoring.score_job(job, preset=a.preset)
    print(f"\n{job.title or '(no title)'} @ {job.company or '(no company)'}  [{job.source}/{job.parser}]")
    print(f"  location : {job.location or '—'}  ·  {job.work_mode}  ·  clearance: {job.clearance or 'none'}")
    print(f"  salary   : {job.salary or '—'}   ·  job id: {job.job_id or '—'}")
    print(f"  FIT {fb.overall}/100 → {fb.recommendation.upper()}  ·  resume: {fb.resume_variant}")
    print(
        f"  skill {fb.skill_match.score} · clearance {fb.clearance_match.score} · salary {fb.salary_fit.score}"
        f" · remote {fb.remote_fit.score} · seniority {fb.seniority_fit.score} · passion {fb.passion_fit.score}"
    )
    if fb.risk_flags:
        print(f"  risks    : {', '.join(fb.risk_flags)}")
    print(f"  why apply: {fb.why_apply}")
    if not a.score_only:
        conn = store.connect(a.db)
        jid = store.record_capture(conn, job, fb)
        print(f"\nSaved as job #{jid} → {a.db}")
    else:
        print("\n(score-only; not saved)")
    return 0


def _cmd_check_links(argv):
    p = argparse.ArgumentParser(
        prog="src.cli check-links",
        description="Verify each stored posting's URL; flag dead (404/410) links so they drop from the queue.",
    )
    p.add_argument("--db", default=store.DEFAULT_DB)
    p.add_argument("--only-unchecked", action="store_true", help="Skip links already checked.")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    conn = store.connect(a.db)
    summary = links.check_jobs(conn, only_unchecked=a.only_unchecked, limit=a.limit)
    print(
        f"Checked {summary['checked']} link(s): "
        f"{summary['alive']} alive · {summary['dead']} dead · {summary['unknown']} unknown."
    )
    for d in summary["dead_jobs"]:
        print(f"  DEAD  [{d['job_id']}] {d['title']} @ {d['company']} — {d['url']}")
    return 0


def _cmd_serve(argv):
    p = argparse.ArgumentParser(prog="src.cli serve", description="Run the FastAPI web UI / API.")
    p.add_argument("--host", default="127.0.0.1")
    # Honor $PORT (Railway / Render / most PaaS inject it) so the container needs no start command.
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    a = p.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("The web UI needs fastapi + uvicorn: pip install -r requirements.txt")
    print(f"Serving on http://{a.host}:{a.port}  (Ctrl-C to stop)")
    uvicorn.run("src.api:app", host=a.host, port=a.port, reload=False)
    return 0


COMMANDS = {
    "index": _cmd_index,
    "tailor": _cmd_tailor,
    "ingest": _cmd_ingest,
    "import-targets": _cmd_import_targets,
    "demo": _cmd_demo,
    "kit": _cmd_kit,
    "run": _cmd_run,
    "eval": _cmd_eval,
    "report": _cmd_report,
    "capture": _cmd_capture,
    "check-links": _cmd_check_links,
    "serve": _cmd_serve,
}


def main(argv=None):
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    return _cmd_tailor(argv)  # default: bare path still tailors


if __name__ == "__main__":
    raise SystemExit(main())
