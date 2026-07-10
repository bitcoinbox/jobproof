"""Evaluation harness: does RAG actually help, and by how much?

For each labeled (job, ideal_keywords) pair, tailor the resume twice — once with the
full master profile, once with RAG-retrieved experience — and measure:
  - keyword-match rate: fraction of the human-labeled ideal keywords the tailored
    resume surfaces (recall of what a strong candidate's resume should hit)
  - fit_score: the model's own honest 0-100 self-assessment

Aggregate per arm and report the delta. The keyword metric is a pure function
(``keyword_match_rate``) so it's unit-testable without any model call.

Honesty note: the committed dataset is tiny and fictional — this demonstrates the
harness and yields a reproducible number on the sample corpus, not a benchmark. On a
one-page corpus RAG often ties the full-profile path (it pays off as the corpus grows);
the harness is the instrument that tells you which regime you're in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from . import judge, rag, tailor

if TYPE_CHECKING:
    import anthropic

    from .tailor import TailoredApplication


def load_dataset(path: str | Path) -> list[dict]:
    """Read a JSONL dataset of {id, job_file, ideal_keywords, min_expected_fit?} rows."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def keyword_match_rate(ideal: list[str], result: TailoredApplication) -> tuple[float, list[str], list[str]]:
    """Fraction of ideal keywords present in the tailored resume (+ matched_keywords).

    Case-insensitive substring match. Returns (rate, hit, missed). Pure — no model.
    """
    if not ideal:
        return 1.0, [], []
    haystack = (result.resume_markdown + " " + " ".join(result.matched_keywords)).lower()
    hit = [kw for kw in ideal if kw.lower() in haystack]
    missed = [kw for kw in ideal if kw.lower() not in haystack]
    return len(hit) / len(ideal), hit, missed


# ---- Retrieval quality metrics (pure; no model, no index) -------------------
# Retrieval is the half of RAG you can measure *deterministically*. Given a job and a
# human label of which corpus sources are actually relevant, did the retriever rank them
# near the top? These are the standard information-retrieval metrics, computed at the
# *source* level — chunks from the same file collapse to one source, since relevance is
# labeled per source. All pure functions: unit-testable with zero API calls.


def sources_of(chunks) -> list[str]:
    """Ordered source ids from a list of ``rag.Retrieved`` (anything with ``.source``)."""
    return [c.source for c in chunks]


def _unique(retrieved: list[str]) -> list[str]:
    """Collapse retrieved chunks to unique sources, preserving first-seen (rank) order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in retrieved:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the top-k retrieved sources, the fraction that are relevant.

    Low precision = you're spending the context window (and tokens) on irrelevant chunks.
    """
    if k <= 0:
        return 0.0
    topk = _unique(retrieved)[:k]
    if not topk:
        return 0.0
    return sum(1 for s in topk if s in relevant) / len(topk)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the sources that *should* have been found, the fraction that made the top-k.

    Low recall = the retriever silently dropped a chunk the answer needed. The metric
    that catches the most dangerous RAG failure: a confident answer missing its evidence.
    """
    if not relevant:
        return 1.0
    topk = set(_unique(retrieved)[:k])
    return len(topk & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant source (0.0 if none). Rewards ranking a hit high."""
    for i, s in enumerate(_unique(retrieved), start=1):
        if s in relevant:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant source is in the top-k, else 0.0."""
    return 1.0 if set(_unique(retrieved)[:k]) & relevant else 0.0


def retrieval_metrics(retrieved: list[str], relevant: set[str], ks: tuple[int, ...] = (3, 5)) -> dict:
    """Aggregate the standard IR metrics for one query into a flat dict."""
    out: dict[str, float] = {"mrr": round(reciprocal_rank(retrieved, relevant), 3)}
    for k in ks:
        out[f"precision@{k}"] = round(precision_at_k(retrieved, relevant, k), 3)
        out[f"recall@{k}"] = round(recall_at_k(retrieved, relevant, k), 3)
        out[f"hit@{k}"] = round(hit_rate_at_k(retrieved, relevant, k), 3)
    return out


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def evaluate(
    dataset_path: str | Path,
    *,
    client: anthropic.Anthropic | None = None,
    master_text: str,
    index_path: str = rag.DEFAULT_INDEX_PATH,
    k: int = rag.DEFAULT_K,
    base_dir: Path | None = None,
    score_faithfulness: bool = False,
) -> dict:
    """Run both arms over the dataset; return {summary, delta, rows}.

    With ``score_faithfulness=True`` the RAG résumé is additionally judged for grounding
    against the retrieved evidence (an extra model call per job — see ``judge.py``).
    """
    import anthropic

    client = client or anthropic.Anthropic()
    base_dir = base_dir or Path(dataset_path).resolve().parent.parent
    rows_in = load_dataset(dataset_path)

    rows_out = []
    per_arm = {"full-profile": {"kw": [], "fit": []}, "rag": {"kw": [], "fit": [], "faith": []}}

    for row in rows_in:
        job_text = (base_dir / row["job_file"]).read_text(encoding="utf-8")
        ideal = row.get("ideal_keywords", [])

        full_res, _ = tailor.tailor(job_text, master_profile=master_text, client=client)
        retrieved = rag.retrieve(job_text, index_path=index_path, k=k)
        rag_context = rag.format_context(retrieved)
        rag_res, _ = tailor.tailor(job_text, experience=rag_context, client=client)

        full_kw, _, full_missed = keyword_match_rate(ideal, full_res)
        rag_kw, _, rag_missed = keyword_match_rate(ideal, rag_res)
        per_arm["full-profile"]["kw"].append(full_kw)
        per_arm["full-profile"]["fit"].append(full_res.fit_score)
        per_arm["rag"]["kw"].append(rag_kw)
        per_arm["rag"]["fit"].append(rag_res.fit_score)

        rag_row = {"keyword_match": round(rag_kw, 3), "fit": rag_res.fit_score, "missed": rag_missed}
        if score_faithfulness:
            faith, frep = judge.judge_faithfulness(rag_res.resume_markdown, rag_context, client=client)
            per_arm["rag"]["faith"].append(faith)
            rag_row["faithfulness"] = faith
            rag_row["unsupported"] = judge.unsupported_claims(frep)

        rows_out.append(
            {
                "id": row.get("id"),
                "full_profile": {
                    "keyword_match": round(full_kw, 3),
                    "fit": full_res.fit_score,
                    "missed": full_missed,
                },
                "rag": rag_row,
                "min_expected_fit": row.get("min_expected_fit"),
            }
        )

    summary = {
        arm: {"mean_keyword_match": _mean(d["kw"]), "mean_fit": _mean(d["fit"]), "n": len(d["kw"])}
        for arm, d in per_arm.items()
    }
    if per_arm["rag"]["faith"]:
        summary["rag"]["mean_faithfulness"] = _mean(per_arm["rag"]["faith"])
    delta = {
        "keyword_match": round(
            summary["rag"]["mean_keyword_match"] - summary["full-profile"]["mean_keyword_match"], 3
        ),
        "fit": round(summary["rag"]["mean_fit"] - summary["full-profile"]["mean_fit"], 3),
    }
    return {"summary": summary, "delta": delta, "rows": rows_out}


def format_report(results: dict) -> str:
    """Render the eval results as a console table + a one-line headline."""
    s = results["summary"]
    d = results["delta"]
    lines = [
        "arm           |  kw-match | mean-fit |  n",
        "--------------+----------+----------+----",
    ]
    for arm in ("full-profile", "rag"):
        a = s[arm]
        lines.append(f"{arm:<13} |   {a['mean_keyword_match']:.3f}  |  {a['mean_fit']:.1f}   | {a['n']:>2}")
    sign = "+" if d["keyword_match"] >= 0 else ""
    fsign = "+" if d["fit"] >= 0 else ""
    lines.append("")
    lines.append(
        f"RAG vs full-profile: keyword-match {sign}{d['keyword_match']:.3f}, "
        f"fit {fsign}{d['fit']:.1f}  (fictional sample corpus)"
    )
    return "\n".join(lines)


def evaluate_retrieval(
    dataset_path: str | Path,
    *,
    index_path: str = rag.DEFAULT_INDEX_PATH,
    k: int = rag.DEFAULT_K,
    base_dir: Path | None = None,
    ks: tuple[int, ...] = (3, 5),
) -> dict:
    """Score retrieval quality against the labeled ``relevant_sources``.

    This is the deterministic half of RAG and needs **no API key** — only the vector
    index. For each labeled job it retrieves top-k, then compares the retrieved sources
    to the human ground truth with precision/recall/MRR/hit-rate. Returns
    ``{per_query, mean, ks, k, n}``; rows without ``relevant_sources`` are skipped.
    """
    base_dir = base_dir or Path(dataset_path).resolve().parent.parent
    rows = [r for r in load_dataset(dataset_path) if r.get("relevant_sources")]

    per_query: list[dict] = []
    agg: dict[str, list[float]] = {}
    for row in rows:
        job_text = (base_dir / row["job_file"]).read_text(encoding="utf-8")
        retrieved = sources_of(rag.retrieve(job_text, index_path=index_path, k=k))
        m = retrieval_metrics(retrieved, set(row["relevant_sources"]), ks=ks)
        per_query.append({"id": row.get("id"), **m})
        for key, val in m.items():
            agg.setdefault(key, []).append(val)

    mean = {key: _mean(vals) for key, vals in agg.items()}
    return {"per_query": per_query, "mean": mean, "ks": list(ks), "k": k, "n": len(rows)}


def format_retrieval_report(results: dict) -> str:
    """Render per-query + mean retrieval metrics as a console table."""
    rows = results["per_query"]
    if not rows:
        return "retrieval eval: no rows have `relevant_sources` labeled."
    keys = [c for c in rows[0] if c != "id"]
    w = 26
    header = "query".ljust(w) + " | " + " | ".join(c.ljust(11) for c in keys)
    out = [header, "-" * len(header)]
    for r in rows:
        out.append(str(r["id"]).ljust(w) + " | " + " | ".join(f"{r[c]:<11.3f}" for c in keys))
    out.append("-" * len(header))
    mean = results["mean"]
    out.append("MEAN".ljust(w) + " | " + " | ".join(f"{mean[c]:<11.3f}" for c in keys))
    out.append(f"\n(top-k={results['k']}, n={results['n']} labeled jobs, fictional sample corpus)")
    return "\n".join(out)


def gate(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    """Regression gate: assert each measured metric meets its floor.

    ``metrics`` is a flat ``{name: value}`` (e.g. ``results["mean"]`` plus an optional
    ``mean_faithfulness``); ``thresholds`` is ``{name: floor}``. Returns
    ``(passed, failures)``. A threshold whose metric is missing is itself a failure —
    you don't get to silently skip a gate by not measuring it.
    """
    failures: list[str] = []
    for name, floor in thresholds.items():
        val = metrics.get(name)
        if val is None:
            failures.append(f"{name}: not measured (gate expected ≥ {floor})")
        elif val < floor:
            failures.append(f"{name}: {val:.3f} < {floor} floor")
    return (not failures), failures
