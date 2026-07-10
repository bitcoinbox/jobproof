from src import eval as ev
from tests.conftest import AUTO_APPLY, make_tailored

# --- retrieval metrics (pure, offline) ---------------------------------------
# Worked example used across the cases below:
#   retrieved (by rank, with a duplicate source A): A, A, B, C, D
#   unique-by-source:                               A, B, C, D
#   relevant set:                                   {A, D, Z}   (Z is never retrieved)

RETRIEVED = ["A", "A", "B", "C", "D"]
RELEVANT = {"A", "D", "Z"}


def test_precision_at_k():
    # top-3 unique = [A, B, C]; only A is relevant -> 1/3
    assert abs(ev.precision_at_k(RETRIEVED, RELEVANT, 3) - 1 / 3) < 1e-9
    # top-5 unique is only [A, B, C, D] (dedup) -> A,D relevant of 4 -> 2/4
    assert ev.precision_at_k(RETRIEVED, RELEVANT, 5) == 0.5


def test_recall_at_k():
    # top-3 unique = {A, B, C}; relevant {A, D, Z}; found A -> 1/3
    assert abs(ev.recall_at_k(RETRIEVED, RELEVANT, 3) - 1 / 3) < 1e-9
    # top-4 finds A and D -> 2/3 (Z is unretrievable)
    assert abs(ev.recall_at_k(RETRIEVED, RELEVANT, 4) - 2 / 3) < 1e-9


def test_reciprocal_rank():
    assert ev.reciprocal_rank(RETRIEVED, RELEVANT) == 1.0  # A is rank 1
    assert abs(ev.reciprocal_rank(["X", "Y", "D"], RELEVANT) - 1 / 3) < 1e-9  # D is rank 3
    assert ev.reciprocal_rank(["X", "Y"], RELEVANT) == 0.0  # no relevant hit


def test_hit_rate():
    assert ev.hit_rate_at_k(RETRIEVED, RELEVANT, 3) == 1.0
    assert ev.hit_rate_at_k(["X", "Y", "Z2"], RELEVANT, 3) == 0.0


def test_recall_empty_relevant_is_one():
    # no relevant sources labeled -> vacuously perfect recall (don't divide by zero)
    assert ev.recall_at_k(RETRIEVED, set(), 3) == 1.0


def test_retrieval_metrics_aggregate():
    m = ev.retrieval_metrics(RETRIEVED, RELEVANT, ks=(3,))
    assert m["mrr"] == 1.0
    assert m["hit@3"] == 1.0
    assert abs(m["precision@3"] - round(1 / 3, 3)) < 1e-9
    assert abs(m["recall@3"] - round(1 / 3, 3)) < 1e-9


def test_sources_of_adapter():
    class R:  # minimal stand-in for rag.Retrieved
        def __init__(self, source):
            self.source = source

    assert ev.sources_of([R("a.md"), R("b.md")]) == ["a.md", "b.md"]


def test_gate_passes_when_all_floors_met():
    ok, failures = ev.gate({"recall@5": 0.82, "mrr": 1.0}, {"recall@5": 0.7, "mrr": 0.9})
    assert ok and failures == []


def test_gate_fails_below_floor():
    ok, failures = ev.gate({"recall@5": 0.5, "mrr": 1.0}, {"recall@5": 0.7, "mrr": 0.9})
    assert not ok
    assert len(failures) == 1 and "recall@5" in failures[0]


def test_gate_fails_when_metric_missing():
    # A gate you can't pass by simply not measuring the metric.
    ok, failures = ev.gate({"recall@5": 0.9}, {"mean_faithfulness": 0.9})
    assert not ok and "not measured" in failures[0]


def test_committed_thresholds_are_valid():
    import json

    data = json.loads((AUTO_APPLY / "evals" / "thresholds.json").read_text())
    floors = {k: v for k, v in data.items() if not k.startswith("_")}
    assert floors and all(isinstance(v, (int, float)) for v in floors.values())


def test_evaluate_retrieval_hermetic(monkeypatch, tmp_path):
    """evaluate_retrieval() scores against labels without any index or API call."""

    class R:
        def __init__(self, source):
            self.source = source

    # Fake retriever: perfect for job A (returns the relevant source first),
    # poor for job B (returns only an irrelevant source).
    def fake_retrieve(job_text, *, index_path=None, k=12):
        return [R("00.md"), R("noise.md")] if "AAA" in job_text else [R("noise.md")]

    monkeypatch.setattr(ev.rag, "retrieve", fake_retrieve)
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "a.txt").write_text("AAA role")
    (tmp_path / "jobs" / "b.txt").write_text("BBB role")
    ds = tmp_path / "evals" / "ds.jsonl"
    ds.parent.mkdir()
    ds.write_text(
        '{"id":"a","job_file":"jobs/a.txt","relevant_sources":["00.md"]}\n'
        '{"id":"b","job_file":"jobs/b.txt","relevant_sources":["05.md"]}\n'
        '{"id":"c","job_file":"jobs/a.txt","ideal_keywords":["x"]}\n'  # no labels -> skipped
    )
    res = ev.evaluate_retrieval(ds, base_dir=tmp_path, ks=(3,))
    assert res["n"] == 2  # row c skipped (no relevant_sources)
    by_id = {r["id"]: r for r in res["per_query"]}
    assert by_id["a"]["recall@3"] == 1.0 and by_id["a"]["mrr"] == 1.0  # found it, rank 1
    assert by_id["b"]["recall@3"] == 0.0  # missed entirely
    assert res["mean"]["recall@3"] == 0.5
    assert "MEAN" in ev.format_retrieval_report(res)


def test_keyword_match_rate():
    r = make_tailored(
        resume_markdown="Built FastAPI services with Docker and LLM APIs.", matched_keywords=["Python", "RAG"]
    )
    rate, hit, miss = ev.keyword_match_rate(
        ["Python", "RAG", "Docker", "FastAPI", "Kubernetes", "LLM APIs"], r
    )
    assert round(rate, 3) == round(5 / 6, 3)
    assert "Kubernetes" in miss and "Docker" in hit


def test_keyword_match_rate_empty_ideal():
    assert ev.keyword_match_rate([], make_tailored())[0] == 1.0


def test_load_dataset_skips_comments(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text('# comment\n{"id":"a","job_file":"jobs/x.txt","ideal_keywords":["Python"]}\n\n')
    rows = ev.load_dataset(p)
    assert len(rows) == 1 and rows[0]["id"] == "a"


def test_committed_dataset_is_valid():
    rows = ev.load_dataset(AUTO_APPLY / "evals" / "labeled.jsonl")
    assert rows
    for row in rows:
        assert (AUTO_APPLY / row["job_file"]).exists(), row["job_file"]
        assert isinstance(row["ideal_keywords"], list)


def test_evaluate_orchestration_hermetic(monkeypatch, tmp_path):
    """evaluate() runs both arms without Chroma or the API (both patched)."""
    monkeypatch.setattr(ev.rag, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(ev.rag, "format_context", lambda chunks: "EXPERIENCE")

    def fake_tailor(job_text, master_profile=None, *, experience=None, client=None):
        # rag arm scores a touch higher so the delta is exercised
        score = 85 if experience else 80
        return make_tailored(fit_score=score), {"output_tokens": 1}

    monkeypatch.setattr(ev.tailor, "tailor", fake_tailor)

    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        '{"id":"a","job_file":"jobs/applied-ai-engineer-remote.txt","ideal_keywords":["Python","RAG"]}\n'
    )
    res = ev.evaluate(ds, master_text="MASTER", base_dir=AUTO_APPLY, client=object())
    assert set(res["summary"]) == {"full-profile", "rag"}
    assert res["delta"]["fit"] == 5.0
    assert "RAG vs full-profile" in ev.format_report(res)


def test_evaluate_with_faithfulness_hermetic(monkeypatch, tmp_path):
    """score_faithfulness=True attaches a grounding score via the judge (mocked)."""
    monkeypatch.setattr(ev.rag, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(ev.rag, "format_context", lambda chunks: "EXPERIENCE")
    monkeypatch.setattr(
        ev.tailor, "tailor", lambda *a, **k: (make_tailored(fit_score=80), {"output_tokens": 1})
    )
    monkeypatch.setattr(
        ev.judge,
        "judge_faithfulness",
        lambda resume, evidence, *, client=None: (0.75, ev.judge.FaithfulnessReport(verdicts=[])),
    )

    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"id":"a","job_file":"jobs/applied-ai-engineer-remote.txt","ideal_keywords":["Python"]}\n')
    res = ev.evaluate(
        ds, master_text="M", base_dir=AUTO_APPLY, client=object(), score_faithfulness=True
    )
    assert res["summary"]["rag"]["mean_faithfulness"] == 0.75
    assert res["rows"][0]["rag"]["faithfulness"] == 0.75
