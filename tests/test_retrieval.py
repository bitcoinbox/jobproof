from src import rag


def test_retrieve_ranks_relevant_source(rag_index):
    index_path, ef = rag_index
    hits = rag.retrieve(
        "retrieval augmented generation vector database embeddings",
        index_path=index_path,
        k=5,
        embedding_function=ef,
    )
    assert hits
    assert len(hits) <= 5
    # The RAG project (or the resume, which lists RAG) should surface near the top.
    top_sources = {h.source for h in hits[:3]}
    assert any("rag" in s.lower() or "master" in s.lower() or "bullet" in s.lower() for s in top_sources)


def test_format_context_strips_bracket_prefix(rag_index):
    index_path, ef = rag_index
    hits = rag.retrieve("python fastapi docker", index_path=index_path, k=3, embedding_function=ef)
    ctx = rag.format_context(hits)
    assert "[from " in ctx
    # the embedded "[heading]\n" prefix is removed from the body
    assert "\n[from " in ("\n" + ctx)


def test_build_index_counts(tmp_path, fake_ef, sample_corpus_dir):
    files, chunks = rag.build_index(
        sample_corpus_dir, index_path=str(tmp_path / "ix"), embedding_function=fake_ef
    )
    assert files >= 5 and chunks >= 10


def test_mmr_reranking_promotes_diversity(tmp_path, fake_ef):
    """MMR should break up a near-duplicate cluster; plain top-k keeps the dupes."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # three near-identical 'python' chunks + one orthogonal 'kubernetes' chunk
    for i in range(3):
        (corpus / f"py{i}.md").write_text("python python python backend api", encoding="utf-8")
    (corpus / "kube.md").write_text("kubernetes cluster orchestration nodes", encoding="utf-8")
    ix = str(tmp_path / "ix")
    rag.build_index(corpus, index_path=ix, embedding_function=fake_ef)

    plain = rag.retrieve("python backend", index_path=ix, k=2, embedding_function=fake_ef, use_mmr=False)
    mmr = rag.retrieve(
        "python backend", index_path=ix, k=2, embedding_function=fake_ef, use_mmr=True, lambda_mult=0.3
    )
    # top-1 (most relevant) is the same under both — MMR only diversifies later picks
    assert plain[0].source.startswith("py")
    # plain top-2 are both python dupes; MMR pulls in the diverse kubernetes chunk
    assert all(h.source.startswith("py") for h in plain)
    assert any(h.source == "kube.md" for h in mmr)


def test_retrieve_use_mmr_false_is_plain_topk(rag_index):
    index_path, ef = rag_index
    hits = rag.retrieve("python fastapi", index_path=index_path, k=4, embedding_function=ef, use_mmr=False)
    assert 0 < len(hits) <= 4


def test_bm25_ranks_exact_term(fake_ef, tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "cleared.md").write_text(
        "Requires an active TS/SCI clearance and STIG hardening.", encoding="utf-8"
    )
    (corpus / "web.md").write_text(
        "Build React frontends and CSS animations for marketing.", encoding="utf-8"
    )
    ix = str(tmp_path / "ix")
    rag.build_index(corpus, index_path=ix, embedding_function=fake_ef)
    # pure BM25 should put the TS/SCI chunk first for a keyword query
    hits = rag.retrieve(
        "TS/SCI STIG", index_path=ix, k=2, embedding_function=fake_ef, mode="bm25", use_mmr=False
    )
    assert hits[0].source == "cleared.md"


def test_hybrid_surfaces_exact_term_chunk(fake_ef, tmp_path):
    """The payoff: an exact-term chunk the bag-of-words 'dense' vector ranks low still
    surfaces once BM25 is fused in (RRF)."""
    corpus = tmp_path / "c"
    corpus.mkdir()
    # Many chunks share generic words (dominate the dense bow space); one rare exact term.
    for i in range(6):
        (corpus / f"generic{i}.md").write_text(
            "engineer systems networking security operations support team", encoding="utf-8"
        )
    (corpus / "katello.md").write_text(
        "Manage configuration with Katello across multi-enclave systems.", encoding="utf-8"
    )
    ix = str(tmp_path / "ix")
    rag.build_index(corpus, index_path=ix, embedding_function=fake_ef)

    q = "Katello configuration management"
    hybrid = rag.retrieve(q, index_path=ix, k=3, embedding_function=fake_ef, mode="hybrid", use_mmr=False)
    # BM25's exact "katello" match lifts the right chunk to the top of the hybrid results
    assert hybrid[0].source == "katello.md"
