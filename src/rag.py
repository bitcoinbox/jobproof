"""RAG over the candidate's experience corpus.

The MVP sends the *entire* master profile to the LLM on every job. That works
for a one-pager, but it doesn't scale: as you accumulate project deep-dives,
past bullet variants, and accomplishment notes, stuffing all of it into every
prompt gets noisy and expensive, and the model has to find the relevant bits
itself.

This module adds retrieval. It chunks an experience corpus (one or many
markdown/text files), embeds the chunks into a local, persisted Chroma
collection, and for a given job posting returns only the most relevant chunks.
The tailoring prompt then carries *retrieved experience for this job* instead of
the whole profile.

Embeddings: Chroma's built-in local model (all-MiniLM-L6-v2 via onnxruntime).
No extra API key, runs offline after the one-time model download. Tailoring
itself still uses an LLM (see tailor.py). To swap in a hosted embedder
(e.g. Voyage AI, Anthropic's recommended partner) pass a Chroma
`embedding_function` to `_collection` — the rest of the pipeline is unchanged.

Trade-off worth knowing (and worth saying in an interview): the MVP caches the
master profile as a stable prompt prefix, so every job after the first reuses it
cheaply. Retrieved context changes per job, so the RAG path gives that cache up
in exchange for a smaller, sharper prompt. Use RAG when the corpus is larger
than one page; use the cached full-profile path when it isn't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_SUFFIXES = {".md", ".txt"}
COLLECTION_NAME = "experience"
DEFAULT_INDEX_PATH = ".chroma"
DEFAULT_K = 12

# Chunking targets (characters). ~1000 chars ≈ 250 tokens — small enough that a
# retrieved chunk is one coherent idea, large enough to keep context with it.
TARGET_CHARS = 1000
MAX_CHARS = 1600
OVERLAP_CHARS = 150


@dataclass
class Chunk:
    text: str  # the chunk body, prefixed with its heading trail
    source: str  # file the chunk came from
    heading: str  # nearest markdown heading ("" if none)


def _heading_level(line: str) -> int:
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else 0


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Split one document into heading-aware chunks.

    Tracks the current heading trail as it walks the lines, packs paragraphs up
    to TARGET_CHARS (never past MAX_CHARS), and carries a small character
    overlap between consecutive chunks so a fact split across a boundary still
    retrieves. Each chunk's stored text is prefixed with its heading trail so the
    embedding captures section context, not just the raw paragraph.
    """
    chunks: list[Chunk] = []
    trail: list[str] = []  # current heading path, e.g. ["Experience", "DoD"]
    buf: list[str] = []
    buf_len = 0

    def heading_str() -> str:
        return " › ".join(trail)

    def flush() -> None:
        nonlocal buf, buf_len
        body = "\n".join(buf).strip()
        if not body:
            buf, buf_len = [], 0
            return
        head = heading_str()
        prefixed = f"[{head}]\n{body}" if head else body
        chunks.append(Chunk(text=prefixed, source=source, heading=head))
        # Keep a tail of the body as overlap for the next chunk.
        tail = body[-OVERLAP_CHARS:] if len(body) > OVERLAP_CHARS else ""
        buf = [tail] if tail else []
        buf_len = len(tail)

    for raw in text.splitlines():
        line = raw.rstrip()
        level = _heading_level(line)
        if level:
            # Heading boundary: flush what we have, then update the trail.
            flush()
            buf, buf_len = [], 0
            title = line[level:].strip()
            trail = trail[: level - 1] + [title]
            continue

        if not line.strip():
            # Paragraph break — flush if we're already over target.
            if buf_len >= TARGET_CHARS:
                flush()
            elif buf:
                buf.append("")
                buf_len += 1
            continue

        # A single line longer than the max: hard-split it so one giant paragraph
        # can't become one giant chunk.
        if len(line) > MAX_CHARS:
            if buf:
                flush()
            for i in range(0, len(line), TARGET_CHARS):
                buf.append(line[i : i + TARGET_CHARS])
                buf_len += len(line[i : i + TARGET_CHARS])
                flush()
            continue

        if buf_len + len(line) > MAX_CHARS and buf:
            flush()
        buf.append(line)
        buf_len += len(line) + 1
        if buf_len >= TARGET_CHARS:
            flush()

    flush()
    return [c for c in chunks if c.text.strip()]


def collect_corpus(path: Path) -> list[Path]:
    """Resolve a corpus path (file or directory) to a list of source files."""
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in CORPUS_SUFFIXES)
        if not files:
            raise SystemExit(f"No .md/.txt files found in corpus directory {path}")
        return files
    if not path.exists():
        raise SystemExit(
            f"Corpus not found: {path}. Put your experience files in experience/ "
            "(master resume + project notes + bullet variants), or pass a path. "
            "A generic example lives in sample-corpus/."
        )
    return [path]


def _collection(index_path: str, embedding_function=None):
    """Open (or create) the persisted Chroma collection.

    Imported lazily so the MVP path doesn't pay chromadb's import cost.
    `embedding_function` is a seam: pass a deterministic one in tests to keep
    retrieval hermetic (no model download); leave None to use Chroma's local default.
    """
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit("RAG needs chromadb. Install it with: pip install -r requirements.txt") from exc
    client = chromadb.PersistentClient(path=index_path)
    kw = {"name": COLLECTION_NAME}
    if embedding_function is not None:
        kw["embedding_function"] = embedding_function
    return client.get_or_create_collection(**kw)


def build_index(
    corpus: Path, index_path: str = DEFAULT_INDEX_PATH, embedding_function=None
) -> tuple[int, int]:
    """Chunk the corpus and (re)build the Chroma collection. Returns (files, chunks)."""
    import chromadb

    files = collect_corpus(corpus)
    all_chunks: list[Chunk] = []
    for f in files:
        all_chunks.extend(chunk_markdown(f.read_text(encoding="utf-8"), f.name))
    if not all_chunks:
        raise SystemExit(f"Corpus {corpus} produced no chunks.")

    client = chromadb.PersistentClient(path=index_path)
    # Rebuild from scratch so re-indexing never leaves stale or duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    kw = {"name": COLLECTION_NAME}
    if embedding_function is not None:
        kw["embedding_function"] = embedding_function
    collection = client.get_or_create_collection(**kw)

    collection.add(
        ids=[f"{c.source}::{i}" for i, c in enumerate(all_chunks)],
        documents=[c.text for c in all_chunks],
        metadatas=[{"source": c.source, "heading": c.heading} for c in all_chunks],
    )
    return len(files), len(all_chunks)


@dataclass
class Retrieved:
    text: str
    source: str
    heading: str
    distance: float


def _cosine(a, b) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _mmr_order(rels: list[float], embs: list, k: int, lambda_mult: float) -> list[int]:
    """Maximal Marginal Relevance: pick k items balancing relevance vs. diversity.

    relevance(i)   = caller-supplied score (min-max normalized here)
    diversity pen. = max cosine similarity to anything already selected
    score(i)       = λ·relevance(i) − (1−λ)·diversity_penalty(i)

    First pick is the most relevant (λ·rel, no penalty), so top-1 is unchanged; later
    picks avoid near-duplicates of what's already in the context.
    """
    lo, hi = min(rels), max(rels)
    span = (hi - lo) or 1.0
    rels = [(r - lo) / span for r in rels]

    selected: list[int] = []
    candidates = list(range(len(embs)))
    while candidates and len(selected) < k:
        best, best_score = candidates[0], -1e9
        for i in candidates:
            penalty = max((_cosine(embs[i], embs[j]) for j in selected), default=0.0)
            score = lambda_mult * rels[i] - (1 - lambda_mult) * penalty
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        candidates.remove(best)
    return selected


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#./]+", (text or "").lower())


def _bm25_rank(query: str, ids: list[str], docs: list[str]) -> list[str]:
    """Rank chunk ids by BM25 over the whole corpus (exact-term / keyword matching).

    BM25 catches what dense embeddings blur — literal tokens like 'TS/SCI', 'STIG',
    'Katello', a specific cert — which is exactly what defense/infra postings hinge on.
    Pure-Python, no extra dependency.
    """
    import math
    from collections import Counter

    k1, b = 1.5, 0.75
    toks = [_tokenize(d) for d in docs]
    n = len(docs)
    avgdl = (sum(len(t) for t in toks) / n) if n else 1.0
    df: Counter = Counter()
    for t in toks:
        df.update(set(t))
    q_terms = set(_tokenize(query))

    scored: list[tuple[float, str]] = []
    for idx, t in enumerate(toks):
        tf = Counter(t)
        dl = len(t)
        s = 0.0
        for w in q_terms:
            f = tf.get(w, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scored.append((s, ids[idx]))
    scored.sort(key=lambda x: -x[0])
    return [i for _, i in scored]


def _rrf(*ranked_lists: list[str], rrf_k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: combine ranked id-lists without score normalization."""
    score: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, i in enumerate(lst):
            score[i] = score.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


def retrieve(
    job_text: str,
    index_path: str = DEFAULT_INDEX_PATH,
    k: int = DEFAULT_K,
    embedding_function=None,
    *,
    mode: str = "hybrid",
    use_mmr: bool = True,
    fetch_k: int | None = None,
    lambda_mult: float = 0.7,
) -> list[Retrieved]:
    """Return the k experience chunks most relevant to this job posting.

    Pipeline (all on by default): **hybrid** retrieval fuses dense (vector) and BM25
    (keyword) rankings with Reciprocal Rank Fusion — so semantic matches AND exact terms
    (TS/SCI, STIG, a cert) both surface — then **MMR** reranks the fused pool for a
    relevant *and* non-redundant final set. `mode` ∈ {"hybrid","dense","bm25"};
    `use_mmr=False` skips diversity reranking.
    """
    collection = _collection(index_path, embedding_function=embedding_function)
    n = collection.count()
    if n == 0:
        raise SystemExit(
            f"No experience index at {index_path}. Build one first:\n"
            "  python -m src.cli index experience/        # or a single file"
        )

    # Whole corpus (texts + embeddings) — personal-scale, so one read is fine and gives
    # BM25 its document frequencies and MMR its vectors without a second round-trip.
    allc = collection.get(include=["documents", "metadatas", "embeddings"])
    ids, docs, metas = allc["ids"], allc["documents"], allc["metadatas"]
    embs = allc.get("embeddings")
    if embs is None or len(embs) != len(ids):
        embs = [None] * len(ids)
    by_id = {i: (d, m, e) for i, d, m, e in zip(ids, docs, metas, embs)}

    dense_ids: list[str] = []
    dist_by_id: dict[str, float] = {}
    if mode in ("hybrid", "dense"):
        res = collection.query(query_texts=[job_text], n_results=n, include=["distances"])
        dense_ids = res["ids"][0]
        for i, d in zip(dense_ids, res.get("distances", [[0.0] * len(dense_ids)])[0]):
            dist_by_id[i] = d

    bm25_ids = _bm25_rank(job_text, ids, docs) if mode in ("hybrid", "bm25") else []

    if mode == "dense":
        ranked = dense_ids
    elif mode == "bm25":
        ranked = bm25_ids
    else:
        ranked = _rrf(dense_ids, bm25_ids)

    fetch = min(max(fetch_k or k * 3, k), n)
    cand_ids = ranked[:fetch]

    has_embs = all(by_id[i][2] is not None for i in cand_ids)
    if use_mmr and has_embs and len(cand_ids) > k:
        # Relevance = fused rank (best first); diversity from chunk embeddings.
        rels = [1.0 / (1.0 + r) for r in range(len(cand_ids))]
        order = _mmr_order(rels, [by_id[i][2] for i in cand_ids], k, lambda_mult)
        sel_ids = [cand_ids[o] for o in order]
    else:
        sel_ids = cand_ids[:k]

    return [
        Retrieved(
            text=by_id[i][0],
            source=(by_id[i][1] or {}).get("source", "?"),
            heading=(by_id[i][1] or {}).get("heading", ""),
            distance=dist_by_id.get(i, 0.0),
        )
        for i in sel_ids
    ]


def format_context(chunks: list[Retrieved]) -> str:
    """Render retrieved chunks into the experience block for the tailoring prompt."""
    parts = []
    for c in chunks:
        loc = f"{c.source} › {c.heading}" if c.heading else c.source
        # Strip the embedded heading prefix; we re-state the location in the tag.
        body = re.sub(r"^\[[^\]]*\]\n", "", c.text)
        parts.append(f"[from {loc}]\n{body}")
    return "\n\n".join(parts)
