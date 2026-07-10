from src import rag

DOC = """# Title

Intro paragraph about the candidate.

## Experience

Worked on Python services and RAG pipelines for a while, shipping production code.

### Sub role

More detail about the sub role and what was built here.
"""


def test_chunks_track_heading_trail():
    chunks = rag.chunk_markdown(DOC, "doc.md")
    assert chunks, "should produce chunks"
    headings = {c.heading for c in chunks}
    assert any("Experience" in h for h in headings)
    assert any("Sub role" in h for h in headings)
    assert all(c.source == "doc.md" for c in chunks)


def test_empty_input_yields_no_chunks():
    assert rag.chunk_markdown("", "empty.md") == []
    assert rag.chunk_markdown("   \n\n  ", "blank.md") == []


def test_long_doc_splits_into_multiple_chunks():
    big = "# Big\n\n" + ("Python RAG FastAPI Docker evaluation pipeline. " * 200)
    chunks = rag.chunk_markdown(big, "big.md")
    assert len(chunks) >= 2
    assert all(len(c.text) <= rag.MAX_CHARS + 200 for c in chunks)


def test_chunk_text_prefixed_with_heading():
    chunks = rag.chunk_markdown(DOC, "doc.md")
    exp = next(c for c in chunks if "Experience" in c.heading)
    assert exp.text.startswith("[")  # heading trail embedded for the embedding
