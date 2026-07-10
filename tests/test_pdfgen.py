"""Tests for the kit -> PDF renderer. Markdown conversion is pure; rendering is best-effort."""

from src import pdfgen

MD = """# Robin Vega
**Applied AI Engineer**

## Experience
- Built **RAG** systems in `Python`
- Shipped evals

## Skills
Python, FastAPI
"""


def test_markdown_to_html_structures():
    h = pdfgen.markdown_to_html(MD, title="CV")
    assert "<h1>Robin Vega</h1>" in h
    assert "<h2>Experience</h2>" in h
    assert "<li>Built <strong>RAG</strong> systems in <code>Python</code></li>" in h
    assert "<ul>" in h and "</ul>" in h
    assert "<p>Python, FastAPI</p>" in h
    assert "<title>CV</title>" in h


def test_markdown_escapes_html():
    h = pdfgen.markdown_to_html("Built <script>alert(1)</script> & more")
    assert "<script>" not in h and "&lt;script&gt;" in h and "&amp;" in h


def test_resume_pdf_graceful_without_chrome(tmp_path):
    # Force "no chrome" → returns None, doesn't raise, writes nothing.
    out = pdfgen.resume_pdf(MD, tmp_path / "resume.pdf", chrome="/nonexistent/chrome-xyz")
    assert out is None
    assert not (tmp_path / "resume.pdf").exists()


def test_resume_pdf_renders_if_chrome_present(tmp_path):
    chrome = pdfgen.find_chrome()
    if not chrome:
        import pytest

        pytest.skip("no headless Chrome on this machine")
    out = pdfgen.resume_pdf(MD, tmp_path / "resume.pdf", title="CV", chrome=chrome)
    assert out is not None and out.exists() and out.stat().st_size > 1000
    assert out.read_bytes()[:4] == b"%PDF"
