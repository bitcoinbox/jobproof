from src import ingest, store
from tests.conftest import AUTO_APPLY

FIX = AUTO_APPLY / "fixtures"


def test_strip_html():
    assert ingest.strip_html("<p>Hello</p><ul><li>a</li><li>b</li></ul>") == "Hello\na\nb"
    assert ingest.strip_html("plain text") == "plain text"
    assert ingest.strip_html(None) == ""


def test_fixture_source_and_heuristic_parse():
    src = ingest.get_source("fixture", fixtures_dir=FIX, provider="remoteok")
    raw = src.fetch("engineer", limit=10)
    assert raw, "fixture should return postings"
    posting = ingest.parse_posting(raw[0], use_llm=False)
    assert posting.title and posting.company
    assert "<" not in posting.description  # HTML stripped
    assert isinstance(posting.required_skills, list)


def test_query_filter():
    src = ingest.get_source("fixture", fixtures_dir=FIX, provider="remoteok")
    phd = src.fetch("PhD", limit=10)
    titles = [r["title"] for r in phd]
    assert any("Research" in t for t in titles)


def test_ingest_dedup(tmp_path):
    conn = store.connect(str(tmp_path / "db.sqlite"))
    first = ingest.ingest("fixture", "engineer", limit=10, conn=conn, use_llm=False, fixtures_dir=FIX)
    assert first
    again = ingest.ingest("fixture", "engineer", limit=10, conn=conn, use_llm=False, fixtures_dir=FIX)
    assert again == []  # everything already seen


def test_llm_parse_uses_tool_call(fake_client):
    src = ingest.get_source("fixture", fixtures_dir=FIX, provider="remoteok")
    raw = src.fetch("engineer", limit=1)[0]
    posting = ingest.parse_posting(raw, client=fake_client, use_llm=True)
    # values come from the faked tool_use block
    assert posting.title == "Applied AI Engineer"
    assert "Python" in posting.required_skills
    assert posting.source_id == str(raw["source_id"])  # carried from raw, not the model
