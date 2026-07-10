from src import pipeline
from tests.conftest import AUTO_APPLY

FIX = AUTO_APPLY / "fixtures"


def test_pipeline_run_offline(tmp_path, fake_client):
    out = tmp_path / "out"
    summary = pipeline.run(
        source="fixture",
        query="engineer",
        limit=5,
        use_rag=False,
        db_path=str(tmp_path / "db.sqlite"),
        out=out,
        client=fake_client,
        master_text="MASTER PROFILE",
        use_llm_parse=False,
        fixtures_dir=FIX,
        write_report=False,
    )
    assert summary["ingested"] >= 1
    assert summary["applications"] == summary["ingested"]
    # each application wrote its three artifacts
    dirs = [p for p in out.iterdir() if p.is_dir()]
    assert dirs
    for d in dirs:
        assert (d / "resume.md").exists()
        assert (d / "cover-letter.md").exists()
        assert (d / "report.json").exists()


def test_slug():
    assert pipeline.slug("Northwind Labs, Inc.") == "northwind-labs-inc"
    assert pipeline.slug("") == "job"
