from src import judge
from tests.conftest import FakeAnthropic


def _report(*pairs):
    """pairs of (claim, supported) -> FaithfulnessReport."""
    return judge.FaithfulnessReport(
        verdicts=[judge.ClaimVerdict(claim=c, supported=s, evidence="…") for c, s in pairs]
    )


def test_faithfulness_score_fraction():
    rep = _report(("Python", True), ("RAG", True), ("Kubernetes", False), ("Rust", False))
    assert judge.faithfulness_score(rep) == 0.5


def test_faithfulness_score_all_supported():
    assert judge.faithfulness_score(_report(("Python", True), ("Docker", True))) == 1.0


def test_faithfulness_score_no_claims_is_vacuously_grounded():
    assert judge.faithfulness_score(judge.FaithfulnessReport(verdicts=[])) == 1.0


def test_unsupported_claims_lists_fabrication_risks():
    rep = _report(("Python", True), ("led a 40-person team", False), ("PhD", False))
    assert judge.unsupported_claims(rep) == ["led a 40-person team", "PhD"]


def test_judge_faithfulness_hermetic():
    """judge_faithfulness uses the injected client — no network, no key."""
    rep = _report(("Built a RAG pipeline", True), ("Managed a $10M budget", False))
    fake = FakeAnthropic(parsed=rep)
    score, out = judge.judge_faithfulness("resume text", "evidence text", client=fake)
    assert score == 0.5
    assert judge.unsupported_claims(out) == ["Managed a $10M budget"]
