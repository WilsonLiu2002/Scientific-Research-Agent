from app.models import EvidencePassage, Paper
from evaluation.run import citation_metrics, load_cases, ranking_metrics, run_evaluation


def test_evaluation_dataset_covers_broad_and_deep_questions() -> None:
    cases = load_cases()

    assert len(cases) == 10
    assert any(case.expect_deep_read for case in cases)
    assert any(not case.expect_deep_read for case in cases)
    assert all(case.expected_paper_ids for case in cases)
    assert all(case.expected_answer_terms for case in cases)


def test_offline_evaluation_meets_quality_floor() -> None:
    report = run_evaluation(load_cases())
    aggregate = report["aggregate"]

    assert aggregate["retrieval_quality"] >= 0.8
    assert aggregate["expected_content_coverage"] >= 0.8
    assert aggregate["citation_quality"] >= 0.9
    assert aggregate["deep_evidence_coverage"] >= 0.9
    assert aggregate["route_accuracy"] == 1.0
    assert aggregate["error_free_rate"] == 1.0
    assert aggregate["overall_score"] >= 0.85
    assert aggregate["average_context_tokens"] > 0
    assert aggregate["average_context_papers"] >= 1
    assert len(report["benchmark_fingerprint"]) == 64


def test_ranking_metrics_penalize_distractor_above_relevant_paper() -> None:
    metrics = ranking_metrics(["relevant-1", "distractor", "relevant-2"], {"relevant-1", "relevant-2"})

    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["reciprocal_rank"] == 1.0
    assert 0.5 < metrics["ndcg"] < 1.0


def test_citation_metrics_reject_unrelated_evidence() -> None:
    paper = Paper(id="p1", title="Scheduling", abstract="Room allocation.", source="test")
    passages = [
        EvidencePassage(
            id="p1::p1",
            paper=paper,
            text="The scheduler allocates rooms to weekly time slots.",
        )
    ]
    answer = "Main Approaches\n- Graph networks predict crystal energy and band gaps. [P1]\nReferences\n[P1] Scheduling"

    metrics = citation_metrics(answer, passages, [])

    assert metrics["coverage"] == 1.0
    assert metrics["correctness"] == 0.0
    assert metrics["labels_valid"] is True
