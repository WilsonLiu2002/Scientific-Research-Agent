from app.llm import generate_research_plan_locally


def test_local_planner_interprets_purpose_before_queries() -> None:
    """Preserve the evidence need and generate a non-redundant query portfolio offline."""

    plan = generate_research_plan_locally(
        "What architecture and experimental setup are used for crystal property prediction?"
    )

    assert plan.intent.purpose == "methods"
    assert "architecture details" in plan.intent.evidence_requirements
    assert 3 <= len(plan.queries) <= 5
    assert len({item.query for item in plan.queries}) == len(plan.queries)
    assert all(item.rationale for item in plan.queries)
