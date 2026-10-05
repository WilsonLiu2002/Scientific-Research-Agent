from app.critic import LocalResearchCritic


def test_local_critic_requests_targeted_retry_for_missing_evidence() -> None:
    """Request a bounded retry and produce concrete replacement queries."""

    critique = LocalResearchCritic().critique(
        {
            "question": "How do models predict superconducting temperature?",
            "search_iteration": 1,
            "evidence_sufficient": False,
            "missing_concepts": ["superconducting", "temperature"],
        },
        max_search_iterations=3,
        max_critic_retries=2,
    )

    assert critique.verdict == "retry"
    assert critique.recommended_queries
    assert critique.retry_budget_remaining == 1


def test_local_critic_stops_repeated_retry_at_budget() -> None:
    """Prevent a repeated reflection from creating an unbounded search loop."""

    critic = LocalResearchCritic()
    state = {
        "question": "How do models predict superconducting temperature?",
        "search_iteration": 1,
        "evidence_sufficient": False,
        "missing_concepts": ["superconducting"],
    }
    first = critic.critique(state, max_search_iterations=3, max_critic_retries=1)
    state.update({"search_iteration": 2, "critique_history": [first]})

    second = critic.critique(state, max_search_iterations=3, max_critic_retries=1)

    assert second.verdict == "stop"
    assert second.retry_budget_remaining == 0


def test_local_critic_routes_detailed_question_to_deep_read() -> None:
    """Prefer selected full text when abstracts pass but requested detail needs more evidence."""

    critique = LocalResearchCritic().critique(
        {
            "question": "What architecture was used?",
            "search_iteration": 1,
            "evidence_sufficient": True,
            "deep_read_requested": True,
            "fulltext_attempted": False,
            "retrieved_papers": [],
        },
        max_search_iterations=3,
        max_critic_retries=2,
    )

    assert critique.verdict == "deep_read"
