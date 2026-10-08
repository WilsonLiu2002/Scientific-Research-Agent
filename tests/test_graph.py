import asyncio
import threading
import time
import pytest

from app.graph import ResearchGraphBuilder, ResearchRunCancelled, build_research_graph
from app.checkpointing import create_durable_checkpointer
from app.models import CitationExpansionResult, DocumentSection, FullTextDocument, Paper, PaperIntegrityResult
from app.rag import create_vector_store
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class FakeSearchClient:
    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        return [
            Paper(
                id=f"{query}-1",
                title="Crystal Graph Neural Networks",
                abstract="Graph neural networks represent crystals as graphs for property prediction.",
                authors=["A. Author"],
                year=2018,
                source="fake",
            ),
            Paper(
                id="duplicate",
                title="Duplicate Paper",
                abstract="Formation energy prediction with graph models.",
                source="fake",
            ),
        ]


class ForbiddenSearchClient:
    """Fail loudly if a local-only authorization tier reaches external search."""

    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        raise AssertionError("External search must not run for a local reader")


def test_local_reader_retrieves_from_embedded_corpus_without_openalex() -> None:
    """Exercise RAG against durable local documents without calling an external tool."""

    builder = ResearchGraphBuilder(
        search_client=ForbiddenSearchClient(),
        vector_store=create_vector_store(force_memory=True),
    )
    state = {
        "question": "How do graph neural networks predict crystal properties?",
        "papers": [],
        "errors": [],
        "access_profile": {
            "token_id": "sim-local-reader",
            "level": "local_reader",
            "label": "Local Reader",
            "capabilities": ["rag:read"],
        },
    }
    state.update(builder.plan_research(state))
    state.update(builder.search_papers(state))
    state.update(builder.index_papers(state))
    state.update(builder.retrieve_relevant_papers(state))

    assert state["local_corpus_paper_count"] >= 4
    assert state["retrieved_passages"]
    assert all(passage.paper.source == "local-curated-corpus" for passage in state["retrieved_passages"])
    assert all(task.status == "skipped" for task in state["subtasks"])


def test_graph_honors_cancellation_before_planning() -> None:
    cancellation = threading.Event()
    cancellation.set()
    graph = build_research_graph(
        search_client=FakeSearchClient(),
        vector_store=create_vector_store(force_memory=True),
        cancellation_event=cancellation,
    )

    with pytest.raises(ResearchRunCancelled):
        graph.invoke({"question": "How are crystals modeled?", "errors": []})


def test_graph_runs_complete_workflow_with_fake_search() -> None:
    graph = build_research_graph(
        search_client=FakeSearchClient(),
        vector_store=create_vector_store(force_memory=True),
    )

    result = graph.invoke(
        {
            "question": "How are GNNs used for crystal property prediction?",
            "papers": [],
            "retrieved_papers": [],
            "answer": None,
            "errors": [],
        }
    )

    assert len(result["search_queries"]) >= 3
    assert result["indexed_paper_count"] > 0
    assert result["retrieved_papers"]
    assert result["retrieved_passages"]
    assert "References" in (result["answer"] or "")
    assert "[P1]" in (result["answer"] or "")
    assert result["citation_warnings"] == []
    assert result["context_stats"]["selected_passages"] > 0
    assert "CONTEXT CONTRACT" in result["synthesis_context"]
    assert result["evidence_sufficient"] is True
    assert result["critique"].verdict == "sufficient"
    assert len(result["critique_history"]) == 1
    assert result["search_iteration"] == 1
    assert result["goal"].status == "completed"
    assert result["goal"].progress == 1.0
    assert all(task.status == "completed" for task in result["subtasks"])


class SparseSearchClient:
    """Search double that never supplies enough distinct or relevant evidence."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        self.queries.append(query)
        return [
            Paper(
                id="same-paper",
                title="General Materials Study",
                abstract="A broad materials study with limited detail.",
                source="fake",
            )
        ]


def test_graph_runs_followup_searches_until_iteration_budget() -> None:
    client = SparseSearchClient()
    graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        max_search_iterations=3,
    )

    result = graph.invoke(
        {
            "question": "How do transformers predict superconducting critical temperature?",
            "papers": [],
            "retrieved_passages": [],
            "errors": [],
        }
    )

    assert result["search_iteration"] == 3
    assert result["evidence_sufficient"] is False
    assert len(result["searched_queries"]) > 4
    assert len(client.queries) == len(set(client.queries))
    assert result["evidence_notes"]
    assert result["goal"].status == "budget_exhausted"
    assert [item.verdict for item in result["critique_history"]] == ["retry", "retry", "stop"]


def test_evidence_evaluation_reports_missing_question_concepts() -> None:
    builder = ResearchGraphBuilder(
        search_client=SparseSearchClient(),
        vector_store=create_vector_store(force_memory=True),
    )

    evaluation = builder.evaluate_evidence(
        {
            "question": "transformers for superconducting critical temperature",
            "retrieved_papers": [],
            "retrieved_passages": [],
        }
    )

    assert evaluation["evidence_sufficient"] is False
    assert "superconducting" in evaluation["missing_concepts"]
    assert evaluation["evidence_coverage"] == 0.0


class FullTextSearchClient(FakeSearchClient):
    def get_full_text(self, paper: Paper) -> FullTextDocument:
        return FullTextDocument(
            document_id=f"doc-{paper.id}",
            paper_id=paper.id,
            title=paper.title,
            source_url="https://example.org/full.xml",
            media_type="application/xml",
            sections=[
                DocumentSection(
                    heading="Methods",
                    text=(
                        "The model uses message passing over crystal graphs with three interaction "
                        "layers for detailed crystal property prediction."
                    ),
                    page=4,
                )
            ],
        )


def test_detailed_question_selectively_deep_reads_top_papers() -> None:
    graph = build_research_graph(
        search_client=FullTextSearchClient(),
        vector_store=create_vector_store(force_memory=True),
        max_fulltext_papers=2,
    )

    result = graph.invoke(
        {
            "question": "What methods are used for crystal property prediction?",
            "papers": [],
            "retrieved_passages": [],
            "errors": [],
        }
    )

    assert result["deep_read_requested"] is True
    assert result["fulltext_attempted"] is True
    assert len(result["fulltext_documents"]) == 2
    assert result["retrieved_passages"][0].section == "Methods"
    assert result["retrieved_passages"][0].page == 4
    deep_tasks = [task for task in result["subtasks"] if task.kind == "deep_read"]
    assert len(deep_tasks) == 2
    assert all(task.status == "completed" for task in deep_tasks)


class ReviewTrackingClient(FakeSearchClient):
    """Search double that records whether external actions happened before approval."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        self.queries.append(query)
        return super().search_papers(query, limit)


def test_human_review_can_edit_search_plan_before_any_search() -> None:
    client = ReviewTrackingClient()
    graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=InMemorySaver(),
        min_relevant_papers=1,
        min_evidence_passages=1,
        min_concept_coverage=0.0,
    )
    config = {"configurable": {"thread_id": "review-search-edit"}}

    paused = graph.invoke(
        {"question": "How are GNNs used for crystal property prediction?", "errors": []},
        config=config,
    )

    assert client.queries == []
    assert paused["__interrupt__"][0].value["review_type"] == "search_plan"

    result = graph.invoke(
        Command(resume={"action": "edit", "queries": ["edited crystal graph query"]}),
        config=config,
    )

    assert client.queries == ["edited crystal graph query"]
    assert result["human_review_history"][0]["action"] == "edit"
    assert result["cancelled"] is False


def test_human_review_can_skip_deep_reading() -> None:
    graph = build_research_graph(
        search_client=FullTextSearchClient(),
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=InMemorySaver(),
        max_fulltext_papers=2,
    )
    config = {"configurable": {"thread_id": "review-deep-skip"}}
    paused = graph.invoke(
        {"question": "What methods are used for crystal property prediction?", "errors": []},
        config=config,
    )
    deep_review = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert deep_review["__interrupt__"][0].value["review_type"] == "deep_read"

    result = graph.invoke(Command(resume={"action": "skip"}), config=config)

    assert result["fulltext_attempted"] is True
    assert result.get("fulltext_documents", []) == []
    assert [item["action"] for item in result["human_review_history"]] == ["approve", "skip"]


def test_human_review_can_cancel_before_search() -> None:
    client = ReviewTrackingClient()
    graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-cancel"}}
    graph.invoke({"question": "How are GNNs used in materials?", "errors": []}, config=config)

    result = graph.invoke(Command(resume={"action": "cancel"}), config=config)

    assert result["cancelled"] is True
    assert client.queries == []
    assert "cancelled" in result["answer"].lower()


def test_followup_searches_require_a_new_human_review() -> None:
    client = SparseSearchClient()
    graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=InMemorySaver(),
        max_search_iterations=2,
    )
    config = {"configurable": {"thread_id": "review-followup"}}
    graph.invoke(
        {"question": "How do transformers predict superconducting critical temperature?", "errors": []},
        config=config,
    )

    followup_review = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert followup_review["__interrupt__"][0].value["review_type"] == "search_plan"
    initial_query_count = len(client.queries)
    assert initial_query_count > 0

    result = graph.invoke(Command(resume={"action": "cancel"}), config=config)

    assert result["cancelled"] is True
    assert len(client.queries) == initial_query_count
    assert result["goal"].status == "cancelled"


def test_human_review_stop_alias_prevents_external_work() -> None:
    client = ReviewTrackingClient()
    graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "review-stop"}}
    graph.invoke({"question": "How are GNNs used in materials?", "errors": []}, config=config)

    result = graph.invoke(Command(resume={"action": "stop"}), config=config)

    assert result["cancelled"] is True
    assert result["human_review_history"][0]["action"] == "stop"
    assert client.queries == []


def test_reviewer_instruction_changes_search_and_synthesis_context() -> None:
    client = ReviewTrackingClient()
    graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=InMemorySaver(),
        min_relevant_papers=1,
        min_evidence_passages=1,
        min_concept_coverage=0.0,
    )
    config = {"configurable": {"thread_id": "review-instruction"}}
    graph.invoke(
        {"question": "How are GNNs used for crystal property prediction?", "errors": []},
        config=config,
    )

    result = graph.invoke(
        Command(
            resume={
                "action": "add_instruction",
                "instruction": "Prioritize experimental validation after 2020.",
            }
        ),
        config=config,
    )

    assert result["reviewer_instructions"] == ["Prioritize experimental validation after 2020."]
    assert any("experimental validation after 2020" in query for query in client.queries)
    assert "REVIEWER GUIDANCE" in result["synthesis_context"]
    assert "Prioritize experimental validation after 2020." in result["synthesis_context"]


def test_sqlite_checkpoint_resumes_across_graph_instances(tmp_path) -> None:
    database = tmp_path / "research.sqlite"
    client = ReviewTrackingClient()
    config = {"configurable": {"thread_id": "durable-review"}}
    first_graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=create_durable_checkpointer(database),
    )
    paused = first_graph.invoke(
        {"question": "How are GNNs used in materials?", "errors": []}, config=config
    )

    assert paused["__interrupt__"]
    assert client.queries == []

    resumed_graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        enable_human_review=True,
        checkpointer=create_durable_checkpointer(database),
        min_relevant_papers=1,
        min_evidence_passages=1,
        min_concept_coverage=0.0,
    )
    result = resumed_graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert result["cancelled"] is False
    assert client.queries
    assert result["human_review_history"][0]["action"] == "approve"


class ConcurrentSearchClient(FakeSearchClient):
    """Search double that records the maximum number of overlapping calls."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        try:
            return super().search_papers(query, limit)
        finally:
            with self.lock:
                self.active -= 1


def test_search_subtasks_execute_concurrently() -> None:
    client = ConcurrentSearchClient()
    graph = build_research_graph(
        search_client=client,
        vector_store=create_vector_store(force_memory=True),
        max_concurrent_tasks=3,
    )

    result = graph.invoke(
        {"question": "How are GNNs used for crystal property prediction?", "errors": []}
    )

    assert client.max_active >= 2
    assert len([task for task in result["subtasks"] if task.kind == "literature_search"]) >= 3


def test_graph_supports_async_invocation() -> None:
    graph = build_research_graph(
        search_client=FakeSearchClient(),
        vector_store=create_vector_store(force_memory=True),
    )

    result = asyncio.run(
        graph.ainvoke(
            {"question": "How are GNNs used for crystal property prediction?", "errors": []}
        )
    )

    assert result["goal"].status == "completed"
    assert result["answer"]


class DiscoveryTrustClient(FakeSearchClient):
    """Provide deterministic integrity and citation results for graph routing tests."""

    def check_research_integrity(self, doi: str) -> PaperIntegrityResult:
        status = "retracted" if doi.endswith("bad") else "clear"
        return PaperIntegrityResult(doi=doi, status=status)

    def expand_citation_graph(self, paper: Paper, limit: int = 5) -> CitationExpansionResult:
        discovered = Paper(
            id="citation-1", title="Citation graph result", abstract="Additional relevant evidence.",
            doi="10.1000/good", source="semantic-scholar-citation", discovered_from=paper.id,
        )
        return CitationExpansionResult(seed_paper_id=paper.id, papers=[discovered], citations_found=1)


def test_integrity_check_excludes_retracted_papers_from_index() -> None:
    builder = ResearchGraphBuilder(
        search_client=DiscoveryTrustClient(), vector_store=create_vector_store(force_memory=True)
    )
    state = {
        "question": "test",
        "papers": [
            Paper(id="bad", title="Retracted", abstract="Do not use.", doi="10.1000/bad", source="test"),
            Paper(id="good", title="Valid", abstract="Usable evidence.", doi="10.1000/good", source="test"),
        ],
        "errors": [],
        "access_profile": {"capabilities": ["rag:read", "literature:search"]},
    }
    state.update(builder.check_paper_integrity(state))
    state.update(builder.index_papers(state))
    passages = builder.vector_store.retrieve("Do not use", top_k=20)

    assert state["retracted_paper_count"] == 1
    assert all(passage.paper.id != "bad" for passage in passages)


def test_weak_evidence_routes_to_one_bounded_citation_expansion() -> None:
    builder = ResearchGraphBuilder(search_client=DiscoveryTrustClient())
    seed = Paper(id="seed", title="Seed", abstract="Initial evidence.", source="test")
    state = {
        "papers": [seed], "retrieved_papers": [seed], "errors": [],
        "evidence_sufficient": False, "citation_expansion_attempted": False,
        "access_profile": {"capabilities": ["literature:search"]},
    }

    assert builder.route_after_evaluation(state) == "expand_citation_graph"
    state.update(builder.expand_citation_graph(state))
    assert state["citation_expansion_attempted"] is True
    assert state["citation_expansion_count"] == 1
    assert builder.route_after_evaluation(state) == "critique_research"
