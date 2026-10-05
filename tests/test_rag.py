from app.models import Paper, ResearchIntent
from app.rag import create_vector_store, retrieve_for_intent


def test_rag_indexes_and_retrieves_documents() -> None:
    store = create_vector_store(force_memory=True)
    papers = [
        Paper(
            id="gnn",
            title="Crystal Graph Networks",
            abstract="Graph neural networks predict crystal formation energy and band gaps.",
            source="test",
        ),
        Paper(
            id="other",
            title="Unrelated",
            abstract="A study of laboratory scheduling.",
            source="test",
        ),
    ]

    assert store.index_papers(papers) == 2
    retrieved = store.retrieve("crystal graph neural network formation energy", top_k=1)

    assert retrieved[0].paper.id == "gnn"
    assert "formation energy" in retrieved[0].text


def test_rag_splits_long_abstracts_into_citable_passages() -> None:
    store = create_vector_store(force_memory=True)
    paper = Paper(
        id="long",
        title="Long Abstract",
        abstract=("Crystal graphs predict formation energy. " * 30).strip(),
        source="test",
    )

    assert store.index_papers([paper]) == 1
    retrieved = store.retrieve("formation energy", top_k=8)

    assert len(retrieved) > 1
    assert all(passage.id.startswith("long::p") for passage in retrieved)


def test_intent_aware_retrieval_favors_requested_evidence_role() -> None:
    """Rank methods evidence above a generic overview for a methods-purpose question."""

    store = create_vector_store(force_memory=True)
    store.index_papers(
        [
            Paper(
                id="overview",
                title="Crystal Prediction Review",
                abstract="A broad review discusses applications of machine learning in crystal prediction.",
                source="test",
            ),
            Paper(
                id="method",
                title="Crystal Graph Architecture",
                abstract="The method uses three message-passing layers and a pooled crystal representation with a held-out experimental protocol.",
                source="test",
            ),
        ]
    )
    intent = ResearchIntent(
        purpose="methods",
        objective="Identify architecture and experimental setup.",
        entities=["crystal", "graph"],
        methods=["message passing"],
        evidence_requirements=["architecture details", "experimental setup"],
    )

    passages, probes = retrieve_for_intent(store, "How are crystal models built?", intent, top_k=2)

    assert passages[0].paper.id == "method"
    assert any("architecture" in probe for probe in probes)
