from app.models import Paper, deduplicate_papers


def test_paper_model_validates_and_renders_reference() -> None:
    paper = Paper(
        id="p1",
        title="Crystal Graph Neural Networks",
        abstract="Graphs represent atoms and bonds.",
        authors=["A. Researcher", "B. Scientist"],
        year=2018,
        url="https://example.org/p1",
        source="test",
    )

    assert "Crystal Graph Neural Networks" in paper.document_text()
    assert paper.reference() == (
        "A. Researcher, B. Scientist (2018). Crystal Graph Neural Networks. https://example.org/p1"
    )


def test_papers_with_the_same_doi_are_deduplicated() -> None:
    first = Paper(id="openalex-1", title="A Paper", doi="https://doi.org/10.1/test", source="a")
    second = Paper(id="other-2", title="A Paper Elsewhere", doi="10.1/test", source="b")

    assert first.dedupe_key() == second.dedupe_key()


def test_deduplicate_papers_prefers_first_occurrence() -> None:
    first = Paper(id="same", title="A", abstract="one", source="test")
    second = Paper(id="same", title="B", abstract="two", source="test")

    assert deduplicate_papers([first, second]) == [first]
