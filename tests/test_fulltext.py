from mcp_server.fulltext import (
    FullTextService,
    extract_sections,
    is_safe_article_url,
    open_access_candidates,
)
from app.evidence import PaperEvidenceExtractor
from app.models import DocumentSection, FullTextDocument


def test_extracts_sections_from_jats_xml() -> None:
    xml = b"""
    <article><body><sec><title>Methods</title><p>
    We trained a graph neural network on a crystal dataset and evaluated formation energy
    prediction using a held-out test split with documented hyperparameters and baselines.
    </p></sec></body></article>
    """

    sections = extract_sections(xml, "application/xml", "https://example.org/article.xml")

    assert sections[0].heading == "Methods"
    assert "graph neural network" in sections[0].text


def test_open_access_candidates_reject_closed_and_local_urls() -> None:
    closed = {"open_access": {"is_oa": False}, "best_oa_location": {"pdf_url": "https://x/p.pdf"}}
    unsafe = {
        "open_access": {"is_oa": True},
        "best_oa_location": {"is_oa": True, "pdf_url": "https://localhost/paper.pdf"},
    }

    assert open_access_candidates(closed) == []
    assert open_access_candidates(unsafe) == []
    assert is_safe_article_url("https://example.org/paper.pdf") is True
    assert is_safe_article_url("file:///tmp/paper.pdf") is False
    assert is_safe_article_url("https://10.0.0.1/paper.pdf") is False


def test_evidence_extractor_returns_traceable_claims() -> None:
    document = FullTextDocument(
        document_id="doc-1",
        paper_id="paper-1",
        title="Crystal Model",
        source_url="https://example.org/article.xml",
        media_type="application/xml",
        license="cc-by",
        sections=[
            DocumentSection(
                heading="Methods",
                text=(
                    "The method uses a graph neural network architecture for crystal property prediction. "
                    "It trains on a curated materials dataset with a held-out evaluation split."
                ),
            )
        ],
    )
    reading = PaperEvidenceExtractor().extract(
        document, "What architecture predicts crystal properties?"
    )

    assert reading is not None
    assert reading.methods
    assert reading.methods[0].section == "Methods"
    assert reading.methods[0].source_url == document.source_url
