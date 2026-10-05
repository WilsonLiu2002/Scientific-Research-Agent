from mcp_server.openalex import OpenAlexSearchClient, reconstruct_abstract


def test_reconstruct_abstract_from_openalex_inverted_index() -> None:
    abstract = reconstruct_abstract({"Graph": [0], "networks": [1], "materials": [2]})

    assert abstract == "Graph networks materials"


def test_openalex_work_parser_returns_paper() -> None:
    client = OpenAlexSearchClient()
    paper = client.parse_work(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Crystal Graph Convolutional Neural Networks",
            "publication_year": 2018,
            "authorships": [{"author": {"display_name": "Tian Xie"}}],
            "abstract_inverted_index": {"Crystal": [0], "graphs": [1]},
            "cited_by_count": 1000,
            "doi": "https://doi.org/10.1000/example",
            "primary_location": {
                "landing_page_url": "https://example.org",
                "source": {"display_name": "Nature Materials"},
            },
        }
    )

    assert paper is not None
    assert paper.title == "Crystal Graph Convolutional Neural Networks"
    assert paper.abstract == "Crystal graphs"
    assert paper.authors == ["Tian Xie"]
    assert paper.doi == "https://doi.org/10.1000/example"
    assert paper.venue == "Nature Materials"
