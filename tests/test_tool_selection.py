from app.llm import generate_research_plan_locally
from app.graph import ResearchGraphBuilder
from app.materials_ml import MaterialsScreeningTool
from app.tool_selection import material_screen_request, select_tools


ALL_CAPABILITIES = {"rag:read", "literature:search", "materials:read", "fulltext:read"}


def selected(question: str, capabilities: set[str] = ALL_CAPABILITIES) -> set[str]:
    """Return selected IDs from the deterministic intent-aware policy."""

    intent = generate_research_plan_locally(question).intent
    return select_tools(question, intent, capabilities).selected_ids()


def test_overview_avoids_unneeded_full_text_and_material_screening() -> None:
    tools = selected("How does machine learning support solid-state battery material discovery?")
    assert "search_papers" in tools
    assert "get_full_text" not in tools
    assert "screen_material_candidates" not in tools


def test_detailed_candidate_question_selects_specialized_tools() -> None:
    tools = selected("Screen material candidates with a band gap from 1.2 to 1.8 eV and compare methods")
    assert "screen_material_candidates" in tools
    assert "get_full_text" in tools
    request = material_screen_request("Find candidates between 1.2-1.8 eV")
    assert request.min_band_gap_ev == 1.2
    assert request.max_band_gap_ev == 1.8


def test_authorization_removes_tools_the_user_cannot_call() -> None:
    tools = selected("Compare detailed band-gap methods", {"rag:read"})
    assert tools == {"local_rag"}


class MaterialToolClient:
    """Expose only the local screening tool needed by this routing test."""

    def __init__(self) -> None:
        self.tool = MaterialsScreeningTool()

    def screen_material_candidates(self, request):
        return self.tool.screen_candidates(request)


def test_selected_material_tool_executes_and_contributes_rag_document() -> None:
    question = "Screen material candidates with a band gap from 1.2 to 1.8 eV"
    intent = generate_research_plan_locally(question).intent
    plan = select_tools(question, intent, {"rag:read", "materials:read"})
    builder = ResearchGraphBuilder(search_client=MaterialToolClient())
    state = {
        "question": question, "papers": [], "errors": [], "search_queries": [],
        "searched_queries": [], "subtasks": [], "tool_plan": plan.model_dump(),
        "access_profile": {"capabilities": ["rag:read", "materials:read"]},
    }

    result = builder.search_papers(state)

    assert result["material_screening_result"]["candidates"]
    assert any(paper.source == "materials-database" for paper in result["papers"])
