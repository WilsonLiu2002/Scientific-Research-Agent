from app.materials_ml import (
    CandidateScreenRequest,
    MatbenchBandGapDatabase,
    MaterialsScreeningTool,
    parse_formula,
)


def test_matbench_dataset_loads_official_experimental_gap_records() -> None:
    """Load the vendored benchmark and retain measured-value provenance."""

    records = MatbenchBandGapDatabase().records()
    assert len(records) == 4604
    assert all(record.value_type == "measured" for record in records)
    assert all(record.provenance == "Matbench matbench_expt_gap" for record in records)


def test_formula_parser_handles_flat_compositions_without_guessing_groups() -> None:
    """Parse ordinary formulas and reject notation unsupported by the baseline."""

    assert parse_formula("TiO2") == {"Ti": 1.0, "O": 2.0}
    assert parse_formula("Fe0.5Ni0.5") == {"Fe": 0.5, "Ni": 0.5}
    assert parse_formula("Ca(OH)2") == {}


def test_candidate_screen_applies_constraints_and_builds_rag_context() -> None:
    """Return only measured candidates satisfying every explicit constraint."""

    result = MaterialsScreeningTool().screen_candidates(
        CandidateScreenRequest(
            min_band_gap_ev=1.0,
            max_band_gap_ev=2.5,
            include_elements=["O"],
            exclude_elements=["Pb"],
            top_k=8,
        )
    )

    assert result.dataset_size == 4604
    assert len(result.candidates) <= 8
    assert result.matched_count >= len(result.candidates) > 0
    assert all("O" in parse_formula(item.formula) for item in result.candidates)
    assert all("Pb" not in parse_formula(item.formula) for item in result.candidates)
    assert all(item.value_type == "measured" for item in result.candidates)
    assert "database measurements" in result.rag_context
    assert "[M1]" in result.rag_context


def test_composition_model_labels_predictions_and_is_deterministic() -> None:
    """Keep model estimates separate from measurements and reproducible locally."""

    tool = MaterialsScreeningTool()
    first = tool.predict_band_gaps(["TiO2"])[0]
    second = tool.predict_band_gaps(["TiO2"])[0]

    assert first == second
    assert first.model_name == "composition-knn-matbench-expt-gap-v1"
    assert first.predicted_band_gap_ev >= 0
    assert first.uncertainty_ev >= 0
    assert len(first.neighbor_formulas) == 7
