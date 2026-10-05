from app.experiments import SyntheticExperimentService


def test_synthetic_experiment_is_reproducible_and_labeled() -> None:
    """Keep demo metrics deterministic and unmistakably separate from real evidence."""

    first = SyntheticExperimentService(seed=7).run_crystal_benchmark()
    second = SyntheticExperimentService(seed=7).run_crystal_benchmark()

    assert first == second
    assert first["status"] == "synthetic"
    assert len(first["models"]) == 4
    assert "not measured results" in first["provenance"]
