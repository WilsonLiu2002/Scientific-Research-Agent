from app.skill_loader import load_skill


def test_skill_file_loads() -> None:
    skill = load_skill("literature-research")

    assert "Base scientific claims only on retrieved literature" in skill
