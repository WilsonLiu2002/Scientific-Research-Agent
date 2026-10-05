from app.settings import get_chat_provider


def test_offline_mode_disables_configured_chat_provider() -> None:
    """Keep tests and explicit offline runs from making accidental paid API calls."""

    assert get_chat_provider() is None
