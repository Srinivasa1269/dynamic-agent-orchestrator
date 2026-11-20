"""
test_config.py - Checks that our settings are loaded correctly.

A "test" is a small piece of code that checks other code works.
pytest finds every function whose name starts with "test_" and runs it.
If an "assert" line is False, that test fails.

None of these tests need Ollama, the internet, or any API key.
"""

from app import config


def test_ollama_model_is_never_blank() -> None:
    assert config.OLLAMA_MODEL.strip() != ""


def test_ollama_url_looks_like_a_web_address() -> None:
    assert config.OLLAMA_BASE_URL.startswith("http")
    assert not config.OLLAMA_BASE_URL.endswith("/")


def test_blank_settings_fall_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SOME_TEST_SETTING", "   ")
    assert config._text_setting("SOME_TEST_SETTING", "default") == "default"
    assert config._number_setting("SOME_TEST_SETTING", 7) == 7


def test_bad_numbers_fall_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SOME_TEST_SETTING", "ten")
    assert config._number_setting("SOME_TEST_SETTING", 7) == 7
    monkeypatch.setenv("SOME_TEST_SETTING", " 42 ")
    assert config._number_setting("SOME_TEST_SETTING", 7) == 42


def test_ollama_is_running_says_no_for_a_closed_port(monkeypatch) -> None:
    # Port 9 on this computer has nothing listening, so the answer is False.
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:9")
    assert config.ollama_is_running(timeout_seconds=0.5) is False


def test_agent_limits_are_sensible() -> None:
    assert config.MIN_AGENTS == 1
    assert config.MAX_AGENTS == 6
    assert config.MIN_AGENTS <= config.MAX_AGENTS


def test_timeouts_are_positive() -> None:
    assert config.WORKER_TIMEOUT_SECONDS > 0
    assert config.PLANNER_TIMEOUT_SECONDS > 0
    assert config.PLANNER_MAX_ATTEMPTS >= 1
