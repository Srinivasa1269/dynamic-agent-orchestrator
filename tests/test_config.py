"""
test_config.py - Checks that our settings are loaded correctly.

A "test" is a small piece of code that checks other code works.
pytest finds every function whose name starts with "test_" and runs it.
If an "assert" line is False, that test fails.

IMPORTANT: these tests never print the API key. They only check it exists.
"""

from app import config


def test_api_key_is_loaded() -> None:
    # We check the key exists without ever showing it.
    assert config.has_api_key(), (
        "ANTHROPIC_API_KEY is missing. Copy .env.example to .env "
        "and paste your key there."
    )


def test_api_key_looks_like_anthropic_key() -> None:
    # Anthropic keys start with "sk-ant-". We only look at the first 7 letters.
    assert config.ANTHROPIC_API_KEY.startswith("sk-ant-"), (
        "The key in .env does not look like an Anthropic key."
    )


def test_agent_limits_are_sensible() -> None:
    assert config.MIN_AGENTS == 1
    assert config.MAX_AGENTS == 6
    assert config.MIN_AGENTS <= config.MAX_AGENTS


def test_worker_timeout() -> None:
    assert config.WORKER_TIMEOUT_SECONDS == 60
