"""
test_config.py - Checks that our settings are loaded correctly.

A "test" is a small piece of code that checks other code works.
pytest finds every function whose name starts with "test_" and runs it.
If an "assert" line is False, that test fails.

IMPORTANT: these tests never print the API key. They only ask config for a
True/False answer. (Asserting on the key directly would make pytest print
the key when the test fails.)
"""

import pytest

from app import config

# Tests marked with this are skipped (not failed) when there is no real key,
# so the other tests still run on a fresh copy of the project.
needs_api_key = pytest.mark.skipif(
    not config.has_api_key(),
    reason="No real ANTHROPIC_API_KEY. Copy .env.example to .env and paste your key there.",
)


@needs_api_key
def test_api_key_looks_like_anthropic_key() -> None:
    # Anthropic keys start with "sk-ant-". We only get True/False back.
    looks_valid = config.api_key_looks_valid()
    assert looks_valid, "The key in .env does not look like an Anthropic key."


def test_has_api_key_rejects_missing_and_placeholder(monkeypatch) -> None:
    # monkeypatch swaps a value for one test only, then puts it back.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    assert config.has_api_key() is False
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "your-key-here")
    assert config.has_api_key() is False


def test_key_checks_accept_a_well_formed_key(monkeypatch) -> None:
    # A made-up key, not a real one.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-fake-for-tests")
    assert config.has_api_key() is True
    assert config.api_key_looks_valid() is True


def test_model_name_is_never_blank() -> None:
    assert config.MODEL_NAME.strip() != ""


def test_agent_limits_are_sensible() -> None:
    assert config.MIN_AGENTS == 1
    assert config.MAX_AGENTS == 6
    assert config.MIN_AGENTS <= config.MAX_AGENTS


def test_worker_timeout() -> None:
    assert config.WORKER_TIMEOUT_SECONDS == 60
