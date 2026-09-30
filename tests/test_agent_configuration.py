"""
Tests for agent_configuration.py's Settings loading and chat-model construction

- Every test sets environment variables via pytest's monkeypatch fixture, never relying on
  whatever happens to be in the developer's real .env file
- No test makes a live network call - get_chat_model() only constructs a client object,
  it never invokes it
"""

# ---------------------------------
# Imports
# ---------------------------------
import pytest
from cloud_agent.agent_configuration import get_settings, get_chat_model


# ---------------------------------
# get_settings tests
# ---------------------------------
def test_get_settings_reads_environment_variables(monkeypatch):
    """
    get_settings() should read values from environment variables, not just defaults
    """

    monkeypatch.setenv("LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("API_KEY", "test-key")

    settings = get_settings()

    assert (settings.llm_provider == "test-provider")
    assert (settings.llm_model == "test-model")
    assert (settings.api_key == "test-key")


def test_get_settings_returns_fresh_instance_each_call(monkeypatch):
    """
    Each call to get_settings() should return its own Settings object, not a shared singleton
    """

    monkeypatch.setenv("LLM_PROVIDER", "google")

    first = get_settings()
    second = get_settings()

    assert (first is not second)


# ---------------------------------
# get_chat_model tests
# ---------------------------------
def test_get_chat_model_raises_on_unknown_provider(monkeypatch):
    """
    An unsupported LLM_PROVIDER value should fail clearly instead of doing nothing
    """

    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")

    settings = get_settings()

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_chat_model(settings)


def test_get_chat_model_raises_when_google_api_key_missing(monkeypatch):
    """
    Choosing google without an API key should fail clearly instead of erroring deep inside the SDK
    """

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("API_KEY", "")

    settings = get_settings()

    with pytest.raises(ValueError, match="API_KEY"):
        get_chat_model(settings)


def test_get_chat_model_returns_google_chat_model(monkeypatch):
    """
    A valid google configuration should successfully construct a chat model object

    Construction only - get_chat_model() never calls .invoke(), so no real key is needed
    """

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("API_KEY", "fake-key-for-construction-only")

    settings = get_settings()
    chat_model = get_chat_model(settings)

    assert (chat_model is not None)
