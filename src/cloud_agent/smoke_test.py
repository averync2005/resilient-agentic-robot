"""
Manual verification that the configured LLM provider is reachable

- Not an automated test: nothing runs it as part of a test suite
- Makes a real, billable (though free-tier) API call
- Exists to check the full chain end to end
- Chain: `.env` -> `Settings` -> `get_chat_model()` -> an actual response from the provider

Run it with:
    uv run python -m cloud_agent.smoke_test
"""

# ---------------------------------
# Imports
# ---------------------------------
from cloud_agent.config import get_settings
from cloud_agent.models import get_chat_model


# ---------------------------------
# main
# ---------------------------------
def main() -> None:
    """
    Entry point for the smoke test

    - Loads settings from the environment
    - Builds a chat model for the configured provider
    - Sends one fixed test message and prints the response
    """
    settings = get_settings()
    print(f"Using provider={settings.llm_provider!r} model={settings.llm_model!r}")

    chat_model = get_chat_model(settings)
    response = chat_model.invoke("Reply with exactly: cloud agent online")

    print("Model replied:")
    print(response.content)


if __name__ == "__main__":
    main()
