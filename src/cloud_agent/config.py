"""
Configuration for the cloud agent

- Loaded from environment variables, via a `.env` file in the project root
- The LLM provider must be swappable through configuration alone, never by editing source code
- `Settings.llm_provider` selects a provider
- `models.get_chat_model()` reads it to decide which vendor SDK to construct
- Adding a new provider later means adding one branch to that function
- Calling code never has to change
- `.env` is git-ignored, so API keys are never committed to source control
- See `.env.example` for the expected keys
"""

# ---------------------------------
# Imports
# ---------------------------------
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------
# Settings
# ---------------------------------
class Settings(BaseSettings):
    """
    Typed, validated configuration for the cloud agent

    - Each field is populated from an environment variable of the same name
    - Matching is case-insensitive
    - Example: `llm_provider` below is read from `LLM_PROVIDER`
    - pydantic-settings handles this mapping, plus type validation, automatically
    """

    model_config = SettingsConfigDict (
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    llm_provider: str = "google"  # Must match a branch in get_chat_model()
    llm_model: str = "gemini-2.5-flash"  # Model name for the selected provider

    google_api_key: str | None = None
    """
    Google AI Studio API key

    - Required when `llm_provider` is "google"
    - Optional here rather than required, so this module can be imported before a `.env` file exists
    - The real error is raised where the key is actually needed: `models.get_chat_model()`
    - That error message explains how to fix it
    """


# ---------------------------------
# get_settings
# ---------------------------------
def get_settings() -> Settings:
    """
    Load configuration from the environment

    Returns:
        A fresh Settings instance, loaded from the environment on every call
    """
    return Settings()
