"""
Configuration and chat-model construction for the cloud agent

Environment loading:
- Loaded from environment variables, via a `.env` file in the project root
- `.env` is git-ignored, so API keys are never committed to source control
- See `.env.example` for the expected keys

Common interface:
- LangChain defines a common interface, `BaseChatModel`, for every provider's chat model class
- Sending messages and receiving a response looks identical across providers
- Applies whether the underlying model is `ChatGoogleGenerativeAI`, `ChatAnthropic`, or `ChatOpenAI`

Model-agnostic design:
- `Settings.llm_provider` selects a provider
- `get_chat_model()` reads it to decide which vendor SDK to construct
- Adding a new provider later means adding one branch to that function
- The rest of the codebase depends only on `BaseChatModel`, via `get_chat_model()`
- Swapping providers is a configuration change only - nothing that uses the model needs to change
"""

# ---------------------------------
# Imports
# ---------------------------------
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.language_models.chat_models import BaseChatModel


# ---------------------------------
# Defining/Loading settings
# ---------------------------------
class Settings(BaseSettings):
    """
    Typed, validated configuration for the cloud agent provider/model configuration

    - Each field is populated from an environment variable of the same name
    - Matching is case-insensitive
    - pydantic-settings handles this mapping, plus type validation, automatically
    """

    model_config = SettingsConfigDict (
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    llm_provider: str = ""  # Must match an existing LLM provider name (see get_chat_model() below)
    llm_model: str = ""  # Must match an existing LLM model name (see get_chat_model() below)
    api_key: str = ""

def get_settings() -> Settings:
    """
    Load the provider(s)/model(s) configuration from the .env file

    Returns:
        A fresh Settings instance, loaded from the environment on every call
    """

    return Settings()


# ---------------------------------
# Building a chat model
# ---------------------------------
def get_chat_model(settings: Settings) -> BaseChatModel:
    """
    Build the chat model selected by `settings.llm_provider`

    Args:
        settings: Configuration produced by `get_settings()`

    Returns:
        A LangChain chat model ready to receive messages via `.invoke()`
    """

    # Google AI Models
    if (settings.llm_provider == "google"):
        if (not settings.api_key):
            raise ValueError (
                "LLM_PROVIDER=google requires API_KEY to be set in the .env file. "
                "A free-tier model key is available at `https://aistudio.google.com/apikey`"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI (
            model=settings.llm_model,
            google_api_key=settings.api_key
        )

    raise ValueError (
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'"
    )
