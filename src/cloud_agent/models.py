"""
Constructs a LangChain chat model from configuration

- LangChain defines a common interface, `BaseChatModel`, for every provider's chat model class
- Sending messages and receiving a response looks identical across providers
- Applies whether the underlying model is `ChatGoogleGenerativeAI`, `ChatAnthropic`, or `ChatOpenAI`
- That shared interface is what makes a model-agnostic design possible
- The rest of the codebase should depend only on `BaseChatModel`, via this function
- It should never import a specific provider's SDK directly
- Switching providers then becomes a configuration change plus one new branch here
- Nothing that uses the model needs to change
"""

# ---------------------------------
# Imports
# ---------------------------------
from langchain_core.language_models.chat_models import BaseChatModel

from cloud_agent.config import Settings


# ---------------------------------
# get_chat_model
# ---------------------------------
def get_chat_model(settings: Settings) -> BaseChatModel:
    """
    Build the chat model selected by `settings.llm_provider`

    Args:
        settings: Configuration produced by `config.get_settings()`

    Returns:
        A LangChain chat model ready to receive messages via `.invoke()`
    """
    if settings.llm_provider == "google":
        if not settings.google_api_key:
            raise ValueError (
                "LLM_PROVIDER=google requires GOOGLE_API_KEY to be set in "
                ".env. A free-tier key is available at "
                "https://aistudio.google.com/apikey"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI (
            model=settings.llm_model,
            google_api_key=settings.google_api_key
        )

    raise ValueError (
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
        "Supported providers: 'google'."
    )
