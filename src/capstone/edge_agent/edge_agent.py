"""
Edge-tier agent, backed by the local Gemma 4 model running on the Jetson Orin Nano

Local model server:
- Gemma 4 E2B (4-bit GGUF) runs inside llama.cpp's llama-server, in a docker container named `gemma`
- llama-server exposes an OpenAI-compatible API, so the standard `openai` client talks to it directly
- No API key and no internet connection needed (the model runs fully on the Jetson)
- Gemma 4 can "think" before answering, but that's off by default (EDGE_THINKING=true turns it on)

Activation:
- Every entry point first checks llama-server's /health endpoint
- If the server isn't up, it runs `docker start gemma` and waits for the model to finish loading
- If that fails too, it exits with instructions for starting the container by hand

Configuration:
- Every setting has a working default, so no `.env` entries are required
- Override any of them in `.env` with an `EDGE_` prefix (ex. EDGE_BASE_URL, EDGE_SYSTEM_PROMPT)

LangGraph agent:
- `buildGraph()` wraps Gemma (via LangChain's ChatOpenAI, pointed at llama-server) in a LangGraph graph
- No tools yet - safety validation and ROS 2 tools register onto it later, the same way as the cloud agent's
- Stateless: no checkpointer, so each `generateReply()` call starts a fresh conversation
- Used by the A2A server in `edge_agent_server.py` to answer the cloud agent

Terminal chat:
- `main()` runs a chat loop in the terminal, streaming each reply as it generates
- Talks to llama-server directly (not through the graph), for local debugging on the Jetson
- Conversation history lives in memory for the length of the session only
- Registered as the `edge-agent` CLI command in `pyproject.toml`
- Run it with:
    uv run edge-agent

Speed check:
- `check()` sends one fixed prompt and reports tokens/sec from llama-server's own timings
- Registered as the `edge-agent-check` CLI command in `pyproject.toml`
- Run it with:
    uv run edge-agent-check
"""

# ---------------------------------
# Imports
# ---------------------------------
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Annotated, NoReturn
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI, OpenAIError
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import TypedDict




# ---------------------------------
# Settings
# ---------------------------------

class EdgeSettings(BaseSettings):
    """
    Typed configuration for the local model server and the edge agent's chat behavior

    - Each field is populated from an environment variable of the same name, prefixed with EDGE_
    - Shares the project's `.env` file with the cloud agent's settings
    """

    model_config = SettingsConfigDict (
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EDGE_",
        extra="ignore"
    )

    base_url: str = "http://127.0.0.1:8080/v1"  # llama-server's OpenAI-compatible API root

    model: str = "gemma"  # llama-server serves one model and ignores this, but the API requires it

    container_name: str = "gemma"  # Docker container running llama-server

    startup_timeout: float = 300.0  # Seconds to wait for the model to load after a container start

    thinking: bool = False  # Gemma 4's reasoning mode (off by default, since it multiplies reply latency several times over)

    system_prompt: str = (
        "You are the edge agent of a resilient cloud-to-edge robotic system, "
        "running locally on an NVIDIA Jetson Orin Nano. Keep answers short and concrete."
    )

    a2a_host: str = "0.0.0.0"  # Interface the A2A server binds to (0.0.0.0 = every interface, including the Ethernet link)

    a2a_port: int = 8443  # Port the A2A server listens on

    public_url: str = "https://192.168.10.2:8443"  # Address advertised in the agent card - where the cloud agent reaches this server

    tls_cert: str = "certs/edge.crt"  # Server certificate from `uv run make-certs`

    tls_key: str = "certs/edge.key"  # Private key for tls_cert




# ---------------------------------
# Activating the local model server
# ---------------------------------

def isServerReady(baseUrl: str) -> bool:
    """
    Check whether llama-server is up and finished loading its model

    Args:
        baseUrl: The API root (ex. http://127.0.0.1:8080/v1)

    Returns:
        True only once /health answers 200 (it answers 503 while the model is still loading)
    """

    # /health lives at the server root, not under the /v1 API prefix
    healthUrl = baseUrl.removesuffix("/").removesuffix("/v1") + "/health"

    try:
        with urllib.request.urlopen(healthUrl, timeout=2) as response:
            return (response.status == 200)
    except (urllib.error.URLError, OSError):
        return False


def exitWithStartInstructions(settings: EdgeSettings, reason: str) -> NoReturn:
    """
    Print why the model server couldn't be activated, plus how to start it by hand, then exit

    Args:
        settings: Used to name the container in the printed instructions
        reason: One line describing what went wrong
    """

    print(f"[ERROR] {reason}")
    print(f"Start it with: docker start {settings.container_name}")
    print(f"(Before re-logging in after joining the docker group: sg docker -c \"docker start {settings.container_name}\")")
    print(f"Watch it load with: docker logs -f {settings.container_name}")

    sys.exit(1)


def activateServer(settings: EdgeSettings) -> None:
    """
    Make sure llama-server is running and ready, starting its container if it isn't

    - Returns immediately if the server is already healthy (the normal case)
    - The container is created with --restart unless-stopped, so it usually comes up on boot by itself
    - Otherwise starts the container and polls /health until the model finishes loading
    - Exits the program with instructions if the server can't be brought up

    Args:
        settings: Supplies the server URL, container name, and startup timeout
    """

    if (isServerReady(settings.base_url)):
        return

    print(f"Model server not responding - starting the '{settings.container_name}' container...")

    try:
        result = subprocess.run (
            ["docker", "start", settings.container_name],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        exitWithStartInstructions(settings, "Docker isn't installed or isn't on PATH")

    if (result.returncode != 0):
        exitWithStartInstructions(settings, f"docker start failed: {result.stderr.strip()}")

    deadline = time.monotonic() + settings.startup_timeout

    # Polling every 2s (loading the model from the microSD card takes roughly 30-60s)
    while (time.monotonic() < deadline):
        if (isServerReady(settings.base_url)):
            print("Model server ready")
            return

        time.sleep(2)

    exitWithStartInstructions (
        settings,
        f"Container started, but the model didn't finish loading within {settings.startup_timeout:.0f}s"
    )




# ---------------------------------
# Thinking mode
# ---------------------------------

def thinkingOptions(settings: EdgeSettings) -> dict:
    """
    Build the request field that turns Gemma 4's reasoning mode on or off

    - llama-server passes `chat_template_kwargs` through to the model's chat template
    - Gemma 4's template reads `enable_thinking` from it
    - With thinking on, llama-server returns the reasoning separately as `reasoning_content`, not in `content`

    Args:
        settings: Supplies the `thinking` flag

    Returns:
        A dict for the openai client's `extra_body` argument
    """

    return {"chat_template_kwargs": {"enable_thinking": settings.thinking}}




# ---------------------------------
# LangGraph agent
# ---------------------------------

class EdgeState(TypedDict):
    """
    LangGraph state for one edge agent request

    - `messages` accumulates via `add_messages`, so each node appends rather than replaces
    """

    messages: Annotated[list, add_messages]


def buildChatModel(settings: EdgeSettings) -> ChatOpenAI:
    """
    Build a LangChain chat model pointed at the local llama-server

    Args:
        settings: Supplies the server URL, model name, and thinking flag

    Returns:
        A ChatOpenAI model, since llama-server speaks the OpenAI API
    """

    # llama-server doesn't check API keys, but the openai client refuses to run without one
    return ChatOpenAI (
        base_url=settings.base_url,
        api_key="not-needed",
        model=settings.model,
        extra_body=thinkingOptions(settings)
    )


def buildGraph(chatModel):
    """
    Wire up the edge agent's graph around a chat model

    - A single "chatbot" node for now - a "tools" node joins it once edge tools exist
    - No checkpointer, so nothing carries over between requests

    Args:
        chatModel: The model to answer with (from `buildChatModel()`, or a stand-in in tests)

    Returns:
        A compiled LangGraph graph ready for `.ainvoke()`
    """

    def chatbot(state: EdgeState) -> EdgeState:
        return {"messages": [chatModel.invoke(state["messages"])]}

    builder = StateGraph(EdgeState)

    builder.add_node(chatbot)

    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    return builder.compile()


async def generateReply(graph, settings: EdgeSettings, text: str) -> str:
    """
    Answer one message with the edge agent's graph, starting from a fresh conversation

    Args:
        graph: A compiled graph from `buildGraph()`
        settings: Supplies the system prompt
        text: The incoming message

    Returns:
        The model's reply text
    """

    state = await graph.ainvoke (
        {"messages": [SystemMessage(content=settings.system_prompt), HumanMessage(content=text)]}
    )

    return state["messages"][-1].text




# ---------------------------------
# Terminal chat
# ---------------------------------

def streamReply(client: OpenAI, settings: EdgeSettings, messages: list) -> str:
    """
    Send the conversation to the local model, printing the reply token by token as it generates

    Args:
        client: An OpenAI client pointed at llama-server
        settings: Supplies the model name and thinking flag
        messages: The full conversation so far, system prompt first

    Returns:
        The complete reply text, for appending to the conversation history
    """

    stream = client.chat.completions.create (
        model=settings.model,
        messages=messages,
        stream=True,
        extra_body=thinkingOptions(settings)
    )

    replyParts = []
    shownThinkingNotice = False

    for chunk in stream:
        if (not chunk.choices):
            continue

        delta = chunk.choices[0].delta

        # Reasoning tokens arrive in a llama-server-specific field, so they land in model_extra
        if ((delta.model_extra or {}).get("reasoning_content") and not shownThinkingNotice):
            print("(thinking...) ", end="", flush=True)
            shownThinkingNotice = True

        # Skipping chunks with no visible text (reasoning chunks, and the final stop-reason chunk)
        if (not delta.content):
            continue

        token = delta.content
        replyParts.append(token)

        print(token, end="", flush=True)

    print()

    return "".join(replyParts)


def main() -> None:
    """
    Activate the local model server, then run a streaming chat loop in the terminal

    - Type a message and press Enter to send it
    - Type `exit`, or press Ctrl+D / Ctrl+C, to quit
    """

    settings = EdgeSettings()
    activateServer(settings)

    # llama-server doesn't check API keys, but the openai client refuses to run without one
    client = OpenAI(base_url=settings.base_url, api_key="not-needed")
    messages = [{"role": "system", "content": settings.system_prompt}]

    print("Edge agent ready (local Gemma 4). Type 'exit' to quit.\n")

    while (True):
        try:
            userMessage = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if (userMessage.lower() in ("exit", "quit")):
            break

        if (not userMessage):
            continue

        messages.append({"role": "user", "content": userMessage})
        print("gemma> ", end="", flush=True)

        try:
            reply = streamReply(client, settings, messages)
        except OpenAIError as error:
            # A failed turn (server restarted mid-reply, context full) shouldn't end the session
            print(f"\n[ERROR] Model call failed: {error}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})




# ---------------------------------
# Speed check
# ---------------------------------

def check() -> None:
    """
    Activate the local model server, send one fixed prompt, and report generation speed

    - Prints the reply plus prompt-processing and generation speeds in tokens/sec
    - Uses llama-server's own `timings` field (more accurate than wall-clock time with HTTP overhead)
    - Reference point: ~20 tok/s generation on the Orin Nano Super in MAXN SUPER mode
    """

    settings = EdgeSettings()
    activateServer(settings)

    client = OpenAI(base_url=settings.base_url, api_key="not-needed")

    response = client.chat.completions.create (
        model=settings.model,
        messages=[{"role": "user", "content": "In two sentences, what does an edge AI agent on a robot do?"}],
        max_tokens=512,  # Room for reasoning tokens too, when EDGE_THINKING is on
        extra_body=thinkingOptions(settings)
    )

    reply = response.choices[0].message.content

    # With thinking on, the token limit can run out mid-reasoning, before any visible answer
    if (not reply):
        print(f"[WARNING] Empty reply (finish reason: {response.choices[0].finish_reason})")
    else:
        print(reply)

    print()

    # `timings` is a llama-server extension to the OpenAI response format, so it lands in model_extra
    timings = (response.model_extra or {}).get("timings")

    if (not timings):
        print("Reply received, but the server didn't report timings")
        return

    print(f"Prompt:     {timings['prompt_n']} tokens at {timings['prompt_per_second']:.1f} tok/s")
    print(f"Generation: {timings['predicted_n']} tokens at {timings['predicted_per_second']:.1f} tok/s")




if (__name__ == "__main__"):
    main()
