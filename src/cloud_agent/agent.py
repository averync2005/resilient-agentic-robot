"""
Cloud agent conversational graph, and a Gradio web window to talk to it in real time

Graph structure:
- A LangGraph StateGraph with a "chatbot" node and a placeholder "tools" node
- The chatbot node calls the chat model from `agent_configuration.py/get_chat_model()`
- No tools are registered yet - the "tools" node exists so real MCP/A2A tools slot in later without restructuring the graph
- Conversation history persists per conversation via a MemorySaver checkpointer, keyed by a thread id

Web window:
- `main()` launches a Gradio ChatInterface backed by the compiled graph
- Each browser session gets its own thread id, so separate tabs don't share conversation history
- Registered as the `cloud-agent` CLI command in `pyproject.toml`
- Run it with:
    uv run cloud-agent

Development:
- `dev()` launches the same web window through Gradio's own auto-reload CLI
- Watches the whole src/ tree, so any source change restarts the running app
- Registered as the `cloud-agent-dev` CLI command in `pyproject.toml`
- Also turns on debug logging (timing for each reply, graph build time) - `cloud-agent`
  never does, since `dev()` is the only place CLOUD_AGENT_DEBUG gets set
- Logs print to the terminal only, never to a file
- Run it with:
    uv run cloud-agent-dev
"""

# ---------------------------------
# Imports
# ---------------------------------
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Annotated
import gradio as gr
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from cloud_agent.agent_configuration import get_settings, get_chat_model


# ---------------------------------
# Debug logging
# ---------------------------------
# Only on when CLOUD_AGENT_DEBUG is set, which only dev() ever sets (for its own
# subprocess) - cloud-agent never turns this on. Only affects this module's own logger,
# not third-party library logging, and only ever prints to the terminal (a plain
# StreamHandler), never to a file
logger = logging.getLogger("cloud_agent")

# getLogger("cloud_agent") returns the SAME logger object every time, even across a
# reload re-executing this module's top-level code - without the `not logger.handlers`
# guard, each reload would stack another handler onto it, duplicating every log line
if (os.environ.get("CLOUD_AGENT_DEBUG") and not logger.handlers):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


# ---------------------------------
# Conversation state
# ---------------------------------
class State(TypedDict):
    """
    LangGraph state for one conversation

    - `messages` accumulates via `add_messages`, so each graph invocation appends rather than replaces
    """

    messages: Annotated[list, add_messages]


# ---------------------------------
# Chatbot node
# ---------------------------------
# No MCP/A2A tools exist yet - this is where real ones get registered once they do
tools = []


def chatbot(state: State) -> State:
    """
    Run one chat-model turn over the current conversation state

    Args:
        state: The conversation so far

    Returns:
        A new state containing only the model's reply, merged into the running history by `add_messages`
    """

    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ---------------------------------
# Compiling the graph
# ---------------------------------
def build_graph():
    """
    Wire up the chatbot/tools graph with a memory checkpointer

    Returns:
        A compiled LangGraph graph ready for `.invoke()`, keyed per conversation by a thread id
    """

    builder = StateGraph(State)

    builder.add_node(chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    builder.add_edge("chatbot", END)  # Redundant given tools_condition, kept for clarity

    return builder.compile(checkpointer=MemorySaver())


_setup_start = time.monotonic()

settings = get_settings()
llm = get_chat_model(settings)
llm_with_tools = llm.bind_tools(tools) if (tools) else llm
graph = build_graph()

logger.debug(f"Agent configuration and graph built in {time.monotonic() - _setup_start:.2f}s")


# ---------------------------------
# Web window
# ---------------------------------
def new_thread_id() -> str:
    """
    Generate a fresh conversation id

    Returns:
        A random id, unique per browser session, so separate chat windows don't share history
    """

    return str(uuid.uuid4())


def extract_text(content) -> str:
    """
    Pull plain text out of a chat message's content, whichever shape it comes in

    Args:
        content: `message.content` - some providers (Gemini 3 included) return a list of
            content blocks instead of a plain string, mixing text blocks with non-text
            extras like reasoning signatures

    Returns:
        The message's visible text, with any non-text blocks dropped
    """

    if (isinstance(content, str)):
        return content

    return "".join(block.get("text", "") for block in content if (block.get("type") == "text"))


def respond(message: str, history: list, thread_id: str) -> str:
    """
    Answer one chat message, using the conversation's own thread id for memory

    Args:
        message: The operator's latest message
        history: Gradio's own display history - unused, the graph's checkpointer is the real memory
        thread_id: This browser session's conversation id

    Returns:
        The agent's reply text
    """

    config = {"configurable": {"thread_id": thread_id}}
    start_time = time.monotonic()

    logger.debug(f"Thread {thread_id!r} - received: {message!r}")

    try:
        state = graph.invoke (
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )
    except Exception as error:
        # A model-provider failure (rate limit, outage, network blip) shouldn't crash the
        # whole chat window - show it in the transcript instead, and keep the full error
        # on the console for debugging
        print(f"Chat model call failed: {error}")

        return f"Sorry, the model call failed: {error}"

    elapsed = time.monotonic() - start_time
    logger.debug(f"Thread {thread_id!r} - replied in {elapsed:.2f}s")

    return extract_text(state["messages"][-1].content)


# Module-level, not built inside main() - Gradio's own reload CLI (`gradio agent.py`)
# needs to find this variable by import, without calling main() itself
demo = gr.ChatInterface (
    fn=respond,
    additional_inputs=[gr.State(new_thread_id)],
    title="Cloud Agent"
)


def main() -> None:
    """
    Launch the Gradio web window
    """

    demo.launch()


def dev() -> None:
    """
    Launch the Gradio web window with auto-reload, for local development

    - Shells out to Gradio's own reload CLI rather than calling demo.launch() directly
    - Watches this file's whole src/ tree, so a change anywhere in the package triggers a reload
    - Sets CLOUD_AGENT_DEBUG for the subprocess it launches, turning on this module's debug logging
    - Equivalent to running: CLOUD_AGENT_DEBUG=1 uv run gradio src/cloud_agent/agent.py --watch-dirs src
    """

    agent_file = Path(__file__)
    src_dir = agent_file.parent.parent

    env = os.environ.copy()
    env["CLOUD_AGENT_DEBUG"] = "1"

    subprocess.run(["gradio", str(agent_file), "--watch-dirs", str(src_dir)], env=env)


if (__name__ == "__main__"):
    main()
