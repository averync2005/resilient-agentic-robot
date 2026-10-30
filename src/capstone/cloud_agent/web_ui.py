"""
Gradio web window for talking to the cloud agent in real time

Web window:
- `main()` launches a Gradio ChatInterface backed by the compiled graph from `cloud_agent.py`
- Each browser session gets its own thread id, so separate tabs don't share conversation history
- Registered as the `cloud-agent` CLI command in `pyproject.toml`
- Run it with:
    uv run cloud-agent

Development:
- `dev()` launches the same web window through Gradio's own auto-reload CLI
- Watches the whole src/ tree, so any source change restarts the running app
- Registered as the `cloud-agent-dev` CLI command in `pyproject.toml`
- Also turns on debug logging (timing for each reply, graph build time, edge agent replies) - `cloud-agent`
  never does, since `dev()` is the only place CLOUD_AGENT_DEBUG gets set
- Logs print to the terminal only, never to a file
- Run it with:
    uv run cloud-agent-dev
"""

# ---------------------------------
# Imports
# ---------------------------------
import os
import subprocess
import time
import uuid
from pathlib import Path
import gradio as gr
from capstone.cloud_agent.cloud_agent import graph, logger




# ---------------------------------
# Conversation helpers
# ---------------------------------

def newThreadId() -> str:
    """
    Generate a fresh conversation id

    Returns:
        A random id, unique per browser session, so separate chat windows don't share history
    """

    return str(uuid.uuid4())


def extractText(content) -> str:
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




# ---------------------------------
# Responding to a message
# ---------------------------------

async def respond(message: str, history: list, threadId: str) -> str:
    """
    Answer one chat message, using the conversation's own thread id for memory

    - Async, since the graph's edge agent tool makes its A2A call asynchronously

    Args:
        message: The operator's latest message
        history: Gradio's own display history - unused, the graph's checkpointer is the real memory
        threadId: This browser session's conversation id

    Returns:
        The agent's reply text
    """

    config = {"configurable": {"thread_id": threadId}}
    startTime = time.monotonic()

    logger.debug(f"Thread {threadId!r} - received: {message!r}")

    try:
        state = await graph.ainvoke (
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )
    except Exception as error:
        # A model-provider failure (rate limit, outage, network blip) shouldn't crash the
        # whole chat window - show it in the transcript instead, and keep the full error
        # on the console for debugging
        print(f"Chat model call failed: {error}")

        return f"Sorry, the model call failed: {error}"

    elapsed = time.monotonic() - startTime
    logger.debug(f"Thread {threadId!r} - replied in {elapsed:.2f}s")

    return extractText(state["messages"][-1].content)




# ---------------------------------
# Web window
# ---------------------------------

# Module-level, not built inside main() - Gradio's own reload CLI (`gradio web_ui.py`)
# needs to find this variable by import, without calling main() itself
demo = gr.ChatInterface (
    fn=respond,
    additional_inputs=[gr.State(newThreadId)],
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
    - Sets CLOUD_AGENT_DEBUG for the subprocess it launches, turning on the cloud agent's debug logging
    - Equivalent to running: CLOUD_AGENT_DEBUG=1 uv run gradio src/capstone/cloud_agent/web_ui.py --watch-dirs src
    """

    uiFile = Path(__file__)
    srcDir = uiFile.parent.parent.parent

    env = os.environ.copy()
    env["CLOUD_AGENT_DEBUG"] = "1"

    subprocess.run(["gradio", str(uiFile), "--watch-dirs", str(srcDir)], env=env)


if (__name__ == "__main__"):
    main()
