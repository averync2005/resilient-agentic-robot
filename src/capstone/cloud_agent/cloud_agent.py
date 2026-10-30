"""
Cloud agent conversational graph, plus the A2A connection it uses to reach the edge agent

Graph structure:
- A LangGraph StateGraph with a "chatbot" node and a "tools" node
- The chatbot node calls the chat model from `agent_configuration.py/getChatModel()`
- The only tool so far is `askEdgeAgent`, which forwards a message to the edge agent over A2A
- The chat model decides on its own when to call it, the same way it would any other tool
- Conversation history persists per conversation via a MemorySaver checkpointer, keyed by a thread id

Edge agent connection:
- The edge agent runs an A2A server over HTTPS on the Jetson (see `edge_agent/edge_agent_server.py`)
- Each `askEdgeAgent` call fetches the edge agent's card, then sends one message and waits for the full reply
- The edge agent's certificate is verified against the project CA only (`CLOUD_CA_CERT`), never the system trust store
- The edge agent is stateless, so anything it needs to know has to be in the message itself

Web window:
- Lives in `web_ui.py`, which imports the compiled graph from here
"""

# ---------------------------------
# Imports
# ---------------------------------
import logging
import os
import ssl
import time
from pathlib import Path
from typing import Annotated
import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest
from a2a.utils.errors import A2AError
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from capstone.cloud_agent.agent_configuration import getSettings, getChatModel




# ---------------------------------
# Debug logging
# ---------------------------------

# Only on when CLOUD_AGENT_DEBUG is set, which only web_ui.dev() ever sets (for its own
# subprocess) - cloud-agent never turns this on. Only affects this package's own logger,
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
# Edge agent connection
# ---------------------------------

def buildSslContext(caCertPath: str) -> ssl.SSLContext:
    """
    Build a TLS context that trusts only the project CA

    Args:
        caCertPath: Path to the CA certificate written by `uv run make-certs`

    Returns:
        An SSL context for httpx's `verify` argument

    Raises:
        FileNotFoundError: If the CA certificate doesn't exist yet
    """

    if (not Path(caCertPath).is_file()):
        raise FileNotFoundError (
            f"CA certificate not found at '{caCertPath}'. "
            "Generate it with `uv run make-certs` (see README.md, 'Connecting the two agents')"
        )

    return ssl.create_default_context(cafile=caCertPath)


async def sendToEdgeAgent(text: str, edgeUrl: str, httpClient: httpx.AsyncClient) -> str:
    """
    Send one message to the edge agent over A2A and wait for its complete reply

    - Fetches the edge agent's card from `<edgeUrl>/.well-known/agent-card.json` first
    - The card says where to send messages, so only the base URL needs configuring

    Args:
        text: The message for the edge agent
        edgeUrl: The edge agent's base URL (ex. https://192.168.10.2:8443)
        httpClient: The HTTP client to send through, already set up with TLS verification and a timeout

    Returns:
        The edge agent's reply text
    """

    clientConfig = ClientConfig(httpx_client=httpClient, streaming=False)
    request = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))
    replyParts = []

    async with await create_client(edgeUrl, client_config=clientConfig) as client:
        async for response in client.send_message(request):
            replyParts.append(get_stream_response_text(response))

    return "\n".join(part for part in replyParts if (part))


@tool
async def askEdgeAgent(message: str) -> str:
    """
    Send a message to the edge agent, which runs a local Gemma 4 model on the NVIDIA Jetson Orin Nano mounted on the robot, and return its reply

    - Use this whenever the user wants to reach the edge agent, the Jetson, or the robot
    - The edge agent keeps no memory between calls, so include any context it needs in the message
    """

    try:
        sslContext = buildSslContext(settings.cloud_ca_cert)

        async with httpx.AsyncClient(verify=sslContext, timeout=settings.cloud_edge_timeout) as httpClient:
            reply = await sendToEdgeAgent(message, settings.cloud_edge_url, httpClient)
    except (A2AError, httpx.HTTPError, OSError) as error:
        # An unreachable edge agent (cable unplugged, server not running, missing certs) is
        # reported back to the chat model as the tool's result instead of crashing the turn -
        # it can then tell the operator what went wrong
        print(f"Edge agent call failed: {error}")

        return f"The edge agent couldn't be reached: {error}"

    logger.debug(f"Edge agent replied: {reply!r}")

    return reply




# ---------------------------------
# Chatbot node
# ---------------------------------

tools = [askEdgeAgent]


def chatbot(state: State) -> State:
    """
    Run one chat-model turn over the current conversation state

    Args:
        state: The conversation so far

    Returns:
        A new state containing only the model's reply, merged into the running history by `add_messages`
    """

    return {"messages": [llmWithTools.invoke(state["messages"])]}




# ---------------------------------
# Compiling the graph
# ---------------------------------

def buildGraph():
    """
    Wire up the chatbot/tools graph with a memory checkpointer

    Returns:
        A compiled LangGraph graph ready for `.ainvoke()`, keyed per conversation by a thread id
    """

    builder = StateGraph(State)

    builder.add_node(chatbot)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    builder.add_edge("chatbot", END)  # Redundant given tools_condition, kept for clarity

    return builder.compile(checkpointer=MemorySaver())


_setupStart = time.monotonic()

settings = getSettings()
llm = getChatModel(settings)
llmWithTools = llm.bind_tools(tools) if (tools) else llm
graph = buildGraph()

logger.debug(f"Agent configuration and graph built in {time.monotonic() - _setupStart:.2f}s")
