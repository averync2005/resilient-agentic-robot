"""
The edge agent's A2A server - how the cloud agent reaches the edge agent over the network

Server:
- A Starlette app serving two routes: the agent card (GET /.well-known/agent-card.json) and A2A JSON-RPC (POST /)
- Runs under uvicorn over HTTPS, using the server certificate from `uv run make-certs`
- Binds to EDGE_A2A_HOST:EDGE_A2A_PORT (0.0.0.0:8443 by default, so the Ethernet link can reach it)

Handling a message:
- `EdgeAgentExecutor` passes the message text to the edge agent's LangGraph graph (`edge_agent.py`)
- The graph's reply goes back as a single A2A message - no task lifecycle, since every reply is immediate and complete
- Stateless: nothing from one request carries over to the next

Running it:
- `serve()` makes sure llama-server is up (same activation as the other edge commands), then starts the server
- Registered as the `edge-agent-serve` CLI command in `pyproject.toml`
- Run it with:
    uv run edge-agent-serve
"""

# ---------------------------------
# Imports
# ---------------------------------
import logging
import sys
from pathlib import Path
import uvicorn
from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils.errors import UnsupportedOperationError
from starlette.applications import Starlette
from capstone.edge_agent.edge_agent import EdgeSettings, activateServer, buildChatModel, buildGraph, generateReply
from capstone.edge_agent.edge_agent_card import buildAgentCard




logger = logging.getLogger("edge_agent")




# ---------------------------------
# Agent executor
# ---------------------------------

class EdgeAgentExecutor(AgentExecutor):
    """
    Answers each A2A message with the edge agent's graph

    - The A2A request handler calls `execute()` once per incoming message
    """

    def __init__(self, graph, settings: EdgeSettings):
        self.graph = graph
        self.settings = settings

    async def execute(self, context: RequestContext, eventQueue: EventQueue) -> None:
        """
        Generate a reply to the incoming message and publish it as one A2A message

        - An exception raised here is caught by the A2A framework and sent back to the cloud agent as an error

        Args:
            context: The incoming request, including the message
            eventQueue: Where the reply gets published
        """

        text = context.get_user_input()
        logger.info(f"Received: {text!r}")

        reply = await generateReply(self.graph, self.settings, text)
        logger.info(f"Replying: {reply!r}")

        await eventQueue.enqueue_event (
            new_text_message(reply, context_id=context.context_id, task_id=context.task_id)
        )

    async def cancel(self, context: RequestContext, eventQueue: EventQueue) -> None:
        """
        Cancelling isn't supported - every reply is generated in one short, uninterruptible call
        """

        raise UnsupportedOperationError()




# ---------------------------------
# Building the app
# ---------------------------------

def buildApp(settings: EdgeSettings, graph) -> Starlette:
    """
    Assemble the A2A server app: agent card route plus JSON-RPC route

    Args:
        settings: Supplies the card's public URL and the system prompt
        graph: A compiled graph from `edge_agent.buildGraph()`

    Returns:
        A Starlette app, ready for uvicorn (or for an in-process test client)
    """

    agentCard = buildAgentCard(settings)

    requestHandler = DefaultRequestHandler (
        agent_executor=EdgeAgentExecutor(graph, settings),
        task_store=InMemoryTaskStore(),
        agent_card=agentCard
    )

    routes = create_agent_card_routes(agentCard) + create_jsonrpc_routes(requestHandler, rpc_url="/")

    return Starlette(routes=routes)




# ---------------------------------
# Running the server
# ---------------------------------

def requireCertFiles(settings: EdgeSettings) -> None:
    """
    Make sure the TLS certificate and key exist, exiting with instructions if they don't

    Args:
        settings: Supplies the certificate and key paths
    """

    missing = [path for path in (settings.tls_cert, settings.tls_key) if (not Path(path).is_file())]

    if (not missing):
        return

    print(f"[ERROR] TLS file(s) not found: {', '.join(missing)}")
    print("Generate them on the laptop with: uv run make-certs")
    print("Then copy them to the Jetson with: scp certs/edge.crt certs/edge.key <user>@192.168.10.2:<repo>/certs/")

    sys.exit(1)


def serve() -> None:
    """
    Activate the local model server, then run the edge agent's A2A server over HTTPS
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    settings = EdgeSettings()
    requireCertFiles(settings)
    activateServer(settings)

    app = buildApp(settings, buildGraph(buildChatModel(settings)))

    print(f"Edge agent A2A server listening on https://{settings.a2a_host}:{settings.a2a_port} (advertised as {settings.public_url})")

    uvicorn.run (
        app,
        host=settings.a2a_host,
        port=settings.a2a_port,
        ssl_certfile=settings.tls_cert,
        ssl_keyfile=settings.tls_key
    )




if (__name__ == "__main__"):
    serve()
