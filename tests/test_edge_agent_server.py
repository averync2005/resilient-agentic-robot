"""
Tests for edge_agent_server.py's A2A executor, app, and serve() entry point

- The app runs in-process through httpx's ASGI transport - no sockets, no certificates
- The model behind the graph is always a fake - nothing here needs the real `gemma` container
- EdgeSettings is always built with `_env_file=None`, so the developer's real .env never leaks in
"""

# ---------------------------------
# Imports
# ---------------------------------
import asyncio
import httpx
import pytest
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest
from a2a.utils.errors import A2AError
from langchain_core.messages import AIMessage
import capstone.edge_agent.edge_agent_server as server
from capstone.edge_agent.edge_agent import EdgeSettings, buildGraph




# ---------------------------------
# Test helpers
# ---------------------------------

class EchoModel:
    """
    Stand-in for the Gemma chat model - echoes the last message back
    """

    def invoke(self, messages):
        return AIMessage(content=f"echo: {messages[-1].content}")


class BrokenModel:
    """
    Stand-in for a Gemma chat model whose server has gone away
    """

    def invoke(self, messages):
        raise RuntimeError("llama-server unreachable")


def clean_settings(**overrides) -> EdgeSettings:
    """
    Build EdgeSettings from defaults plus explicit overrides only, ignoring the real .env
    """

    return EdgeSettings(_env_file=None, **overrides)


def appClient(chatModel) -> httpx.AsyncClient:
    """
    Build an HTTP client wired straight into the edge agent's app, with `chatModel` behind the graph
    """

    app = server.buildApp(clean_settings(), buildGraph(chatModel))

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://192.168.10.2:8443")


async def sendThroughA2A(chatModel, text: str) -> str:
    """
    Send one message to the app with the real A2A client, returning the reply text
    """

    async with appClient(chatModel) as httpClient:
        clientConfig = ClientConfig(httpx_client=httpClient, streaming=False)

        async with await create_client("https://192.168.10.2:8443", client_config=clientConfig) as client:
            request = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))

            return "".join([get_stream_response_text(response) async for response in client.send_message(request)])




# ---------------------------------
# App tests
# ---------------------------------

def test_app_serves_the_agent_card():
    """
    GET /.well-known/agent-card.json should return the edge agent's card
    """

    async def run():
        async with appClient(EchoModel()) as httpClient:
            return await httpClient.get("/.well-known/agent-card.json")

    response = asyncio.run(run())

    assert (response.status_code == 200)
    assert (response.json()["name"] == "Edge Agent")


def test_app_answers_a_message_with_the_graph_reply():
    """
    A message sent over A2A should come back as the graph's reply
    """

    assert (asyncio.run(sendThroughA2A(EchoModel(), "hello")) == "echo: hello")


def test_app_reports_a_model_failure_as_an_a2a_error():
    """
    A model failure inside the executor should reach the client as an A2A error, not hang or crash the server
    """

    with pytest.raises(A2AError):
        asyncio.run(sendThroughA2A(BrokenModel(), "hello"))




# ---------------------------------
# EdgeAgentExecutor tests
# ---------------------------------

def test_executor_cancel_is_unsupported():
    """
    Cancelling should raise UnsupportedOperationError - replies are one short call
    """

    executor = server.EdgeAgentExecutor(buildGraph(EchoModel()), clean_settings())

    with pytest.raises(server.UnsupportedOperationError):
        asyncio.run(executor.cancel(None, None))




# ---------------------------------
# requireCertFiles tests
# ---------------------------------

def test_require_cert_files_exits_with_instructions_when_missing(tmp_path, capsys):
    """
    Missing TLS files should exit with a pointer to `uv run make-certs`
    """

    settings = clean_settings(tls_cert=str(tmp_path / "edge.crt"), tls_key=str(tmp_path / "edge.key"))

    with pytest.raises(SystemExit):
        server.requireCertFiles(settings)

    output = capsys.readouterr().out

    assert ("edge.crt" in output and "edge.key" in output)
    assert ("uv run make-certs" in output)


def test_require_cert_files_passes_when_present(tmp_path):
    """
    Existing TLS files should let startup continue
    """

    (tmp_path / "edge.crt").write_text("cert")
    (tmp_path / "edge.key").write_text("key")

    server.requireCertFiles(clean_settings(tls_cert=str(tmp_path / "edge.crt"), tls_key=str(tmp_path / "edge.key")))




# ---------------------------------
# serve tests
# ---------------------------------

def test_serve_runs_uvicorn_over_https_after_activating_the_model(monkeypatch):
    """
    serve() should check certs, activate llama-server, then run uvicorn with the TLS files on the configured host/port
    """

    calls = []

    monkeypatch.setattr(server, "EdgeSettings", lambda: clean_settings())
    monkeypatch.setattr(server, "requireCertFiles", lambda settings: calls.append("certs"))
    monkeypatch.setattr(server, "activateServer", lambda settings: calls.append("activate"))
    monkeypatch.setattr(server, "buildChatModel", lambda settings: EchoModel())
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))

    server.serve()

    assert (calls[:2] == ["certs", "activate"])
    assert (calls[2] == {"host": "0.0.0.0", "port": 8443, "ssl_certfile": "certs/edge.crt", "ssl_keyfile": "certs/edge.key"})
