"""
Tests for cloud_agent.py's graph and its A2A connection to the edge agent

- Every graph/model call is mocked - nothing here makes a live network call
- The A2A tests run the real edge agent server app in-process (httpx's ASGI transport), with
  a fake model behind it - no sockets, no certificates, no Jetson
- Env vars (CLOUD_AGENT_DEBUG, LLM_PROVIDER, API_KEY) are set via monkeypatch, never via
  the developer's real .env
- The reload-regression test near the bottom reloads the real module in place - see its
  own docstring before adding tests that assume a particular agent.graph identity
"""

# ---------------------------------
# Imports
# ---------------------------------
import asyncio
import importlib
import ssl
from types import SimpleNamespace
import httpx
from langchain_core.messages import AIMessage, HumanMessage
import capstone.cloud_agent.cloud_agent as agent
from capstone.edge_agent.edge_agent import EdgeSettings, buildGraph as buildEdgeGraph
from capstone.edge_agent.edge_agent_server import buildApp as buildEdgeApp




# ---------------------------------
# Test helpers
# ---------------------------------

class FakeLLM:
    """
    Stand-in for `llmWithTools` - records every call so tests can inspect what the graph
    actually sent it, and replies with a distinguishable message each time
    """

    def __init__(self, reply_text="mocked reply"):
        self.reply_text = reply_text
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        return AIMessage(content=f"{self.reply_text} #{len(self.calls)}")


class ToolCallingLLM:
    """
    Stand-in for `llmWithTools` that calls askEdgeAgent on its first turn, then answers with
    whatever the tool returned
    """

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        if (len(self.calls) == 1):
            toolCall = {"name": "askEdgeAgent", "args": {"message": "are you there?"}, "id": "call-1"}

            return AIMessage(content="", tool_calls=[toolCall])

        return AIMessage(content=f"Edge said: {messages[-1].content}")


class FakeEdgeModel:
    """
    Stand-in for the edge agent's Gemma chat model - echoes the last message back
    """

    def invoke(self, messages):
        return AIMessage(content=f"edge echo: {messages[-1].content}")


def edgeHttpClient() -> httpx.AsyncClient:
    """
    Build an HTTP client wired straight into an in-process edge agent server app

    - The app runs the real A2A routes and executor, with FakeEdgeModel in place of Gemma
    """

    edgeSettings = EdgeSettings(_env_file=None)
    edgeApp = buildEdgeApp(edgeSettings, buildEdgeGraph(FakeEdgeModel()))

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=edgeApp))




# ---------------------------------
# chatbot tests
# ---------------------------------

def test_chatbot_wraps_model_reply_in_a_messages_list(monkeypatch):
    """
    chatbot() should return a state dict containing exactly the model's reply
    """

    fake_llm = FakeLLM()
    monkeypatch.setattr(agent, "llmWithTools", fake_llm)

    result = agent.chatbot({"messages": [HumanMessage(content="hi")]})

    assert (list(result.keys()) == ["messages"])
    assert (len(result["messages"]) == 1)
    assert (result["messages"][0].content == "mocked reply #1")




# ---------------------------------
# buildGraph tests
# ---------------------------------

def test_build_graph_has_chatbot_and_tools_nodes():
    """
    The compiled graph should expose a chatbot node and a tools node
    """

    compiled = agent.buildGraph()
    node_names = set(compiled.get_graph().nodes.keys())

    assert ({"chatbot", "tools"}.issubset(node_names))


def test_ask_edge_agent_is_registered_as_a_tool():
    """
    askEdgeAgent should be in the tool list the chat model is bound to
    """

    assert ([registered.name for registered in agent.tools] == ["askEdgeAgent"])


def test_graph_persists_conversation_across_calls_on_same_thread(monkeypatch):
    """
    Two invokes with the same thread id should see the full conversation, not just the latest message
    """

    fake_llm = FakeLLM()
    monkeypatch.setattr(agent, "llmWithTools", fake_llm)

    compiled = agent.buildGraph()
    config = {"configurable": {"thread_id": "shared-thread"}}

    compiled.invoke({"messages": [HumanMessage(content="first")]}, config=config)
    compiled.invoke({"messages": [HumanMessage(content="second")]}, config=config)

    # By the second call, the fake LLM should have seen 3 messages: first human, first
    # reply, second human - proof the checkpointer is actually accumulating history
    assert (len(fake_llm.calls[-1]) == 3)


def test_graph_does_not_share_memory_across_different_threads(monkeypatch):
    """
    Two different thread ids should each start with their own clean conversation
    """

    fake_llm = FakeLLM()
    monkeypatch.setattr(agent, "llmWithTools", fake_llm)

    compiled = agent.buildGraph()
    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    compiled.invoke({"messages": [HumanMessage(content="hi from a")]}, config=config_a)
    compiled.invoke({"messages": [HumanMessage(content="hi from b")]}, config=config_b)

    assert (len(fake_llm.calls[0]) == 1)
    assert (len(fake_llm.calls[1]) == 1)


def test_graph_routes_a_tool_call_through_ask_edge_agent(monkeypatch):
    """
    A model turn that calls askEdgeAgent should run the tool, then feed its result back to the model
    """

    sent = []

    async def fake_send(text, edgeUrl, httpClient):
        sent.append(text)

        return "edge is online"

    monkeypatch.setattr(agent, "llmWithTools", ToolCallingLLM())
    monkeypatch.setattr(agent, "buildSslContext", lambda path: ssl.create_default_context())
    monkeypatch.setattr(agent, "sendToEdgeAgent", fake_send)

    compiled = agent.buildGraph()
    state = asyncio.run (
        compiled.ainvoke (
            {"messages": [HumanMessage(content="ping the robot")]},
            config={"configurable": {"thread_id": "tool-thread"}}
        )
    )

    assert (sent == ["are you there?"])
    assert (state["messages"][-1].content == "Edge said: edge is online")




# ---------------------------------
# buildSslContext tests
# ---------------------------------

def test_build_ssl_context_raises_with_instructions_when_ca_missing(tmp_path):
    """
    A missing CA certificate should raise, pointing at `uv run make-certs`
    """

    missing = tmp_path / "ca.crt"

    try:
        agent.buildSslContext(str(missing))
    except FileNotFoundError as error:
        assert ("make-certs" in str(error))
    else:
        raise AssertionError("expected FileNotFoundError")


def test_build_ssl_context_trusts_the_given_ca(tmp_path):
    """
    An existing CA certificate should load into a verifying TLS context
    """

    from capstone.certificates import makeCerts

    makeCerts(tmp_path, "192.168.10.2", "jetson-edge")
    context = agent.buildSslContext(str(tmp_path / "ca.crt"))

    assert (context.verify_mode == ssl.CERT_REQUIRED)
    assert (len(context.get_ca_certs()) == 1)




# ---------------------------------
# sendToEdgeAgent tests
# ---------------------------------

def test_send_to_edge_agent_round_trips_through_the_real_edge_server():
    """
    The A2A client should fetch the edge agent's card, send the message, and return the reply text
    """

    async def run():
        async with edgeHttpClient() as httpClient:
            return await agent.sendToEdgeAgent("hello edge", "https://192.168.10.2:8443", httpClient)

    assert (asyncio.run(run()) == "edge echo: hello edge")




# ---------------------------------
# askEdgeAgent tests
# ---------------------------------

def test_ask_edge_agent_reports_missing_certs_instead_of_raising(monkeypatch, tmp_path):
    """
    No CA certificate yet should come back as the tool's result, so the chat model can explain it
    """

    fake_settings = SimpleNamespace (
        cloud_ca_cert=str(tmp_path / "missing.crt"),
        cloud_edge_url="https://192.168.10.2:8443",
        cloud_edge_timeout=5.0
    )
    monkeypatch.setattr(agent, "settings", fake_settings)

    result = asyncio.run(agent.askEdgeAgent.ainvoke({"message": "hi"}))

    assert ("couldn't be reached" in result)
    assert ("make-certs" in result)


def test_ask_edge_agent_reports_an_unreachable_edge_agent(monkeypatch):
    """
    A network failure (cable unplugged, server down) should come back as the tool's result, not raise
    """

    async def unreachable(text, edgeUrl, httpClient):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(agent, "buildSslContext", lambda path: ssl.create_default_context())
    monkeypatch.setattr(agent, "sendToEdgeAgent", unreachable)

    result = asyncio.run(agent.askEdgeAgent.ainvoke({"message": "hi"}))

    assert ("couldn't be reached" in result)
    assert ("connection refused" in result)


def test_ask_edge_agent_uses_configured_url_and_timeout(monkeypatch):
    """
    The tool should send to CLOUD_EDGE_URL, through a client using CLOUD_EDGE_TIMEOUT
    """

    captured = {}

    async def fake_send(text, edgeUrl, httpClient):
        captured["url"] = edgeUrl
        captured["timeout"] = httpClient.timeout.read

        return "ok"

    fake_settings = SimpleNamespace(cloud_ca_cert="unused", cloud_edge_url="https://10.0.0.5:9000", cloud_edge_timeout=42.0)
    monkeypatch.setattr(agent, "settings", fake_settings)
    monkeypatch.setattr(agent, "buildSslContext", lambda path: ssl.create_default_context())
    monkeypatch.setattr(agent, "sendToEdgeAgent", fake_send)

    assert (asyncio.run(agent.askEdgeAgent.ainvoke({"message": "hi"})) == "ok")
    assert (captured == {"url": "https://10.0.0.5:9000", "timeout": 42.0})




# ---------------------------------
# Logging regression
# ---------------------------------

def test_logger_handlers_dont_duplicate_across_reload(monkeypatch):
    """
    Gradio's reload CLI re-executes this module's top-level code multiple times in one
    process - reloading with CLOUD_AGENT_DEBUG set must not stack a new handler onto the
    shared "cloud_agent" logger each time, or every log line duplicates

    Reloads the real capstone.cloud_agent.cloud_agent module, rebuilding its graph in place -
    keep this isolated from tests that assume a particular agent.graph identity
    """

    monkeypatch.setenv("CLOUD_AGENT_DEBUG", "1")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("API_KEY", "fake-key-for-reload-test")

    importlib.reload(agent)
    importlib.reload(agent)

    assert (len(agent.logger.handlers) <= 1)
