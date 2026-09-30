"""
Tests for agent.py's graph, web window, and CLI entry points

- Every graph/model call is mocked - nothing here makes a live network call
- Env vars (CLOUD_AGENT_DEBUG, LLM_PROVIDER, API_KEY) are set via monkeypatch, never via
  the developer's real .env
- The reload-regression test near the bottom reloads the real module in place - see its
  own docstring before adding tests that assume a particular agent.graph/agent.demo identity
"""

# ---------------------------------
# Imports
# ---------------------------------
import importlib
import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
import cloud_agent.agent as agent


# ---------------------------------
# Test helpers
# ---------------------------------
class FakeMessage:
    """
    Stand-in for a LangChain message - just enough to carry a `.content` value
    """

    def __init__(self, content):
        self.content = content


class FakeLLM:
    """
    Stand-in for `llm_with_tools` - records every call so tests can inspect what the graph
    actually sent it, and replies with a distinguishable message each time
    """

    def __init__(self, reply_text="mocked reply"):
        self.reply_text = reply_text
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        return AIMessage(content=f"{self.reply_text} #{len(self.calls)}")


# ---------------------------------
# chatbot tests
# ---------------------------------
def test_chatbot_wraps_model_reply_in_a_messages_list(monkeypatch):
    """
    chatbot() should return a state dict containing exactly the model's reply
    """

    fake_llm = FakeLLM()
    monkeypatch.setattr(agent, "llm_with_tools", fake_llm)

    result = agent.chatbot({"messages": [HumanMessage(content="hi")]})

    assert (list(result.keys()) == ["messages"])
    assert (len(result["messages"]) == 1)
    assert (result["messages"][0].content == "mocked reply #1")


# ---------------------------------
# build_graph tests
# ---------------------------------
def test_build_graph_has_chatbot_and_tools_nodes():
    """
    The compiled graph should expose a chatbot node and a tools node, ready for real tools
    to register onto later
    """

    compiled = agent.build_graph()
    node_names = set(compiled.get_graph().nodes.keys())

    assert ({"chatbot", "tools"}.issubset(node_names))


def test_graph_persists_conversation_across_calls_on_same_thread(monkeypatch):
    """
    Two invokes with the same thread id should see the full conversation, not just the latest message
    """

    fake_llm = FakeLLM()
    monkeypatch.setattr(agent, "llm_with_tools", fake_llm)

    compiled = agent.build_graph()
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
    monkeypatch.setattr(agent, "llm_with_tools", fake_llm)

    compiled = agent.build_graph()
    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    compiled.invoke({"messages": [HumanMessage(content="hi from a")]}, config=config_a)
    compiled.invoke({"messages": [HumanMessage(content="hi from b")]}, config=config_b)

    assert (len(fake_llm.calls[0]) == 1)
    assert (len(fake_llm.calls[1]) == 1)


# ---------------------------------
# new_thread_id tests
# ---------------------------------
def test_new_thread_id_returns_a_string():
    """
    new_thread_id() should return a plain string, suitable for use as a LangGraph thread id
    """

    assert (isinstance(agent.new_thread_id(), str))


def test_new_thread_id_returns_unique_values():
    """
    Repeated calls should never return the same id, so separate chat sessions never collide
    """

    ids = {agent.new_thread_id() for _ in range(100)}

    assert (len(ids) == 100)


# ---------------------------------
# extract_text tests
# ---------------------------------
def test_extract_text_passes_through_a_plain_string():
    """
    A plain string content should be returned unchanged
    """

    assert (agent.extract_text("plain string") == "plain string")


def test_extract_text_joins_text_blocks_from_a_list():
    """
    Multiple text blocks should be concatenated into one string
    """

    content = [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]

    assert (agent.extract_text(content) == "part one part two")


def test_extract_text_skips_non_text_blocks():
    """
    Non-text blocks (e.g. reasoning signatures) should be dropped, not included in the result
    """

    content = [
        {"type": "text", "text": "visible"},
        {"type": "thought", "text": "should not appear", "extras": {"signature": "xyz"}}
    ]

    assert (agent.extract_text(content) == "visible")


def test_extract_text_returns_empty_string_for_an_empty_list():
    """
    An empty content-block list should produce an empty string, not an error
    """

    assert (agent.extract_text([]) == "")


# ---------------------------------
# respond tests
# ---------------------------------
def test_respond_returns_model_reply_text(monkeypatch):
    """
    A successful graph.invoke() should have its reply text extracted and returned
    """

    monkeypatch.setattr (
        agent.graph, "invoke",
        lambda *args, **kwargs: {"messages": [FakeMessage("hello there")]}
    )

    assert (agent.respond("hi", [], "test-thread") == "hello there")


def test_respond_extracts_text_from_content_blocks(monkeypatch):
    """
    A Gemini-style structured content reply should still come back as plain text
    """

    content = [{"type": "text", "text": "structured reply"}]
    monkeypatch.setattr (
        agent.graph, "invoke",
        lambda *args, **kwargs: {"messages": [FakeMessage(content)]}
    )

    assert (agent.respond("hi", [], "test-thread") == "structured reply")


def test_respond_returns_a_friendly_message_on_failure(monkeypatch):
    """
    A model-provider failure should come back as a chat message, not raise
    """

    def broken_invoke(*args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(agent.graph, "invoke", broken_invoke)

    result = agent.respond("hi", [], "test-thread")

    assert ("Sorry, the model call failed" in result)
    assert ("503 UNAVAILABLE" in result)


def test_respond_prints_the_real_error_to_the_console_on_failure(monkeypatch, capsys):
    """
    The full error should still reach the console, even though the chat window gets a friendlier message
    """

    def broken_invoke(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(agent.graph, "invoke", broken_invoke)
    agent.respond("hi", [], "test-thread")

    captured = capsys.readouterr()

    assert ("Chat model call failed: simulated failure" in captured.out)


def test_respond_logs_received_and_replied_messages(monkeypatch, caplog):
    """
    A successful call should log both the incoming message and the elapsed reply time
    """

    monkeypatch.setattr (
        agent.graph, "invoke",
        lambda *args, **kwargs: {"messages": [FakeMessage("ok")]}
    )

    with caplog.at_level("DEBUG", logger="cloud_agent"):
        agent.respond("test message", [], "log-thread")

    assert (any("received: 'test message'" in record.message for record in caplog.records))
    assert (any("replied in" in record.message for record in caplog.records))


# ---------------------------------
# demo tests
# ---------------------------------
def test_demo_is_a_chat_interface():
    """
    demo should be a Gradio ChatInterface - the type Gradio's reload CLI looks for by default
    """

    assert (isinstance(agent.demo, gr.ChatInterface))


def test_demo_is_wired_to_respond():
    """
    demo's callback should be respond() itself, not a copy or a wrapper
    """

    assert (agent.demo.fn is agent.respond)


# ---------------------------------
# main tests
# ---------------------------------
def test_main_launches_demo(monkeypatch):
    """
    main() should launch the module-level demo, nothing more
    """

    calls = []
    monkeypatch.setattr(agent.demo, "launch", lambda *args, **kwargs: calls.append(True))
    agent.main()

    assert (calls == [True])


# ---------------------------------
# dev tests
# ---------------------------------
def test_dev_runs_gradio_reload_cli_on_this_file(monkeypatch):
    """
    dev() should shell out to `gradio <this file> --watch-dirs <src>`, not launch demo directly
    """

    captured = {}

    def fake_run(command, env=None):
        captured["command"] = command

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    agent.dev()

    assert (captured["command"][0] == "gradio")
    assert (captured["command"][1].endswith("agent.py"))
    assert (captured["command"][2] == "--watch-dirs")
    assert (captured["command"][3].endswith("src"))


def test_dev_turns_on_debug_logging_for_its_subprocess(monkeypatch):
    """
    dev()'s subprocess should inherit the current environment, plus CLOUD_AGENT_DEBUG=1
    """

    monkeypatch.setenv("SOME_UNRELATED_VAR", "still-here")

    captured = {}

    def fake_run(command, env=None):
        captured["env"] = env

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    agent.dev()

    assert (captured["env"]["CLOUD_AGENT_DEBUG"] == "1")
    assert (captured["env"]["SOME_UNRELATED_VAR"] == "still-here")


# ---------------------------------
# Logging regression
# ---------------------------------
def test_logger_handlers_dont_duplicate_across_reload(monkeypatch):
    """
    Gradio's reload CLI re-executes this module's top-level code multiple times in one
    process - reloading with CLOUD_AGENT_DEBUG set must not stack a new handler onto the
    shared "cloud_agent" logger each time, or every log line duplicates

    Reloads the real cloud_agent.agent module, rebuilding its graph/demo/etc in place -
    keep this isolated from tests that assume a particular agent.graph/agent.demo identity
    """

    monkeypatch.setenv("CLOUD_AGENT_DEBUG", "1")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("API_KEY", "fake-key-for-reload-test")

    importlib.reload(agent)
    importlib.reload(agent)

    assert (len(agent.logger.handlers) <= 1)
