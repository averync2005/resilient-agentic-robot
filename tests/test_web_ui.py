"""
Tests for web_ui.py's conversation helpers, Gradio window, and CLI entry points

- Every graph call is mocked - nothing here makes a live network call
- respond() is async, so its tests drive it with asyncio.run()
"""

# ---------------------------------
# Imports
# ---------------------------------
import asyncio
import gradio as gr
import capstone.cloud_agent.web_ui as ui




# ---------------------------------
# Test helpers
# ---------------------------------

class FakeMessage:
    """
    Stand-in for a LangChain message - just enough to carry a `.content` value
    """

    def __init__(self, content):
        self.content = content


def fake_ainvoke(result=None, error=None):
    """
    Build a stand-in for graph.ainvoke() that returns `result` or raises `error`
    """

    async def ainvoke(*args, **kwargs):
        if (error):
            raise error

        return result

    return ainvoke


def respond(message="hi"):
    """
    Run ui.respond() to completion and return its reply
    """

    return asyncio.run(ui.respond(message, [], "test-thread"))




# ---------------------------------
# newThreadId tests
# ---------------------------------

def test_new_thread_id_returns_a_string():
    """
    newThreadId() should return a plain string, suitable for use as a LangGraph thread id
    """

    assert (isinstance(ui.newThreadId(), str))


def test_new_thread_id_returns_unique_values():
    """
    Repeated calls should never return the same id, so separate chat sessions never collide
    """

    ids = {ui.newThreadId() for _ in range(100)}

    assert (len(ids) == 100)




# ---------------------------------
# extractText tests
# ---------------------------------

def test_extract_text_passes_through_a_plain_string():
    """
    A plain string content should be returned unchanged
    """

    assert (ui.extractText("plain string") == "plain string")


def test_extract_text_joins_text_blocks_from_a_list():
    """
    Multiple text blocks should be concatenated into one string
    """

    content = [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]

    assert (ui.extractText(content) == "part one part two")


def test_extract_text_skips_non_text_blocks():
    """
    Non-text blocks (e.g. reasoning signatures) should be dropped, not included in the result
    """

    content = [
        {"type": "text", "text": "visible"},
        {"type": "thought", "text": "should not appear", "extras": {"signature": "xyz"}}
    ]

    assert (ui.extractText(content) == "visible")


def test_extract_text_returns_empty_string_for_an_empty_list():
    """
    An empty content-block list should produce an empty string, not an error
    """

    assert (ui.extractText([]) == "")




# ---------------------------------
# respond tests
# ---------------------------------

def test_respond_returns_model_reply_text(monkeypatch):
    """
    A successful graph.ainvoke() should have its reply text extracted and returned
    """

    monkeypatch.setattr(ui.graph, "ainvoke", fake_ainvoke({"messages": [FakeMessage("hello there")]}))

    assert (respond() == "hello there")


def test_respond_extracts_text_from_content_blocks(monkeypatch):
    """
    A Gemini-style structured content reply should still come back as plain text
    """

    content = [{"type": "text", "text": "structured reply"}]
    monkeypatch.setattr(ui.graph, "ainvoke", fake_ainvoke({"messages": [FakeMessage(content)]}))

    assert (respond() == "structured reply")


def test_respond_returns_a_friendly_message_on_failure(monkeypatch):
    """
    A model-provider failure should come back as a chat message, not raise
    """

    monkeypatch.setattr(ui.graph, "ainvoke", fake_ainvoke(error=RuntimeError("503 UNAVAILABLE")))

    result = respond()

    assert ("Sorry, the model call failed" in result)
    assert ("503 UNAVAILABLE" in result)


def test_respond_prints_the_real_error_to_the_console_on_failure(monkeypatch, capsys):
    """
    The full error should still reach the console, even though the chat window gets a friendlier message
    """

    monkeypatch.setattr(ui.graph, "ainvoke", fake_ainvoke(error=RuntimeError("simulated failure")))
    respond()

    captured = capsys.readouterr()

    assert ("Chat model call failed: simulated failure" in captured.out)


def test_respond_logs_received_and_replied_messages(monkeypatch, caplog):
    """
    A successful call should log both the incoming message and the elapsed reply time
    """

    monkeypatch.setattr(ui.graph, "ainvoke", fake_ainvoke({"messages": [FakeMessage("ok")]}))

    with caplog.at_level("DEBUG", logger="cloud_agent"):
        respond("test message")

    assert (any("received: 'test message'" in record.message for record in caplog.records))
    assert (any("replied in" in record.message for record in caplog.records))




# ---------------------------------
# demo tests
# ---------------------------------

def test_demo_is_a_chat_interface():
    """
    demo should be a Gradio ChatInterface - the type Gradio's reload CLI looks for by default
    """

    assert (isinstance(ui.demo, gr.ChatInterface))


def test_demo_is_wired_to_respond():
    """
    demo's callback should be respond() itself, not a copy or a wrapper
    """

    assert (ui.demo.fn is ui.respond)




# ---------------------------------
# main tests
# ---------------------------------

def test_main_launches_demo(monkeypatch):
    """
    main() should launch the module-level demo, nothing more
    """

    calls = []
    monkeypatch.setattr(ui.demo, "launch", lambda *args, **kwargs: calls.append(True))
    ui.main()

    assert (calls == [True])




# ---------------------------------
# dev tests
# ---------------------------------

def test_dev_runs_gradio_reload_cli_on_this_file(monkeypatch):
    """
    dev() should shell out to `gradio <web_ui.py> --watch-dirs <src>`, not launch demo directly
    """

    captured = {}

    def fake_run(command, env=None):
        captured["command"] = command

    monkeypatch.setattr(ui.subprocess, "run", fake_run)
    ui.dev()

    assert (captured["command"][0] == "gradio")
    assert (captured["command"][1].endswith("web_ui.py"))
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

    monkeypatch.setattr(ui.subprocess, "run", fake_run)
    ui.dev()

    assert (captured["env"]["CLOUD_AGENT_DEBUG"] == "1")
    assert (captured["env"]["SOME_UNRELATED_VAR"] == "still-here")
