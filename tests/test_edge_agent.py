"""
Tests for edge_agent.py's settings, model-server activation, LangGraph agent, terminal chat, and speed check

- Every network, docker, and model call is mocked - nothing here needs the real `gemma` container
- EdgeSettings is always built with `_env_file=None`, so the developer's real .env never leaks in
- For a live end-to-end check against the real server, run `uv run edge-agent-check` instead
"""

# ---------------------------------
# Imports
# ---------------------------------
import asyncio
import subprocess
import urllib.error
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage
from openai import APIConnectionError
import capstone.edge_agent.edge_agent as edge




# ---------------------------------
# Test helpers
# ---------------------------------

# Held separately, since main()/check() tests monkeypatch edge.EdgeSettings itself
RealEdgeSettings = edge.EdgeSettings


def clean_settings(**overrides) -> RealEdgeSettings:
    """
    Build EdgeSettings from defaults plus explicit overrides only, ignoring the real .env
    """

    return RealEdgeSettings(_env_file=None, **overrides)


class FakeHealthResponse:
    """
    Stand-in for the object urllib.request.urlopen() returns - only `.status` and the context-manager protocol
    """

    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_chunk(content=None, reasoning=None, has_choices=True):
    """
    Build one streamed chat-completion chunk, shaped like the openai client's objects

    - `reasoning` lands in `delta.model_extra`, matching how llama-server's reasoning_content arrives
    """

    if (not has_choices):
        return SimpleNamespace(choices=[])

    extra = {"reasoning_content": reasoning} if (reasoning) else {}
    delta = SimpleNamespace(content=content, model_extra=extra)

    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeClient:
    """
    Stand-in for an OpenAI client - records every chat.completions.create() call and returns a canned result
    """

    def __init__(self, result):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.result = result

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return self.result




# ---------------------------------
# EdgeSettings tests
# ---------------------------------

def test_settings_defaults_point_at_local_gemma_container():
    """
    With no .env entries, the edge agent should target the local llama-server with thinking off
    """

    settings = clean_settings()

    assert (settings.base_url == "http://127.0.0.1:8080/v1")
    assert (settings.container_name == "gemma")
    assert (settings.thinking is False)


def test_settings_read_edge_prefixed_env_vars(monkeypatch):
    """
    EDGE_-prefixed environment variables should override the defaults
    """

    monkeypatch.setenv("EDGE_THINKING", "true")
    monkeypatch.setenv("EDGE_CONTAINER_NAME", "other-gemma")

    settings = clean_settings()

    assert (settings.thinking is True)
    assert (settings.container_name == "other-gemma")


def test_settings_ignore_unprefixed_env_vars(monkeypatch):
    """
    The cloud agent's unprefixed variables (ex. LLM_MODEL) must not bleed into the edge settings
    """

    monkeypatch.setenv("MODEL", "should-be-ignored")

    assert (clean_settings().model == "gemma")


def test_settings_a2a_defaults_match_the_ethernet_setup():
    """
    With no .env entries, the A2A server should bind every interface on 8443 and advertise the Jetson's static IP
    """

    settings = clean_settings()

    assert (settings.a2a_host == "0.0.0.0")
    assert (settings.a2a_port == 8443)
    assert (settings.public_url == "https://192.168.10.2:8443")
    assert ((settings.tls_cert, settings.tls_key) == ("certs/edge.crt", "certs/edge.key"))




# ---------------------------------
# isServerReady tests
# ---------------------------------

def test_server_ready_when_health_returns_200(monkeypatch):
    """
    A 200 from /health means the model has finished loading
    """

    monkeypatch.setattr(edge.urllib.request, "urlopen", lambda url, timeout: FakeHealthResponse(200))

    assert (edge.isServerReady("http://127.0.0.1:8080/v1") is True)


def test_server_not_ready_when_connection_refused(monkeypatch):
    """
    No server listening (container stopped) should read as not ready, not raise
    """

    def refused(url, timeout):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(edge.urllib.request, "urlopen", refused)

    assert (edge.isServerReady("http://127.0.0.1:8080/v1") is False)


def test_server_not_ready_while_model_still_loading(monkeypatch):
    """
    llama-server answers 503 while loading, which urlopen raises as an HTTPError
    """

    def loading(url, timeout):
        raise urllib.error.HTTPError(url, 503, "Loading model", None, None)

    monkeypatch.setattr(edge.urllib.request, "urlopen", loading)

    assert (edge.isServerReady("http://127.0.0.1:8080/v1") is False)


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1/"])
def test_health_url_drops_the_v1_api_prefix(monkeypatch, base_url):
    """
    /health lives at the server root, so the /v1 suffix (with or without a trailing slash) must be stripped
    """

    requested = []

    def record(url, timeout):
        requested.append(url)

        return FakeHealthResponse(200)

    monkeypatch.setattr(edge.urllib.request, "urlopen", record)
    edge.isServerReady(base_url)

    assert (requested == ["http://127.0.0.1:8080/health"])




# ---------------------------------
# activateServer tests
# ---------------------------------

def test_activate_does_nothing_when_server_already_ready(monkeypatch):
    """
    The normal case: a healthy server means no docker call at all
    """

    docker_calls = []

    monkeypatch.setattr(edge, "isServerReady", lambda url: True)
    monkeypatch.setattr(edge.subprocess, "run", lambda *args, **kwargs: docker_calls.append(args))

    edge.activateServer(clean_settings())

    assert (docker_calls == [])


def test_activate_starts_container_and_waits_until_ready(monkeypatch):
    """
    A down server should trigger `docker start <container>`, then polling until /health passes
    """

    readiness = iter([False, False, True])  # Down, still loading, ready
    docker_calls = []

    def fake_run(command, **kwargs):
        docker_calls.append(command)

        return subprocess.CompletedProcess(command, 0, stdout="gemma", stderr="")

    monkeypatch.setattr(edge, "isServerReady", lambda url: next(readiness))
    monkeypatch.setattr(edge.subprocess, "run", fake_run)
    monkeypatch.setattr(edge.time, "sleep", lambda seconds: None)

    edge.activateServer(clean_settings())

    assert (docker_calls == [["docker", "start", "gemma"]])


def test_activate_exits_with_instructions_when_docker_start_fails(monkeypatch, capsys):
    """
    A failed `docker start` (ex. permission denied) should exit 1 and print how to start it by hand
    """

    monkeypatch.setattr(edge, "isServerReady", lambda url: False)
    monkeypatch.setattr (
        edge.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, stdout="", stderr="permission denied")
    )

    with pytest.raises(SystemExit) as exit_info:
        edge.activateServer(clean_settings())

    output = capsys.readouterr().out

    assert (exit_info.value.code == 1)
    assert ("permission denied" in output)
    assert ("docker start gemma" in output)


def test_activate_exits_when_docker_not_installed(monkeypatch, capsys):
    """
    No docker binary at all should exit 1 with a clear message, not a raw FileNotFoundError
    """

    def missing_docker(command, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(edge, "isServerReady", lambda url: False)
    monkeypatch.setattr(edge.subprocess, "run", missing_docker)

    with pytest.raises(SystemExit) as exit_info:
        edge.activateServer(clean_settings())

    assert (exit_info.value.code == 1)
    assert ("isn't installed" in capsys.readouterr().out)


def test_activate_exits_when_model_never_finishes_loading(monkeypatch, capsys):
    """
    A container that starts but never passes /health should exit once startup_timeout runs out
    """

    monkeypatch.setattr(edge, "isServerReady", lambda url: False)
    monkeypatch.setattr (
        edge.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="gemma", stderr="")
    )

    with pytest.raises(SystemExit) as exit_info:
        edge.activateServer(clean_settings(startup_timeout=0))

    assert (exit_info.value.code == 1)
    assert ("didn't finish loading" in capsys.readouterr().out)




# ---------------------------------
# thinkingOptions tests
# ---------------------------------

@pytest.mark.parametrize("thinking", [False, True])
def test_thinking_options_pass_flag_through_chat_template_kwargs(thinking):
    """
    The thinking flag should reach Gemma 4's chat template as enable_thinking
    """

    options = edge.thinkingOptions(clean_settings(thinking=thinking))

    assert (options == {"chat_template_kwargs": {"enable_thinking": thinking}})




# ---------------------------------
# streamReply tests
# ---------------------------------

def test_stream_reply_prints_and_returns_full_text(capsys):
    """
    Streamed tokens should print as they arrive and join into the returned reply
    """

    chunks = [make_chunk("Hel"), make_chunk("lo"), make_chunk(None)]
    client = FakeClient(iter(chunks))

    reply = edge.streamReply(client, clean_settings(), [{"role": "user", "content": "hi"}])

    assert (reply == "Hello")
    assert (capsys.readouterr().out == "Hello\n")


def test_stream_reply_requests_streaming_with_thinking_options():
    """
    The request should stream, and carry the thinking flag in extra_body
    """

    client = FakeClient(iter([]))
    messages = [{"role": "user", "content": "hi"}]

    edge.streamReply(client, clean_settings(), messages)

    call = client.calls[0]

    assert (call["stream"] is True)
    assert (call["messages"] is messages)
    assert (call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}})


def test_stream_reply_skips_chunks_without_choices(capsys):
    """
    Chunks with an empty choices list (ex. a trailing usage chunk) shouldn't crash or print anything
    """

    client = FakeClient(iter([make_chunk(has_choices=False), make_chunk("ok")]))

    assert (edge.streamReply(client, clean_settings(), []) == "ok")


def test_stream_reply_shows_thinking_notice_once_and_hides_reasoning(capsys):
    """
    Reasoning chunks should print one "(thinking...)" notice, never the reasoning text itself
    """

    chunks = [make_chunk(reasoning="Let me"), make_chunk(reasoning=" think"), make_chunk("Answer")]
    client = FakeClient(iter(chunks))

    reply = edge.streamReply(client, clean_settings(thinking=True), [])
    output = capsys.readouterr().out

    assert (reply == "Answer")
    assert (output.count("(thinking...)") == 1)
    assert ("Let me" not in output)




# ---------------------------------
# main tests
# ---------------------------------

def run_main_with_inputs(monkeypatch, inputs, fake_stream_reply):
    """
    Run main() against scripted terminal input, with server activation and the model mocked out

    Returns:
        Every messages list streamReply() was called with, copied at call time
    """

    scripted = iter(inputs)
    seen_messages = []

    def fake_input(prompt):
        try:
            return next(scripted)
        except StopIteration:
            raise EOFError

    def recording_stream_reply(client, settings, messages):
        seen_messages.append(list(messages))

        return fake_stream_reply(messages)

    monkeypatch.setattr(edge, "EdgeSettings", lambda: clean_settings())
    monkeypatch.setattr(edge, "activateServer", lambda settings: None)
    monkeypatch.setattr(edge, "streamReply", recording_stream_reply)
    monkeypatch.setattr("builtins.input", fake_input)

    edge.main()

    return seen_messages


def test_main_keeps_conversation_history_across_turns(monkeypatch):
    """
    The second turn should see the system prompt, first exchange, and second question
    """

    seen = run_main_with_inputs(monkeypatch, ["first", "second", "exit"], lambda messages: "reply")

    assert (len(seen) == 2)
    assert ([message["role"] for message in seen[1]] == ["system", "user", "assistant", "user"])
    assert (seen[1][-1]["content"] == "second")


def test_main_skips_blank_input(monkeypatch):
    """
    Pressing Enter on an empty line shouldn't send anything to the model
    """

    seen = run_main_with_inputs(monkeypatch, ["", "   ", "exit"], lambda messages: "reply")

    assert (seen == [])


def test_main_exits_cleanly_on_ctrl_d(monkeypatch):
    """
    EOF (Ctrl+D) with no `exit` typed should end the loop, not raise
    """

    seen = run_main_with_inputs(monkeypatch, ["hello"], lambda messages: "reply")

    assert (len(seen) == 1)


def test_main_survives_a_failed_turn_and_drops_its_message(monkeypatch, capsys):
    """
    A model error should print, remove the failed question from history, and keep the session going
    """

    attempts = []

    def flaky_reply(messages):
        attempts.append(True)

        if (len(attempts) == 1):
            raise APIConnectionError(request=None)

        return "recovered"

    seen = run_main_with_inputs(monkeypatch, ["fails", "works", "exit"], flaky_reply)

    assert ("[ERROR] Model call failed" in capsys.readouterr().out)
    assert ([message["content"] for message in seen[1][1:]] == ["works"])




# ---------------------------------
# check tests
# ---------------------------------

def run_check_with_response(monkeypatch, response) -> FakeClient:
    """
    Run check() with server activation mocked out and a canned completion response

    Returns:
        The fake client, so tests can inspect the request check() sent
    """

    client = FakeClient(response)

    monkeypatch.setattr(edge, "EdgeSettings", lambda: clean_settings())
    monkeypatch.setattr(edge, "activateServer", lambda settings: None)
    monkeypatch.setattr(edge, "OpenAI", lambda **kwargs: client)

    edge.check()

    return client


def make_response(content, timings=None, finish_reason="stop"):
    """
    Build a non-streamed chat completion, with llama-server's `timings` in model_extra when given
    """

    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)

    return SimpleNamespace(choices=[choice], model_extra={"timings": timings} if (timings) else {})


def test_check_prints_reply_and_speeds(monkeypatch, capsys):
    """
    check() should print the reply, then prompt and generation speeds from llama-server's timings
    """

    timings = {"prompt_n": 24, "prompt_per_second": 125.9, "predicted_n": 51, "predicted_per_second": 24.3}
    run_check_with_response(monkeypatch, make_response("An edge agent...", timings))

    output = capsys.readouterr().out

    assert ("An edge agent..." in output)
    assert ("24 tokens at 125.9 tok/s" in output)
    assert ("51 tokens at 24.3 tok/s" in output)


def test_check_warns_on_empty_reply(monkeypatch, capsys):
    """
    An empty reply (token limit hit mid-reasoning) should print a warning with the finish reason
    """

    run_check_with_response(monkeypatch, make_response("", finish_reason="length"))

    assert ("[WARNING] Empty reply (finish reason: length)" in capsys.readouterr().out)


def test_check_handles_missing_timings(monkeypatch, capsys):
    """
    A server that doesn't report timings shouldn't crash the check
    """

    run_check_with_response(monkeypatch, make_response("reply"))

    assert ("didn't report timings" in capsys.readouterr().out)


def test_check_leaves_room_for_reasoning_tokens(monkeypatch):
    """
    check() should send its thinking flag and a token limit large enough for reasoning plus a reply
    """

    client = run_check_with_response(monkeypatch, make_response("reply"))
    call = client.calls[0]

    assert (call["max_tokens"] >= 512)
    assert ("chat_template_kwargs" in call["extra_body"])




# ---------------------------------
# LangGraph agent tests
# ---------------------------------

class FakeChatModel:
    """
    Stand-in for the ChatOpenAI model - records every call and replies with a canned message
    """

    def __init__(self, reply="graph reply"):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        return AIMessage(content=self.reply)


def test_build_chat_model_points_at_llama_server():
    """
    The LangChain model should target llama-server's URL and carry the thinking flag in extra_body
    """

    chatModel = edge.buildChatModel(clean_settings(base_url="http://127.0.0.1:9999/v1", thinking=True))

    assert (str(chatModel.openai_api_base) == "http://127.0.0.1:9999/v1")
    assert (chatModel.model_name == "gemma")
    assert (chatModel.extra_body == {"chat_template_kwargs": {"enable_thinking": True}})


def test_build_graph_has_a_chatbot_node():
    """
    The edge graph should expose a chatbot node, ready for tools to join it later
    """

    compiled = edge.buildGraph(FakeChatModel())

    assert ("chatbot" in compiled.get_graph().nodes)


def test_generate_reply_sends_system_prompt_then_message():
    """
    generateReply() should hand the model the system prompt followed by the incoming text, and return its reply
    """

    chatModel = FakeChatModel("hello from gemma")
    settings = clean_settings(system_prompt="be brief")

    reply = asyncio.run(edge.generateReply(edge.buildGraph(chatModel), settings, "hi"))
    sent = chatModel.calls[0]

    assert (reply == "hello from gemma")
    assert ([message.type for message in sent] == ["system", "human"])
    assert ((sent[0].content, sent[1].content) == ("be brief", "hi"))


def test_generate_reply_keeps_no_memory_between_calls():
    """
    Each call should start fresh - the edge agent is stateless
    """

    chatModel = FakeChatModel()
    graph = edge.buildGraph(chatModel)

    asyncio.run(edge.generateReply(graph, clean_settings(), "first"))
    asyncio.run(edge.generateReply(graph, clean_settings(), "second"))

    assert (len(chatModel.calls[1]) == 2)
