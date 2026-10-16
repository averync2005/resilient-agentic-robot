# capstone

Code for the "Resilient Cloud-to-Edge Agentic Robotic System" (ECE 257/258, Lehigh University, 2026-2027). If you only read one file in this repo, make it this one. Everything else assumes you have read this first.

## The two tiers

The system splits its reasoning across two tiers that talk to each other over a network. Both tiers live in this repo, as two modules of the same `capstone` Python package.

| | Cloud agent (`cloud_agent.py`) | Edge agent (`edge_agent.py`) |
|---|---|---|
| Runs on | An always-on cloud host | An NVIDIA Jetson Orin Nano Super, mounted on the robot |
| Model | A hosted LLM (currently Google Gemini) | Gemma 4 E2B, running locally on the Jetson's GPU |
| Job | Turn a natural-language command into an ordered sequence of robot actions | Receive that sequence, check that it is safe, and move the robot arm via ROS 2 |
| Without internet | Unavailable | Keeps working (the local model needs no network) |

The two tiers will talk to each other over A2A (Agent-to-Agent) protocol. The A2A "contract" (the shape of the messages the cloud agent sends) will be documented here once it exists.

## What this code does, in plain English

**Right now:**

- **Cloud agent:** a person types into a web chat window, and a LangGraph-backed agent running on Gemini replies, with memory of the conversation so far
- **Edge agent:** a person types into a terminal chat, and Gemma 4 replies from the Jetson itself, with no cloud involved

Neither agent generates robot action plans, talks to the other over A2A, or moves a robot yet. Those come later in the two-semester schedule (see the project proposal for the full timeline).

**The finished picture:** the cloud agent turns a command into an ordered list of robot actions from a fixed, pre-approved vocabulary. That vocabulary is a set of named poses like `switch_1`, `button_2`, `home`, and `safe_retract`, and the robot is never told to do anything outside it. The cloud agent sends that plan to the Jetson over A2A, and the edge agent (not the cloud agent) validates and executes it. The cloud agent does not trust itself to be the final safety check. When the cloud is unreachable, the edge agent's local model is the fallback.

## Repository layout

```
src/capstone/
    __init__.py               Package docstring (a short version of this README, read by tools that
                              inspect the package)
    agent_configuration.py    The cloud agent's Settings (reads .env into a validated object: LLM provider,
                              model, API key) and get_chat_model() (builds a LangChain chat model from
                              Settings). This is the ONE place that knows about specific LLM vendor SDKs.
                              Everything else calls get_chat_model() instead of importing e.g.
                              langchain_google_genai directly. That is what "model-agnostic" means in
                              practice: swapping providers touches this file and .env, nothing else.
    cloud_agent.py            The LangGraph conversational graph (chatbot node, placeholder tools node,
                              per-conversation memory) and the Gradio web window that talks to it. main()
                              is the `cloud-agent` command's entry point. dev() is `cloud-agent-dev`'s
                              (same app, auto-reloads on source changes).
    edge_agent.py             The edge agent: EdgeSettings (EDGE_* variables from .env, all optional),
                              startup of the local model server, a streaming terminal chat (main(), the
                              `edge-agent` command), and a speed check (check(), the `edge-agent-check`
                              command).
tests/
    test_cloud_agent.py          pytest tests for cloud_agent.py's graph, web window, and entry points
    test_edge_agent.py           pytest tests for edge_agent.py's settings, server startup, chat, and speed check
    test_agent_configuration.py  pytest tests for agent_configuration.py
pyproject.toml                Project metadata, dependency list, and the four CLI commands, managed by uv
.env.example                  Template for the .env file you create locally. .env itself is git-ignored,
                              so real API keys are never committed.
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. You do not need to create a virtualenv or pip install anything by hand.

1. Install uv if you do not have it (see the link above).
2. Clone this repo and `cd` into it.
3. Install dependencies:
   ```bash
   uv sync
   ```
4. Create your local config file:
   ```bash
   cp .env.example .env
   ```
5. For the cloud agent, get a free Google AI Studio API key from https://aistudio.google.com/apikey, then fill in `LLM_PROVIDER`, `LLM_MODEL`, and `API_KEY` in `.env`.
6. For the edge agent, nothing in `.env` is required. Every `EDGE_*` setting has a working default. The edge agent does need the local model server, set up once per Jetson (see below).

## Running the cloud agent

Launch the chat window:
```bash
uv run cloud-agent
```
This starts a local Gradio server (by default at `http://127.0.0.1:7860`) and prints the URL to open. Each browser tab gets its own conversation memory.

### Live-reloading while developing

Instead of `uv run cloud-agent`, use:
```bash
uv run cloud-agent-dev
```
This runs Gradio's own dev-reload CLI, watching the whole `src/` tree. An edit to any file in the package restarts the running app automatically, with no manual stop and restart. It also turns on debug logging (reply timing, graph build time). Under the hood it runs `uv run gradio src/capstone/cloud_agent.py --watch-dirs src` (see `dev()` in `cloud_agent.py`).

## Running the edge agent (Jetson only)

### One-time setup: the local model server

The edge agent talks to Gemma 4 E2B (Unsloth's 4-bit `Q4_K_S` GGUF), served by llama.cpp's `llama-server` inside a Docker container named `gemma`. `llama-server` exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, so the edge agent uses the standard `openai` client with no API key.

Create the container once:
```bash
docker run -d --name gemma --restart unless-stopped --runtime=nvidia --network host \
  -v $HOME/.cache/huggingface:/data/models/huggingface \
  ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin \
  llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S \
  --ctx-size 8192 --parallel 1 --n-gpu-layers 99 --host 127.0.0.1 --port 8080
```

The first start downloads the model (about 3.8 GB, including the 942 MB vision projector) into `~/.cache/huggingface`. Later starts load it from there. Because of `--restart unless-stopped`, the container comes back up on its own after a reboot.

Details that matter:

- **Use the `gemma4-jetson-orin` image tag, not `latest-jetson-orin`.** The `latest` tag is built for CUDA 13 (JetPack 7). On our JetPack 6.2 board it generated 2-3 tokens/sec. The `gemma4-jetson-orin` tag is built for CUDA 12.6 and generates about 20-24 tokens/sec.
- **`--runtime=nvidia` is required.** The Jetson's default Docker runtime is `runc`, which has no GPU access.
- **Mount the cache at `/data/models/huggingface`.** That is where this image looks for models. Mounting anywhere else re-downloads the model on every start.
- **Keep `--ctx-size` small.** The Jetson's 8 GB of RAM is shared between CPU and GPU. Without the flag, the server reserves a 131K-token context.
- **Docker permission errors:** if `docker` says "permission denied" right after you were added to the `docker` group, log out and back in. Until then, `sg docker -c "docker ps"` works.

Day-to-day control:
```bash
docker stop gemma          # Free the RAM (about 1.5 GB) for something else
docker start gemma         # Bring it back
docker logs -f gemma       # Watch it load, or see request timings
```

The llama.cpp web UI is at http://127.0.0.1:8080 while the container is running.

### Chatting with the edge agent

```bash
uv run edge-agent
```
This opens a chat in the terminal. Replies stream in as Gemma generates them, and the conversation is remembered until you quit. Type `exit` or press Ctrl+D to quit.

If the model server is not running, the edge agent runs `docker start gemma` itself and waits for the model to load. If that fails, it exits with the commands for starting the container by hand.

### Checking speed

```bash
uv run edge-agent-check
```
This sends one fixed prompt, prints the reply, and reports prompt and generation speed in tokens/sec, taken from `llama-server`'s own timings. Generation should land around 20-24 tokens/sec. Much lower usually means the wrong image tag, or layers not fully on the GPU (check `docker logs gemma 2>&1 | grep offloaded` for `36/36 layers`).

### Thinking mode

Gemma 4 can reason before answering. That reasoning returns separately from the reply, and it can use the whole token budget before any visible answer appears. It also multiplies reply time several times over. It is off by default. Set `EDGE_THINKING=true` in `.env` to turn it on, and the terminal chat shows `(thinking...)` while the model reasons.

## Running the tests

```bash
uv run pytest
```
No test makes a live network call. Model, docker, and HTTP calls are all mocked, so the tests run without the `gemma` container or an internet connection.

- `test_agent_configuration.py` and `test_edge_agent.py` never depend on your real `.env`. They set environment variables directly via pytest's `monkeypatch` fixture.
- `test_cloud_agent.py` imports `cloud_agent.py`, which builds the chat model at import time, so it needs the cloud agent's `.env` settings filled in.
- For a live end-to-end check of the edge agent against the real model server, run `uv run edge-agent-check`.

## Why Gemini for the cloud agent?

Google AI Studio has a free tier, which matters for a two-person capstone budget. Its daily free-tier request quota is easy to burn through while testing. A `RESOURCE_EXHAUSTED` error means the quota ran out, not that the code has a bug. Nothing in the design is Gemini-specific: see `agent_configuration.py` for how a second provider would be added, per the project's "model-agnostic" requirement.

## Adding a new dependency

```bash
uv add <package-name>
```

Add `--dev` for a development-only dependency (like `pytest`), e.g. `uv add --dev <package-name>`. This updates `pyproject.toml` and `uv.lock` automatically. Commit both files after running it.
