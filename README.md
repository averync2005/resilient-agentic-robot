# cloud-agent

This is the **cloud-tier half** of a two-repo capstone project. If you only read one file in this repo, make it this one -- everything else assumes you've read this first.

## The two-repo split (read this first)

The full project ("Resilient Cloud-to-Edge Agentic Robotic System," ECE 257/258, Lehigh University, 2026-2027) has two codebases that talk to each other over a network:

| | This repo (`cloud-agent`) | The other repo (edge agent, not here) |
|---|---|---|
| Runs on | An always-on cloud host | An NVIDIA Jetson Orin Nano, bolted to the robot |
| Job | Turn a natural-language command into an ordered sequence of robot actions | Receive that sequence, check it's safe, actually move the robot arm via ROS 2 |
| Talks to the other side via | A2A (Agent-to-Agent) protocol | A2A (Agent-to-Agent) protocol |

A contributor working only on the edge/hardware side does not need to run or read this repository's code. The one thing that matters across the boundary is the A2A "contract" -- the shape of the messages this repo sends -- which will be documented here once it exists (it doesn't yet; see "What this code does" below).

## What this code does, in plain English

Right now: a person types a message into a web chat window, and a LangGraph-backed agent (currently running on Google's Gemini) replies -- with memory of the conversation so far. That's the foundation, not the finished picture. It does **not** yet generate real robot action plans, talk to the edge repo over A2A, or do anything with an actual robot -- those come later as the two-semester schedule progresses (see the project proposal for the full timeline). Concretely, once the rest of the pipeline exists: this cloud agent will turn a natural-language command into an ordered list of robot actions from a fixed, pre-approved vocabulary (named poses like `switch_1`, `button_2`, `home`, `safe_retract` -- the robot will never be told to do anything outside that list), send that plan to the Jetson over A2A, and let the Jetson -- not this repo -- validate and execute it. This repo does not trust itself to be the final safety check.

## Repository layout

```
src/cloud_agent/
    __init__.py              Package docstring -- a short version of this README, read by tools that inspect
                              the package.
    agent_configuration.py    Defines Settings (reads .env into a validated object: LLM provider/model, API
                              key) and get_chat_model() (builds a LangChain chat model from Settings). This is
                              the ONE place that knows about specific LLM vendor SDKs -- everything else in the
                              codebase should call get_chat_model() instead of importing e.g.
                              langchain_google_genai directly. That's what "model-agnostic" means in practice:
                              swapping providers later touches this file and .env, nothing else.
    agent.py                  The LangGraph conversational graph (chatbot node, placeholder tools node,
                              per-conversation memory) and the Gradio web window that talks to it. `main()` is
                              the `cloud-agent` CLI command's entry point; `dev()` is `cloud-agent-dev`'s
                              (same app, auto-reloads on source changes).
tests/
    test_agent_configuration.py  pytest tests for agent_configuration.py -- see "Running the tests" below.
pyproject.toml               Project metadata and dependency list, managed by uv (see "Setup" below).
.env.example                  Template for the .env file you need to create locally. .env itself is
                              git-ignored -- never commit real API keys.
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies -- you don't need to manually create a virtualenv or pip install anything.

1. Install uv if you don't have it (see the link above).
2. Clone this repo and `cd` into it.
3. Install dependencies:
   ```bash
   uv sync
   ```

4. Create your local config file:
   ```bash
   cp .env.example .env
   ```

5. Get a free Google AI Studio API key from https://aistudio.google.com/apikey and paste it into `.env` as `API_KEY=...`.

## Running it

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
This runs Gradio's own dev-reload CLI under the hood, watching the whole `src/` tree -- so an edit to either `agent.py` or `agent_configuration.py` automatically restarts the running app, no manual stop/restart needed. (It's a thin wrapper around `uv run gradio src/cloud_agent/agent.py --watch-dirs src` -- see `dev()` in `agent.py` if you want to see exactly what it runs.)

## Running the tests

```bash
uv run pytest
```
Tests never depend on your real `.env` -- they set environment variables directly via pytest's `monkeypatch` fixture, and never make a live network call (chat-model tests only verify object construction, not `.invoke()`).

## Why Gemini?

Google AI Studio has a genuinely free tier, which matters for a two-person capstone budget (and its daily free-tier request quota is easy to burn through while testing -- if you hit a `RESOURCE_EXHAUSTED` error, that's why, not a real bug). Nothing in the design is Gemini-specific -- see `agent_configuration.py` for how a second provider would be added later, per the project's "model-agnostic" requirement.

## Adding a new dependency

```bash
uv add <package-name>
```

Add `--dev` for a development-only dependency (like `pytest`), e.g. `uv add --dev <package-name>`. This updates `pyproject.toml` and `uv.lock` automatically -- commit both files after running it.
