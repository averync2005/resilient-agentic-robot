# cloud-agent

This is the **cloud-tier half** of a two-repo capstone project. If you only read one file in this repo, make it this one -- everything else assumes you've read this first.

## The two-repo split (read this first)

The full project ("Resilient Cloud-to-Edge Agentic Robotic System," ECE 257/258, Lehigh University, 2026-2027) has two codebases that talk to each other over a network:

| | This repo (`cloud-agent`) | The other repo (edge agent, not here) |
|---|---|---|
| Runs on | An always-on cloud host | An NVIDIA Jetson Orin Nano, bolted to the robot |
| Job | Turn a natural-language command into an ordered sequence of robot actions | Receive that sequence, check it's safe, actually move the robot arm via ROS 2 |
| Talks to the other side via | A2A (Agent-to-Agent) protocol | A2A (Agent-to-Agent) protocol |

A contributor working only on the edge/hardware side does not need to run or read this repository's code. The one thing that matters across the boundary is the A2A "contract" -- the shape of the messages this repo sends -- which will be documented here once it exists (it doesn't yet; see Status below).

## What this code does, in plain English

1. A person types a command in plain English (e.g. "flip switch 2 and press button 1").
2. This cloud agent sends that text to an LLM (currently Google's Gemini) and gets back a plan: an ordered list of robot actions from a fixed, pre-approved vocabulary (named poses like `switch_1`, `button_2`, `home`, `safe_retract` -- the robot cannot be told to do anything outside this list).
3. That plan gets sent to the Jetson over A2A. The Jetson is the one that actually validates and executes it -- this repo does not talk to the robot directly, and does not trust itself to be the final safety check.

## Status

Early scaffolding only. Right now this repo can: load API keys/config from a `.env` file, and send one test message to Gemini to confirm the whole chain works. It does **not** yet: talk to the edge repo over A2A, generate real action plans, or do anything with an actual robot. Those come later as the two-semester schedule progresses -- see the project proposal for the full timeline.

## Repository layout

```
src/cloud_agent/
    __init__.py       Package docstring -- a short version of this README, read by tools that inspect the package.
    config.py          Reads .env into a validated Settings object (which LLM provider/model to use, API keys).
    models.py           Builds a LangChain chat model from Settings. This is the ONE place that knows about
                        specific LLM vendor SDKs -- everything else in the codebase should call get_chat_model()
                        instead of importing e.g. langchain_google_genai directly. That's what "model-agnostic"
                        means in practice: swapping providers later touches this file and .env, nothing else.
    smoke_test.py       A manual script (not an automated test) that proves .env -> Settings -> get_chat_model()
                        -> a real Gemini response all works. Run it after setting up your .env file.
notebooks/ (if present) Scratch exploration of LangChain/LangGraph/A2A/MCP tutorials -- not part of the shipped
                        agent, safe to ignore.
pyproject.toml          Project metadata and dependency list, managed by uv (see "Setup" below).
.env.example             Template for the .env file you need to create locally. .env itself is git-ignored --
                        never commit real API keys.
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

5. Get a free Google AI Studio API key from https://aistudio.google.com/apikey and paste it into `.env` as `GOOGLE_API_KEY=...`.

6. Confirm everything's wired up correctly:
   ```bash
   uv run python -m cloud_agent.smoke_test
   ```

   You should see `Model replied:` followed by a short message from Gemini. If instead you see a `ValueError` about a missing API key, double check step 5.

## Why Gemini?

Google AI Studio has a genuinely free tier, which matters for a two-person capstone budget. Nothing in the design is Gemini-specific -- see `models.py` for how a second provider would be added later, per the project's "model-agnostic" requirement.

## Adding a new dependency

```bash
uv add <package-name>
```

This updates `pyproject.toml` and `uv.lock` automatically -- commit both files after running it.
