# capstone

Code for the "Resilient Cloud-to-Edge Agentic Robotic System" (ECE 257/258, Lehigh University, 2026-2027). If you only read one file in this repo, make it this one. Everything else assumes you have read this first.

## The two tiers

The system splits its reasoning across two tiers that talk to each other over a network. Both tiers live in this repo, as two subpackages of the same `capstone` Python package.

| | Cloud agent (`cloud_agent/`) | Edge agent (`edge_agent/`) |
|---|---|---|
| Runs on | A laptop for now (eventually an always-on cloud host) | An NVIDIA Jetson Orin Nano Super, mounted on the robot |
| Model | A hosted LLM (currently Google Gemini) | Gemma 4 E2B, running locally on the Jetson's GPU |
| Job | Turn a natural-language command into an ordered sequence of robot actions | Receive that sequence, check that it is safe, and move the robot arm via ROS 2 |
| Without internet | Unavailable | Keeps working (the local model needs no network) |

Both agents are LangGraph agents. They talk to each other over A2A (Agent-to-Agent) protocol, using Google's `a2a-sdk`, over HTTPS on a direct Ethernet cable between the laptop and the Jetson. See "Connecting the two agents" below. For now the A2A "contract" is plain text: the cloud agent sends a message and the edge agent replies. The structured robot-action contract will be documented here once it exists.

## What this code does, in plain English

**Right now:**

- **Cloud agent:** a person types into a web chat window, and a LangGraph-backed agent running on Gemini replies, with memory of the conversation so far. When the person asks for the edge agent, the Jetson, or the robot, Gemini calls its `askEdgeAgent` tool, which sends a message to the edge agent over A2A and returns its reply
- **Edge agent:** runs an HTTPS A2A server on the Jetson (`edge-agent-serve`). Each incoming message is answered by Gemma 4 through a small LangGraph graph, with no memory between messages. A terminal chat (`edge-agent`) is still there for talking to Gemma directly on the Jetson

Neither agent generates robot action plans or moves a robot yet. Those come later in the two-semester schedule (see the project proposal for the full timeline).

**The finished picture:** the cloud agent turns a command into an ordered list of robot actions from a fixed, pre-approved vocabulary. That vocabulary is a set of named poses like `switch_1`, `button_2`, `home`, and `safe_retract`, and the robot is never told to do anything outside it. The cloud agent sends that plan to the Jetson over A2A, and the edge agent (not the cloud agent) validates and executes it. The cloud agent does not trust itself to be the final safety check. When the cloud is unreachable, the edge agent's local model is the fallback.

## Repository layout

```
src/capstone/
    __init__.py               Package docstring (a short version of this README, read by tools that
                              inspect the package)
    certificates.py           `make-certs`: creates the project's private certificate authority (CA) and
                              the Jetson's HTTPS server certificate, signed by it (see "Connecting the
                              two agents").
    cloud_agent/
        agent_configuration.py  The cloud agent's Settings (reads .env into a validated object: LLM
                                provider, model, API key, edge agent URL, CA path) and getChatModel()
                                (builds a LangChain chat model from Settings). This is the ONE place that
                                knows about specific LLM vendor SDKs. Everything else calls
                                getChatModel() instead of importing e.g. langchain_google_genai directly.
                                That is what "model-agnostic" means in practice: swapping providers
                                touches this file and .env, nothing else.
        cloud_agent.py          The LangGraph conversational graph (chatbot node, tools node,
                                per-conversation memory) and the A2A client that reaches the edge agent,
                                exposed to the model as the `askEdgeAgent` tool.
        web_ui.py               The Gradio web window. main() is the `cloud-agent` command's entry point.
                                dev() is `cloud-agent-dev`'s (same app, auto-reloads on source changes).
    edge_agent/
        edge_agent.py           EdgeSettings (EDGE_* variables from .env, all optional), startup of the
                                local model server, the edge agent's LangGraph graph (Gemma via
                                ChatOpenAI), a streaming terminal chat (main(), the `edge-agent` command),
                                and a speed check (check(), the `edge-agent-check` command).
        edge_agent_card.py      The A2A agent card: who the edge agent is, its skills, and where to reach it.
        edge_agent_server.py    The HTTPS A2A server (serve(), the `edge-agent-serve` command) and the
                                executor that answers each message with the graph.
tests/                        One pytest file per module above (test_<module>.py)
pyproject.toml                Project metadata, dependency list, and the six CLI commands, managed by uv
.env.example                  Template for the .env file you create locally. .env itself is git-ignored,
                              so real API keys are never committed.
certs/                        Created by `uv run make-certs`. Git-ignored, so keys are never committed.
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

To reach the edge agent from the chat window, the Jetson needs to be running `uv run edge-agent-serve` (see "Connecting the two agents"). Without it, the cloud agent still chats normally, and any edge agent request comes back explaining that the edge agent couldn't be reached.

### Live-reloading while developing

Instead of `uv run cloud-agent`, use:
```bash
uv run cloud-agent-dev
```
This runs Gradio's own dev-reload CLI, watching the whole `src/` tree. An edit to any file in the package restarts the running app automatically, with no manual stop and restart. It also turns on debug logging (reply timing, graph build time, edge agent replies). Under the hood it runs `uv run gradio src/capstone/cloud_agent/web_ui.py --watch-dirs src` (see `dev()` in `web_ui.py`).

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

## Connecting the two agents

The cloud agent runs on a laptop, and the edge agent runs on the Jetson. They are joined by a direct Ethernet cable and talk A2A over HTTPS:

```text
Laptop (192.168.10.1)                                    Jetson (192.168.10.2)
  uv run cloud-agent                                       uv run edge-agent-serve
  Gemini decides to call askEdgeAgent  --- HTTPS :8443 --> A2A server -> LangGraph -> Gemma 4
                                       <-- reply --------
```

- The cloud agent first fetches the edge agent's **agent card** from `https://192.168.10.2:8443/.well-known/agent-card.json`. The card says who the edge agent is, what it can do (one `chat` skill for now), and where to send messages.
- It then sends the message as an A2A JSON-RPC request and waits for the complete reply. Gemma on the Jetson can take a while, so the timeout is 120 seconds (`CLOUD_EDGE_TIMEOUT`).
- The edge agent is stateless. Each message is answered on its own, and the cloud agent's own memory is what holds the conversation.
- The link is encrypted, but there is no authentication yet. Anything that can reach port 8443 can send the edge agent messages, which over a direct cable means only the laptop.

### One-time setup: static IPs on the Ethernet link

There is no router on a direct cable, so each side needs a fixed address. Neither side gets a gateway, so both keep their internet through Wi-Fi.

**Laptop (Windows):** open Settings > Network & internet > Ethernet > IP assignment > Edit. Choose Manual, turn on IPv4, and set IP address `192.168.10.1` and subnet mask `255.255.255.0`. Leave the gateway and DNS blank. Or, from an administrator PowerShell:

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.10.1 -PrefixLength 24
```

**Jetson:** find the Ethernet interface name with `nmcli device` (often `eth0` or `enP8p1s0`), then:

```bash
sudo nmcli con add type ethernet ifname <interface> con-name capstone-link ipv4.method manual ipv4.addresses 192.168.10.2/24
sudo nmcli con up capstone-link
```

Check the link from the laptop with `ping 192.168.10.2`.

### One-time setup: HTTPS certificates

The Jetson has no domain name, so no public certificate authority can issue it a certificate. Instead, `make-certs` creates a private certificate authority (CA) just for this project and uses it to sign the Jetson's certificate. The cloud agent trusts only that CA, so nothing else can pretend to be the Jetson.

On the **laptop**, from the project root:

```bash
uv run make-certs
```

This writes four files to `certs/` (git-ignored):

| File | Stays on | Purpose |
|---|---|---|
| `ca.crt` | Laptop | The cloud agent checks the Jetson's certificate against this (`CLOUD_CA_CERT`) |
| `ca.key` | Laptop | Signs certificates. Never copy it anywhere |
| `edge.crt` | Jetson | The Jetson's server certificate, valid for `192.168.10.2` and `localhost` (`EDGE_TLS_CERT`) |
| `edge.key` | Jetson | The server certificate's private key (`EDGE_TLS_KEY`) |

Copy the server pair into the repo on the Jetson:

```bash
ssh <user>@192.168.10.2 "mkdir -p <repo>/certs"
scp certs/edge.crt certs/edge.key <user>@192.168.10.2:<repo>/certs/
```

If the Jetson's IP changes, run `uv run make-certs --ip <new ip>` and copy the two files again. The CA is reused, so the laptop doesn't need anything new.

### Running both agents

1. On the **Jetson**: `uv run edge-agent-serve`. This starts the `gemma` container if needed, then listens on `https://0.0.0.0:8443`. Every message it receives and every reply it sends is logged to the terminal.
2. On the **laptop**: `uv run cloud-agent`, open the chat window, and ask for the edge agent, e.g. "Ask the edge agent if it's online."

If the cloud agent says the edge agent couldn't be reached, the error text says why. The usual causes are:

- The cable or a static IP isn't set up. Check with `ping 192.168.10.2`.
- `edge-agent-serve` isn't running.
- `certs/ca.crt` is missing on the laptop.
- The Jetson's certificate came from a different `make-certs` run than the laptop's `ca.crt`.

Every address, port, and certificate path can be overridden in `.env`. See the `CLOUD_*` and `EDGE_*` entries in `.env.example`.

## Running the tests

```bash
uv run pytest
```
No test makes a live network call. Model, docker, and HTTP calls are all mocked, so the tests run without the `gemma` container, the Jetson, or an internet connection. The A2A tests run the real edge agent server in-process, through httpx's ASGI transport, with a fake model behind it, so the cloud agent's A2A client is tested against the real server code.

- `test_agent_configuration.py`, the `test_edge_agent*.py` files, and `test_certificates.py` never depend on your real `.env`. They set values directly, via pytest's `monkeypatch` fixture or `_env_file=None`.
- `test_cloud_agent.py` and `test_web_ui.py` import `cloud_agent.py`, which builds the chat model at import time, so they need the cloud agent's `.env` settings filled in.
- For a live end-to-end check of the edge agent against the real model server, run `uv run edge-agent-check`.

## Why Gemini for the cloud agent?

Google AI Studio has a free tier, which matters for a two-person capstone budget. Its daily free-tier request quota is easy to burn through while testing. A `RESOURCE_EXHAUSTED` error means the quota ran out, not that the code has a bug. Nothing in the design is Gemini-specific: see `cloud_agent/agent_configuration.py` for how a second provider would be added, per the project's "model-agnostic" requirement.

## Adding a new dependency

```bash
uv add <package-name>
```

Add `--dev` for a development-only dependency (like `pytest`), e.g. `uv add --dev <package-name>`. This updates `pyproject.toml` and `uv.lock` automatically. Commit both files after running it.
