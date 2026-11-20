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

Both agents are LangGraph agents. They talk to each other over A2A (Agent-to-Agent) protocol, using Google's `a2a-sdk`, over HTTPS on the USB-C link between the laptop and the Jetson. See "Connecting the two agents" below. For now the A2A "contract" is plain text: the cloud agent sends a message and the edge agent replies. The structured robot-action contract will be documented here once it exists.

## What this code does, in plain English

**Right now:**

- **Cloud agent:** a person types into a web chat window, and a LangGraph-backed agent running on Gemini replies, with memory of the conversation so far. When the person asks for the edge agent, the Jetson, or the robot, Gemini calls its `askEdgeAgent` tool, which sends a message to the edge agent over A2A and returns its reply.
- **Edge agent:** runs an HTTPS A2A server on the Jetson (`edge-agent-serve`). Each incoming message is answered by Gemma 4 through a small LangGraph graph, with no memory between messages. A terminal chat (`edge-agent`) is still there for talking to Gemma directly on the Jetson.

Neither agent generates robot action plans or moves a robot yet. Those come later in the two-semester schedule (see the project proposal for the full timeline).

**The finished picture:** the cloud agent turns a command into an ordered list of robot actions from a fixed, pre-approved vocabulary. That vocabulary is a set of named poses like `switch_1`, `button_2`, `home`, and `safe_retract`, and the robot is never told to do anything outside it. The cloud agent sends that plan to the Jetson over A2A, and the edge agent (not the cloud agent) validates and executes it. The cloud agent does not trust itself to be the final safety check. When the cloud is unreachable, the edge agent's local model is the fallback.

## Repository layout

```
src/capstone/
    __init__.py               Package docstring (a short version of this README, read by tools that
                              inspect the package).
    certificates.py           `make-certs`: creates the project's private certificate authority (CA) and
                              the Jetson's HTTPS server certificate, signed by it (see "Connecting the
                              two agents").
    cloud_agent/
        __init__.py             Subpackage docstring listing the modules below.
        agent_configuration.py  The cloud agent's Settings (reads .env into a validated object: LLM
                                provider, model, API key, edge agent URL, CA path) and getChatModel()
                                (builds a LangChain chat model from Settings). This is the ONE place that
                                knows about specific LLM vendor SDKs. Everything else calls
                                getChatModel() instead of importing ex. langchain_google_genai directly.
                                That is what "model-agnostic" means in practice: swapping providers
                                touches this file and .env, nothing else.
        cloud_agent.py          The LangGraph conversational graph (chatbot node, tools node,
                                per-conversation memory) and the A2A client that reaches the edge agent,
                                exposed to the model as the `askEdgeAgent` tool.
        web_ui.py               The Gradio web window. main() is the `cloud-agent` command's entry point.
                                dev() is `cloud-agent-dev`'s (same app, auto-reloads on source changes).
    edge_agent/
        __init__.py             Subpackage docstring listing the modules below.
        edge_agent.py           EdgeSettings (EDGE_* variables from .env, all optional), startup of the
                                local model server, the edge agent's LangGraph graph (Gemma via
                                ChatOpenAI), a streaming terminal chat (main(), the `edge-agent` command),
                                and a speed check (check(), the `edge-agent-check` command).
        edge_agent_card.py      The A2A agent card: who the edge agent is, its skills, and where to reach it.
        edge_agent_server.py    The HTTPS A2A server (serve(), the `edge-agent-serve` command) and the
                                executor that answers each message with the graph.
    arm_controller/
        __init__.py             Subpackage docstring listing the modules below.
        arm_driver.py           ArmDriver: moves, reads, and stops the xArm 1S's six servos over USB, in
                                degrees, with per-servo limits (SERVO_LIMITS). openArm() connects to it.
        poses.py                Loads and saves poses.yaml (named poses and button sequences), checking
                                every angle against SERVO_LIMITS.
        arm_actions.py          The named-pose interlock: runs "POSE <name>" / "PRESS <name>" only for
                                names in poses.yaml. parseAction() reads them out of Gemma's reply.
        arm_cli.py              The `arm` command (status, move, pose, press, teach).
tests/                        One pytest file per module above (test_<module>.py).
pyproject.toml                Project metadata, dependency list, and the six CLI commands, managed by uv.
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
docker run -d --name gemma --restart no --runtime=nvidia --network host \
  -v $HOME/.cache/huggingface:/data/models/huggingface \
  ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin \
  llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S \
  --ctx-size 8192 --parallel 1 --n-gpu-layers 99 --host 127.0.0.1 --port 8080
```

The first start downloads the model (about 3.8 GB, including the 942 MB vision projector) into `~/.cache/huggingface`. Later starts load it from there. Every edge command starts the container itself when it isn't running, so it doesn't need to come up on boot.

**Use `--restart no`, not `--restart unless-stopped`.** When the model fails to load (usually out of GPU memory), `llama-server` exits cleanly, and `unless-stopped` restarts it forever. On 11/5/26 that loop ran 214 times in the background, holding memory and hiding the real error behind a generic "didn't finish loading" timeout. To change an existing container: `docker update --restart=no gemma`.

Details that matter:

- **Use the `gemma4-jetson-orin` image tag, not `latest-jetson-orin`.** The `latest` tag is built for CUDA 13 (JetPack 7). On our JetPack 6.2 board it generated 2-3 tokens/sec. The `gemma4-jetson-orin` tag is built for CUDA 12.6 and generates about 20-24 tokens/sec.
- **`--runtime=nvidia` is required.** The Jetson's default Docker runtime is `runc`, which has no GPU access.
- **Mount the cache at `/data/models/huggingface`.** That is where this image looks for models. Mounting anywhere else re-downloads the model on every start.
- **Keep `--ctx-size` small.** The Jetson's 8 GB of RAM is shared between CPU and GPU. Without the flag, the server reserves a 131K-token context.
- **Run the Jetson without its desktop.** See "Freeing memory for Gemma" below. With the desktop up, Gemma may not load at all.
- **Docker permission errors:** if `docker` says "permission denied" right after you were added to the `docker` group, log out and back in. Until then, `sg docker -c "docker ps"` works.

Day-to-day control:
```bash
docker stop gemma          # Free the RAM (about 1.5 GB) for something else
docker start gemma         # Bring it back
docker logs -f gemma       # Watch it load, or see request timings
```

The llama.cpp web UI is at http://127.0.0.1:8080 while the container is running.

### Freeing memory for Gemma

Loading Gemma needs about 1.4 GB of GPU memory in one piece for the weights, on top of its context cache. GPU memory on the Jetson comes out of the same 8 GB as everything else, and swap doesn't count. The Ubuntu desktop plus a browser, VS Code, and the Claude desktop app used about 3 GB on 11/5/26, which was enough to stop Gemma from loading. With the desktop off, the board idles around 420 MB and Gemma loads in seconds. Table 2's S-4 edge memory envelope (7 GB threshold, 6 GB target) assumes this.

The symptom is `edge-agent` or `edge-agent-serve` giving up with "the model didn't finish loading within 300s". Confirm it in the logs:
```bash
docker logs gemma 2>&1 | grep -iE "out of memory|NvMap.*error 12|failed to load"
```

The fix is to boot the Jetson without the desktop and work on it over SSH from a laptop (see "Working on the Jetson from a laptop"):
```bash
sudo systemctl set-default multi-user.target    # Boot to a text console from now on
sudo reboot
```
To get the desktop back: `sudo systemctl set-default graphical.target`, then reboot. Avoid `sudo systemctl isolate multi-user.target`, which tears the desktop down live and can cut your SSH session with it. Change the default and reboot instead.

If memory is tight and a reboot isn't an option, clearing the file cache before starting can help: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`. `free -h` reports cache as "available", but the GPU driver doesn't reliably reclaim it.

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

## Working on the Jetson from a laptop

With the Jetson running without its desktop (see "Freeing memory for Gemma"), all work on it happens over SSH. The SSH server is installed with JetPack. Turn it on once with `sudo systemctl enable --now ssh`.

### The USB-C link

The Jetson's USB-C port doubles as a network link. Plug a data-capable USB-C cable (not a charge-only one) into a laptop, and the Jetson is at `192.168.55.1`. No setup is needed on the Jetson.

On Windows, two new adapters appear in `Get-NetAdapter`: "Remote NDIS Compatible Device" (RNDIS) and "UsbNcm Host Device" (NCM). **Use the NCM one.** On 11/5/26 the RNDIS adapter dropped about a quarter of all packets, in 3-4 second outages every 10-15 seconds, which kills SSH sessions. The Jetson's address handout over USB was also unreliable, so give the laptop a fixed address. From an administrator PowerShell, with the adapter names from `Get-NetAdapter`:

```powershell
Disable-NetAdapter -Name "Ethernet 2" -Confirm:$false     # The RNDIS adapter
New-NetIPAddress -InterfaceAlias "Ethernet 3" -IPAddress 192.168.55.100 -PrefixLength 24   # The NCM adapter
ping -t 192.168.55.1     # Should reply steadily. Ctrl+C to stop
```

Windows may rename the adapters if the cable moves to a different port. If the link stops working, check `Get-NetAdapter` again.

The Jetson's Wi-Fi address also works for SSH (`hostname -I` on the Jetson lists it), but Lehigh's public Wi-Fi is slow, and the address changes.

### SSH

```powershell
ssh capstone@192.168.55.1
```

To skip the password, copy your key over once (Windows has no `ssh-copy-id`):
```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh capstone@192.168.55.1 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```
Run `ssh-keygen -t ed25519` first if you don't have a key yet. If it asks to overwrite an existing key, answer `n`.

Practical notes:

- **Pasting:** in the classic PowerShell window, right-click pastes and Ctrl+V doesn't. In Windows Terminal, Ctrl+V works. If a pasted line comes out as `^[[200~...`, paste it again.
- **Paste one command at a time.** Lines pasted together get typed into whatever is running, including a `sudo` password prompt.
- **Viewing a Jetson web page on the laptop** (for example the cloud agent's chat window, when testing both agents on the Jetson): connect with `ssh -L 7860:127.0.0.1:7860 capstone@192.168.55.1`, then open `http://127.0.0.1:7860` on the laptop.
- **VS Code:** the Remote-SSH extension connects to `capstone@192.168.55.1` and gives you the editor and terminals on the Jetson.

## Connecting the two agents

The cloud agent runs on a laptop, and the edge agent runs on the Jetson. They are joined by the USB-C link (see "Working on the Jetson from a laptop") and talk A2A over HTTPS:

```text
Laptop (192.168.55.100)                                  Jetson (192.168.55.1)
  uv run cloud-agent                                       uv run edge-agent-serve
  Gemini decides to call askEdgeAgent  --- HTTPS :8443 --> A2A server -> LangGraph -> Gemma 4
                                       <-- reply --------
```

- The cloud agent first fetches the edge agent's **agent card** from `https://192.168.55.1:8443/.well-known/agent-card.json`. The card says who the edge agent is, what it can do (one `chat` skill for now), and where to send messages.
- It then sends the message as an A2A JSON-RPC request and waits for the complete reply. Gemma on the Jetson can take a while, so the timeout is 120 seconds (`CLOUD_EDGE_TIMEOUT`).
- The edge agent is stateless. Each message is answered on its own, and the cloud agent's own memory is what holds the conversation.
- The link is encrypted, but there is no authentication yet. Anything that can reach port 8443 can send the edge agent messages. The server listens on every network the Jetson is on by default (`EDGE_A2A_HOST=0.0.0.0`), so while the Jetson is also on Wi-Fi, set `EDGE_A2A_HOST=192.168.55.1`. Then only the laptop can reach it.
- A2A first ran laptop-to-Jetson over the USB-C link on 11/5/26. The defaults (`CLOUD_EDGE_URL`, `EDGE_PUBLIC_URL`, and the `make-certs` IP) all point at `192.168.55.1`, so no `.env` entries are needed for the link itself.

**Alternate link (Ethernet):** the Jetson also has an Ethernet port. To use it, give both sides static addresses on the cable, reissue the certificate with `uv run make-certs --ip <jetson address>`, and set `CLOUD_EDGE_URL` and `EDGE_PUBLIC_URL` to that address in `.env`.

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
| `edge.crt` | Jetson | The Jetson's server certificate, valid for `192.168.55.1` and `localhost` (`EDGE_TLS_CERT`) |
| `edge.key` | Jetson | The server certificate's private key (`EDGE_TLS_KEY`) |

Copy the server pair into the repo on the Jetson:

```bash
ssh <user>@192.168.55.1 "mkdir -p <repo>/certs"
scp certs/edge.crt certs/edge.key <user>@192.168.55.1:<repo>/certs/
```

If the Jetson's IP changes, run `uv run make-certs --ip <new ip>` and copy the two files again. The CA is reused, so the laptop doesn't need anything new.

**Where our CA actually lives:** on 11/5/26 the CA was created on the Jetson, not the laptop, so `ca.key` is on the Jetson and certificates get issued there. That works the same way with the roles reversed: run `make-certs` on the Jetson, and copy only `ca.crt` to the laptop:
```powershell
mkdir certs -Force
scp capstone@192.168.55.1:~/Desktop/resilient-agentic-robot/certs/ca.crt certs/
```
What matters is that there is only one CA. Running `make-certs` on both machines creates two unrelated CAs, and the laptop then rejects the Jetson's certificate.

### Running both agents

1. On the **Jetson**: `uv run edge-agent-serve`. This starts the `gemma` container if needed, then listens on port 8443 (on every network, or only on `EDGE_A2A_HOST` if set). Wait for the line `Edge agent A2A server listening on ...` before going on. The first load can take a few minutes. Every message it receives and every reply it sends is logged to the terminal.
2. On the **laptop**: `uv run cloud-agent`, open the chat window, and ask for the edge agent, ex. "Ask the edge agent if it's online."

It worked if the Jetson's terminal logs `Received: ...` and then `Replying: ...`. If Gemini answers but nothing is logged, it answered without calling `askEdgeAgent`. On 11/5/26, Gemma took under a second from `Received` to `Replying`.

**Setting variables on Windows:** the bash form `CLOUD_EDGE_URL=... uv run cloud-agent` doesn't work in PowerShell. Put the setting in `.env`, or set it on its own line first:
```powershell
$env:CLOUD_EDGE_URL = "https://192.168.55.1:8443"
uv run cloud-agent
```

**Testing both agents on the Jetson alone,** with no laptop link: run `EDGE_PUBLIC_URL=https://127.0.0.1:8443 uv run edge-agent-serve` in one SSH session, and `CLOUD_EDGE_URL=https://127.0.0.1:8443 uv run cloud-agent` in a second one opened with `ssh -L 7860:127.0.0.1:7860 ...`. This needs `EDGE_A2A_HOST` unset, or set to `127.0.0.1`. The certificate always covers `127.0.0.1`.

If the cloud agent says the edge agent couldn't be reached, the error text says why. The usual causes are:

- The USB-C link is down, or the laptop's fixed address on the NCM adapter is missing. Check with `ping 192.168.55.1`.
- `edge-agent-serve` isn't running, or hasn't finished loading Gemma.
- `certs/ca.crt` is missing on the laptop.
- The Jetson's certificate came from a different `make-certs` run than the laptop's `ca.crt`.
- `EDGE_PUBLIC_URL` is wrong. The cloud agent fetches the agent card from `CLOUD_EDGE_URL`, but then sends messages to whatever address the card advertises. A wrong advertised address shows up only as "All connection attempts failed", with no address in the message.
- Not a link problem: if the reply says the edge agent "was reached but failed to answer", the link worked and the edge agent itself failed (ex. Gemma crashed or ran out of memory). The Jetson's terminal has the full error.

Every address, port, and certificate path can be overridden in `.env`. See the `CLOUD_*` and `EDGE_*` entries in `.env.example`.

## The robot arm (xArm 1S)

The arm is a Hiwonder xArm 1S. Its controller board (on the bottom plate) takes a 7.5 V supply on the barrel jack and talks to a computer over its micro-USB port. The board shows up as a USB HID device, not a serial port. `arm_controller/arm_driver.py` drives it through the `xarm` package:

```python
from capstone.arm_controller.arm_driver import openArm

arm = openArm()
print(arm.readBatteryVoltage(), arm.readPositions())
arm.moveServos({6: 30.0, 4: -15.0}, durationMs=1500)
arm.stop()
```

- **Units:** degrees from each servo's center, -120 to 120. Servo IDs are 1 gripper, 2 wrist rotation, 3 wrist, 4 elbow, 5 shoulder, 6 base.
- **Limits:** every move is checked against `SERVO_LIMITS` before anything is sent. They start at ±110 degrees for every servo, 10 degrees inside the physical range. Narrow each one once it's measured on the real arm.
- **Move time:** `durationMs` is how long the move takes to arrive (1000 ms by default). Longer is slower and gentler on the arm and the supply.
- **`stop()`** holds every servo where it is. **`servosOff()`** makes the arm go limp, so support it first.
- Close Hiwonder's PC software first. Only one program can have the board open at a time.
- **On the Jetson (Linux),** a new USB device is readable only by root by default, so `openArm()` fails with "open failed". A udev rule (below) gives the `plugdev` group read/write access to the board. It's untested so far.

**One-time setup on the Jetson: the udev rule.** udev is the Linux service that creates a device file in `/dev` each time something is plugged in. A rule file tells it which permissions to give a matching device. The `xarm` package opens the board through libusb (a library for raw USB access) rather than `/dev/hidraw*`, so the rule covers both device types:

```bash
sudo tee /etc/udev/rules.d/99-xarm.rules > /dev/null <<'EOF'
# Hiwonder xArm 1S controller board (USB vendor 0483, product 5750)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5750", MODE="0660", GROUP="plugdev"
KERNEL=="hidraw*", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5750", MODE="0660", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then check that your user is in `plugdev` with `groups`. If it isn't, run `sudo usermod -aG plugdev $USER` and log out and back in. Unplug and replug the arm, then check with `lsusb -d 0483:5750` (the board should be listed) and `uv run python -c "import hid; print(hid.enumerate(0x0483, 0x5750))"` (should print one entry, not `[]`).

### Named poses, the `arm` command, and the edge agent

Approved poses and buttons live in `poses.yaml` at the repo root. Only names in that file can reach the arm. Every angle in it is checked against `SERVO_LIMITS` when it loads.

| Command | What it does |
|---|---|
| `uv run arm status` | Supply voltage and every servo's angle |
| `uv run arm move 6 20` | One servo move, printed as ACCEPTED or REJECTED (`uv run arm move 6 150` is rejected and sends nothing) |
| `uv run arm teach` | Teach poses by hand: `free 3 4 5` frees those servos, you position the arm, `save button_a_above` records it. `button button_a button_a_above button_a_press button_a_above` saves a press sequence |
| `uv run arm pose home` / `uv run arm press button_a` | Run a pose or button directly, no model |
| `uv run edge-agent-arm` | Terminal chat where Gemma picks the action ("go to home", "press button 1"). Lines starting with `/` skip the model (`/pose home`) |
| `EDGE_ARM_ENABLED=true uv run edge-agent-serve` | The A2A server with the arm, so cloud agent requests can move it |

Gemma never sends angles. It replies with one line, `POSE <name>` or `PRESS <name>`, and `arm_actions.py` looks the name up in `poses.yaml` before anything is sent. An unapproved name comes back as `REJECTED` with nothing sent. This is parsed by our code rather than through llama-server's tool calling, because Gemma 4's native tool-call format isn't reliably parsed there.

## Running the tests

```bash
uv run pytest
```
No test makes a live network call or needs the arm. Model, docker, HTTP, and arm calls are all mocked, so the tests run without the `gemma` container, the Jetson, or an internet connection. The A2A tests run the real edge agent server in-process, through httpx's ASGI transport, with a fake model behind it, so the cloud agent's A2A client is tested against the real server code.

- `test_agent_configuration.py`, the `test_edge_agent*.py` files, and `test_certificates.py` never depend on your real `.env`. They set values directly, via pytest's `monkeypatch` fixture or `_env_file=None`.
- `test_cloud_agent.py` and `test_web_ui.py` import `cloud_agent.py`, which builds the chat model at import time, so they need the cloud agent's `.env` settings filled in. The key is never used, so placeholder values work on a machine without a real `.env`: `LLM_PROVIDER=google LLM_MODEL=gemini-2.5-flash API_KEY=dummy uv run pytest`.
- For a live end-to-end check of the edge agent against the real model server, run `uv run edge-agent-check`.

## Why Gemini for the cloud agent?

Google AI Studio has a free tier, which matters for a two-person capstone budget. Its daily free-tier request quota is easy to burn through while testing. A `RESOURCE_EXHAUSTED` error means the quota ran out, not that the code has a bug. Nothing in the design is Gemini-specific: see `cloud_agent/agent_configuration.py` for how a second provider would be added, per the project's "model-agnostic" requirement.

## Adding a new dependency

```bash
uv add <package-name>
```

Add `--dev` for a development-only dependency (like `pytest`), ex. `uv add --dev <package-name>`. This updates `pyproject.toml` and `uv.lock` automatically. Commit both files after running it.
