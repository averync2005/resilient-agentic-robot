# agentic-fallback

A cloud-to-edge agentic robot system with defined behavior across complete loss of the reasoning tier.

## The problem

Agentic robot systems increasingly run their reasoning in the cloud for capability and cost. Existing cloud-edge robotics work tolerates latency and recovers from individual server failures, but treats a fully unreachable cloud as out of scope. This leaves no defined behavior when the link is gone: the robot either stops and waits, or acts unpredictably.

This matters for robots in environments with spotty connectivity—warehouse floors with signal dead zones, field equipment at remote sites, educational makerspaces with unreliable WiFi. When the cloud becomes unreachable mid-task, the robot should gracefully degrade rather than freeze or fail unsafely.

## What this does

A cloud-hosted reasoning agent communicates with a local agent on an NVIDIA Jetson Orin Nano. When the link is lost:

1. **Fault detection** (target: 1000 ms) — heartbeat-based detection with a bounded timeout
2. **Controlled stop** — the robot completes any in-flight atomic action, then moves to a safe configuration
3. **Local autonomy** — a restricted set of pre-authorized recipes can execute locally without cloud reasoning
4. **Durable queuing** — deferred actions are persisted with timestamps; expired or invalid actions are dropped on reconnect
5. **State reconciliation** — on reconnect, the cloud model and local agent align without duplicating or dropping actions

The system is model-agnostic: swap the cloud model, the local model, or the target hardware without modifying the robot's control code.

## Architecture

**Cloud tier** (persistent service, any host)  
Runs a reasoning agent. Communicates over the A2A (Agent-to-Agent) protocol.

**Local tier** (NVIDIA Jetson Orin Nano, ROS 2 Humble)  
Runs a local model and exposes hardware capability as MCP (Model Context Protocol) tools on ROS 2. Safety interlocks are enforced here, not in agent reasoning: every tool call is validated against pose constraints, ordering preconditions, speed limits, and safety invariants before motion.

**Hardware** (demonstration platform)  
A 6-DOF manipulator operates a fixtured benchtop control panel (3 toggle switches, 3 buttons). The panel state is read directly over GPIO. Eight named poses stand in for a vocabulary of skills. This is a stand-in for industrial equipment with no network interface.

## Key performance targets

| Spec | Threshold | Target |
|---|---|---|
| End-to-end command latency | 5000 ms | 2500 ms |
| Fault detection + safe stop | 1500 ms | 1000 ms |
| Recipe sequence accuracy | 80% | 90% |
| Reconciliation correctness | 95% | 100% |
| Interlock enforcement | 100% | 100% |

## Status

Demonstration expected Nov 20, 2026. Code released TBD. Repository in active development.

## Getting started

**Prerequisites**
- NVIDIA Jetson Orin Nano (8 GB) or compatible edge device
- ROS 2 Humble
- Python 3.10+
- A cloud host for the reasoning agent (any small always-on service; no serverless)

**Installation**

```bash
git clone https://github.com/[user]/agentic-fallback
cd agentic-fallback
pip install -r requirements.txt
```

**Run the demo**  
(Instructions TBD—repository in active development)

## Implementation notes

- **Recipes** are ordered sequences of named primitives with preconditions. Sequence dependence is intentional: it makes the fallback behavior substantive (outages mid-recipe leave the machine half-configured, which is the problem we solve).
- **Safety enforcement** lives in the MCP tool layer, not the agent. This decouples safety from model reasoning and ensures the interlock holds even under local autonomy.
- **Local model** emits only named poses, never raw numeric control. This constrains the action space and makes authorization tractable.
- **Perception and reasoning run as separate processes** to avoid GPU contention on resource-constrained edge hardware.

## Contributing

(Guidelines TBD)

## License

(TBD)

## Team

Avery Nudell-Cook (cloud agent, MCP tools, state machine, queue/reconciliation)  
Ali Mohrez (hardware, Jetson, ROS 2 integration)

ECE 257/258 Senior Design Capstone, Lehigh University, 2026–2027.
