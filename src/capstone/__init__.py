"""
Cloud and edge agents for the Resilient Cloud-to-Edge Agentic Robotic System

Both tiers live in this package, as subpackages, and communicate over HTTPS via A2A (Agent-to-Agent) protocol

Cloud agent (`cloud_agent/`):
- Runs on a laptop for now (eventually an always-on cloud host), backed by a hosted LLM (currently Google Gemini)
- Takes a natural-language command and turns it into an ordered sequence of robot actions
- Sends that sequence to the edge agent over A2A

Edge agent (`edge_agent/`):
- Runs on an NVIDIA Jetson Orin Nano connected to the physical robot arm
- Backed by Gemma 4 E2B running locally on the Jetson's GPU (keeps working without internet)
- Receives commands from the cloud agent
- Validates them against safety rules before any hardware moves
- Executes them via ROS 2

TLS certificates for the link come from `certificates.py` (the `make-certs` command)

The cloud agent is not trusted to be the final safety authority:
- It proposes a plan
- The edge agent enforces whether that plan is allowed to run
- This separation is a core design requirement of the project, not an implementation detail

See the top-level README.md for setup instructions, current status, and a description of what each module in this package is responsible for.
"""
