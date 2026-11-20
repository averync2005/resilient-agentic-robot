"""
Cloud and edge agents for the Resilient Cloud-to-Edge Agentic Robotic System

Package layout:
- Both tiers live in this package, as subpackages
- The tiers communicate over HTTPS via A2A (Agent-to-Agent) protocol
- TLS certificates for the link come from `certificates.py` (the `make-certs` command)
- The robot arm driver lives in `arm_controller/`
- See the top-level README.md for setup instructions, current status, and what each module is responsible for

Cloud agent (`cloud_agent/`):
- Runs on a laptop for now (eventually an always-on cloud host), backed by a hosted LLM (currently Google Gemini)
- Takes a natural-language command and passes robot requests to the edge agent over A2A (`askEdgeAgent`)
- Planned: turning a command into an ordered sequence of named robot actions

Edge agent (`edge_agent/`):
- Runs on an NVIDIA Jetson Orin Nano connected to the physical robot arm
- Backed by Gemma 4 E2B running locally on the Jetson's GPU (keeps working without internet)
- Receives commands from the cloud agent
- Gemma names one action (a pose or a button press), and only names in poses.yaml are sent to the arm
- Moves the arm directly through `arm_controller/` for now (ROS 2 Actions planned)

Arm controller (`arm_controller/`):
- Drives the Hiwonder xArm 1S over its controller board's micro-USB port
- Checks every move against per-servo limits before anything is sent
- Named poses and button sequences come from poses.yaml (taught by hand with `uv run arm teach`)

Safety authority:
- The cloud agent is not trusted to be the final safety authority
- It proposes a plan
- The edge agent enforces whether that plan is allowed to run
- This separation is a core design requirement of the project, not an implementation detail
"""