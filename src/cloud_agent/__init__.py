"""
Cloud-tier agent for the Resilient Cloud-to-Edge Agentic Robotic System

This project is split across two repositories that communicate over the network:
- This repository (cloud-agent) runs on an always-on cloud host
- It takes a natural-language command and turns it into an ordered sequence of robot actions
- It sends that sequence to the edge agent over the A2A (Agent-to-Agent) protocol

- The edge agent is a separate repository, not included here
- It runs on an NVIDIA Jetson Orin Nano mounted to the physical robot arm
- It receives commands from this cloud agent
- It validates them against safety rules before any hardware moves
- It executes them via ROS 2

The cloud agent is not trusted to be the final safety authority:
- It proposes a plan
- The edge agent enforces whether that plan is allowed to run
- This separation is a core design requirement of the project, not an implementation detail

See the top-level README.md for setup instructions and a description of what each module in this package is responsible for.
"""
