"""
Edge agent: a local Gemma 4 LangGraph agent on the Jetson Orin Nano, served to the cloud agent over A2A

- `edge_agent.py`         Settings, model-server activation, the LangGraph agent, terminal chat, and speed check
- `edge_agent_card.py`    The A2A agent card the cloud agent discovers
- `edge_agent_server.py`  The HTTPS A2A server and the `edge-agent-serve` command
"""
