"""
Cloud agent: a hosted-LLM LangGraph agent with a Gradio chat window, which reaches the edge agent over A2A

- `agent_configuration.py`  Settings and chat-model construction
- `cloud_agent.py`          The LangGraph graph, plus the A2A connection to the edge agent (`askEdgeAgent` tool)
- `web_ui.py`               The Gradio chat window and the `cloud-agent` / `cloud-agent-dev` commands
"""
