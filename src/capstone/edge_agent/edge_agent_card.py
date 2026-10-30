"""
The edge agent's A2A agent card - how the cloud agent discovers what the edge agent is and how to reach it

Agent card:
- Served by the A2A server at https://<jetson>:8443/.well-known/agent-card.json
- Advertises EDGE_PUBLIC_URL as the address to send messages to, not the bind address (0.0.0.0 isn't reachable)
- One skill for now, `chat`: a message goes to the local Gemma 4 model and its reply comes back
- No streaming and no push notifications - each request gets one complete reply
- No security scheme yet - the HTTPS link is encrypted, but the server doesn't authenticate who's calling
"""

# ---------------------------------
# Imports
# ---------------------------------
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
from capstone.edge_agent.edge_agent import EdgeSettings




# ---------------------------------
# Agent card
# ---------------------------------

SUPPORTED_CONTENT_TYPES = ["text/plain"]  # Plain text in, plain text out


def buildAgentCard(settings: EdgeSettings) -> AgentCard:
    """
    Build the edge agent's card

    Args:
        settings: Supplies the public URL the card advertises

    Returns:
        The AgentCard the A2A server publishes
    """

    chatSkill = AgentSkill (
        id="chat",
        name="Local Gemma chat",
        description="Answers a message with Gemma 4 E2B, running locally on the Jetson Orin Nano (works without internet)",
        tags=["chat", "edge", "gemma"],
        examples=["Are you online?", "Summarize what you can do in one sentence."]
    )

    # Only JSON-RPC is served (see edge_agent_server.py), so it's the only interface listed
    jsonRpcInterface = AgentInterface (
        url=settings.public_url.removesuffix("/") + "/",
        protocol_binding=TransportProtocol.JSONRPC,
        protocol_version=PROTOCOL_VERSION_CURRENT
    )

    return AgentCard (
        name="Edge Agent",
        description="Edge tier of the resilient cloud-to-edge robotic system, running on an NVIDIA Jetson Orin Nano",
        version="0.1.0",
        supported_interfaces=[jsonRpcInterface],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=SUPPORTED_CONTENT_TYPES,
        default_output_modes=SUPPORTED_CONTENT_TYPES,
        skills=[chatSkill]
    )
