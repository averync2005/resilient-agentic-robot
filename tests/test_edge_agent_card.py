"""
Tests for edge_agent_card.py's agent card

- EdgeSettings is always built with `_env_file=None`, so the developer's real .env never leaks in
"""

# ---------------------------------
# Imports
# ---------------------------------
from a2a.utils.constants import TransportProtocol
from capstone.edge_agent.edge_agent import EdgeSettings
from capstone.edge_agent.edge_agent_card import buildAgentCard




# ---------------------------------
# buildAgentCard tests
# ---------------------------------

def test_card_advertises_the_public_url_over_json_rpc():
    """
    The card should point the cloud agent at EDGE_PUBLIC_URL (never the 0.0.0.0 bind address), over JSON-RPC
    """

    card = buildAgentCard(EdgeSettings(_env_file=None))

    assert (len(card.supported_interfaces) == 1)
    assert (card.supported_interfaces[0].url == "https://192.168.10.2:8443/")
    assert (card.supported_interfaces[0].protocol_binding == TransportProtocol.JSONRPC)


def test_card_url_has_exactly_one_trailing_slash():
    """
    A public URL configured with or without a trailing slash should produce the same card URL
    """

    withSlash = buildAgentCard(EdgeSettings(_env_file=None, public_url="https://10.0.0.5:9000/"))
    withoutSlash = buildAgentCard(EdgeSettings(_env_file=None, public_url="https://10.0.0.5:9000"))

    assert (withSlash.supported_interfaces[0].url == "https://10.0.0.5:9000/")
    assert (withoutSlash.supported_interfaces[0].url == "https://10.0.0.5:9000/")


def test_card_has_one_chat_skill():
    """
    The only skill so far should be the local Gemma chat
    """

    card = buildAgentCard(EdgeSettings(_env_file=None))

    assert ([skill.id for skill in card.skills] == ["chat"])


def test_card_declares_no_streaming_and_plain_text_only():
    """
    Replies come back whole, as plain text
    """

    card = buildAgentCard(EdgeSettings(_env_file=None))

    assert (card.capabilities.streaming is False)
    assert (list(card.default_input_modes) == ["text/plain"])
    assert (list(card.default_output_modes) == ["text/plain"])
