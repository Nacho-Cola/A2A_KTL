# utils/agent_card.py
import requests
from urllib.parse import urlparse
from common.types import AgentCard

def get_agent_card(remote_agent_address: str) -> AgentCard:
    """
    Fetch the agent’s metadata from its /.well-known/agent.json.
    - If no scheme is given, default to http://
    - Strip any trailing slash before appending the path
    """
    addr = remote_agent_address.strip()

    # Default to http:// if no scheme present
    parsed = urlparse(addr)
    if not parsed.scheme:
        addr = "http://" + addr

    # Remove any trailing slash
    addr = addr.rstrip("/")

    # Build URL and fetch
    url = f"{addr}/.well-known/agent.json"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return AgentCard(**resp.json())
