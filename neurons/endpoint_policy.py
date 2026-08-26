"""Shared endpoint URL admission policy.

This module is shipped with both miner and validator releases.  Proxy-only
modules are intentionally excluded from the public release tree, so runtime
roles must not import endpoint policy from those modules.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

_BLOCKED_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def is_safe_endpoint(url: str, *, allow_private: bool = False) -> bool:
    """Validate endpoint syntax, public address policy, and mainnet TLS."""

    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    if allow_private:
        return True
    if parsed.scheme != "https" or hostname in ("localhost", ""):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return True
    return not any(address in network for network in _BLOCKED_NETWORKS)
