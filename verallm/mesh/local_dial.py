"""Local transport addresses without changing advertised mesh identities."""

from urllib.parse import urlsplit, urlunsplit


_advertised_host = ""


def configure(endpoint: str) -> None:
    """Set this serving process's own configured advertised host."""
    global _advertised_host
    _advertised_host = urlsplit(endpoint).hostname or ""


def endpoint_for_host(endpoint: str, host: str) -> str:
    if not host:
        return endpoint
    is_url = "://" in endpoint
    try:
        parsed = urlsplit(endpoint if is_url else "//" + endpoint)
        port = parsed.port
    except ValueError:
        return endpoint
    if (
        parsed.hostname != host
        or parsed.scheme not in {"", "http"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return endpoint
    target = f"127.0.0.1:{port}"
    if not is_url:
        return target
    return urlunsplit((parsed.scheme, target, parsed.path, parsed.query, parsed.fragment))


def endpoint(endpoint: str) -> str:
    return endpoint_for_host(endpoint, _advertised_host)


def rpc_command_for_host(command: list[str], host: str) -> list[str]:
    result = list(command)
    for index, value in enumerate(result[:-1]):
        if value == "--rpc":
            result[index + 1] = ",".join(
                endpoint_for_host(item, host) for item in result[index + 1].split(",")
            )
    return result
