"""Mesh onboarding machinery shared by every setup front-end.

One implementation behind two doors: the flag-driven agent path
(``verathos setup mesh-coordinator`` / ``mesh-worker``, neurons/mesh_wizard.py)
and the interactive human path (``verathos mesh setup``,
verallm/mesh/setup_wizard.py). Both build a :class:`MeshCoordinatorOptions`
and call :func:`setup_mesh_coordinator`; neither owns pool creation, PM2
naming, or health polling on its own. The previous split implementation is
how one box ended up with two pools and a manager crash-looping 4117 times
on a taken port.

This module lives in ``verallm`` because package boundaries flow one way:
``neurons`` imports ``verallm``, never the reverse. The manager subprocess
argv still says ``-m neurons.cli`` — that is a child process contract, not
an import.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

POOL_MANAGER_PM2_NAME = "verathos-pool-manager"
DEFAULT_MANAGER_PORT = 9500
_HEALTH_POLL_ATTEMPTS = 30
_HEALTH_POLL_INTERVAL_S = 0.5

# Operator-facing network names. chain_id/netuid are read from the shipped
# chain config files, never trusted from here: these constants exist only to
# map a name to a file and to sanity-check what the file says.
MESH_NETWORKS = {
    "testnet": {"chain_config": "chain_config_testnet.json", "netuid": 405},
    "mainnet": {"chain_config": "chain_config_mainnet.json", "netuid": 96},
}


@dataclass
class MeshCoordinatorOptions:
    serving_mode: str = "dev"  # dev | subnet ("validator" = legacy alias)
    manager_host: str = ""  # address workers dial; default detected public IP
    manager_port: int = DEFAULT_MANAGER_PORT
    manager_endpoint: str = ""  # full URL override (e.g. behind nginx TLS)
    tls_cert: str = ""
    tls_key: str = ""
    # Self-minted HTTPS API listener (additional to the plain --port).
    # This is the only TLS mode whose certificate the manager can PIN in
    # join tokens, so it is the wizard's default for remote-worker pools;
    # BYO cert files and nginx fronts cannot mint pinned join commands.
    api_tls_port: int = 0
    owner_account: str = ""  # SS58, required in subnet (validator) mode
    coordinator_address: str = ""
    validator_shared_state: str = ""
    chain_id: int | None = None
    netuid: int | None = None
    # Operator's OWN subtensor node (ws:// or http://). Recommended for
    # subnet pools: snapshot epoch follows the chain head and the hot
    # capacity audit self-derives windows from block polls - neither is
    # stable against the congested public RPC. Empty = the network's
    # public default.
    subtensor_endpoint: str = ""
    coordinator_uid: int | None = None
    epoch: int | None = None
    snapshot_ttl_seconds: int = 86_400
    pools_root: Path = field(
        default_factory=lambda: Path.home() / ".verathos" / "pools"
    )
    skip_install: bool = False
    non_interactive: bool = False
    # None = ask when interactive (default yes); non-interactive runs need
    # the explicit --join-local-gpus / --no-join-local-gpus choice.
    join_local_gpus: bool | None = None
    # GPU grouping spec forwarded to join_pool.sh --gpus ("0,1,2+3"). Empty
    # = the default topology: ONE worker owning every GPU (fastest; RPC only
    # between physical machines). Interactive setup asks when >1 GPU.
    join_gpu_groups: str = ""
    # None = refuse to mint when this machine already has a pool (the safe
    # default); True = explicitly requested a new pool (--new-pool or the
    # interactive "create another" choice).
    allow_new_pool: bool | None = None
    # Worker signing identity, forwarded to join_pool.sh for locally
    # joined GPUs. A subnet ("validator") pool's DRIVER hard-requires all
    # three at drive time; without them every launch dies after minutes of
    # waiting with no earlier symptom.
    worker_wallet_name: str = ""
    worker_wallet_hotkey: str = ""
    validator_allowlist_path: str = ""
    # Chain config the MANAGER needs for the on-chain lease renewer. A
    # subnet pool whose manager has no wallet + chain config stops renewing
    # `renewModel`, so the registration silently expires (24h lease) and the
    # miner drops off chain with nothing in the pool logs to explain it.
    # Resolved from --network; an explicit path stays an override.
    chain_config: str = ""


@dataclass
class MeshCoordinatorResult:
    pool_dir: Path
    pool_id: str
    manager_endpoint: str
    worker_token: str
    admin_token_file: Path
    pm2_name: str
    dashboard_url: str
    join_command: str
    reused: bool = False


@dataclass
class ManagerProbe:
    """What actually answers on a manager endpoint right now."""

    reachable: bool
    pool_id: str = ""
    error: str = ""


@dataclass
class ExistingSetup:
    """Everything a re-run must see before it may mint anything."""

    pools: list[Path]
    manager: ManagerProbe
    pm2_apps: list[str]

    @property
    def anything(self) -> bool:
        return bool(self.pools or self.manager.reachable or self.pm2_apps)


def _confirm(message: str, default: bool = True) -> bool:
    """Yes/no prompt; the seam tests patch instead of stdin."""

    hint = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {message} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(1)
    if not raw:
        return default
    return raw in ("y", "yes")


def find_repo_root() -> Path:
    """Walk up from cwd or this file to find the verathos pyproject."""

    for start in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
        candidate = start
        for _ in range(5):
            marker = candidate / "pyproject.toml"
            if marker.exists():
                try:
                    if "verathos" in marker.read_text():
                        return candidate
                except OSError:
                    pass
            candidate = candidate.parent
    raise SystemExit(
        "Cannot find the verathos repo root (no pyproject.toml). "
        "Run this from inside the verathos directory."
    )


def detect_public_ip() -> str:
    """Public IPv4 via external services; '' when detection fails."""

    for url in (
        "https://api.ipify.org",
        "https://ipv4.icanhazip.com",
        "https://ifconfig.me",
    ):
        try:
            probe = subprocess.run(
                ["curl", "-s", "-4", "--max-time", "5", url],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                ip = probe.stdout.strip()
                if ":" not in ip:
                    return ip
        except Exception:
            continue
    return ""


def endpoint_is_loopback(endpoint: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(endpoint).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def loopback_equivalent(endpoint: str) -> str:
    """The same manager endpoint dialled via loopback.

    Pool endpoints advertise the PUBLIC address so remote workers can dial
    them, but commands running on the coordinator box itself must not depend
    on hairpin NAT (a box that cannot reach its own public IP would look
    like a dead manager).
    """

    from urllib.parse import urlparse

    parsed = urlparse(endpoint.rstrip("/"))
    port = parsed.port or DEFAULT_MANAGER_PORT
    return f"{parsed.scheme or 'http'}://127.0.0.1:{port}"


def pool_management_request(
    pool_dir: Path,
    route: str,
    body: dict | None = None,
    timeout: float = 5.0,
    *,
    require_pool_id: bool = False,
) -> dict | None:
    """POST a management-authed request to the pool's LIVE manager.

    Returns the JSON payload, or None when no manager answers (down, wrong
    pool, unreadable token). Callers treat None as "manager unavailable"
    and degrade, never crash.
    """

    from verallm.mesh.pool import (
        POOL_ADMIN_TOKEN_FILE,
        load_pool_token_file,
    )

    try:
        token = load_pool_token_file(Path(pool_dir) / POOL_ADMIN_TOKEN_FILE)
    except (OSError, ValueError):
        return None
    for endpoint in dict.fromkeys(
        (loopback_equivalent(token.manager_endpoint), token.manager_endpoint)
    ):
        url = endpoint.rstrip("/") + route
        context = None
        if url.startswith("https://"):
            context = ssl._create_unverified_context()
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {**(body or {}), "management_secret": token.pool_secret}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=context
            ) as resp:
                payload = json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if require_pool_id and str(
            payload.get("pool_id", "") or ""
        ) != Path(pool_dir).name:
            continue
        return payload
    return None


def pool_management_status(
    pool_dir: Path, timeout: float = 5.0
) -> dict | None:
    """The LIVE manager's full status for a local pool dir, or None.

    Reading pool-state.json behind a running manager lies in both
    directions: the manager persists lazily, and an operator editing the
    file changes nothing in memory. Every local decision (is a mesh
    serving? are this machine's GPUs already members?) must prefer this
    view and only fall back to the state file when no manager answers.
    """

    return pool_management_request(
        pool_dir,
        "/v1/pool/status",
        timeout=timeout,
        require_pool_id=True,
    )


def pool_recommend(pool_dir: Path, model_id: str) -> dict:
    """Placement advice for one model from the live manager ({} when down).

    Returns the raw recommend payload: ``suggestions`` (ranked worker sets
    with link class and warnings) and ``reasons`` (per-worker text for
    workers that cannot take part). These strings are already written for
    humans; callers print them verbatim.
    """

    from verallm.mesh.pool import POOL_TOKEN_FILE, load_pool_token_file

    try:
        token = load_pool_token_file(Path(pool_dir) / POOL_TOKEN_FILE)
    except (OSError, ValueError):
        return {}
    for endpoint in dict.fromkeys(
        (loopback_equivalent(token.manager_endpoint), token.manager_endpoint)
    ):
        url = endpoint.rstrip("/") + "/v1/pool/recommend"
        context = None
        if url.startswith("https://"):
            context = ssl._create_unverified_context()
        request = urllib.request.Request(
            url,
            data=json.dumps({"model_id": model_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=10.0, context=context
            ) as resp:
                return json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return {}


def local_member_worker_ids(status: dict) -> list[str]:
    """Live pool workers that are THIS machine's GPUs.

    A worker serves from this machine when its advertised rpc/proof
    endpoint is loopback or this box's own public IP. Stale workers are
    excluded: a dead record must not block a legitimate re-join.
    """

    from urllib.parse import urlparse

    local_hosts = {"127.0.0.1", "localhost", "::1"}
    detected = detect_public_ip()
    if detected:
        local_hosts.add(detected)
    try:
        local_hosts.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    # NAT'd boxes advertise an interface address (`hostname -I`, exactly
    # what join_pool.sh defaults to), which matches neither the WAN IP nor
    # gethostbyname on stock Debian (127.0.1.1). Without these, the
    # double-join guard silently misses on the machine class most likely
    # to need it.
    try:
        probe = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5
        )
        local_hosts.update(
            part.strip().lower()
            for part in (probe.stdout or "").split()
            if part.strip()
        )
    except Exception:
        pass
    found: list[str] = []
    for worker_id, worker in (status.get("workers") or {}).items():
        if bool(worker.get("stale")):
            continue
        endpoints = worker.get("endpoints") or {}
        for endpoint in (endpoints.get("rpc", ""), endpoints.get("proof", "")):
            raw = str(endpoint or "")
            if not raw:
                continue
            if "//" not in raw:
                raw = "//" + raw
            host = (urlparse(raw).hostname or "").lower()
            if host in local_hosts:
                found.append(str(worker_id))
                break
    return found


def resolve_network(name: str, repo_root: Path | None = None) -> dict | None:
    """Map an operator network name to its chain binding, from the shipped file.

    Returns None for "local" (no chain binding at all). The chain_id and
    netuid come from the chain config FILE so they can never drift from what
    deploy/register will actually use; the constant table only names the file
    and cross-checks the netuid.
    """

    key = (name or "").strip().lower()
    if key in ("", "local", "none", "dev"):
        return None
    spec = MESH_NETWORKS.get(key)
    if spec is None:
        raise SystemExit(
            f"unknown network {name!r}; choose local, testnet, or mainnet"
        )
    root = repo_root or find_repo_root()
    config_path = root / spec["chain_config"]
    if not config_path.exists():
        raise SystemExit(f"{spec['chain_config']} not found in {root}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    chain_id = int(payload.get("chain_id") or 0)
    netuid = int(payload.get("netuid") or 0)
    if netuid != spec["netuid"]:
        raise SystemExit(
            f"{config_path} declares netuid {netuid}, expected "
            f"{spec['netuid']} for {key}; refusing a mismatched binding"
        )
    return {
        "network": key,
        "chain_config": str(config_path),
        "chain_id": chain_id,
        "netuid": netuid,
    }


def derive_subnet_setup_defaults(opts: MeshCoordinatorOptions) -> list[str]:
    """Fill derivable subnet-mode fields in place; returns note lines.

    A public operator cannot know --coordinator-address, --coordinator-uid
    or --epoch, and --owner-account is simply their own hotkey: every one
    of them is derivable from the
    worker wallet plus one chain read. Explicit flags always win — nothing
    already set is touched.
    """

    notes: list[str] = []
    if opts.serving_mode not in ("subnet", "validator"):
        return notes
    wallet = str(opts.worker_wallet_name or "").strip()
    hotkey = str(opts.worker_wallet_hotkey or "").strip()

    ss58 = ""
    seed = b""
    if wallet and hotkey:
        try:
            path = (
                Path.home()
                / ".bittensor"
                / "wallets"
                / wallet
                / "hotkeys"
                / hotkey
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            ss58 = str(data.get("ss58Address", "") or "")
            raw_seed = str(data.get("secretSeed", "") or "").replace("0x", "")
            seed = bytes.fromhex(raw_seed) if raw_seed else b""
        except Exception:
            pass

    if not opts.owner_account and ss58:
        opts.owner_account = ss58
        notes.append(f"owner-account = {wallet}/{hotkey} hotkey ({ss58})")

    if not opts.coordinator_address and len(seed) >= 32:
        try:
            from eth_account import Account
            from eth_utils import keccak

            opts.coordinator_address = Account.from_key(
                keccak(seed[:32])
            ).address
            notes.append(
                f"coordinator-address = hotkey EVM mirror "
                f"({opts.coordinator_address})"
            )
        except Exception:
            pass

    needs_uid = opts.coordinator_uid is None
    needs_epoch = opts.epoch is None
    if (needs_uid or needs_epoch) and opts.netuid is not None:
        try:
            import bittensor as bt

            from verallm.mesh.allowlist import normalize_subtensor_network

            network = normalize_subtensor_network(
                opts.subtensor_endpoint
                or ("test" if int(opts.netuid) == 405 else "finney")
            )
            ctor = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
            sub = ctor(network=network)
            if needs_uid and ss58:
                uid = sub.get_uid_for_hotkey_on_subnet(
                    ss58, int(opts.netuid)
                )
                if uid is not None:
                    opts.coordinator_uid = int(uid)
                    notes.append(f"coordinator-uid = {uid} (metagraph)")
            if needs_epoch:
                block = int(sub.get_current_block())
                epoch_blocks = 360
                try:
                    import urllib.request as _url

                    from verallm.mesh.manifest_store import (
                        default_store_urls_for_chain,
                    )

                    for base in default_store_urls_for_chain(
                        int(opts.chain_id or 0)
                    ):
                        try:
                            raw = _url.urlopen(
                                base.rstrip("/") + "/subnet-config.json",
                                timeout=10.0,
                            ).read()
                            blocks = int(
                                (json.loads(raw).get("epoch") or {}).get(
                                    "epoch_blocks", 0
                                )
                                or 0
                            )
                            if blocks > 0:
                                epoch_blocks = blocks
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
                opts.epoch = block // epoch_blocks
                notes.append(
                    f"epoch = {opts.epoch} (block {block} // {epoch_blocks};"
                    " a seed only, the pool follows the chain)"
                )
        except Exception as exc:
            notes.append(f"chain derivation unavailable: {exc}")
    return notes


def resolve_manager_endpoint(opts: MeshCoordinatorOptions) -> str:
    if opts.manager_endpoint:
        return opts.manager_endpoint.rstrip("/")
    host = opts.manager_host or "127.0.0.1"
    if opts.api_tls_port:
        return f"https://{host}:{int(opts.api_tls_port)}"
    scheme = "https" if (opts.tls_cert and opts.tls_key) else "http"
    return f"{scheme}://{host}:{opts.manager_port}"


def validator_preflight(opts: MeshCoordinatorOptions, endpoint: str) -> None:
    """Refuse a subnet-mode setup before any state is written."""

    missing = [
        name
        # --validator-shared-state is deliberately NOT here: it only feeds
        # the operator-board score display, which degrades gracefully to
        # the chain score cache, and public operators have no validator
        # install to point it at.
        for name, value in (
            ("--owner-account", opts.owner_account),
            ("--coordinator-address", opts.coordinator_address),
            ("--chain-id", opts.chain_id),
            ("--netuid", opts.netuid),
            ("--coordinator-uid", opts.coordinator_uid),
            ("--epoch", opts.epoch),
        )
        if value in ("", None)
    ]
    if missing:
        raise SystemExit(
            "subnet serving mode needs: " + ", ".join(missing)
        )
    is_loopback = endpoint.startswith("http://127.") or endpoint.startswith(
        "http://localhost"
    )
    if not endpoint.startswith("https://") and not is_loopback:
        raise SystemExit(
            "subnet serving mode requires an HTTPS manager endpoint.\n"
            "Either pass --tls-cert and --tls-key, or front the manager with\n"
            "nginx TLS and pass the public URL via --manager-endpoint:\n"
            "  sudo bash scripts/setup_https.sh --port 9543 --backend-port "
            f"{opts.manager_port} --append"
        )


def _mesh_python(repo: Path) -> str:
    """The interpreter the pool manager runs under."""

    venv_python = repo / ".venv-mesh" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_setup_script(repo: Path) -> None:
    """Install mesh dependencies; works in both release and source trees."""

    # scripts/setup_mesh.sh exists in every tree (a thin wrapper in the
    # source tree, the real script in release checkouts).
    for candidate in (
        repo / "scripts" / "setup_mesh.sh",
    ):
        if candidate.exists():
            subprocess.run(
                ["bash", str(candidate), "--role", "coordinator"],
                check=True,
                cwd=repo,
            )
            return
    raise SystemExit("scripts/setup_mesh.sh not found; is this a verathos repo?")


def probe_manager(endpoint: str, timeout: float = 3.0) -> ManagerProbe:
    """One /healthz round-trip: who is actually serving this endpoint?"""

    url = endpoint.rstrip("/") + "/healthz"
    context = None
    if url.startswith("https://"):
        context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as resp:
            payload = json.loads(resp.read() or b"{}")
        return ManagerProbe(
            reachable=True, pool_id=str(payload.get("pool_id", "") or "")
        )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return ManagerProbe(reachable=False, error=str(exc))


def _poll_manager_health(endpoint: str, expected_pool_id: str) -> None:
    """Wait for /healthz and require the pool id to match.

    A stale manager from a previous pool answering on the same port is the
    silent failure this catches; a bare port check would call it healthy.
    """

    last_error = "no response"
    for _ in range(_HEALTH_POLL_ATTEMPTS):
        probe = probe_manager(endpoint)
        if probe.reachable and probe.pool_id == expected_pool_id:
            return
        if probe.reachable:
            last_error = (
                f"manager on {endpoint} serves pool {probe.pool_id!r}, "
                f"expected {expected_pool_id!r} (stale manager?)"
            )
        else:
            last_error = probe.error
        time.sleep(_HEALTH_POLL_INTERVAL_S)
    raise SystemExit(f"pool manager did not become healthy: {last_error}")


def manager_crash_loop_error(
    pm2_name: str, settle_seconds: float = 8.0
) -> str:
    """Return the unit's recent error output if it is crash-looping, else ''.

    A crash-looping PM2 unit reports "online" between exits and can pass a
    single health probe inside an up-window. The restart counter is the only signal PM2 cannot fake:
    sample it twice across a settle window and fail on movement.
    """

    def _restarts() -> int:
        out = subprocess.run(
            ["pm2", "jlist"], capture_output=True, text=True
        )
        try:
            for app in json.loads(getattr(out, "stdout", "") or "[]"):
                if app.get("name") == pm2_name:
                    return int(
                        app.get("pm2_env", {}).get("restart_time", 0)
                    )
        except (ValueError, TypeError, AttributeError):
            pass
        return -1

    first = _restarts()
    if first < 0:
        return ""
    time.sleep(settle_seconds)
    second = _restarts()
    if second < 0 or second - first < 2:
        return ""
    logs = subprocess.run(
        ["pm2", "logs", pm2_name, "--nostream", "--lines", "10", "--err"],
        capture_output=True,
        text=True,
    )
    tail = [
        line
        for line in (getattr(logs, "stdout", "") or "").strip().splitlines()
        if line.strip()
    ][-6:]
    detail = "\n  ".join(tail) if tail else "no error output captured"
    return (
        f"{pm2_name} is crash-looping "
        f"(restarts {first} -> {second} in {settle_seconds:.0f}s):\n  "
        + detail
    )


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, int(port))) == 0


def pm2_mesh_apps() -> list[str]:
    """Names of mesh-related PM2 apps on this machine, [] without PM2."""

    if shutil.which("pm2") is None:
        return []
    try:
        listing = subprocess.run(
            ["pm2", "jlist"], capture_output=True, text=True, timeout=15
        )
        apps = json.loads(listing.stdout or "[]")
    except Exception:
        return []
    names = []
    for app in apps:
        name = str(app.get("name", ""))
        if (
            name == POOL_MANAGER_PM2_NAME
            or name.startswith(POOL_MANAGER_PM2_NAME + "-")
            or name.startswith("verathos-mesh-")
        ):
            names.append(name)
    return names


def scan_existing_setup(
    pools_root: Path, manager_port: int = DEFAULT_MANAGER_PORT
) -> ExistingSetup:
    """Everything already on this machine that a setup re-run must respect."""

    from verallm.mesh.pool import known_pool_dirs

    pools: list[Path] = []
    seen: set[str] = set()
    for candidate in known_pool_dirs():
        resolved = str(Path(candidate).resolve())
        if resolved not in seen and (Path(candidate) / "pool-state.json").exists():
            seen.add(resolved)
            pools.append(Path(candidate))
    for root in {Path(pools_root), Path.home() / ".verathos" / "mesh-pool"}:
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob("pool-*")):
            resolved = str(candidate.resolve())
            if resolved not in seen and (candidate / "pool-state.json").exists():
                seen.add(resolved)
                pools.append(candidate)
    return ExistingSetup(
        pools=pools,
        manager=probe_manager(f"http://127.0.0.1:{manager_port}", timeout=1.5),
        pm2_apps=pm2_mesh_apps(),
    )


def join_command_for_token(token_text: str) -> str:
    return (
        "curl -fsSL https://verathos.ai/install.sh | bash -s -- "
        f"--mesh-worker --token {token_text}"
    )


def result_from_existing_pool(
    pool_dir: Path, *, reused: bool = True
) -> MeshCoordinatorResult:
    """Rebuild the coordinator result for a pool that already exists."""

    from verallm.mesh.pool import (
        POOL_ADMIN_TOKEN_FILE,
        POOL_TOKEN_FILE,
        load_pool_token_file,
    )

    token = load_pool_token_file(pool_dir / POOL_TOKEN_FILE)
    token_text = token.encode()
    return MeshCoordinatorResult(
        pool_dir=pool_dir,
        pool_id=token.pool_id,
        manager_endpoint=token.manager_endpoint,
        worker_token=token_text,
        admin_token_file=pool_dir / POOL_ADMIN_TOKEN_FILE,
        pm2_name=POOL_MANAGER_PM2_NAME,
        dashboard_url=token.manager_endpoint + "/operator",
        join_command=join_command_for_token(token_text),
        reused=reused,
    )


def manager_pm2_name_for_pool(pool_id: str) -> str:
    """The PM2 app name a SECOND pool's manager must use.

    The default name is a singleton; `pm2 delete` + `pm2 start` under the
    same name would silently kill the FIRST pool's manager whenever an
    additional pool is minted (`--new-pool`), leaving its workers
    heartbeating into the void.
    """

    suffix = pool_id.removeprefix("pool-")[:12] or pool_id
    return f"{POOL_MANAGER_PM2_NAME}-{suffix}"


def _pm2_start_manager(
    repo: Path,
    pool_dir: Path,
    opts: MeshCoordinatorOptions,
    pm2_name: str = POOL_MANAGER_PM2_NAME,
) -> list[str]:
    """Start the pool manager under PM2 and return the argv used."""

    from verallm.mesh.pool import (
        POOL_SERVING_MODE_SUBNET,
        _normalize_pool_serving_mode,
        subtensor_network_for_chain_id,
    )

    serve_args = [
        "-m",
        "neurons.cli",
        "mesh",
        "pool",
        "serve",
        "--pool",
        str(pool_dir),
        "--host",
        "0.0.0.0",
        "--port",
        str(opts.manager_port),
    ]
    if opts.tls_cert and opts.tls_key:
        serve_args += ["--tls-cert", opts.tls_cert, "--tls-key", opts.tls_key]
    if opts.api_tls_port:
        serve_args += ["--api-tls-port", str(int(opts.api_tls_port))]
    # A subnet pool's manager owns the on-chain lease: without the signing
    # wallet and chain config it comes up healthy and serves fine, then the
    # registration expires a day later. The operator already supplies this
    # identity for the driver, so carry it through instead of making them
    # restart the unit by hand.
    if (
        _normalize_pool_serving_mode(opts.serving_mode)
        == POOL_SERVING_MODE_SUBNET
        and opts.worker_wallet_name
    ):
        serve_args += [
            "--wallet-name",
            opts.worker_wallet_name,
            "--wallet-hotkey",
            opts.worker_wallet_hotkey or "default",
        ]
        if opts.chain_config:
            serve_args += ["--chain-config", opts.chain_config]
        subtensor_network = (
            opts.subtensor_endpoint
            or subtensor_network_for_chain_id(opts.chain_id)
        )
        if subtensor_network:
            serve_args += ["--subtensor-network", subtensor_network]
    # Autorestart is correct here, unlike workers: the manager owns no GPU
    # children and has no in-process crash breaker to reset.
    argv = [
        "pm2",
        "start",
        _mesh_python(repo),
        "--name",
        pm2_name,
        "--interpreter",
        "none",
        "--",
        *serve_args,
    ]
    if shutil.which("pm2") is None:
        raise SystemExit(
            "PM2 not found. Install it (npm install -g pm2) and re-run,\n"
            "or start the manager manually:\n  "
            + " ".join([_mesh_python(repo), *serve_args])
        )
    subprocess.run(
        ["pm2", "delete", pm2_name],
        capture_output=True,
        cwd=repo,
    )
    subprocess.run(argv, check=True, cwd=repo)
    subprocess.run(["pm2", "save"], check=True, capture_output=True, cwd=repo)
    return argv


def start_manager_for_pool(
    pool_dir: Path, opts: MeshCoordinatorOptions
) -> None:
    """Bring the manager up (or back up) for an existing pool dir."""

    repo = find_repo_root()
    _pm2_start_manager(repo, pool_dir, opts)
    local_scheme = "https" if (opts.tls_cert and opts.tls_key) else "http"
    _poll_manager_health(
        f"{local_scheme}://127.0.0.1:{opts.manager_port}", pool_dir.name
    )
    # The health poll can land inside a crash loop's up-window; only the
    # restart counter proves the manager actually STAYS up.
    crash = manager_crash_loop_error(POOL_MANAGER_PM2_NAME)
    if crash:
        raise SystemExit(f"pool manager did not stay healthy: {crash}")


def setup_mesh_coordinator(opts: MeshCoordinatorOptions) -> MeshCoordinatorResult:
    from verallm.mesh.pool import POOL_ADMIN_TOKEN_FILE, create_pool_state

    repo = find_repo_root()
    endpoint = resolve_manager_endpoint(opts)
    if opts.serving_mode in ("subnet", "validator"):
        validator_preflight(opts, endpoint)
    elif opts.serving_mode != "dev":
        raise SystemExit(f"unknown serving mode: {opts.serving_mode}")

    # Never mint a second pool silently. A re-run on a configured machine is
    # almost always an accident (this exact accident once left a manager
    # crash-looping 4117 times against a taken port); requiring the explicit
    # choice costs one flag when a second pool really is wanted.
    if opts.allow_new_pool is not True:
        existing = scan_existing_setup(opts.pools_root, opts.manager_port)
        if existing.anything:
            details = []
            if existing.pools:
                details.append(
                    "pools: " + ", ".join(str(p) for p in existing.pools)
                )
            if existing.manager.reachable:
                details.append(
                    f"a manager already answers on port {opts.manager_port} "
                    f"for pool {existing.manager.pool_id!r}"
                )
            if existing.pm2_apps:
                details.append("pm2 apps: " + ", ".join(existing.pm2_apps))
            raise SystemExit(
                "this machine already has mesh state:\n  "
                + "\n  ".join(details)
                + "\nReuse it (verathos mesh setup offers this), or pass "
                "--new-pool to mint another pool anyway."
            )
    elif port_in_use(opts.manager_port):
        # Explicit new pool, but the port is held by something PM2 does not
        # own for us: refuse rather than crash-loop against it.
        probe = probe_manager(f"http://127.0.0.1:{opts.manager_port}")
        holder = (
            f"pool {probe.pool_id!r}" if probe.reachable else "another process"
        )
        if POOL_MANAGER_PM2_NAME not in pm2_mesh_apps():
            raise SystemExit(
                f"port {opts.manager_port} is already held by {holder} and "
                "PM2 does not manage it, so it cannot be replaced safely. "
                "Stop that process first or pick another --port."
            )

    if not opts.skip_install:
        _run_setup_script(repo)

    pool_dir, worker_token = create_pool_state(
        opts.pools_root,
        manager_endpoint=endpoint,
        serving_mode=opts.serving_mode,
        owner_account=opts.owner_account,
        coordinator_address=opts.coordinator_address,
        validator_shared_state_path=opts.validator_shared_state,
        chain_id=opts.chain_id,
        netuid=opts.netuid,
        coordinator_uid=opts.coordinator_uid,
        epoch=opts.epoch,
        snapshot_ttl_seconds=opts.snapshot_ttl_seconds,
    )
    pool_id = pool_dir.name
    pm2_name = POOL_MANAGER_PM2_NAME
    if pm2_name in pm2_mesh_apps():
        # An ADDITIONAL pool must never `pm2 delete` the first pool's
        # manager out from under its workers; give it its own app name.
        pm2_name = manager_pm2_name_for_pool(pool_id)
    _pm2_start_manager(repo, pool_dir, opts, pm2_name=pm2_name)
    # Poll locally: the public endpoint may only resolve from outside.
    local_scheme = "https" if (opts.tls_cert and opts.tls_key) else "http"
    _poll_manager_health(
        f"{local_scheme}://127.0.0.1:{opts.manager_port}", pool_id
    )
    crash = manager_crash_loop_error(pm2_name)
    if crash:
        raise SystemExit(f"pool manager did not stay healthy: {crash}")

    if opts.api_tls_port:
        # The self-minted certificate exists only after the manager's
        # first start, so the token from create_pool_state cannot carry
        # its pin. Bake it in now so remote joins do not fail certificate
        # verification.
        cert_path = pool_dir / "api-tls-cert.pem"
        if cert_path.is_file():
            import hashlib as _hashlib
            from dataclasses import replace as _dc_replace

            der = ssl.PEM_cert_to_DER_cert(
                cert_path.read_text(encoding="utf-8")
            )
            worker_token = _dc_replace(
                worker_token,
                manager_endpoint=endpoint,
                manager_ca_sha256=_hashlib.sha256(der).hexdigest(),
            )

    token_text = worker_token.encode()
    return MeshCoordinatorResult(
        pool_dir=pool_dir,
        pool_id=pool_id,
        manager_endpoint=endpoint,
        worker_token=token_text,
        admin_token_file=pool_dir / POOL_ADMIN_TOKEN_FILE,
        pm2_name=pm2_name,
        dashboard_url=endpoint + "/operator",
        join_command=join_command_for_token(token_text),
    )


def print_coordinator_result(result: MeshCoordinatorResult) -> None:
    from verallm.mesh import render

    print()
    verb = "is live (reused)" if result.reused else "is live"
    print(render.bold(f"  Pool {result.pool_id} {verb}."))
    print(f"    Dashboard:   {result.dashboard_url}")
    print(f"    Manager PM2: {result.pm2_name} (pm2 logs {result.pm2_name})")
    print(f"    Admin token: {result.admin_token_file}")
    print(render.dim("    The admin token controls the pool. Never share it."))
    print()
    if endpoint_is_loopback(result.manager_endpoint):
        print(
            render.warn(
                "the manager endpoint is loopback, so this join token only "
                "works on THIS machine."
            )
        )
        print(
            render.dim(
                "    To add other machines, re-create with a reachable "
                "--host (or --manager-endpoint behind TLS)."
            )
        )
        print()
        print(render.bold("  Add this machine's GPUs (local only):"))
    else:
        print(render.bold("  Add a GPU machine (run on that machine):"))
    print(f"    bash scripts/join_pool.sh --token {result.worker_token}")
    print(
        render.dim(
            "    (from a verathos checkout; the curl one-liner below works "
            "after the next public release)"
        )
    )
    print(render.dim(f"    {result.join_command}"))
    print()
    print(render.dim("  The worker token only lets a machine join and serve."))
    print(render.dim("  Print it again later: verathos mesh pool join-token"))
    print()


def maybe_join_local_gpus(
    result: MeshCoordinatorResult, opts: MeshCoordinatorOptions
) -> bool:
    """Offer to enroll this machine's own GPUs as worker units.

    The coordinator role needs no GPU, but the common first pool is one GPU
    box that is also the coordinator. Both roles share .venv-mesh and run
    as separate PM2 units, so joining locally is the same join_pool.sh flow
    every other machine uses; the worker units simply dial their own
    manager. Returns True when a local join ran.
    """

    from verallm.mesh import render
    from verallm.mesh import units
    from verallm.mesh.units import detect_gpus

    gpus = detect_gpus()
    if not gpus:
        if opts.join_local_gpus:
            print("  No CUDA GPU detected on this machine; nothing to join.")
        return False
    # Re-running setup on a machine whose GPUs already serve in this pool
    # must not enroll them AGAIN: that once stacked a second pair of worker
    # units onto two busy GPUs (4 workers on 2 GPUs). Membership is read
    # from the LIVE manager, never guessed from local files.
    status = pool_management_status(result.pool_dir)
    if status is not None:
        local_members = local_member_worker_ids(status)
        # A worker may own several GPUs; compare GPU COVERAGE, not worker
        # count, or one all-GPU worker would look like 1-of-4 enrolled.
        covered_gpus = sum(
            max(
                1,
                int(
                    (status.get("workers") or {})
                    .get(worker_id, {})
                    .get("capability", {})
                    .get("gpu_count", 1)
                    or 1
                ),
            )
            for worker_id in local_members
        )
        if local_members and covered_gpus >= len(gpus):
            print(
                render.ok(
                    "this machine's GPU(s) already serve in the pool as "
                    + ", ".join(sorted(local_members))
                )
            )
            print(
                render.dim(
                    "  skipping the local join. To re-enroll, stop those "
                    "workers first (verathos mesh status shows them)."
                )
            )
            return False
        if local_members:
            print(
                render.warn(
                    f"{covered_gpus} of this machine's {len(gpus)} "
                    "GPU(s) already serve in the pool ("
                    + ", ".join(sorted(local_members))
                    + ")"
                )
            )
            print(
                render.dim(
                    "  joining again would double-book the busy GPU(s); "
                    "skipping. Stop the existing workers first to re-enroll "
                    "everything cleanly."
                )
            )
            return False
    # Units already enrolled for THIS pool but not currently joined
    # (crashed, wedged, or unable to reach the manager) need a RESTART,
    # not a reinstall: re-running the join offer over live units reads
    # like the double-join bug even when unit names make it idempotent.
    try:
        registry = units.load_unit_registry()
    except Exception:
        registry = None
    if registry and str(registry.get("pool_id", "") or "") == result.pool_id:
        apps = pm2_mesh_apps()
        enrolled = [
            unit.pm2_name
            for unit in units.units_from_registry(registry)
            if unit.pm2_name in apps
        ]
        if enrolled:
            print(
                render.warn(
                    "this machine's GPUs are already enrolled as PM2 units "
                    f"({', '.join(enrolled)}) but not joined to the pool "
                    "right now."
                )
            )
            restart = opts.join_local_gpus
            if restart is None and not opts.non_interactive:
                restart = _confirm(
                    "restart the existing worker units?", default=True
                )
            if restart:
                for name in enrolled:
                    subprocess.run(
                        ["pm2", "restart", name], capture_output=True
                    )
                print(render.ok(f"restarted {len(enrolled)} worker unit(s)"))
            else:
                print(
                    render.dim(
                        "  restart them later: verathos mesh start --all"
                    )
                )
            return bool(restart)
    join = opts.join_local_gpus
    if join is None and not opts.non_interactive:
        join = _confirm(
            f"{len(gpus)} GPU(s) detected on this machine. "
            "Join them to the pool as worker units?",
            default=True,
        )
    if not join:
        print(
            render.dim(
                f"  Skipped joining this machine's {len(gpus)} GPU(s). "
                "Join later with the printed one-liner."
            )
        )
        return False
    gpu_groups = _choose_gpu_grouping(gpus, opts)
    print(f"  Joining this machine's {len(gpus)} GPU(s) to the pool...")
    passthrough = worker_passthrough(opts)
    if gpu_groups:
        passthrough += ["--gpus", gpu_groups]
    setup_mesh_worker(result.worker_token, passthrough)
    return True


def _choose_gpu_grouping(gpus: list, opts: MeshCoordinatorOptions) -> str:
    """Pick the worker topology for this machine's GPUs.

    Default (and the non-interactive fallback) is one worker owning every
    GPU: the model tensor-splits across local devices inside one process,
    and RPC is only used between physical machines. Splitting into groups
    trades that speed for running several independent models on one box.
    """

    if opts.join_gpu_groups:
        return str(opts.join_gpu_groups)
    if len(gpus) <= 1 or opts.non_interactive:
        return ""
    from verallm.mesh import render

    n = len(gpus)
    print(f"  How should this machine's {n} GPUs join?")
    print(
        f"    1) one worker with all {n} GPUs "
        + render.dim("(recommended: fastest, fits the biggest models)")
    )
    print(
        "    2) one worker per GPU "
        + render.dim(f"(run up to {n} smaller models independently)")
    )
    print(
        "    3) custom groups "
        + render.dim("(e.g. 0,1,2+3 = two single-GPU workers + one 2-GPU worker)")
    )
    while True:
        raw = input("  choice [1]: ").strip() or "1"
        if raw == "1":
            return ""
        if raw == "2":
            return ",".join(str(gpu.index) for gpu in gpus)
        if raw == "3":
            spec = input("  groups (e.g. 0,1,2+3): ").strip()
            from verallm.mesh.units import parse_gpu_groups

            try:
                parse_gpu_groups(spec, [gpu.index for gpu in gpus])
            except ValueError as exc:
                print(render.warn(f"  {exc}"))
                continue
            return spec
        print(render.warn("  pick 1, 2, or 3"))


def set_manager_api_tls_port(port: int) -> int:
    """Reconfigure the running manager's https API listener (day 2).

    Rebuilds the PM2 unit with its CURRENT argv (wallet/chain flags
    preserved byte for byte) plus the new ``--api-tls-port``, because
    ``pm2 restart`` reuses cached args and would silently ignore the
    change. ``port=0`` disables the listener. Returns the manager's plain
    http port for the health poll. Raises SystemExit with guidance when
    no manager unit runs on this box.
    """

    import json as json_module

    completed = subprocess.run(
        ["pm2", "jlist"], capture_output=True, text=True
    )
    try:
        procs = json_module.loads(completed.stdout or "[]")
    except ValueError:
        procs = []
    unit = next(
        (
            proc
            for proc in procs
            if proc.get("name") == POOL_MANAGER_PM2_NAME
        ),
        None,
    )
    if unit is None:
        raise SystemExit(
            "no running pool manager PM2 unit on this box; this command "
            "reconfigures the coordinator (verathos mesh setup starts it)"
        )
    env = unit.get("pm2_env") or {}
    script = str(env.get("pm_exec_path") or "")
    argv = [str(item) for item in (env.get("args") or [])]
    if not script or not argv:
        raise SystemExit("could not read the manager unit's argv from pm2")
    cleaned: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--api-tls-port":
            skip_next = True
            continue
        cleaned.append(item)
    if int(port):
        cleaned += ["--api-tls-port", str(int(port))]
    manager_port = DEFAULT_MANAGER_PORT
    for index, item in enumerate(cleaned):
        if item == "--port" and index + 1 < len(cleaned):
            try:
                manager_port = int(cleaned[index + 1])
            except ValueError:
                pass
    subprocess.run(
        ["pm2", "delete", POOL_MANAGER_PM2_NAME], capture_output=True
    )
    subprocess.run(
        [
            "pm2", "start", script,
            "--name", POOL_MANAGER_PM2_NAME,
            "--interpreter", "none",
            "--", *cleaned,
        ],
        check=True,
    )
    subprocess.run(["pm2", "save"], capture_output=True)
    return manager_port


def test_api_tls_endpoint(host: str, port: int, timeout: float = 10.0):
    """One live request against the exposed https API: (ok, detail).

    A 401 with the API-key hint is the PASS signature: the listener is
    reachable from this network path, terminates TLS, and enforces key
    auth. Certificate verification is skipped (self-signed by default;
    the operator pins the printed fingerprint)."""

    import urllib.error
    import urllib.request

    url = f"https://{host}:{int(port)}/v1/models"
    context = ssl._create_unverified_context()
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            return False, (
                f"HTTP {response.status} WITHOUT a key: auth is not "
                "enforced, do not expose this"
            )
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            pass
        if exc.code == 401 and "API key" in body:
            return True, "reachable, TLS terminated, key auth enforced (401)"
        return False, f"HTTP {exc.code} with unexpected body {body[:160]!r}"
    except Exception as exc:
        return False, (
            f"unreachable: {str(exc)[:200]} (is the port mapped/open in "
            "the provider firewall?)"
        )


def api_tls_pubkey_pin(pool_dir: Path) -> str:
    """curl-compatible SPKI pin (base64) of the pool's API TLS cert.

    Usable directly as ``--pinnedpubkey 'sha256//<pin>'``. Self-signed
    endpoints still require the client to disable CA verification; the
    public-key pin then authenticates the expected manager key.
    """

    certfile = Path(pool_dir) / "api-tls-cert.pem"
    if not certfile.is_file():
        return ""
    try:
        extracted = subprocess.run(
            ["openssl", "x509", "-pubkey", "-noout", "-in", str(certfile)],
            capture_output=True,
        )
        if extracted.returncode != 0 or not extracted.stdout:
            return ""
        encoded = subprocess.run(
            ["openssl", "pkey", "-pubin", "-outform", "der"],
            input=extracted.stdout,
            capture_output=True,
        )
    except OSError:
        return ""
    if encoded.returncode != 0 or not encoded.stdout:
        return ""
    digest = hashlib.sha256(encoded.stdout).digest()
    return base64.b64encode(digest).decode("ascii")


def api_tls_cert_fingerprint(pool_dir: Path) -> str:
    """SHA256 fingerprint of the pool's API TLS cert ("" if absent)."""

    certfile = Path(pool_dir) / "api-tls-cert.pem"
    if not certfile.is_file():
        return ""
    completed = subprocess.run(
        [
            "openssl", "x509", "-in", str(certfile),
            "-noout", "-fingerprint", "-sha256",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip().split("=", 1)[-1]


def worker_passthrough(opts: MeshCoordinatorOptions) -> list[str]:
    """join_pool.sh flags derived from the coordinator options."""

    passthrough: list[str] = []
    if opts.worker_wallet_name and opts.worker_wallet_hotkey:
        passthrough += [
            "--wallet-name",
            opts.worker_wallet_name,
            "--wallet-hotkey",
            opts.worker_wallet_hotkey,
        ]
    if opts.validator_allowlist_path:
        passthrough += [
            "--validator-allowlist-path",
            opts.validator_allowlist_path,
        ]
    # A subnet worker keeps its own validators.json fresh (the in-worker
    # allowlist refresher needs network + netuid); without these flags the
    # allowlist goes stale and the pool refuses the box as a driver.
    if opts.worker_wallet_name and opts.netuid is not None:
        from verallm.mesh.pool import subtensor_network_for_chain_id

        network = subtensor_network_for_chain_id(opts.chain_id)
        if network:
            passthrough += [
                "--subtensor-network",
                network,
                "--netuid",
                str(opts.netuid),
            ]
            if opts.chain_config:
                passthrough += ["--chain-config", opts.chain_config]
    return passthrough


def materialize_worker_token(token_text: str) -> Path:
    """Persist a pasted vtpool_ token as an owner-only file, like join_pool.sh.

    The secret must never sit in shell history or a managed process argv;
    everything downstream takes --pool-token-file.
    """

    target_dir = Path.home() / ".verathos"
    target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = target_dir / "pool-token.txt"
    fd, tmp_name = tempfile.mkstemp(dir=str(target_dir), prefix=".pool-token-")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token_text.strip() + "\n")
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def describe_worker_token(token_text: str) -> dict:
    """Decode a vtpool_ token and probe where it points, before acting.

    A wrong or loopback token used to fail as an endless silent 5s retry
    loop inside the worker; surfacing pool id, endpoint, and reachability
    up front turns hours of confusion into one line.
    """

    from verallm.mesh.pool import MeshPoolToken

    token = MeshPoolToken.decode(token_text.strip())
    probe = probe_manager(token.manager_endpoint, timeout=5.0)
    return {
        "pool_id": token.pool_id,
        "manager_endpoint": token.manager_endpoint,
        "loopback": endpoint_is_loopback(token.manager_endpoint),
        "reachable": probe.reachable,
        "reachable_pool_id": probe.pool_id,
        "error": probe.error,
    }


def setup_mesh_worker(token: str, passthrough: list[str]) -> None:
    """Hand off to join_pool.sh, which owns the worker install end to end."""

    repo = find_repo_root()
    script = repo / "scripts" / "join_pool.sh"
    if not script.exists():
        raise SystemExit("scripts/join_pool.sh not found; is this a verathos repo?")
    argv = ["bash", str(script)]
    if token:
        argv += ["--token", token]
    argv += list(passthrough)
    completed = subprocess.run(argv, cwd=repo)
    if completed.returncode != 0:
        # SystemExit(int) exits SILENTLY - the wizard used to just stop
        # here with nothing on screen when the join script died without
        # output of its own. Always say what failed and how loudly.
        raise SystemExit(
            f"worker join failed: join_pool.sh exited {completed.returncode}. "
            "Scroll up for its last output; if it ended with no message, "
            "rerun as 'bash -x scripts/join_pool.sh ...' to trace the "
            "aborting command."
        )
