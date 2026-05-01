#!/usr/bin/env python3
"""Interactive post-install wizard for Verathos miners and validators.

Guides new users through wallet creation, subnet registration, EVM funding,
HTTPS setup, PM2 installation, and ecosystem.config.js generation.

Each step is idempotent — detects current state and skips completed steps.

Usage::

    python -m neurons.wizard miner       # miner setup wizard
    python -m neurons.wizard validator   # validator setup wizard
    python -m neurons.wizard status      # preflight readiness check

Or via the ``verathos`` CLI::

    verathos setup              # defaults to miner
    verathos setup validator
    verathos status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Terminal helpers (ANSI — no external deps)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: int, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str:
    return _c(32, t)


def red(t: str) -> str:
    return _c(31, t)


def yellow(t: str) -> str:
    return _c(33, t)


def bold(t: str) -> str:
    return _c(1, t)


def dim(t: str) -> str:
    return _c(2, t)


def _ok() -> str:
    return green("✓")


def _fail() -> str:
    return red("✗")


def _skip() -> str:
    return yellow("—")


def _prompt(msg: str, default: str = "") -> str:
    """Prompt user for input with an optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return val or default


def _confirm(msg: str, default: bool = True) -> bool:
    """Yes/no prompt."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        val = input(f"  {msg} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not val:
        return default
    return val in ("y", "yes")


def _header(step: int, total: int, title: str) -> None:
    print()
    print(bold(f"  [{step}/{total}] {title}"))
    print(f"  {'─' * 50}")


def _banner(role: str) -> None:
    print()
    print(bold("  ═══════════════════════════════════════════════"))
    print(bold(f"   Verathos {role.title()} Setup Wizard"))
    print(bold("  ═══════════════════════════════════════════════"))
    print()


# ---------------------------------------------------------------------------
# Repo detection
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk up from cwd or script location to find pyproject.toml with verathos."""
    for start in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        d = start
        for _ in range(5):
            if (d / "pyproject.toml").exists():
                try:
                    if "verathos" in (d / "pyproject.toml").read_text():
                        return d
                except OSError:
                    pass
            d = d.parent
    print(red("  ERROR: Cannot find Verathos repo root (no pyproject.toml)."))
    print("  Run this from inside the verathos directory.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Network detection
# ---------------------------------------------------------------------------

def step_network(repo: Path) -> dict:
    """Detect mainnet vs testnet chain config."""
    mainnet = repo / "chain_config_mainnet.json"
    testnet = repo / "chain_config_testnet.json"
    # Also check plain chain_config.json
    plain = repo / "chain_config.json"

    has_mainnet = mainnet.exists() or plain.exists()
    has_testnet = testnet.exists()

    if has_mainnet and has_testnet:
        print("  Found both mainnet and testnet chain configs.")
        choice = _prompt("Network", "mainnet")
        if choice.startswith("test"):
            return {"network": "test", "netuid": 405, "config_path": str(testnet)}
        return {"network": "finney", "netuid": 96, "config_path": str(mainnet if mainnet.exists() else plain)}
    elif has_mainnet:
        path = str(mainnet if mainnet.exists() else plain)
        print(f"  Using mainnet config: {path}")
        return {"network": "finney", "netuid": 96, "config_path": path}
    elif has_testnet:
        print(f"  Using testnet config: {testnet}")
        return {"network": "test", "netuid": 405, "config_path": str(testnet)}
    else:
        print(yellow("  No chain config found — defaulting to mainnet."))
        return {"network": "finney", "netuid": 96, "config_path": ""}


# ---------------------------------------------------------------------------
# Step 2: Wallet detection
# ---------------------------------------------------------------------------

def _list_wallets() -> list[dict]:
    """List existing Bittensor wallets with their hotkeys."""
    wallets_dir = Path.home() / ".bittensor" / "wallets"
    if not wallets_dir.exists():
        return []
    result = []
    for w in sorted(wallets_dir.iterdir()):
        if not w.is_dir():
            continue
        hotkeys_dir = w / "hotkeys"
        hotkeys = []
        if hotkeys_dir.exists():
            hotkeys = sorted(f.name for f in hotkeys_dir.iterdir()
                            if f.is_file() and not f.name.endswith('.txt'))
        if hotkeys:
            result.append({"name": w.name, "hotkeys": hotkeys})
    return result


def step_wallet(role: str) -> dict:
    """Detect or create a Bittensor wallet."""
    wallets = _list_wallets()

    if wallets:
        print("  Existing wallets:")
        for i, w in enumerate(wallets):
            hks = ", ".join(w["hotkeys"][:3])
            if len(w["hotkeys"]) > 3:
                hks += f", ... (+{len(w['hotkeys']) - 3})"
            print(f"    [{i + 1}] {w['name']}  (hotkeys: {hks})")
        print()
        choice = _prompt(f"Select wallet", "1")

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(wallets):
                wallet = wallets[idx]
                name = wallet["name"]
                if len(wallet["hotkeys"]) == 1:
                    hotkey = wallet["hotkeys"][0]
                    print(f"  Using wallet: {name}/{hotkey}")
                else:
                    print(f"  Hotkeys for {name}:")
                    for j, hk in enumerate(wallet["hotkeys"]):
                        print(f"    [{j + 1}] {hk}")
                    hk_choice = _prompt("Select hotkey", "1")
                    if hk_choice.isdigit():
                        hk_idx = int(hk_choice) - 1
                        if 0 <= hk_idx < len(wallet["hotkeys"]):
                            hotkey = wallet["hotkeys"][hk_idx]
                        else:
                            hotkey = wallet["hotkeys"][0]
                    else:
                        hotkey = hk_choice  # typed name directly
                return {"wallet": name, "hotkey": hotkey}

    # No wallets found
    print(yellow("  No wallets found on this machine."))
    print()
    print("  Set up your wallet first, then re-run verathos setup.")
    print("  Never store the coldkey private key on a rented GPU server.")
    print()
    print("  Option 1: Regenerate on this server (recommended)")
    print("    btcli wallet regen-coldkeypub --wallet.name <NAME> --ss58-address <YOUR_COLDKEY_SS58>")
    print("    btcli wallet regen-hotkey --wallet.name <NAME> --hotkey <HOTKEY>")
    print()
    print("  Option 2: Copy from your local machine")
    print("    scp ~/.bittensor/wallets/<NAME>/coldkeypub.txt  root@<THIS_SERVER>:~/.bittensor/wallets/<NAME>/")
    print("    scp ~/.bittensor/wallets/<NAME>/hotkeys/<HOTKEY> root@<THIS_SERVER>:~/.bittensor/wallets/<NAME>/hotkeys/")
    print()
    return {"wallet": "", "hotkey": ""}


# ---------------------------------------------------------------------------
# Step 3: Subnet registration check
# ---------------------------------------------------------------------------

def step_registration(wallet_info: dict, net_info: dict) -> dict:
    """Check if the hotkey is registered on the subnet."""
    wallet_name = wallet_info["wallet"]
    hotkey_name = wallet_info["hotkey"]
    network = net_info["network"]
    netuid = net_info["netuid"]

    hk_path = Path.home() / f".bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}"
    if not hk_path.exists():
        print(red(f"  Hotkey file not found: {hk_path}"))
        print("  Create it first, then re-run the wizard.")
        return {"registered": False, "uid": None}

    # Try to query metagraph
    try:
        import bittensor as bt
        sub = bt.Subtensor(network=network)
        mg = sub.metagraph(netuid=netuid)

        # Read hotkey SS58
        hk_data = json.loads(hk_path.read_text())
        ss58 = hk_data.get("ss58Address", "")
        if not ss58:
            # Try alternative field names
            ss58 = hk_data.get("address", "")

        if ss58 and ss58 in mg.hotkeys:
            uid = mg.hotkeys.index(ss58)
            print(f"  Found UID {uid} on netuid {netuid} ({network}) {_ok()}")
            return {"registered": True, "uid": uid}
        else:
            print(yellow(f"  Not registered on netuid {netuid} ({network})"))
    except Exception as e:
        print(yellow(f"  Could not query metagraph: {e}"))
        print("  Will check registration status when starting the miner.")
        return {"registered": False, "uid": None}

    # Not registered — show instructions and offer retry
    print()
    print("  Register with (requires coldkey — run on your local machine):")
    print(f"    btcli subnet register --wallet.name {wallet_name} --hotkey {hotkey_name} "
          f"--netuid {netuid} --subtensor.network {network}")
    print()
    while True:
        if _confirm("Check registration again?"):
            try:
                mg = bt.Subtensor(network=network).metagraph(netuid=netuid)
                if ss58 in mg.hotkeys:
                    uid = mg.hotkeys.index(ss58)
                    print(f"  Found UID {uid} on netuid {netuid} ({network}) {_ok()}")
                    return {"registered": True, "uid": uid}
                print(yellow("  Still not registered."))
            except Exception as e:
                print(yellow(f"  Check failed: {e}"))
        else:
            break

    return {"registered": False, "uid": None}


# ---------------------------------------------------------------------------
# Step 4: EVM funding
# ---------------------------------------------------------------------------

def _check_evm_balance(
    hotkey_seed: bytes,
    network: str,
    chain_config_path: str = "",
) -> tuple[str, str, float]:
    """Derive EVM address, SS58 mirror, and check balance.

    Returns (evm_address, ss58_mirror, balance_tao).
    Balance is -1.0 if RPC check fails.
    """
    from verallm.chain.wallet import derive_evm_address, h160_to_ss58_mirror

    evm_addr = derive_evm_address(hotkey_seed)
    ss58_mirror = h160_to_ss58_mirror(evm_addr)

    balance = -1.0
    try:
        from verallm.chain.config import ChainConfig

        rpc_url = ChainConfig.resolve_rpc_url(None, network)
        if chain_config_path:
            cc = ChainConfig.from_json(
                chain_config_path,
                **({"rpc_url": rpc_url} if rpc_url else {}),
            )
        else:
            cc = ChainConfig(rpc_url=rpc_url or "https://lite.chain.opentensor.ai")

        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(cc.rpc_url))
        balance_wei = w3.eth.get_balance(evm_addr)
        balance = balance_wei / 1e18  # wei -> TAO
    except Exception:
        pass

    return evm_addr, ss58_mirror, balance


def _read_hotkey_seed(wallet_name: str, hotkey_name: str) -> Optional[bytes]:
    """Read hotkey seed from wallet keyfile."""
    hk_path = Path.home() / f".bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}"
    if not hk_path.exists():
        return None
    try:
        hk_data = json.loads(hk_path.read_text())
        return bytes.fromhex(hk_data["secretSeed"].replace("0x", ""))
    except Exception:
        return None


def step_evm_funding(wallet_info: dict, net_info: dict, role: str = "miner") -> dict:
    """Check EVM balance and guide funding."""
    seed = _read_hotkey_seed(wallet_info["wallet"], wallet_info["hotkey"])
    if seed is None:
        print(red(f"  Cannot read hotkey seed — wallet file missing."))
        return {"evm_funded": False, "evm_address": "", "ss58_mirror": ""}

    evm_addr, ss58_mirror, balance = _check_evm_balance(
        seed, net_info["network"], net_info.get("config_path", ""),
    )

    print(f"  EVM address: {evm_addr}")
    print(f"  SS58 mirror: {ss58_mirror}")

    # ~0.001 TAO per transaction. Validators only register EVM once;
    # miners also renew every 12h + model updates → suggest more.
    MIN_BALANCE = 0.001
    suggested_amount = 0.01 if role == "validator" else 0.05

    if balance >= MIN_BALANCE:
        print(f"  Balance:     {green(f'{balance:.4f} TAO')} (sufficient)")
        return {"evm_funded": True, "evm_address": evm_addr, "ss58_mirror": ss58_mirror}
    elif balance >= 0:
        print(f"  Balance:     {red(f'{balance:.4f} TAO')} (need >= {MIN_BALANCE} for gas)")
    else:
        print(f"  Balance:     {yellow('unable to check')} (RPC timeout)")

    print()
    print(f"  Fund your EVM address by sending TAO to the SS58 mirror:")
    print(f"    btcli wallet transfer --dest {ss58_mirror} --amount {suggested_amount} "
          f"--subtensor.network {net_info['network']}")
    print()
    if role == "validator":
        print(dim("  Gas costs: ~0.001 TAO per transaction (one-time EVM registration)."))
    else:
        print(dim("  Gas costs: ~0.001 TAO per transaction (register, renew, etc.)"))
        print(dim("  The miner renews its on-chain lease every 12h (~0.001 TAO each)."))
    print()

    while True:
        if not _confirm("Check balance again?", default=True):
            return {"evm_funded": False, "evm_address": evm_addr, "ss58_mirror": ss58_mirror}
        print("  Checking...", end="", flush=True)
        _, _, balance = _check_evm_balance(
            seed, net_info["network"], net_info.get("config_path", ""),
        )
        if balance >= MIN_BALANCE:
            print(f" {green(f'{balance:.4f} TAO')} — funded!")
            return {"evm_funded": True, "evm_address": evm_addr, "ss58_mirror": ss58_mirror}
        elif balance >= 0:
            print(f" {red(f'{balance:.4f} TAO')} — not yet.")
        else:
            print(f" {yellow('unable to check')}")


# ---------------------------------------------------------------------------
# Step 5: HTTPS setup
# ---------------------------------------------------------------------------

def _detect_public_ip() -> str:
    """Detect public IPv4 via external service. Prefers IPv4 over IPv6."""
    # Try IPv4-only services first
    for url in ["https://api.ipify.org", "https://ipv4.icanhazip.com", "https://ifconfig.me"]:
        try:
            r = subprocess.run(
                ["curl", "-s", "-4", "--max-time", "5", url],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                ip = r.stdout.strip()
                if ":" not in ip:  # skip IPv6
                    return ip
        except Exception:
            continue
    # Fallback: any IP (may be IPv6)
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "5", "ifconfig.me"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _detect_https_config() -> Optional[dict]:
    """Detect existing Verathos HTTPS nginx config."""
    for path in [
        Path("/etc/nginx/sites-available/verathos-miner"),
        Path("/etc/nginx/sites-enabled/verathos-miner"),
    ]:
        if path.exists():
            try:
                text = path.read_text()
                # Extract port from "listen <PORT> ssl"
                m = re.search(r"listen\s+(\d+)\s+ssl", text)
                port = int(m.group(1)) if m else 443
                return {"port": port, "config_path": str(path)}
            except Exception:
                pass
    # Check inline nginx.conf for verathos pattern
    nginx_conf = Path("/etc/nginx/nginx.conf")
    if nginx_conf.exists():
        try:
            text = nginx_conf.read_text()
            if "miner" in text and "ssl" in text:
                m = re.search(r"listen\s+(\d+)\s+ssl", text)
                port = int(m.group(1)) if m else 443
                return {"port": port, "config_path": str(nginx_conf)}
        except Exception:
            pass
    return None


def step_https(repo: Path) -> dict:
    """Set up HTTPS reverse proxy."""
    existing = _detect_https_config()
    public_ip = _detect_public_ip()

    if existing:
        endpoint = f"https://{public_ip}:{existing['port']}" if public_ip else f"https://YOUR_IP:{existing['port']}"
        print(f"  HTTPS already configured (port {existing['port']})")
        print(f"  Endpoint: {endpoint}")
        if public_ip:
            return {"endpoint": endpoint, "port": existing["port"], "host": public_ip}
        host = _prompt("Public IP or domain", public_ip or "")
        return {"endpoint": f"https://{host}:{existing['port']}", "port": existing["port"], "host": host}

    print(f"  Public IP: {public_ip or yellow('not detected')}")

    # Accept full URL or hostname/IP + port separately
    host_input = _prompt("Miner endpoint hostname/IP", public_ip)

    # If user pasted a full URL, parse it
    if host_input.startswith(("https://", "http://")):
        from urllib.parse import urlparse
        _p = urlparse(host_input)
        host = _p.hostname or public_ip
        port = _p.port or 443
    elif host_input.isdigit():
        # User entered just a port number — use detected IP
        host = public_ip or "YOUR_IP"
        port = int(host_input)
        print(yellow(f"  Looks like a port number — using IP {host}"))
    else:
        host = host_input
        while True:
            port_str = _prompt("HTTPS port", "443").strip()
            if port_str.isdigit() and 1 <= int(port_str) <= 65535:
                port = int(port_str)
                break
            print(red(f"  Invalid port: enter a number between 1 and 65535."))

    endpoint = f"https://{host}:{port}" if port != 443 else f"https://{host}"
    print(f"  Endpoint: {endpoint}")

    setup_script = repo / "scripts" / "setup_https.sh"
    if not setup_script.exists():
        print(yellow(f"  setup_https.sh not found — set up HTTPS manually."))
        return {"endpoint": endpoint, "port": port, "host": host}

    if _confirm("Set up HTTPS now (nginx + self-signed cert)?"):
        print(f"  Running setup_https.sh --port {port}...")
        r = subprocess.run(
            ["sudo", "bash", str(setup_script), "--port", str(port)],
            stdin=sys.stdin,
        )
        if r.returncode != 0:
            print(yellow("  HTTPS setup returned non-zero — check output above."))
            print("  You can re-run manually: sudo bash scripts/setup_https.sh --port " + str(port))
    else:
        print(yellow("  Skipped HTTPS setup. Run later:"))
        print(f"    sudo bash scripts/setup_https.sh --port {port}")

    return {"endpoint": endpoint, "port": port, "host": host}


# ---------------------------------------------------------------------------
# Step 6: Model selection
# ---------------------------------------------------------------------------

def step_model(net_info: dict) -> dict:
    """Show model recommendations and let user pick."""
    import threading

    # Import registry + detect GPU in background (torch init is slow)
    _done = threading.Event()
    _result = [None, None, None]  # [gpu_info, recommend_models, ModelCategory]
    _error = [None]

    def _load():
        try:
            from verallm.registry import ModelCategory, recommend_models
            from verallm.registry.gpu import detect_gpu_info
            _result[0] = detect_gpu_info()
            _result[1] = recommend_models
            _result[2] = ModelCategory
        except ImportError as e:
            _error[0] = e
        _done.set()

    threading.Thread(target=_load, daemon=True).start()
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not _done.is_set():
        print(f"\r  Detecting GPU {chars[i % len(chars)]}", end="", flush=True)
        _done.wait(0.1)
        i += 1
    print("\r" + " " * 50 + "\r", end="", flush=True)

    if _error[0] is not None:
        print(yellow("  Registry not available — using auto selection."))
        return {"model_id": "auto", "category": None}

    gpu_info, recommend_models, ModelCategory = _result
    if not gpu_info or not gpu_info.get("available"):
        print(yellow("  No GPU detected — using auto selection."))
        return {"model_id": "auto", "category": None}

    tier = gpu_info["tier"]
    print(f"  GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB, tier {tier.name})")
    print()

    # Get recommendations
    recs = recommend_models(tier)

    # Filter by on-chain models if chain config available
    chain_config_path = net_info.get("config_path", "")
    if chain_config_path and recs:
        try:
            import threading
            from neurons.model_resolve import _get_on_chain_models
            # RPC call can be slow — show a spinner
            _spinner_done = threading.Event()
            def _spin():
                chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                i = 0
                while not _spinner_done.is_set():
                    print(f"\r  Checking on-chain models {chars[i % len(chars)]}", end="", flush=True)
                    _spinner_done.wait(0.1)
                    i += 1
                print("\r" + " " * 40 + "\r", end="", flush=True)
            _t = threading.Thread(target=_spin, daemon=True)
            _t.start()
            on_chain = _get_on_chain_models(chain_config_path, net_info.get("network"))
            _spinner_done.set()
            _t.join()
            if on_chain is not None:
                on_chain_set = {m.lower() for m in on_chain}
                filtered = [
                    r for r in recs
                    if any(tc.checkpoint.lower() in on_chain_set
                           for tc in r.model.tier_configs)
                ]
                if filtered:
                    recs = filtered
                    print(f"  Filtered to {len(recs)} on-chain registered models")
        except Exception:
            pass  # chain query failed — show all recommendations

    if not recs:
        print(yellow("  No models found for this GPU tier."))
        return {"model_id": "auto", "category": None}

    # Show top recommendations
    print(f"  Recommended models:")
    print()
    show = min(len(recs), 8)
    print(f"    {'#':<3s} {'Model':<40s} {'Quant':<6s} {'VRAM':>5s} {'Context':>8s} {'Score':>6s}")
    print(f"    {'─'*3} {'─'*40} {'─'*6} {'─'*5} {'─'*8} {'─'*6}")
    for i, r in enumerate(recs[:show]):
        print(f"    [{i+1}] {r.model.name:<40s} {r.quant:<6s} {r.est_weight_gb:>4.0f}G {r.est_context:>7,d} {r.utility:>6.1f}")

    print()

    # Category filter option
    categories = sorted(set(
        cat.value for r in recs for cat in r.model.categories
    ))
    if len(categories) > 1:
        cat_str = ", ".join(categories)
        print(f"  Categories: {dim(cat_str)}")
        print()

    while True:
        choice = _prompt("Select model (number or 'auto')", "1").strip()

        if choice.lower() == "auto":
            return {"model_id": "auto", "category": None}

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(recs):
                r = recs[idx]
                print(f"  Selected: {green(r.model.name)} ({r.quant}, ~{r.est_weight_gb:.0f}GB)")
                return {
                    "model_id": r.config.checkpoint,
                    "quant": r.quant,
                    "category": None,
                }

        print(red(f"  Invalid selection. Enter 1-{len(recs)} or 'auto'."))


# ---------------------------------------------------------------------------
# Step 7: PM2
# ---------------------------------------------------------------------------

def step_pm2() -> dict:
    """Check and install PM2."""
    import threading
    pm2 = shutil.which("pm2")
    if pm2:
        # Get version
        try:
            r = subprocess.run([pm2, "--version"], capture_output=True, text=True, timeout=5)
            version = r.stdout.strip() if r.returncode == 0 else "installed"
        except Exception:
            version = "installed"
        print(f"  PM2: {green(version)}")
        return {"pm2_installed": True, "pm2_path": pm2}

    node = shutil.which("node")
    npm = shutil.which("npm")

    if not node or not npm:
        print(yellow("  Node.js and PM2 not found."))
        if _confirm("Install Node.js and PM2?"):
            _install_done = threading.Event()
            _install_ok = [False]
            def _do_node_install():
                if shutil.which("apt-get"):
                    subprocess.run(["sudo", "apt-get", "update", "-qq"], capture_output=True)
                    r = subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "nodejs", "npm"], capture_output=True)
                    _install_ok[0] = r.returncode == 0
                elif shutil.which("dnf"):
                    r = subprocess.run(["sudo", "dnf", "install", "-y", "nodejs", "npm"], capture_output=True)
                    _install_ok[0] = r.returncode == 0
                _install_done.set()
            threading.Thread(target=_do_node_install, daemon=True).start()
            _chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            _i = 0
            while not _install_done.is_set():
                print(f"\r  Installing Node.js {_chars[_i % len(_chars)]}", end="", flush=True)
                _install_done.wait(0.1)
                _i += 1
            print(f"\r  Installing Node.js {green('done')}    ")
            if not _install_ok[0]:
                print(red("  Node.js installation failed. Install manually:"))
                print("    https://nodejs.org/en/download/")
                return {"pm2_installed": False, "pm2_path": ""}
        else:
            print(yellow("  Skipped. Install later: sudo apt install nodejs npm && sudo npm install -g pm2"))
            return {"pm2_installed": False, "pm2_path": ""}

    # Install PM2 globally — rehash PATH in case apt just installed npm
    npm = shutil.which("npm")
    if not npm:
        # apt may have installed to a path not in current PATH
        for _p in ["/usr/bin/npm", "/usr/local/bin/npm"]:
            if os.path.isfile(_p):
                npm = _p
                break
    if npm:
        _pm2_done = threading.Event()
        def _do_pm2_install():
            subprocess.run(["sudo", npm, "install", "-g", "pm2"], capture_output=True)
            _pm2_done.set()
        threading.Thread(target=_do_pm2_install, daemon=True).start()
        _chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        _i = 0
        while not _pm2_done.is_set():
            print(f"\r  Installing PM2 {_chars[_i % len(_chars)]}", end="", flush=True)
            _pm2_done.wait(0.1)
            _i += 1
        print(f"\r  Installing PM2 {green('done')}    ")

    pm2 = shutil.which("pm2")
    if pm2:
        print(f"  PM2: {green('installed')}")
        return {"pm2_installed": True, "pm2_path": pm2}
    else:
        print(yellow("  PM2 installation may need a shell restart to be on PATH."))
        return {"pm2_installed": False, "pm2_path": ""}


# ---------------------------------------------------------------------------
# Step: Options (subtensor, logging, analytics)
# ---------------------------------------------------------------------------


def _test_subtensor_endpoint(endpoint: str, silent: bool = False) -> bool:
    """Quick connectivity test for a subtensor RPC endpoint."""
    import socket
    from urllib.parse import urlparse
    try:
        p = urlparse(endpoint)
        host = p.hostname or "localhost"
        port = p.port or 9944
        with socket.create_connection((host, port), timeout=3):
            if not silent:
                print(f"  {green('Connected')} to {host}:{port}")
            return True
    except Exception as e:
        if not silent:
            print(f"  {red('Connection failed')}: {e}")
        return False


def step_options(role: str) -> dict:
    """Prompt for optional configuration (subtensor, logging, analytics)."""
    # Local subtensor — auto-detect, then prompt
    print()
    chain_endpoint = ""
    _local_default = "ws://localhost:9944"
    _local_detected = _test_subtensor_endpoint(_local_default, silent=True)
    if _local_detected:
        print(f"  Local subtensor detected at {_local_default}")
        print(f"    [Enter] Use local subtensor (recommended — no rate limits)")
        print(f"    [P]     Use public RPC instead")
        print(f"    [url]   Custom endpoint")
        choice = _prompt("Subtensor", "")
        if not choice:
            chain_endpoint = _local_default
        elif choice.upper() == "P":
            chain_endpoint = ""  # public RPC
        elif choice.startswith(("ws://", "http://", "wss://")):
            if _test_subtensor_endpoint(choice):
                chain_endpoint = choice
            elif _confirm("Use this endpoint anyway?", default=False):
                chain_endpoint = choice
        else:
            print(yellow("  Invalid — using local subtensor"))
            chain_endpoint = _local_default
    else:
        print("  No local subtensor detected at localhost:9944")
        print(f"    [Enter] Use public RPC")
        print(f"    [url]   Custom endpoint (ws://host:port)")
        choice = _prompt("Subtensor", "")
        if choice.startswith(("ws://", "http://", "wss://")):
            if _test_subtensor_endpoint(choice):
                chain_endpoint = choice
            elif _confirm("Use this endpoint anyway?", default=False):
                chain_endpoint = choice

    # Log level — shared across both roles
    print()
    print("  Log level:")
    print(f"    [Enter] INFO (default)")
    print(f"    [1]     DEBUG (verbose — canary details, RPC calls, proof timing)")
    log_choice = _prompt("Log level", "")
    log_level = "debug" if log_choice == "1" else ""

    # Derive EVM RPC URL from chain endpoint.  The local subtensor serves
    # both Substrate (ws://) and EVM (http://) on the same port.  web3.py
    # needs the http:// variant.
    evm_rpc_url = ""
    if chain_endpoint:
        from urllib.parse import urlparse
        p = urlparse(chain_endpoint)
        host = p.hostname or "localhost"
        port = p.port or 9944
        evm_rpc_url = f"http://{host}:{port}"

    result = {"chain_endpoint": chain_endpoint, "evm_rpc_url": evm_rpc_url, "log_level": log_level}

    # Validator-specific options
    if role == "validator":
        print()
        print("  Analytics database (canary results, epoch scores, network receipts):")
        print(f"    [Enter] Enabled (recommended)")
        print(f"    [N]     Disabled")
        analytics = _prompt("Analytics", "").upper() != "N"
        result["analytics"] = analytics

        print()
        print("  Retain analytics backups (keep backup files indefinitely):")
        print(f"    [Enter] No (auto-delete after 7 days — saves disk)")
        print(f"    [Y]     Yes (keep all backups)")
        retain = _prompt("Retain backups", "").upper() == "Y"
        result["retain_backups"] = retain

        print()
        print("  HuggingFace token (avoids rate limits when downloading tokenizers):")
        print(f"    [Enter] Skip (no token)")
        print(f"    [token] Paste your HF token (from https://huggingface.co/settings/tokens)")
        hf_token = _prompt("HF_TOKEN", "")
        result["hf_token"] = hf_token

    return result


# ---------------------------------------------------------------------------
# Step: ecosystem.config.js generation
# ---------------------------------------------------------------------------

def _parse_existing_config(repo: Path) -> list[dict]:
    """Parse existing ecosystem.config.js for miner entries.

    Returns list of dicts with name, wallet, hotkey, endpoint, model args.
    This is a best-effort parser for the JS module.exports format.
    """
    config_path = repo / "ecosystem.config.js"
    if not config_path.exists():
        return []

    try:
        text = config_path.read_text()
    except OSError:
        return []

    miners = []
    # Find miner app entries by looking for neurons.miner in args
    # Simple regex-based parsing of the JS config
    for m in re.finditer(r'name:\s*["\']([^"\']*miner[^"\']*)["\']', text):
        name = m.group(1)
        # Find the args line near this match
        section = text[m.start():m.start() + 500]
        args_m = re.search(r'args:\s*["\']([^"\']*)["\']', section)
        if args_m:
            args_str = args_m.group(1)
            entry = {"name": name, "args": args_str}
            # Extract endpoint
            ep_m = re.search(r"--endpoint\s+(\S+)", args_str)
            if ep_m:
                entry["endpoint"] = ep_m.group(1)
            # Extract wallet + hotkey
            w_m = re.search(r"--wallet\s+(\S+)", args_str)
            if w_m:
                entry["wallet"] = w_m.group(1)
            h_m = re.search(r"--hotkey\s+(\S+)", args_str)
            if h_m:
                entry["hotkey"] = h_m.group(1)
            miners.append(entry)
    return miners


def _get_pm2_processes() -> list[dict]:
    """Get running PM2 processes."""
    pm2 = shutil.which("pm2")
    if not pm2:
        return []
    try:
        r = subprocess.run(
            [pm2, "jlist"], capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except Exception:
        pass
    return []


def _generate_miner_entry(
    name: str,
    wallet: str,
    hotkey: str,
    endpoint: str,
    repo_root: str,
    network: str,
    netuid: int,
    model_id: str = "auto",
    quant: str = "",
    chain_endpoint: str = "",
    cuda_device: str = "",
    extra_args: str = "",
    log_level: str = "",
    evm_rpc_url: str = "",
) -> str:
    """Generate a single PM2 miner app entry as JS text."""
    net_flag = f"--subtensor-network {network}"
    chain_flag = f" --subtensor-chain-endpoint {chain_endpoint}" if chain_endpoint else ""
    evm_flag = f" --evm-rpc-url {evm_rpc_url}" if evm_rpc_url else ""
    log_flag = " --logging.debug" if log_level == "debug" else ""
    quant_flag = f" --quant {quant}" if quant else ""
    args = (
        f"-u -m neurons.miner "
        f"--wallet {wallet} --hotkey {hotkey} "
        f"--netuid {netuid} {net_flag}{chain_flag}{evm_flag} "
        f"--model-id {model_id}{quant_flag} --endpoint {endpoint} "
        f"--auto-update{log_flag}"
    )
    if extra_args:
        args += f" {extra_args}"

    env_block = ""
    if cuda_device:
        env_block = f"""
      env: {{
        CUDA_VISIBLE_DEVICES: "{cuda_device}",
      }},"""

    return f"""    {{
      name: "{name}",
      script: ".venv-vllm/bin/python",
      args: "{args}",
      cwd: "{repo_root}",{env_block}
      autorestart: false,
      max_restarts: 0,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    }}"""


def _generate_validator_entry(
    wallet: str,
    hotkey: str,
    repo_root: str,
    network: str,
    netuid: int,
    chain_endpoint: str = "",
    evm_rpc_url: str = "",
    log_level: str = "",
    analytics: bool = True,
    retain_backups: bool = False,
    hf_token: str = "",
    no_evm: bool = False,
) -> str:
    """Generate a PM2 validator app entry as JS text."""
    net_flag = f"--subtensor-network {network}"
    chain_flag = f" --subtensor-chain-endpoint {chain_endpoint}" if chain_endpoint else ""
    evm_flag = f" --evm-rpc-url {evm_rpc_url}" if evm_rpc_url else ""
    log_flag = " --logging.debug" if log_level == "debug" else ""
    analytics_flag = " --analytics" if analytics else ""
    retain_flag = " --retain-backups" if retain_backups else ""
    no_evm_flag = " --no-evm" if no_evm else ""
    args = (
        f"-u -m neurons.validator "
        f"--wallet {wallet} --hotkey {hotkey} "
        f"--netuid {netuid} {net_flag}{chain_flag}{evm_flag} "
        f"--auto-update{analytics_flag}{retain_flag}{no_evm_flag}{log_flag}"
    )
    return f"""    {{
      name: "validator",
      script: ".venv-validator/bin/python",
      args: "{args}",
      cwd: "{repo_root}",
      env: {{
        HF_TOKEN: "{hf_token}",
      }},
      autorestart: true,
      max_restarts: 5,
      min_uptime: "60s",
      restart_delay: 10000,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    }}"""


def step_config(
    role: str,
    repo: Path,
    wallet_info: dict,
    net_info: dict,
    https_info: dict,
    model_info: Optional[dict] = None,
    options: Optional[dict] = None,
) -> dict:
    """Generate or update ecosystem.config.js."""
    config_path = repo / "ecosystem.config.js"
    existing_miners = _parse_existing_config(repo)
    repo_root = str(repo)
    model_id = (model_info or {}).get("model_id", "auto")
    quant = (model_info or {}).get("quant", "")
    opts = options or {}
    chain_endpoint = opts.get("chain_endpoint", "")
    evm_rpc_url = opts.get("evm_rpc_url", "")
    log_level = opts.get("log_level", "")

    if role == "validator":
        entry = _generate_validator_entry(
            wallet=wallet_info["wallet"],
            hotkey=wallet_info["hotkey"],
            repo_root=repo_root,
            network=net_info["network"],
            netuid=net_info["netuid"],
            chain_endpoint=chain_endpoint,
            evm_rpc_url=evm_rpc_url,
            log_level=log_level,
            analytics=opts.get("analytics", True),
            retain_backups=opts.get("retain_backups", False),
            hf_token=opts.get("hf_token", ""),
            no_evm=opts.get("no_evm", False),
        )
        if config_path.exists() and existing_miners:
            # Append validator to existing config
            print("  Existing config found — will add validator entry.")
        _write_ecosystem_config(config_path, [entry])
        return {"config_generated": True, "pm2_name": "validator"}

    # Miner flow
    endpoint = https_info.get("endpoint", "")

    # Generate fresh config
    if not endpoint:
        endpoint = _prompt("Miner endpoint URL", "https://YOUR_IP")

    entry = _generate_miner_entry(
        name="miner",
        wallet=wallet_info["wallet"],
        hotkey=wallet_info["hotkey"],
        endpoint=endpoint,
        repo_root=repo_root,
        network=net_info["network"],
        netuid=net_info["netuid"],
        model_id=model_id,
        quant=quant,
        chain_endpoint=chain_endpoint,
        evm_rpc_url=evm_rpc_url,
        log_level=log_level,
    )

    print()
    print(dim("  Generated ecosystem.config.js:"))
    print(dim(f"    wallet: {wallet_info['wallet']}/{wallet_info['hotkey']}"))
    print(dim(f"    endpoint: {endpoint}"))
    print(dim(f"    network: {net_info['network']} (netuid {net_info['netuid']})"))
    _model_label = f"{model_id} ({quant})" if quant else model_id
    print(dim(f"    model: {_model_label}"))
    if chain_endpoint:
        print(dim(f"    subtensor: {chain_endpoint}"))
    if log_level:
        print(dim(f"    log level: {log_level}"))
    print()

    if config_path.exists():
        if not _confirm("Overwrite existing ecosystem.config.js?", default=False):
            print(yellow("  Keeping existing config."))
            return {"config_generated": True, "pm2_name": "miner"}

    _write_ecosystem_config(config_path, [entry])
    print(green(f"  Written: {config_path}"))
    return {"config_generated": True, "pm2_name": "miner"}


def _write_ecosystem_config(path: Path, entries: list[str]) -> None:
    """Write a fresh ecosystem.config.js with the given app entries."""
    apps = ",\n".join(entries)
    content = f"""// Auto-generated by: verathos setup
// Manual edits are preserved if you re-run the wizard with "keep current config".
//
// Usage:
//   pm2 start ecosystem.config.js          # start all
//   pm2 start ecosystem.config.js --only miner
//   pm2 logs miner --lines 50
//   pm2 stop all
//
// IMPORTANT: After changing args, delete + recreate the PM2 process:
//   pm2 delete miner && pm2 start ecosystem.config.js --only miner

module.exports = {{
  apps: [
{apps}
  ],
}};
"""
    path.write_text(content)


def _append_to_ecosystem_config(path: Path, entry: str) -> None:
    """Append a new app entry to an existing ecosystem.config.js."""
    text = path.read_text()
    # Find the last closing bracket of the apps array and insert before it
    # Look for the pattern: ],\n};  or  ]\n};
    insertion = re.sub(
        r"(\s*)\],(\s*)\};(\s*)$",
        rf",\n{entry}\n\1],\2}};\3",
        text,
    )
    if insertion != text:
        path.write_text(insertion)
    else:
        # Fallback: rewrite entirely
        print(yellow("  Could not append — regenerating config."))
        _write_ecosystem_config(path, [entry])


# ---------------------------------------------------------------------------
# Status (preflight check)
# ---------------------------------------------------------------------------

def _detect_roles_from_config(repo: Path) -> dict:
    """Detect configured roles from ecosystem.config.js.

    Returns dict with 'has_validator', 'has_miner', 'miner_entries' (list of
    dicts with name/wallet/hotkey/endpoint), 'validator_entry' (dict or None).
    """
    config_path = repo / "ecosystem.config.js"
    result = {"has_validator": False, "has_miner": False,
              "miner_entries": [], "validator_entry": None}
    if not config_path.exists():
        return result
    try:
        text = config_path.read_text()
    except OSError:
        return result

    # Detect validator
    for m in re.finditer(r'name:\s*["\']validator["\']', text):
        section = text[m.start():m.start() + 500]
        args_m = re.search(r'args:\s*["\']([^"\']*)["\']', section)
        if args_m:
            result["has_validator"] = True
            args_str = args_m.group(1)
            entry = {"name": "validator", "args": args_str}
            w_m = re.search(r"--wallet\s+(\S+)", args_str)
            h_m = re.search(r"--hotkey\s+(\S+)", args_str)
            if w_m:
                entry["wallet"] = w_m.group(1)
            if h_m:
                entry["hotkey"] = h_m.group(1)
            result["validator_entry"] = entry
            break

    # Detect miners
    result["miner_entries"] = _parse_existing_config(repo)
    result["has_miner"] = len(result["miner_entries"]) > 0
    return result


def _format_pm2_uptime(proc: dict) -> str:
    """Format a PM2 process status string."""
    name = proc.get("name", "?")
    status = proc.get("pm2_env", {}).get("status", "?")
    pid = proc.get("pid", "?")
    uptime_ms = proc.get("pm2_env", {}).get("pm_uptime", 0)
    if uptime_ms and status == "online":
        uptime_s = (time.time() * 1000 - uptime_ms) / 1000
        hours = int(uptime_s // 3600)
        days = hours // 24
        hours = hours % 24
        uptime_str = f"{days}d {hours}h" if days else f"{hours}h"
        return f"{name}: {green('running')} (pid {pid}, {uptime_str})"
    return f"{name}: {red(status)}"


def run_status(repo: Optional[Path] = None) -> None:
    """Non-interactive preflight readiness check.

    Auto-detects role (miner/validator/both) from ecosystem.config.js and
    shows only relevant checks for each role.
    """
    if repo is None:
        repo = _find_repo_root()

    roles = _detect_roles_from_config(repo)
    is_validator = roles["has_validator"]
    is_miner = roles["has_miner"]

    # If no config, show generic status
    if not is_validator and not is_miner:
        role_label = "unknown (no ecosystem.config.js)"
    elif is_validator and is_miner:
        role_label = "validator + miner"
    elif is_validator:
        role_label = "validator"
    else:
        role_label = "miner"

    print()
    print(bold(f"  Verathos Status ({role_label})"))
    print(f"  {'─' * 50}")

    # Network
    mainnet = repo / "chain_config_mainnet.json"
    testnet = repo / "chain_config_testnet.json"
    plain = repo / "chain_config.json"
    if mainnet.exists() or plain.exists():
        print(f"  Network:     mainnet (netuid 96)              {_ok()}")
        network = "finney"
    elif testnet.exists():
        print(f"  Network:     testnet (netuid 405)             {_ok()}")
        network = "test"
    else:
        print(f"  Network:     {red('no chain config found')}         {_fail()}")
        network = None

    # Subtensor — check local
    if _test_subtensor_endpoint("ws://localhost:9944", silent=True):
        print(f"  Subtensor:   local (ws://localhost:9944)      {_ok()}")
    else:
        print(f"  Subtensor:   public RPC")

    # Collect all unique wallet/hotkey pairs from config
    _wallet_pairs: list[tuple[str, str]] = []
    _seen_pairs: set[tuple[str, str]] = set()
    if is_validator and roles["validator_entry"]:
        w = roles["validator_entry"].get("wallet", "")
        h = roles["validator_entry"].get("hotkey", "")
        if w and h and (w, h) not in _seen_pairs:
            _wallet_pairs.append((w, h))
            _seen_pairs.add((w, h))
    for m in roles.get("miner_entries", []):
        w = m.get("wallet", "")
        h = m.get("hotkey", "")
        if w and h and (w, h) not in _seen_pairs:
            _wallet_pairs.append((w, h))
            _seen_pairs.add((w, h))
    # Fallback: first wallet on disk
    if not _wallet_pairs:
        wallets = _list_wallets()
        if wallets:
            w = wallets[0]["name"]
            h = wallets[0]["hotkeys"][0] if wallets[0]["hotkeys"] else ""
            if w and h:
                _wallet_pairs.append((w, h))

    # Load metagraph once for all registration checks
    _metagraph = None
    _netuid = 96 if network == "finney" else 405
    if _wallet_pairs and network:
        try:
            import bittensor as bt
            sub = bt.Subtensor(network=network)
            _metagraph = sub.metagraph(netuid=_netuid)
        except Exception:
            pass

    # Show wallet / registration / EVM for each unique hotkey
    for _w, _h in _wallet_pairs:
        _label = f"{_w}/{_h}"
        _prefix = "  " if len(_wallet_pairs) == 1 else "    "
        if len(_wallet_pairs) > 1:
            print(f"  {bold(_label)}:")

        # Wallet exists?
        _hk_path = Path.home() / f".bittensor/wallets/{_w}/hotkeys/{_h}"
        if _hk_path.exists():
            if len(_wallet_pairs) == 1:
                print(f"  Wallet:      {_label:<35s} {_ok()}")
        else:
            print(f"{_prefix}Wallet:      {red(f'{_label} (hotkey file missing)')} {_fail()}")
            continue

        # Registration + metagraph stats
        if _metagraph is not None:
            try:
                _hk_data = json.loads(_hk_path.read_text())
                _ss58 = _hk_data.get("ss58Address", _hk_data.get("address", ""))
                if _ss58 in _metagraph.hotkeys:
                    _uid = _metagraph.hotkeys.index(_ss58)
                    print(f"{_prefix}Registered:  UID {_uid:<35d} {_ok()}")

                    # Show role-relevant metagraph stats
                    try:
                        _stake = float(_metagraph.S[_uid])
                        _emission = float(_metagraph.E[_uid])
                        _incentive = float(_metagraph.I[_uid])
                        _trust = float(_metagraph.TS[_uid]) if hasattr(_metagraph, 'TS') else 0.0

                        if is_validator:
                            _vtrust = float(_metagraph.Tv[_uid])
                            _dividends = float(_metagraph.D[_uid])
                            _vt_status = green(f"{_vtrust:.4f}") if _vtrust > 0 else yellow("0.0000 (not setting weights yet)")
                            print(f"{_prefix}Stake:       {_stake:.2f} alpha")
                            print(f"{_prefix}VTrust:      {_vt_status}")
                            print(f"{_prefix}Dividends:   {_dividends:.4f}")
                            print(f"{_prefix}Emission:    {_emission:.4f} alpha/tempo")
                        else:
                            _inc_status = green(f"{_incentive:.4f}") if _incentive > 0 else yellow("0.0000 (not scored yet)")
                            print(f"{_prefix}Stake:       {_stake:.2f} alpha")
                            print(f"{_prefix}Incentive:   {_inc_status}")
                            print(f"{_prefix}Trust:       {_trust:.4f}")
                            print(f"{_prefix}Emission:    {_emission:.4f} alpha/tempo")
                    except Exception:
                        pass  # metagraph field access failed — skip stats
                else:
                    print(f"{_prefix}Registered:  {red('not found in metagraph')}       {_fail()}")
            except Exception as e:
                print(f"{_prefix}Registered:  {yellow(f'check failed ({e})')}")
        else:
            print(f"{_prefix}Registered:  {_skip()}")

        # EVM balance
        _seed = _read_hotkey_seed(_w, _h)
        if _seed and network:
            _evm_addr, _, _balance = _check_evm_balance(_seed, network)
            if _balance >= 0.01:
                print(f"{_prefix}EVM balance: {_balance:.4f} TAO{' ' * (28 - len(f'{_balance:.4f}'))}{_ok()}")
            elif _balance >= 0.001:
                print(f"{_prefix}EVM balance: {yellow(f'{_balance:.4f} TAO (low — consider topping up)')}")
            elif _balance >= 0:
                print(f"{_prefix}EVM balance: {red(f'{_balance:.4f} TAO (critically low)')}{' ' * 5}{_fail()}")
            else:
                print(f"{_prefix}EVM balance: {yellow('unable to check')}")
        else:
            print(f"{_prefix}EVM balance: {_skip()}")

    if not _wallet_pairs:
        print(f"  Wallet:      {red('none found')}                    {_fail()}")

    # HTTPS — only relevant for miners
    if is_miner or (not is_validator and not is_miner):
        https = _detect_https_config()
        public_ip = _detect_public_ip()
        if https:
            ep = f"https://{public_ip}:{https['port']}" if public_ip else f":{https['port']}"
            print(f"  HTTPS:       {ep:<35s} {_ok()}")
        else:
            print(f"  HTTPS:       {red('not configured')}                {_fail()}")

    # PM2
    pm2 = shutil.which("pm2")
    if pm2:
        try:
            r = subprocess.run([pm2, "--version"], capture_output=True, text=True, timeout=5)
            v = r.stdout.strip() if r.returncode == 0 else "yes"
            print(f"  PM2:         v{v:<34s} {_ok()}")
        except Exception:
            print(f"  PM2:         {green('installed')}")
    else:
        print(f"  PM2:         {red('not found')}                     {_fail()}")

    # Config
    config = repo / "ecosystem.config.js"
    if config.exists():
        n_miners = len(roles["miner_entries"])
        parts = []
        if is_validator:
            parts.append("validator")
        if n_miners:
            parts.append(f"{n_miners} miner endpoint{'s' if n_miners != 1 else ''}")
        label = " + ".join(parts) if parts else "present"
        print(f"  Config:      ecosystem.config.js ({label}){' ' * max(0, 15 - len(label))} {_ok()}")
    else:
        print(f"  Config:      {red('ecosystem.config.js missing')}  {_fail()}")

    # GPU — only relevant for miners
    if is_miner or (not is_validator and not is_miner):
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                gpu_name = p.name
                vram = p.total_memory / (1024**3)
                n_gpus = torch.cuda.device_count()
                gpu_str = f"{gpu_name} ({vram:.0f} GB)"
                if n_gpus > 1:
                    gpu_str += f" x{n_gpus}"
                print(f"  GPU:         {gpu_str:<35s} {_ok()}")
            else:
                print(f"  GPU:         {yellow('CUDA not available')}")
        except ImportError:
            print(f"  GPU:         {yellow('torch not installed')}")

    # Version
    try:
        from neurons.version import miner_version_str, spec_version

        remote_ver = None
        try:
            subprocess.run(
                ["git", "fetch", "origin", "--quiet"],
                capture_output=True, timeout=10, cwd=str(repo),
            )
            r = subprocess.run(
                ["git", "show", "origin/main:neurons/version.py"],
                capture_output=True, text=True, timeout=5, cwd=str(repo),
            )
            if r.returncode == 0:
                import re as _re
                m_maj = _re.search(r"MINER_MAJOR\s*=\s*(\d+)", r.stdout)
                m_min = _re.search(r"MINER_MINOR\s*=\s*(\d+)", r.stdout)
                m_pat = _re.search(r"MINER_PATCH\s*=\s*(\d+)", r.stdout)
                if m_maj and m_min and m_pat:
                    remote_ver = f"{m_maj.group(1)}.{m_min.group(1)}.{m_pat.group(1)}"
        except Exception:
            pass

        if remote_ver and remote_ver != miner_version_str:
            print(f"  Version:     {yellow(f'v{miner_version_str}')} (latest: v{remote_ver} — update available)")
        elif remote_ver:
            ver_str = f"v{miner_version_str} (up to date)"
            print(f"  Version:     {ver_str:<35s} {_ok()}")
        else:
            print(f"  Version:     v{miner_version_str}")
    except ImportError:
        print(f"  Version:     {yellow('unknown')}")

    # Running processes — only online processes relevant to this config
    pm2_procs = _get_pm2_processes()
    # Collect names from the ecosystem config
    _config_names = set()
    if is_validator:
        _config_names.add("validator")
    for _me in roles.get("miner_entries", []):
        _config_names.add(_me.get("name", "miner"))
    # Also include proxy/gateway if running (common companion processes)
    _companion = {"proxy", "gateway"}
    relevant = [
        p for p in pm2_procs
        if p.get("pm2_env", {}).get("status") == "online"
        and (p.get("name") in _config_names or p.get("name") in _companion)
    ]
    if relevant:
        print()
        print(bold("  Processes:"))
        for p in relevant:
            print(f"    {_format_pm2_uptime(p)}")

    # Validator: shared state info (epoch, scores)
    if is_validator:
        _state_paths = [
            Path("/tmp/verathos_validator_state.json"),
            repo / "shared_state.json",
        ]
        for _sp in _state_paths:
            if _sp.exists():
                try:
                    _state = json.loads(_sp.read_text())
                    _epoch = _state.get("epoch_number", "?")
                    _scores = _state.get("miner_scores", {})
                    _probation = _state.get("probation_miners", [])
                    # Count unique miners and total entries
                    _n_miners = len(_scores)
                    _n_entries = sum(
                        len(e) for e in _scores.values() if isinstance(e, dict)
                    )
                    _total_score = sum(
                        sc for e in _scores.values() if isinstance(e, dict)
                        for sc in e.values()
                    )
                    print()
                    print(bold("  Epoch Info:"))
                    print(f"    Epoch:           {_epoch}")
                    print(f"    Miners:          {_n_miners} ({_n_entries} endpoints)")

                    # Weight split from actual normalized weights (UID 0 = burn)
                    _weights = _state.get("last_weights", {})
                    if _weights:
                        _burn_w = float(_weights.get("0", 0))
                        _total_w = sum(float(v) for v in _weights.values())
                        if _total_w > 0:
                            _miner_w = _total_w - _burn_w
                            print(f"    Weights:         {_miner_w/_total_w*100:.0f}% to miners, {_burn_w/_total_w*100:.0f}% burned")

                    if _probation:
                        print(f"    Probation:       {yellow(f'{len(_probation)} miner(s)')}")
                except Exception:
                    pass
                break

    # Miner: endpoint health check (merged with endpoint listing)
    if is_miner and roles.get("miner_entries"):
        print()
        print(bold("  Endpoints:"))
        for _me in roles["miner_entries"]:
            _ep = _me.get("endpoint", "")
            _name = _me.get("name", "miner")
            _model_m = re.search(r"--model-id\s+(\S+)", _me.get("args", ""))
            _model_label = _model_m.group(1) if _model_m else "auto"
            if not _ep:
                print(f"    {_name}: {_model_label} — {yellow('no endpoint configured')}")
                continue
            # Try localhost first (pods often can't reach their own external IP),
            # then fall back to the external endpoint.
            _health = None
            for _check_url in [
                "http://localhost:8000/health",  # direct backend
                f"{_ep}/health",                 # external endpoint (via nginx)
            ]:
                try:
                    r = subprocess.run(
                        ["curl", "-sk", "--max-time", "3", _check_url],
                        capture_output=True, text=True, timeout=8,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        _health = json.loads(r.stdout)
                        break
                except Exception:
                    continue
            if _health:
                _model = _health.get("model", "?")
                _max_ctx = _health.get("max_context", 0)
                _gpu = _health.get("hardware", {}).get("gpu_name", "")
                _gpu_str = f" ({_gpu})" if _gpu else ""
                print(f"    {_name}: {_ep} {green('reachable')}")
                print(f"      {_model}, ctx={_max_ctx}{_gpu_str}")
            else:
                print(f"    {_name}: {_ep} {yellow('not reachable from this server (may work externally)')}")

    print()


# ---------------------------------------------------------------------------
# Main wizard flow
# ---------------------------------------------------------------------------

def _wallet_exists(wallet_info: dict) -> bool:
    """Check if the wallet hotkey file actually exists on disk."""
    hk_path = Path.home() / f".bittensor/wallets/{wallet_info['wallet']}/hotkeys/{wallet_info['hotkey']}"
    return hk_path.exists()


def _quick_start(repo: Path, pm2_name: str) -> None:
    """Start miner/validator from existing config, checking if already running."""
    pm2 = shutil.which("pm2")
    if not pm2:
        print(yellow("  PM2 not found — installing..."))
        step_pm2()
        pm2 = shutil.which("pm2")
        if not pm2:
            print(red("  PM2 installation failed. Install manually: sudo npm install -g pm2"))
            return
    procs = _get_pm2_processes()

    # Check if this exact process is already running
    running = [p for p in procs if p.get("name") == pm2_name
               and p.get("pm2_env", {}).get("status") == "online"]
    if running:
        pid = running[0].get("pid", "?")
        print(f"  {pm2_name} is already running (pid {pid})")
        if _confirm("Restart?", default=False):
            subprocess.run(["pm2", "delete", pm2_name], capture_output=True)
        else:
            print(f"  Check logs: pm2 logs {pm2_name} --lines 50")
            print()
            return

    # Check for OTHER miner processes that might be using the GPU
    # (e.g. miner-mainnet, miner-gpu1). Don't check for validators/proxies.
    if "miner" in pm2_name:
        other_miners = [
            p for p in procs
            if "miner" in p.get("name", "")
            and p.get("name") != pm2_name
            and p.get("pm2_env", {}).get("status") == "online"
        ]
        if other_miners:
            print()
            print(yellow(f"  Warning: {len(other_miners)} other miner process(es) running:"))
            for p in other_miners:
                _name = p.get("name", "?")
                _pid = p.get("pid", "?")
                print(f"    {_name} (pid {_pid})")
            print("  Starting another miner may fail if the GPU is already in use.")
            print()
            print(f"    [Enter] Start anyway")
            print(f"    [S]     Stop other miners first, then start")
            print(f"    [N]     Cancel")
            choice = _prompt("Action", "").upper()
            if choice == "N":
                return
            elif choice == "S":
                for p in other_miners:
                    _name = p.get("name", "?")
                    subprocess.run(["pm2", "delete", _name], capture_output=True)
                    print(f"  Stopped {_name}")
                print()

    subprocess.run(
        ["pm2", "start", "ecosystem.config.js", "--only", pm2_name],
        cwd=str(repo),
        stdin=sys.stdin,
    )
    subprocess.run(["pm2", "save"], capture_output=True)
    print()
    print(green(f"  {pm2_name} started! Check logs:"))
    print(f"    pm2 logs {pm2_name} --lines 50")
    print()


def _change_endpoint(repo: Path, miners: list[dict]) -> None:
    """Change endpoint for an existing miner (adds --update-endpoint)."""
    from urllib.parse import urlparse

    m = miners[0]
    old_ep = m.get("endpoint", "?")
    _old_parsed = urlparse(old_ep)
    _old_host = _old_parsed.hostname or ""
    _old_port = _old_parsed.port or 443

    print(f"  Current endpoint: {old_ep}")
    print()

    # Ask for host (default: current)
    public_ip = _detect_public_ip()
    _default_host = _old_host or public_ip or ""
    host = _prompt("Hostname/IP", _default_host)

    # Ask for port (default: current)
    while True:
        port_str = _prompt("HTTPS port", str(_old_port)).strip()
        if port_str.isdigit() and 1 <= int(port_str) <= 65535:
            port = int(port_str)
            break
        print(red(f"  Invalid port: enter a number between 1 and 65535."))

    new_ep = f"https://{host}:{port}" if port != 443 else f"https://{host}"
    print(f"  New endpoint: {new_ep}")

    if new_ep == old_ep:
        print(yellow("  Endpoint unchanged."))
        return

    config_path = repo / "ecosystem.config.js"
    text = config_path.read_text()
    text = text.replace(old_ep, new_ep)
    if "--update-endpoint" not in text:
        text = text.replace("--auto-update", "--auto-update --update-endpoint")
    config_path.write_text(text)
    print(green(f"  Updated: {old_ep} → {new_ep}"))
    print(f"  Config includes --update-endpoint (updates on-chain, no new registration)")
    print()

    # Set up HTTPS on the new port
    if port != _old_port:
        setup_script = repo / "scripts" / "setup_https.sh"
        if setup_script.exists():
            if _confirm(f"Set up HTTPS on port {port}?", default=True):
                subprocess.run(
                    ["sudo", "bash", str(setup_script), "--port", str(port)],
                    stdin=sys.stdin,
                )

    if _confirm("Restart with new endpoint?", default=True):
        pm2 = shutil.which("pm2")
        if not pm2:
            print(yellow("  PM2 not found — installing..."))
            step_pm2()
            pm2 = shutil.which("pm2")
        if pm2:
            subprocess.run([pm2, "delete", m["name"]], capture_output=True)
            subprocess.run(
                [pm2, "start", "ecosystem.config.js", "--only", m["name"]],
                cwd=str(repo), stdin=sys.stdin,
            )
        subprocess.run(["pm2", "save"], capture_output=True)
        print(green(f"  {m['name']} restarted with new endpoint"))
        print(f"  Check logs: pm2 logs {m['name']} --lines 50")


def _check_model_cache(new_model: str) -> None:
    """Check HuggingFace cache for unused models and offer to delete them."""
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if not cache_dir.exists():
        return

    # Find cached models
    cached = []
    for d in cache_dir.iterdir():
        if d.is_dir() and d.name.startswith("models--"):
            # models--Qwen--Qwen3.5-9B → Qwen/Qwen3.5-9B
            parts = d.name.replace("models--", "").split("--", 1)
            hf_name = "/".join(parts) if len(parts) == 2 else parts[0]
            try:
                size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                size_gb = size_bytes / (1024 ** 3)
            except OSError:
                size_gb = 0
            cached.append({"path": d, "name": hf_name, "size_gb": size_gb})

    if not cached:
        return

    # Find models that aren't the new selection
    others = [c for c in cached if c["name"].lower() != new_model.lower()]
    if not others:
        return

    # Check disk space
    import shutil as _shutil
    disk = _shutil.disk_usage(str(cache_dir))
    free_gb = disk.free / (1024 ** 3)

    print()
    print(f"  Cached models ({free_gb:.1f} GB free):")
    for c in cached:
        is_new = c["name"].lower() == new_model.lower()
        label = green("(selected)") if is_new else dim("(unused)")
        print(f"    {c['name']:<45s} {c['size_gb']:>5.1f} GB  {label}")

    total_reclaimable = sum(c["size_gb"] for c in others)
    if total_reclaimable > 0.5:
        print()
        if _confirm(f"Delete unused models to free {total_reclaimable:.1f} GB?", default=free_gb < 5):
            for c in others:
                import shutil as _shutil2
                _shutil2.rmtree(c["path"], ignore_errors=True)
                print(f"    Deleted: {c['name']} ({c['size_gb']:.1f} GB)")
            disk = _shutil.disk_usage(str(cache_dir))
            print(f"  Disk: {disk.free / (1024**3):.1f} GB free")


def _change_model(repo: Path, miners: list[dict]) -> None:
    """Change model/quant for an existing miner, keeping endpoint + wallet."""
    m = miners[0]
    old_args = m.get("args", "")
    old_model_m = re.search(r"--model-id\s+(\S+)", old_args)
    old_model = old_model_m.group(1) if old_model_m else "auto"
    ep = m.get("endpoint", "?")

    # Extract current quant if present
    old_quant_m = re.search(r"--quant\s+(\S+)", old_args)
    old_quant = old_quant_m.group(1) if old_quant_m else "auto"

    print(f"  Current: {bold(old_model)} (quant={old_quant})")
    print(f"  Endpoint: {ep} (unchanged)")
    print()

    # Pass chain config for on-chain model filtering
    net_info = {}
    chain_m = re.search(r"--subtensor-network\s+(\S+)", old_args)
    if chain_m:
        net_info["network"] = chain_m.group(1)
    for name in ("chain_config_mainnet.json", "chain_config.json",
                 "chain_config_testnet.json"):
        p = repo / name
        if p.exists():
            net_info["config_path"] = str(p)
            break

    model_info = step_model(net_info)
    new_model = model_info.get("model_id", "auto")
    new_quant = model_info.get("quant", "")

    if new_model == old_model and new_quant == old_quant:
        print(yellow("  Model unchanged."))
        return

    # Check for cached models that can be cleaned up (skip for "auto" —
    # we don't know which model will be selected at runtime)
    if new_model != "auto":
        _check_model_cache(new_model)

    # Replace --model-id in the config
    config_path = repo / "ecosystem.config.js"
    text = config_path.read_text()
    if old_model_m:
        text = text.replace(f"--model-id {old_model}", f"--model-id {new_model}")
    else:
        text = text.replace(f"--endpoint {ep}", f"--model-id {new_model} --endpoint {ep}")

    # Replace or add --quant
    if old_quant_m:
        if new_quant:
            text = text.replace(f"--quant {old_quant}", f"--quant {new_quant}")
        else:
            # Remove --quant flag (switching to auto quant)
            text = text.replace(f" --quant {old_quant}", "")
    elif new_quant:
        # Add --quant after --model-id
        text = text.replace(f"--model-id {new_model}", f"--model-id {new_model} --quant {new_quant}")

    config_path.write_text(text)
    _label = f"{new_model} ({new_quant})" if new_quant else new_model
    print()
    print(green(f"  Model updated: {old_model} → {_label}"))
    print(f"  On-chain: old entry will be deactivated, new one registered at startup.")
    print()

    if _confirm("Restart miner with new model?", default=True):
        pm2 = shutil.which("pm2")
        if not pm2:
            print(yellow("  PM2 not found — installing..."))
            step_pm2()
            pm2 = shutil.which("pm2")
        if pm2:
            subprocess.run([pm2, "delete", m["name"]], capture_output=True)
            subprocess.run(
                [pm2, "start", "ecosystem.config.js", "--only", m["name"]],
                cwd=str(repo), stdin=sys.stdin,
            )
            subprocess.run([pm2, "save"], capture_output=True)
            print(green(f"  {m['name']} restarted with new model"))
            print(f"  Check logs: pm2 logs {m['name']} --lines 50")


def run_wizard(role: str = "miner") -> None:
    """Run the interactive setup wizard."""
    repo = _find_repo_root()
    _banner(role)

    # ── Early detection: existing config? ─────────────────────────────
    existing_miners = _parse_existing_config(repo) if role == "miner" else []
    if existing_miners:
        m = existing_miners[0]
        ep = m.get("endpoint", "?")
        # Extract model from args
        model_match = re.search(r"--model-id\s+(\S+)", m.get("args", ""))
        model = model_match.group(1) if model_match else "?"
        print(f"  Existing config detected:")
        print(f"    {m['name']}: {model} → {ep}")
        print()
        print(f"  [1] Change endpoint (update on-chain)")
        print(f"  [2] Full reconfigure (re-run all steps)")
        print(f"  [3] Add model endpoint (different model/GPU/port)")
        print(f"  [4] Change model/quant (keep endpoint + wallet)")
        print(f"  [5] Start as-is")
        print()
        choice = _prompt("Choice", "5")

        if choice == "1":
            _change_endpoint(repo, existing_miners)
            return
        elif choice == "5":
            _quick_start(repo, m["name"])
            return
        elif choice == "4":
            _change_model(repo, existing_miners)
            return
        elif choice == "3":
            # Check GPU count before adding a model endpoint
            try:
                import torch
                n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if n_gpus <= len(existing_miners):
                    print()
                    print(yellow(f"  Warning: {n_gpus} GPU{'s' if n_gpus != 1 else ''} detected, "
                                 f"{len(existing_miners)} endpoint{'s' if len(existing_miners) != 1 else ''} already configured."))
                    print("  Each model endpoint needs its own GPU.")
                    if not _confirm("Continue anyway?", default=False):
                        return
                elif n_gpus > 1:
                    print()
                    print(f"  {n_gpus} GPUs detected:")
                    for i in range(n_gpus):
                        p = torch.cuda.get_device_properties(i)
                        # Check if GPU is already assigned to an existing endpoint
                        in_use = False
                        for em in existing_miners:
                            args = em.get("args", "")
                            # If no CUDA_VISIBLE_DEVICES in args, GPU 0 is implied
                            if "CUDA_VISIBLE_DEVICES" not in args and i == 0:
                                in_use = True
                            elif f"CUDA_VISIBLE_DEVICES" in args and str(i) in args:
                                in_use = True
                        status = f" {yellow('(in use)')}" if in_use else ""
                        print(f"    [{i}] {p.name} ({p.total_memory / (1024**3):.0f} GB){status}")
            except ImportError:
                pass
            # Fall through to full wizard
        # choice == "2" or "3" → continue to full wizard

    total_steps = 9 if role == "miner" else 7

    # Step 1: Network
    _header(1, total_steps, "Network")
    net_info = step_network(repo)
    net_label = f"{net_info['network']} (netuid {net_info['netuid']})"
    print(f"  {green(net_label)}")

    # Step 2: Wallet
    _header(2, total_steps, "Wallet")
    wallet_info = step_wallet(role)
    if not wallet_info.get("wallet") or not wallet_info.get("hotkey"):
        return  # No wallet — instructions printed, user needs to set up first
    if not _wallet_exists(wallet_info):
        print(yellow("  Wallet files not found on disk. Set up wallet first."))
        return

    # Step 3: Registration
    _header(3, total_steps, "Subnet Registration")
    reg_info = step_registration(wallet_info, net_info)

    if not reg_info.get("registered"):
        _cmd = "verathos setup" if role == "miner" else "verathos setup validator"
        print()
        print(f"  Register your hotkey on the subnet, then re-run: {_cmd}")
        return

    # Step 4: EVM Funding
    _header(4, total_steps, "EVM Funding")
    evm_info = step_evm_funding(wallet_info, net_info, role=role)

    if not evm_info.get("evm_funded"):
        if role == "validator" and _confirm("Skip EVM funding and run validator without EVM?", default=True):
            print(dim("  Skipping. Validator will run with --no-evm (no on-chain reportOffline)."))
            evm_info["no_evm"] = True
        else:
            _cmd = "verathos setup" if role == "miner" else "verathos setup validator"
            print()
            print(f"  Fund your EVM address, then re-run: {_cmd}")
            return

    if role == "miner":
        # Step 5: HTTPS
        _header(5, total_steps, "HTTPS Endpoint")
        https_info = step_https(repo)

        # Step 6: Model Selection
        _header(6, total_steps, "Model Selection")
        model_info = step_model(net_info)

        # Step 7: PM2
        _header(7, total_steps, "PM2 Process Manager")
        pm2_info = step_pm2()

        # Step 8: Options (subtensor, logging)
        _header(8, total_steps, "Options")
        options = step_options(role)

        # Step 9: Config
        _header(9, total_steps, "ecosystem.config.js")
        config_info = step_config(role, repo, wallet_info, net_info, https_info, model_info, options)
    else:
        # Validator: skip HTTPS + model, still need PM2 + options + config
        https_info = {"endpoint": "", "port": 0, "host": ""}
        model_info = None

        _header(5, total_steps, "PM2 Process Manager")
        pm2_info = step_pm2()

        # Step 6: Options (subtensor, logging, analytics)
        _header(6, total_steps, "Options")
        options = step_options(role)
        if evm_info.get("no_evm"):
            options["no_evm"] = True

        _header(7, total_steps, "ecosystem.config.js")
        config_info = step_config(role, repo, wallet_info, net_info, https_info, model_info, options)

    # ── Summary ───────────────────────────────────────────────────────────
    print()

    # Check what's missing (registration + EVM already blocked above,
    # so only PM2 and config can be incomplete at this point)
    missing = []
    if not pm2_info.get("pm2_installed"):
        missing.append(("PM2", "sudo npm install -g pm2"))
    if not config_info.get("config_generated"):
        _cmd = "verathos setup" if role == "miner" else "verathos setup validator"
        missing.append(("ecosystem.config.js", f"Re-run: {_cmd}"))

    if missing:
        print(bold("  ═══════════════════════════════════════════════"))
        print(bold("   Setup Incomplete"))
        print(bold("  ═══════════════════════════════════════════════"))
        print()
        print("  The following steps still need to be completed:")
        print()
        for label, hint in missing:
            print(f"    {_fail()} {label}")
            print(f"      {dim(hint)}")
        print()
        print(f"  After completing these steps, re-run: {bold('verathos setup')}")
        print()
        return

    # All good — offer to start
    print(bold("  ═══════════════════════════════════════════════"))
    print(bold("   Setup Complete"))
    print(bold("  ═══════════════════════════════════════════════"))
    print()

    pm2_name = config_info.get("pm2_name", role)
    if _confirm(f"Start {pm2_name} now?", default=True):
        _quick_start(repo, pm2_name)
    else:
        print(f"  Start manually: pm2 start ecosystem.config.js --only {pm2_name}")

    print()
    print(f"  Check readiness anytime: {bold('verathos status')}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point (python -m neurons.wizard)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="verathos-wizard",
        description="Interactive setup wizard for Verathos miners and validators.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="miner",
        choices=["miner", "validator", "status"],
        help="What to set up (default: miner)",
    )
    args = parser.parse_args()

    if args.command == "status":
        run_status()
    else:
        run_wizard(args.command)


if __name__ == "__main__":
    main()
