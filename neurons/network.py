#!/usr/bin/env python3
"""Verathos network status — fetch and display network-wide stats.

Uses the proxy's ``/v1/network/stats`` endpoint.  Discovery order:

1. ``--proxy-url`` (explicit override)
2. ``http://localhost:{8080,8081}`` (local proxy)
3. ``https://api.verathos.ai`` (public default)
4. On-chain ValidatorRegistry fallback (requires chain config)
"""
# TODO: organic traffic stats omitted — /v1/network/stats only reflects one
# proxy's traffic.  Network-wide stats could be added by aggregating
# network_receipts from shared_state or validator DB.

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

import re as _re

from neurons.wizard import green, red, yellow, bold, dim

_ANSI_RE = _re.compile(r"\033\[[0-9;]*m")

# ── Constants ─────────────────────────────────────────────────────

DEFAULT_PROXY_URL = "https://api.verathos.ai"
LOCAL_PORTS = [8080, 8081]
FETCH_TIMEOUT = 10  # seconds
DEFAULT_WATCH_INTERVAL = 30
MIN_WATCH_INTERVAL = 5

# Mainnet chain IDs
_MAINNET_CHAIN_ID = 964  # Bittensor finney EVM
_TESTNET_CHAIN_ID = 945  # Bittensor testnet EVM


# ── Data fetching ─────────────────────────────────────────────────

def _try_fetch(url: str, timeout: int = FETCH_TIMEOUT) -> Optional[dict]:
    """GET ``{url}/v1/network/stats``, return parsed JSON or None."""
    target = f"{url.rstrip('/')}/v1/network/stats"
    try:
        ctx = None
        # Skip TLS verification for localhost (self-signed certs are common).
        if "localhost" in url or "127.0.0.1" in url:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(target, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return None


def _discover_from_registry() -> Optional[dict]:
    """Try to discover a proxy via on-chain ValidatorRegistry."""
    try:
        from verallm.chain.config import ChainConfig
        from verallm.chain.validator_registry import ValidatorRegistryClient
    except ImportError:
        return None

    # Search for chain config files in common locations.
    from neurons.wizard import _find_repo_root
    try:
        root = str(_find_repo_root())
    except Exception:
        return None

    for name in ("chain_config_mainnet.json", "chain_config.json",
                 "chain_config_testnet.json"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                cfg = ChainConfig.from_json(path)
                client = ValidatorRegistryClient(cfg)
                validators = client.get_proxy_validators()
                for v in validators:
                    ep = getattr(v, "proxy_endpoint", "") or ""
                    if ep:
                        data = _try_fetch(ep)
                        if data is not None:
                            return data
            except Exception:
                continue
    return None


def fetch_stats(proxy_url: Optional[str] = None) -> dict:
    """Fetch ``/v1/network/stats`` from a proxy.

    Discovery order: explicit URL -> localhost -> api.verathos.ai -> chain registry.

    Returns:
        Parsed JSON dict from the proxy.

    Raises:
        RuntimeError: If no proxy could be reached.
    """
    # 1. Explicit URL
    if proxy_url:
        data = _try_fetch(proxy_url)
        if data is not None:
            return data
        raise RuntimeError(
            f"Could not reach proxy at {proxy_url}. "
            "Check the URL and ensure the proxy is running."
        )

    # 2. Localhost
    for port in LOCAL_PORTS:
        data = _try_fetch(f"http://localhost:{port}")
        if data is not None:
            return data

    # 3. Public default
    data = _try_fetch(DEFAULT_PROXY_URL)
    if data is not None:
        return data

    # 4. On-chain registry
    data = _discover_from_registry()
    if data is not None:
        return data

    raise RuntimeError(
        "Could not reach any Verathos proxy.\n"
        f"  Tried: localhost:{LOCAL_PORTS}, {DEFAULT_PROXY_URL}, on-chain registry\n"
        "  Use:   verathos network --proxy-url <URL>"
    )


# ── Rendering helpers ─────────────────────────────────────────────

def _visible_len(s: str) -> int:
    """Length of string excluding ANSI escape codes."""
    return len(_ANSI_RE.sub("", s))


def _pad(s: str, width: int, align: str = "<") -> str:
    """Pad string to ``width`` visible chars, accounting for ANSI codes."""
    diff = len(s) - _visible_len(s)
    if align == ">":
        return f"{s:>{width + diff}s}"
    return f"{s:<{width + diff}s}"


def _short_model(model_id: str) -> str:
    """Strip HuggingFace org prefix: ``meta-llama/Llama-3.3-70B`` -> ``Llama-3.3-70B``."""
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _short_addr(ss58: str, evm: str = "") -> str:
    """Abbreviate SS58 address: ``5HPPZSm3D4...`` -> ``5HPP..Gxf``.

    Falls back to EVM address if SS58 is not available.
    """
    addr = ss58 or evm
    if not addr:
        return "-"
    if len(addr) > 10:
        return addr[:4] + ".." + addr[-3:]
    return addr


def _format_gpu(gpu_name: str, gpu_count: int, vram_gb: int) -> str:
    """Format GPU info: ``A100x2 160G`` or ``H100 80G`` or dim dash."""
    if not gpu_name:
        return dim("-")
    name = gpu_name
    # Common abbreviations for readability.
    for prefix in ("NVIDIA ", "GeForce "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if gpu_count > 1:
        label = f"{name}x{gpu_count}"
    else:
        label = name
    if vram_gb > 0:
        label += f" {vram_gb}G"
    return label


def _format_ctx(max_context_len: int) -> str:
    """Format context length: 32768 -> ``32k``."""
    if max_context_len <= 0:
        return dim("-")
    return f"{max_context_len // 1024}k"


def _format_score(score: float) -> str:
    """Color-coded score: green >= 0.5, yellow > 0, red/dim = 0."""
    s = f"{score:.3f}"
    if score >= 0.5:
        return green(s)
    elif score > 0:
        return yellow(s)
    return dim(s)


def _format_health(healthy: bool, on_probation: bool) -> str:
    """Health indicator with optional probation badge."""
    if healthy:
        h = green("\u2713")
    else:
        h = red("\u2717")
    if on_probation:
        h += " " + yellow("(P)")
    return h


def _network_name(chain_id: Optional[int]) -> str:
    """Human-readable network name from chain ID."""
    if chain_id == _TESTNET_CHAIN_ID:
        return "Testnet"
    elif chain_id == _MAINNET_CHAIN_ID:
        return "Mainnet"
    elif chain_id is not None:
        return f"Chain {chain_id}"
    return "Unknown"


# ── Main renderer ─────────────────────────────────────────────────

def render_network(data: dict) -> None:
    """Print ANSI-formatted network overview to stdout."""
    miners = data.get("miners", [])
    totals = data.get("totals", {})
    validators = data.get("validators", [])
    epoch = data.get("epoch_number", 0)
    netuid = data.get("netuid")
    chain_id = data.get("chain_id")

    network = _network_name(chain_id)
    netuid_str = f"SN{netuid}" if netuid else "SN?"

    # ── Header ────────────────────────────────────────────────
    print()
    print(bold(f"  Verathos Network \u00b7 {netuid_str} \u00b7 {network} \u00b7 Epoch {epoch}"))
    _line = "\u2500" * 62
    print(f"  {_line}")

    # ── Summary line ──────────────────────────────────────────
    n_miners = totals.get("miners", 0)
    n_healthy = totals.get("healthy", 0)
    n_models = totals.get("models", 0)
    n_validators = totals.get("validators", len(validators))
    n_slots = len(miners)

    # TEE count: unique addresses with at least one TEE slot.
    tee_addrs = set()
    for m in miners:
        if m.get("tee_enabled"):
            tee_addrs.add(m.get("address", "").lower())
    n_tee = len(tee_addrs)

    parts = [
        f"Miners: {bold(str(n_miners))} ({n_healthy} healthy)",
        f"Slots: {bold(str(n_slots))}",
        f"Models: {bold(str(n_models))}",
        f"Validators: {bold(str(n_validators))}",
    ]
    if n_tee > 0:
        parts.append(f"TEE: {n_tee}/{n_miners}")

    print(f"  {('  ').join(parts)}")
    print()

    # ── Slots table ───────────────────────────────────────────
    print(bold("  Slots"))
    if not miners:
        print(dim("  No miners registered."))
    else:
        # Sort: primary by UID (None last), secondary by score descending.
        sorted_miners = sorted(
            miners,
            key=lambda m: (
                m.get("uid") if m.get("uid") is not None else 99999,
                -m.get("score", 0),
            ),
        )

        # Column headers.
        hdr = (
            f"  {'UID':>4s}  {'Addr':<9s}  {'Model':<25s}  {'Quant':<6s}  "
            f"{'Score':>5s}  {'Health':<8s}  {'TEE':<3s}  {'GPU':<16s}  {'Ctx':>4s}"
        )
        print(dim(hdr))
        print(dim(f"  {'─' * 4}  {'─' * 9}  {'─' * 25}  {'─' * 6}  "
                   f"{'─' * 5}  {'─' * 8}  {'─' * 3}  {'─' * 16}  {'─' * 4}"))

        for m in sorted_miners:
            uid = m.get("uid")
            uid_str = f"{uid:4d}" if uid is not None else "   -"
            addr = _short_addr(m.get("ss58_address", ""), m.get("address", ""))
            model = _short_model(m.get("model_id", ""))[:25]
            quant = (m.get("quant") or "-")[:6]
            score = _pad(_format_score(m.get("score", 0)), 5, ">")
            health = _pad(
                _format_health(m.get("healthy", True), m.get("on_probation", False)),
                8,
            )
            tee = _pad(green("\u2713") if m.get("tee_enabled") else " ", 3)
            gpu = _format_gpu(
                m.get("gpu_name", ""), m.get("gpu_count", 0), m.get("vram_gb", 0),
            )
            # Truncate GPU to visible 16 chars.
            if _visible_len(gpu) > 16:
                gpu = _ANSI_RE.sub("", gpu)[:16]
            gpu = _pad(gpu, 16)
            ctx = _pad(_format_ctx(m.get("max_context_len", 0)), 4, ">")

            print(
                f"  {uid_str}  {addr:<9s}  {model:<25s}  {quant:<6s}  "
                f"{score}  {health}  {tee}  {gpu}  {ctx}"
            )

    print()

    # ── Validators table ──────────────────────────────────────
    print(bold("  Validators"))
    if not validators:
        print(dim("  No validators found."))
    else:
        hdr = f"  {'UID':>4s}  {'Proxy Endpoint':<38s}  {'Active':<6s}"
        print(dim(hdr))
        print(dim(f"  {'─' * 4}  {'─' * 38}  {'─' * 6}"))

        sorted_vals = sorted(validators, key=lambda v: v.get("uid", 99999))
        for v in sorted_vals:
            uid = v.get("uid")
            uid_str = f"{uid:4d}" if uid is not None else "   -"
            ep = v.get("proxy_endpoint", "") or ""
            if not ep:
                ep_str = dim("\u2014") + "  " + dim("(no proxy)")
            else:
                ep_str = ep[:38]
            active = green("\u2713") if v.get("active", True) else red("\u2717")
            print(f"  {uid_str}  {ep_str:<38s}  {active}")

    print()

    # ── Models table ──────────────────────────────────────────
    print(bold("  Models"))
    if not miners:
        print(dim("  No models active."))
    else:
        # Aggregate per model.
        model_agg: dict = {}  # model_id -> {slots, miners, healthy, scores}
        for m in miners:
            mid = m.get("model_id", "")
            if not mid:
                continue
            if mid not in model_agg:
                model_agg[mid] = {
                    "slots": 0, "miners": set(), "healthy": set(), "scores": [],
                }
            agg = model_agg[mid]
            agg["slots"] += 1
            addr = m.get("address", "").lower()
            agg["miners"].add(addr)
            is_healthy = m.get("healthy", True)
            if is_healthy:
                agg["healthy"].add(addr)
            score = m.get("score", 0)
            if score > 0 and is_healthy:
                agg["scores"].append(score)

        hdr = (
            f"  {'Model':<25s}  {'Slots':>5s}  {'Active':>6s}  "
            f"{'Miners':>6s}  {'Healthy':>7s}  {'Avg Score':>9s}"
        )
        print(dim(hdr))
        print(dim(
            f"  {'─' * 25}  {'─' * 5}  {'─' * 6}  "
            f"{'─' * 6}  {'─' * 7}  {'─' * 9}"
        ))

        sorted_models = sorted(
            model_agg.items(),
            key=lambda kv: (sum(kv[1]["scores"]) / max(len(kv[1]["scores"]), 1)),
            reverse=True,
        )
        for mid, agg in sorted_models:
            name = _short_model(mid)[:25]
            active = len(agg["scores"])
            avg = sum(agg["scores"]) / max(active, 1)
            print(
                f"  {name:<25s}  {agg['slots']:>5d}  {active:>6d}  "
                f"{len(agg['miners']):>6d}  {len(agg['healthy']):>7d}  "
                f"{_pad(_format_score(avg), 9, '>')}"
            )

    print()


# ── CLI entry point ───────────────────────────────────────────────

def _print_help() -> None:
    print("""
  verathos network                  Show network status
  verathos network --json           JSON output (scriptable)
  verathos network --watch [N]      Auto-refresh every N seconds (default: 30)
  verathos network --proxy-url URL  Use specific proxy endpoint
""")


def cmd_network(args: list[str]) -> None:
    """CLI entry point for ``verathos network``."""
    proxy_url: Optional[str] = None
    json_mode = False
    watch_mode = False
    watch_interval = DEFAULT_WATCH_INTERVAL

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--proxy-url" and i + 1 < len(args):
            proxy_url = args[i + 1]
            i += 2
        elif arg == "--json":
            json_mode = True
            i += 1
        elif arg == "--watch":
            watch_mode = True
            if i + 1 < len(args) and args[i + 1].isdigit():
                watch_interval = max(MIN_WATCH_INTERVAL, int(args[i + 1]))
                i += 2
            else:
                i += 1
        elif arg in ("-h", "--help"):
            _print_help()
            return
        else:
            print(f"Unknown option: {arg}")
            _print_help()
            sys.exit(1)

    if watch_mode and json_mode:
        print(red("  --watch and --json cannot be used together."))
        sys.exit(1)

    # ── One-shot mode ─────────────────────────────────────────
    if not watch_mode:
        try:
            data = fetch_stats(proxy_url)
        except RuntimeError as e:
            print(red(f"  {e}"))
            sys.exit(1)

        if json_mode:
            json.dump(data, sys.stdout, indent=2)
            print()
            return

        render_network(data)
        return

    # ── Watch mode ────────────────────────────────────────────
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            try:
                data = fetch_stats(proxy_url)
                render_network(data)
                print(dim(f"  Refreshing every {watch_interval}s. Press Ctrl+C to exit."))
            except RuntimeError as e:
                print(red(f"  {e}"))
                print(dim(f"  Retrying in {watch_interval}s..."))
            time.sleep(watch_interval)
    except KeyboardInterrupt:
        print()
