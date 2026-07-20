"""Subnet version constants for on-chain weight gating and auto-update.

Three independent version numbers control update behavior:

``spec_version``
    On-chain weight gating.  Passed as ``version_key`` to
    ``subtensor.set_weights()``.  Must match the subnet's
    ``weights_version`` hyperparameter or the chain **rejects** the
    extrinsic.  From now on this is derived as
    ``max(miner_version, validator_version)`` so any role release can set
    weights after the subnet owner updates the on-chain version key.

``miner_version``
    Miner-side code version.  The auto-updater on miners only restarts
    when the remote ``miner_version`` is higher than the local one.
    Bump when miner code changes (server, proofs, registration).

``validator_version``
    Validator/proxy-side code version.  The auto-updater on validators
    and proxies only restarts when the remote ``validator_version`` is
    higher.  Bump when validator, proxy, scoring, or canary code changes.

This means a validator-only bug fix (bump ``validator_version``) does NOT
force miners to restart (and vice versa).  The subnet ``weights_version`` must
track ``spec_version``, which is always the higher role version.

Encoding
--------
Same base-1000 scheme as bittensor core::

    version_key = MAJOR * 1_000_000 + MINOR * 1_000 + PATCH

Examples::

    0.1.0  →     1_000
    0.2.0  →     2_000
    1.0.0  → 1_000_000
    1.2.3  → 1_002_003

Bump workflow
-------------
1. Edit the relevant role version constants below.
2. Commit and push.
3. Miners/validators with ``--auto-update`` will pull and restart
   **only if their role's version increased**.
4. If either role version changed, the subnet owner must also update the
   on-chain ``weights_version`` to the derived ``spec_version``::

       subtensor.sudo_set_weights_version_key(
           netuid=405, weights_version_key=spec_version,
       )
"""

from __future__ import annotations

_VERSION_BASE = 1_000


def _encode(major: int, minor: int, patch: int) -> int:
    return major * _VERSION_BASE * _VERSION_BASE + minor * _VERSION_BASE + patch


def _version_str(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


def _decode(version: int) -> tuple[int, int, int]:
    major = version // (_VERSION_BASE * _VERSION_BASE)
    minor = (version // _VERSION_BASE) % _VERSION_BASE
    patch = version % _VERSION_BASE
    return major, minor, patch


# ── Miner version ────────────────────────────────────────────────
#
# Bump when miner-side code changes: server, proofs, registration,
# heartbeat, model selection.  Only miners with --auto-update restart.

MINER_MAJOR = 0
MINER_MINOR = 1
MINER_PATCH = 0

miner_version: int = _encode(MINER_MAJOR, MINER_MINOR, MINER_PATCH)
miner_version_str: str = _version_str(MINER_MAJOR, MINER_MINOR, MINER_PATCH)


# ── Validator / proxy version ────────────────────────────────────
#
# Bump when validator or proxy code changes: canary testing, scoring,
# weight setting, routing, credits, auth, settlement.
# Both validators and proxies with --auto-update restart.

VALIDATOR_MAJOR = 0
VALIDATOR_MINOR = 1
VALIDATOR_PATCH = 25

validator_version: int = _encode(VALIDATOR_MAJOR, VALIDATOR_MINOR, VALIDATOR_PATCH)
validator_version_str: str = _version_str(VALIDATOR_MAJOR, VALIDATOR_MINOR, VALIDATOR_PATCH)


# ── Subnet protocol version (on-chain weight gating) ─────────────
#
# The subnet weight version tracks the highest role version.  This keeps
# validator-only and miner-only releases compatible with Bittensor weight
# gating without forcing the other role's auto-updater to restart.

spec_version: int = max(miner_version, validator_version)
SPEC_MAJOR, SPEC_MINOR, SPEC_PATCH = _decode(spec_version)
version_str: str = _version_str(SPEC_MAJOR, SPEC_MINOR, SPEC_PATCH)
