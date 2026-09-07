"""Release identity, role update thresholds and weight specification.

``release_version`` / ``release_version_str`` identify the tagged code release.
An opt-in release advances this identity without changing role thresholds.

``miner_version`` and ``validator_version`` are automatic-update thresholds.
The updater adopts newer code only when its role's remote threshold increases;
proxies use the validator threshold. A newer release alone triggers no restart.

``spec_version`` remains max(miner_version, validator_version) and is supplied
as the weight version key. ``version_str`` retains its legacy spec-string
meaning; use ``release_version_str`` to report the installed code release.

Every code release receives a monotonic matching Git tag. Advance affected
role thresholds when automatic adoption is intended. Opt-in releases must be
qualified for manual adoption and future automatic upgrades as well.
"""

_VERSION_BASE = 1_000

# Release identity is independent of automatic-update thresholds. An opt-in
# release advances these constants without advancing MINER_* or VALIDATOR_*.
RELEASE_MAJOR = 0
RELEASE_MINOR = 2
RELEASE_PATCH = 1


def _encode(major: int, minor: int, patch: int) -> int:
    return major * _VERSION_BASE * _VERSION_BASE + minor * _VERSION_BASE + patch


def _version_str(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


def _decode(version: int) -> tuple[int, int, int]:
    major = version // (_VERSION_BASE * _VERSION_BASE)
    minor = (version // _VERSION_BASE) % _VERSION_BASE
    patch = version % _VERSION_BASE
    return major, minor, patch


release_version: int = _encode(RELEASE_MAJOR, RELEASE_MINOR, RELEASE_PATCH)
release_version_str: str = _version_str(RELEASE_MAJOR, RELEASE_MINOR, RELEASE_PATCH)


# ── Miner version ────────────────────────────────────────────────
#
# Advance when automatic miner adoption is required. Opt-in code releases
# retain this threshold; only miners with --auto-update restart on an increase.

MINER_MAJOR = 0
MINER_MINOR = 2
MINER_PATCH = 0

miner_version: int = _encode(MINER_MAJOR, MINER_MINOR, MINER_PATCH)
miner_version_str: str = _version_str(MINER_MAJOR, MINER_MINOR, MINER_PATCH)


# ── Validator / proxy version ────────────────────────────────────
#
# Advance when automatic validator/proxy adoption is required.
# Opt-in code releases retain this threshold. Both roles use this comparison.

VALIDATOR_MAJOR = 0
VALIDATOR_MINOR = 2
VALIDATOR_PATCH = 0

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
