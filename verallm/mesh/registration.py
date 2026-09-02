"""On-chain registration for GGUF mesh miners.

The single writer of a mesh miner's MinerRegistry entry. Everything a
validator will later derive from chain is resolved here FROM chain (the
ModelSpec), never typed by the operator: one hand-typed hex digest in the
pool registry makes every canary fail snapshot validation with no local
symptom, so operator input is limited to the model id, the public endpoint,
and the context length.

Reuses the contract slot-lifecycle planning and the fail-closed mesh
eligibility gate from verallm.chain.miner_lifecycle; wraps them with the
chain clients, the scoring-profile allowlist, and the post-transaction index
verification the mesh's signed snapshots depend on.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from verallm.chain.miner_lifecycle import (
    LifecyclePlan,
    LifecycleRefusal,
    MESH_QUANT_PREFIX,
    RegistrationTarget,
    configuration_matches,
    is_live,
    plan_registration,
    plan_renewal,
    resolve_registered_index,
    target_from_values,
    validate_mesh_model_registry_eligibility,
)

logger = logging.getLogger(__name__)

MESH_REGISTRATION_STATE_FILE = "registration.json"


@dataclass(frozen=True)
class MeshChainAnchors:
    """Everything the validator independently derives from chain."""

    model_id: str
    gguf_scheme: str  # "q4_k_m"
    registry_quant: str  # "gguf_mesh_q4_k_m" (MinerRegistry)
    chain_quantization_scheme: str  # "gguf_q4_k_m" (ModelSpec.quant_mode)
    model_package_hash: str  # weight_file_hash hex
    model_tensor_manifest_root: str  # weight_merkle_root hex
    tokenizer_hash: str  # tokenizer_hash hex
    total_layers: int
    max_context_len: int


@dataclass(frozen=True)
class RegistrationOutcome:
    action: str  # reuse-active | reactivate | append | renew
    index: int
    tx_hash: str  # "" when no transaction was needed
    expires_at: int


def _model_registry_client(chain_config):
    from verallm.chain.model_registry import ModelRegistryClient

    return ModelRegistryClient(chain_config)


def _miner_registry_client(chain_config):
    from verallm.chain.miner_registry import MinerRegistryClient

    return MinerRegistryClient(chain_config)


def resolve_mesh_chain_anchors(
    chain_config,
    model_id: str,
    *,
    max_context_len: int | None = None,
) -> MeshChainAnchors:
    """Fail-closed anchor resolution mirroring the validator's own checks.

    1. The ModelSpec must exist on ModelRegistry with a gguf quant mode.
    2. The (model_id, gguf_mesh quant) pair must be an approved scoring
       profile; the validator refuses canaries for anything else.
    3. The mesh eligibility gate must pass (non-zero digests, empty layer
       roots, sane layer count and context).
    4. Every digest is taken from the ModelSpec, never from operator input.

    ``max_context_len`` may be None during preflight: the registered value
    is the launch's MEASURED KV auto-fit, which does not exist before a
    launch. Eligibility is then validated with a placeholder and the real
    context is validated again by register_mesh_endpoint before the write.
    """
    spec = _model_registry_client(chain_config).get_model_spec(model_id)
    if spec is None:
        raise LifecycleRefusal(
            f"model {model_id!r} is absent from ModelRegistry; register the "
            "ModelSpec first (scripts/register_model_onchain.py)"
        )
    quant_mode = str(getattr(spec, "quant_mode", "") or "").strip().lower()
    if not quant_mode.startswith("gguf_") or len(quant_mode) <= len("gguf_"):
        raise LifecycleRefusal(
            f"ModelRegistry quant_mode {quant_mode!r} is not a GGUF scheme; "
            "this model cannot serve on the mesh path"
        )
    gguf_scheme = quant_mode[len("gguf_") :]
    registry_quant = f"{MESH_QUANT_PREFIX}_{gguf_scheme}"

    from neurons.runtime import get_mesh_model_scoring_profile

    if get_mesh_model_scoring_profile(model_id, registry_quant) is None:
        raise LifecycleRefusal(
            f"({model_id!r}, {registry_quant!r}) is not an approved mesh "
            "scoring profile; the validator would refuse every canary"
        )

    validation_context = int(max_context_len) if max_context_len else 1
    target = target_from_values(
        model_id=model_id,
        endpoint="",
        quant=registry_quant,
        max_context_len=validation_context,
    )
    validate_mesh_model_registry_eligibility(target, spec)

    return MeshChainAnchors(
        model_id=model_id,
        gguf_scheme=gguf_scheme,
        registry_quant=registry_quant,
        chain_quantization_scheme=quant_mode,
        model_package_hash=bytes(spec.weight_file_hash).hex(),
        model_tensor_manifest_root=bytes(spec.weight_merkle_root).hex(),
        tokenizer_hash=bytes(spec.tokenizer_hash).hex(),
        total_layers=int(spec.num_layers),
        max_context_len=int(max_context_len or 0),
    )


def build_registration_target(
    *,
    anchors: MeshChainAnchors,
    endpoint: str,
    max_context_len: int | None = None,
) -> RegistrationTarget:
    context = int(max_context_len or anchors.max_context_len or 0)
    if context <= 0:
        raise LifecycleRefusal(
            "registration context is unresolved; launch the mesh first so "
            "the KV auto-fit measures it"
        )
    return target_from_values(
        model_id=anchors.model_id,
        endpoint=endpoint,
        quant=anchors.registry_quant,
        max_context_len=context,
    )


def substrate_endpoint(subtensor_network: str) -> str:
    """Normalize a node URL for the Substrate (websocket) client.

    Modern subtensor exposes Substrate (WS) and EVM (HTTP) on the same
    host:port, and operators pass ONE node URL in either form. The EVM
    side translates ws[s]:// to http[s]:// (ChainConfig.resolve_rpc_url);
    this is the mirror: http[s]:// becomes ws[s]:// for bittensor, which
    only understands network names and ws endpoints (an http URL sent to
    bt.Subtensor fails during DNS resolution).
    Network names ("test", "finney") and ws URLs pass through unchanged.
    """
    value = str(subtensor_network or "").strip()
    if value.startswith("http://"):
        return "ws://" + value[len("http://"):]
    if value.startswith("https://"):
        return "wss://" + value[len("https://"):]
    return value


def resolve_uid_for_hotkey(
    subtensor_network: str,
    netuid: int,
    hotkey_ss58: str,
) -> int:
    """Resolve the UID a hotkey is registered under on a subnet, or refuse.

    The seamless testnet/mainnet check: before any chain write, prove the
    coordinator hotkey actually holds a UID on the target netuid of the
    target network. Reads the Substrate metagraph the same way the vLLM
    miner does at startup (neurons/miner.py).
    """
    import bittensor as bt

    try:
        subtensor = bt.Subtensor(network=substrate_endpoint(subtensor_network))
        metagraph = subtensor.metagraph(int(netuid))
        hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
    except Exception as exc:
        raise LifecycleRefusal(
            f"could not read the {subtensor_network} metagraph for netuid "
            f"{netuid}: {exc}"
        ) from exc
    if hotkey_ss58 not in hotkeys:
        raise LifecycleRefusal(
            f"hotkey {hotkey_ss58} is not registered on netuid {netuid} of "
            f"{subtensor_network!r}; register the hotkey on the subnet "
            "before deploying (btcli subnet register)"
        )
    return int(hotkeys.index(hotkey_ss58))


def _plan_mesh_registration_from_entries(
    entries,
    target: RegistrationTarget,
    previous_registration: Mapping[str, Any] | None = None,
) -> LifecyclePlan:
    """Plan a first deploy or a pool-owned endpoint replacement.

    ``registerModel`` identifies a slot by ``(model, endpoint, quant)``.  A
    replacement host necessarily changes the endpoint, so the generic planner
    quite correctly calls it an append.  A pool manager, however, persists the
    exact registration it previously committed.  When that stored full tuple
    still matches the entry at its recorded index, it is an unambiguous
    replacement intent and ``updateEndpoint`` can preserve the index and its
    score history.

    The stored record is never trusted on its own: a missing, moved, or changed
    chain tuple fails closed.  A second pool run by the same operator has no
    matching local record and therefore retains the ordinary append behavior.
    """

    ordinary = plan_registration(entries, target)
    if previous_registration is None:
        return ordinary

    try:
        previous = registration_target_from_state(previous_registration)
        previous_index = int(previous_registration["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleRefusal(
            "stored pool registration is malformed; refusing endpoint "
            "replacement"
        ) from exc

    if previous.model_id != target.model_id or previous.quant != target.quant:
        raise LifecycleRefusal(
            "stored pool registration does not identify the requested "
            "model and quantization; refusing endpoint replacement"
        )
    if previous_index < 0 or previous_index >= len(entries):
        raise LifecycleRefusal(
            f"stored pool registration index {previous_index} is outside "
            "the current MinerRegistry entries; refusing endpoint replacement"
        )

    entry = entries[previous_index]
    after_endpoint_update = RegistrationTarget(
        model_id=previous.model_id,
        endpoint=target.endpoint,
        model_spec_ref=previous.model_spec_ref,
        quant=previous.quant,
        max_context_len=previous.max_context_len,
    )
    if configuration_matches(entry, target) or configuration_matches(
        entry, after_endpoint_update
    ):
        # Recovery after updateEndpoint, or its following metadata refresh,
        # committed but deploy did not persist the new registration state.
        # Continue only when the ordinary planner resolves the requested tuple
        # at the same index.
        if ordinary.predicted_index != previous_index:
            raise LifecycleRefusal(
                f"requested endpoint resolves at index "
                f"{ordinary.predicted_index}, not the pool's stored index "
                f"{previous_index}; refusing ambiguous replacement recovery"
            )
        return ordinary

    if not configuration_matches(entry, previous):
        raise LifecycleRefusal(
            f"MinerRegistry entry {previous_index} no longer matches the "
            "pool's stored model, endpoint, quant, ModelSpec reference, and "
            "context; refusing to update an ambiguous slot"
        )
    if ordinary.action != "append":
        raise LifecycleRefusal(
            f"requested endpoint already resolves at index "
            f"{ordinary.predicted_index}, not the pool's stored index "
            f"{previous_index}; refusing to move this pool onto another slot"
        )

    metadata_changed = (
        bytes(previous.model_spec_ref) != bytes(target.model_spec_ref)
        or int(previous.max_context_len) != int(target.max_context_len)
        or not is_live(entry, int(time.time()))
    )
    return LifecyclePlan(
        action=(
            "update-endpoint-refresh"
            if metadata_changed
            else "update-endpoint"
        ),
        predicted_index=previous_index,
        reason=(
            f"pool registration proves index {previous_index} owns the prior "
            "full tuple; updateEndpoint preserves that slot"
            + (
                " and registerModel refreshes its metadata/lease"
                if metadata_changed
                else ""
            )
        ),
    )


def plan_mesh_registration(
    chain_config,
    target: RegistrationTarget,
    signer_address: str,
    *,
    previous_registration: Mapping[str, Any] | None = None,
) -> LifecyclePlan:
    """Read-only preflight; the predicted index is what launch binds to."""
    entries = _miner_registry_client(chain_config).get_miner_models(
        signer_address
    )
    return _plan_mesh_registration_from_entries(
        entries,
        target,
        previous_registration,
    )


def register_mesh_endpoint(
    chain_config,
    target: RegistrationTarget,
    *,
    private_key: str,
    expected_index: int | None = None,
    previous_registration: Mapping[str, Any] | None = None,
) -> RegistrationOutcome:
    """Send registerModel and verify the resulting slot.

    The eligibility gate runs again immediately before the write (chain
    state may have changed since preflight). When ``expected_index`` is
    given and the post-transaction index differs, this raises: the running
    mesh's signed snapshots carry the predicted index and are now wrong,
    so the caller must stop the mesh rather than serve a mismatched
    registration.
    """
    from eth_account import Account

    spec = _model_registry_client(chain_config).get_model_spec(target.model_id)
    validate_mesh_model_registry_eligibility(target, spec)

    signer_address = Account.from_key(private_key).address
    client = _miner_registry_client(chain_config)
    plan = _plan_mesh_registration_from_entries(
        client.get_miner_models(signer_address),
        target,
        previous_registration,
    )
    endpoint_update_tx = ""
    if plan.action in ("update-endpoint", "update-endpoint-refresh"):
        try:
            endpoint_update_tx = client.update_endpoint(
                plan.predicted_index,
                target.endpoint,
                private_key=private_key,
            )
        except Exception as exc:
            raise LifecycleRefusal(
                f"updateEndpoint for index {plan.predicted_index} failed or "
                "its transaction outcome is unknown; stop serving and "
                "re-read the registry before retrying"
            ) from exc
        # Re-read from a fresh client.  The normal lifecycle planner can now
        # see the target endpoint and safely completes any metadata refresh or
        # interrupted replacement reactivation at the same index.
        try:
            client = _miner_registry_client(chain_config)
            entries = client.get_miner_models(signer_address)
        except Exception as exc:
            raise LifecycleRefusal(
                f"updateEndpoint was submitted for index "
                f"{plan.predicted_index}, but fresh registry verification "
                "failed; transaction outcome is unverified, so stop serving "
                "and re-read the registry before retrying"
            ) from exc
        followup = plan_registration(entries, target)
        if (
            followup.predicted_index != plan.predicted_index
            or followup.action == "append"
        ):
            raise LifecycleRefusal(
                f"updateEndpoint sent for index {plan.predicted_index} but "
                "fresh registry state did not resolve the requested tuple at "
                "that index; stop serving and inspect the transaction before "
                "retrying"
            )
        replacement_action = "update-endpoint"
        plan = followup
    else:
        replacement_action = ""

    if plan.action == "reuse-active":
        index = plan.predicted_index
        entry = client.get_miner_models(signer_address)[index]
        outcome = RegistrationOutcome(
            action=replacement_action or plan.action,
            index=index,
            tx_hash=endpoint_update_tx,
            expires_at=int(entry.expires_at),
        )
    else:
        if plan.action == "refresh":
            # An active slot whose contract changed (re-measured context,
            # re-anchored spec): registerModel reverts on an active
            # duplicate but reactivates a deactivated matching tuple IN
            # PLACE, so deactivating first updates the slot while keeping
            # its index and score history.
            client.deactivate_model(
                plan.predicted_index, private_key=private_key
            )
        try:
            tx_hash = client.register_model(
                target.model_id,
                target.endpoint,
                target.model_spec_ref,
                target.quant,
                target.max_context_len,
                private_key=private_key,
            )
        except Exception as exc:
            if plan.action == "refresh":
                # Non-atomic by contract design: the slot is now
                # DEACTIVATED and validators drop it until re-registered.
                # Re-running deploy/registration reactivates the same
                # slot in place (idempotent) - say so, or the operator
                # has a dead index and no recovery instruction.
                raise LifecycleRefusal(
                    f"slot {plan.predicted_index} was deactivated for a "
                    "contract refresh but the re-registration failed "
                    f"({exc}); the model is OFF CHAIN until re-registered. "
                    "Re-run the deploy/registration - registerModel "
                    "reactivates the same slot in place."
                ) from exc
            raise
        # Fresh client: bypass the read cache so the verification sees the
        # post-transaction registry state.
        fresh = _miner_registry_client(chain_config)
        entries = fresh.get_miner_models(signer_address)
        index = resolve_registered_index(entries, target)
        outcome = RegistrationOutcome(
            action=(
                f"{replacement_action}+{plan.action}"
                if replacement_action
                else plan.action
            ),
            index=index,
            tx_hash=tx_hash,
            expires_at=int(entries[index].expires_at),
        )
    if expected_index is not None and outcome.index != expected_index:
        raise LifecycleRefusal(
            f"registered at index {outcome.index} but the running mesh was "
            f"launched against index {expected_index}; its signed snapshots "
            "are wrong. Stop the mesh and re-run deploy."
        )
    return outcome


def renew_once(
    chain_config,
    target: RegistrationTarget,
    index: int,
    *,
    private_key: str,
) -> RegistrationOutcome:
    """Renew one lease with the full-tuple guard; idempotent and timer-safe.

    Refuses when the ModelSpec no longer passes the mesh eligibility gate:
    a removed or altered spec should let the lease lapse instead of keeping
    an unverifiable mesh registered.
    """
    from eth_account import Account

    spec = _model_registry_client(chain_config).get_model_spec(target.model_id)
    validate_mesh_model_registry_eligibility(target, spec)

    signer_address = Account.from_key(private_key).address
    client = _miner_registry_client(chain_config)
    entries = client.get_miner_models(signer_address)
    plan_renewal(entries, target, index)
    tx_hash = client.renew_model(index, private_key=private_key)
    fresh = _miner_registry_client(chain_config)
    entries = fresh.get_miner_models(signer_address)
    verified_index = resolve_registered_index(entries, target)
    if verified_index != index:
        raise LifecycleRefusal(
            f"renewal verified at index {verified_index}, expected {index}"
        )
    return RegistrationOutcome(
        action="renew",
        index=index,
        tx_hash=tx_hash,
        expires_at=int(entries[index].expires_at),
    )


def ensure_evm_registered(
    chain_config,
    *,
    uid: int,
    hotkey_seed: bytes,
    netuid: int,
    private_key: str,
) -> bool:
    """Reconcile the EVM-to-UID binding before any registerModel.

    Mirrors the vLLM miner's reconciliation (read, repair, read back)
    without importing neurons.miner, which drags in vLLM. Returns True when
    a registerEvm transaction was sent.
    """
    from eth_account import Account

    evm_address = Account.from_key(private_key).address
    client = _miner_registry_client(chain_config)

    def _binding_ok() -> bool:
        associated = client.get_associated_uid(evm_address, refresh=True)
        registered = client.get_registered_uid_for_evm(evm_address, refresh=True)
        evm_for_uid = client.get_registered_evm_for_uid(uid, refresh=True)
        return (
            associated == uid
            and registered == uid
            and bool(evm_for_uid)
            and evm_for_uid.lower() == evm_address.lower()
        )

    if _binding_ok():
        return False
    client.register_evm(
        uid, hotkey_seed=hotkey_seed, netuid=netuid, private_key=private_key
    )
    if not _binding_ok():
        raise LifecycleRefusal(
            "registerEvm succeeded but the read-back binding does not show "
            f"UID {uid} bound to {evm_address}"
        )
    return True


def save_mesh_registration_state(
    pool_dir: Path,
    *,
    target: RegistrationTarget,
    index: int,
    mesh_key: str,
    expires_at: int,
) -> Path:
    """Persist what the pool manager's lease renewer needs, owner-only."""
    from verallm.mesh.private_files import write_owner_only_json

    return write_owner_only_json(
        Path(pool_dir) / MESH_REGISTRATION_STATE_FILE,
        {
            "version": 1,
            "model_id": target.model_id,
            "endpoint": target.endpoint,
            "quant": target.quant,
            "max_context_len": target.max_context_len,
            "model_spec_ref": bytes(target.model_spec_ref).hex(),
            "index": int(index),
            "mesh_key": mesh_key,
            "expires_at": int(expires_at),
        },
    )


def load_mesh_registration_state(pool_dir: Path) -> dict[str, Any] | None:
    from verallm.mesh.private_files import read_owner_only_text

    path = Path(pool_dir) / MESH_REGISTRATION_STATE_FILE
    if not path.exists():
        return None
    state = json.loads(read_owner_only_text(path, label="mesh registration state"))
    if not isinstance(state, Mapping):
        raise ValueError("mesh registration state must be a JSON object")
    return dict(state)


def registration_target_from_state(state: Mapping[str, Any]) -> RegistrationTarget:
    return RegistrationTarget(
        model_id=str(state["model_id"]),
        endpoint=str(state["endpoint"]),
        model_spec_ref=bytes.fromhex(str(state["model_spec_ref"])),
        quant=str(state["quant"]),
        max_context_len=int(state["max_context_len"]),
    )


# Renew when less than this much lease remains; the lease itself is 24h.
LEASE_RENEW_WINDOW_S = 12 * 3600
LEASE_RENEWER_INTERVAL_S = 15 * 60

# One "letting the lease lapse" line per (model, mesh, status) transition:
# a permanently-retired model would otherwise repeat the identical warning
# every tick forever, burying the manager log's real signal. In-memory by
# design — a manager restart re-logs each lapse once, which is the useful
# reminder, not spam.
_LAPSE_LOGGED: dict[str, str] = {}


def renew_lease_if_due(
    manager,
    *,
    chain_config,
    private_key: str,
    now: float | None = None,
) -> RegistrationOutcome | None:
    """One renewer tick against a PoolManager's stored registrations.

    A pool holds one registration PER MODEL (glm on one machine, qwen on
    another); every due lease is renewed in the same tick. Driven by each
    persisted on-chain ``expires_at`` (refreshed after every renewal), not
    a wall-clock timer, so a manager restart cannot skip a window.
    Deliberately lets a lease lapse when:
    - the bound mesh is no longer serving (a dead mesh must not stay
      registered past its lease), or
    - more than LEASE_RENEW_WINDOW_S remains.
    Raises LifecycleRefusal when a ModelSpec no longer passes the mesh
    eligibility gate, which also lets that lease lapse.

    Returns the LAST renewed outcome (or None when nothing was due), which
    keeps the single-registration callers and tests unchanged.
    """
    current_time = time.time() if now is None else now
    with manager.lock:
        registrations = {
            model_id: dict(entry)
            for model_id, entry in manager._registrations_locked().items()
        }
        all_meshes = {
            key: dict(mesh)
            for key, mesh in (manager.state.get("meshes") or {}).items()
        }
    meshes = {
        model_id: dict(
            all_meshes.get(str(entry.get("mesh_key", "")), {}) or {}
        )
        for model_id, entry in registrations.items()
    }
    last_outcome: RegistrationOutcome | None = None
    for model_id, state in registrations.items():
        mesh = meshes.get(model_id) or {}
        if mesh.get("status") != "serving":
            # A relaunch mints a NEW mesh_key while the stored registration
            # keeps the old one, so the stale-key lookup alone reads a
            # healthy model as "gone" and lets a live chain entry expire.
            # The registration's contract is the
            # (model, index) chain slot, not one mesh incarnation: adopt a
            # serving mesh that carries the SAME model at the SAME confirmed
            # index, and heal the stored key. A model with no such mesh
            # still lapses, exactly as the dead-mesh rule intends.
            adopted_key = ""
            for key, candidate in all_meshes.items():
                if (
                    str(candidate.get("status", "")) == "serving"
                    and str(candidate.get("model_id", "")) == model_id
                    and candidate.get("model_index") is not None
                    and int(candidate["model_index"])
                    == int(state.get("index", -1))
                ):
                    adopted_key = key
                    mesh = dict(candidate)
                    break
            if not adopted_key:
                lapse_fingerprint = (
                    f"{state.get('mesh_key')}:{mesh.get('status', 'gone')}"
                )
                if _LAPSE_LOGGED.get(model_id) != lapse_fingerprint:
                    _LAPSE_LOGGED[model_id] = lapse_fingerprint
                    logger.warning(
                        "lease renewer: %s mesh %s is %r, letting the lease "
                        "lapse (repeats of this exact state are not "
                        "re-logged)",
                        model_id,
                        state.get("mesh_key"),
                        mesh.get("status", "gone"),
                    )
                continue
            logger.info(
                "lease renewer: %s relaunched as mesh %s (stored key %s); "
                "adopting it for renewal",
                model_id,
                adopted_key,
                state.get("mesh_key"),
            )
            with manager.lock:
                stored = manager._registrations_locked().get(model_id)
                if isinstance(stored, dict):
                    stored["mesh_key"] = adopted_key
                    manager._save()
        # Reaching here means the model has a serving mesh again (directly
        # or adopted): a future lapse is a NEW event worth logging.
        _LAPSE_LOGGED.pop(model_id, None)
        if (
            int(state.get("expires_at", 0)) - current_time
            > LEASE_RENEW_WINDOW_S
        ):
            continue
        target = registration_target_from_state(state)
        outcome = renew_once(
            chain_config, target, int(state["index"]), private_key=private_key
        )
        with manager.lock:
            stored = manager._registrations_locked().get(model_id)
            if isinstance(stored, dict):
                stored["expires_at"] = outcome.expires_at
                manager._save()
        logger.info(
            "lease renewed for %s at index %d until %d (tx %s)",
            model_id,
            outcome.index,
            outcome.expires_at,
            outcome.tx_hash,
        )
        last_outcome = outcome
    return last_outcome


def run_lease_renewer(
    manager,
    *,
    chain_config,
    private_key: str,
    interval_s: float = LEASE_RENEWER_INTERVAL_S,
    stop_event=None,
) -> None:
    """Daemon loop for the pool manager process. Errors are logged, never
    fatal: a transient RPC failure must not kill the manager, and the next
    tick retries well inside the renew window."""
    import threading

    waiter = stop_event if stop_event is not None else threading.Event()
    while not waiter.is_set():
        try:
            renew_lease_if_due(
                manager, chain_config=chain_config, private_key=private_key
            )
        except LifecycleRefusal as exc:
            logger.error("lease renewal refused (letting it lapse): %s", exc)
        except Exception as exc:
            logger.error("lease renewal tick failed (will retry): %s", exc)
        waiter.wait(interval_s)
