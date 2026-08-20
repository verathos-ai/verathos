"""Signed GPU roster for mesh capacity audits.

The roster is the document that pins a mesh chain entry to its physical
GPUs: these tests pin its canonical form, the coordinator-EVM signature,
the global ordinal assignment, and the validation bounds a hostile roster
must not slip past.
"""

from __future__ import annotations

import pytest
from eth_account import Account

from verallm.mesh.capacity_roster import (
    BACKEND_CUDA,
    BACKEND_METAL,
    BACKEND_OTHER,
    build_roster,
    canonical_roster_json,
    host_group_hints,
    normalize_backend,
    recover_roster_signer,
    roster_cuda_gpus,
    roster_cuda_vram_total_gb,
    roster_digest,
    sign_roster,
    validate_roster,
    verify_roster_signature,
)

ACCT = Account.from_key(b"\x21" * 32)
ADDRESS = ACCT.address.lower()


def _workers() -> list[dict]:
    return [
        {
            "worker_id": "w-bbb",
            "backend": "CUDA0",
            "gpu_names": ["NVIDIA A100 80GB PCIe"] * 4,
            "per_gpu_vram_gb": [80] * 4,
            "host": "http://203.0.113.137:20002",
        },
        {
            "worker_id": "w-aaa",
            "backend": "cuda",
            "gpu_names": ["NVIDIA GeForce RTX 4090"],
            "per_gpu_vram_gb": [24],
            "host": "miner.example.com:9000",
        },
        {
            "worker_id": "w-mac",
            "backend": "metal",
            "gpu_names": [],
            "per_gpu_vram_gb": [],
            "host": "",
        },
    ]


def _roster() -> dict:
    return build_roster(
        chain_id=945,
        netuid=405,
        address=ADDRESS,
        model_index=45,
        roster_epoch=21550,
        workers=_workers(),
    )


# ---------------------------------------------------------------------------
# Canonical form + ordinals
# ---------------------------------------------------------------------------


def test_build_roster_is_canonical_and_deterministic():
    a = _roster()
    b = build_roster(
        chain_id=945,
        netuid=405,
        address=ADDRESS,
        model_index=45,
        roster_epoch=21550,
        # Same content, different input ordering: workers sort by id.
        workers=list(reversed(_workers())),
    )
    assert canonical_roster_json(a) == canonical_roster_json(b)
    assert roster_digest(a) == roster_digest(b)
    assert [w["worker_id"] for w in a["workers"]] == ["w-aaa", "w-bbb", "w-mac"]


def test_ordinals_sorted_by_worker_then_local_index_cuda_only():
    gpus = roster_cuda_gpus(_roster())
    assert [gpu.ordinal for gpu in gpus] == [0, 1, 2, 3, 4]
    # w-aaa sorts before w-bbb, so the 4090 takes ordinal 0.
    assert gpus[0].worker_id == "w-aaa" and gpus[0].local_gpu_index == 0
    assert [gpu.worker_id for gpu in gpus[1:]] == ["w-bbb"] * 4
    assert [gpu.local_gpu_index for gpu in gpus[1:]] == [0, 1, 2, 3]
    # The Metal worker contributes no ordinals.
    assert all(gpu.worker_id != "w-mac" for gpu in gpus)
    assert roster_cuda_vram_total_gb(_roster()) == 24 + 4 * 80


def test_all_metal_roster_has_no_ordinals():
    roster = build_roster(
        chain_id=945,
        netuid=405,
        address=ADDRESS,
        model_index=1,
        roster_epoch=5,
        workers=[
            {
                "worker_id": "w-mac",
                "backend": "mps",
                "gpu_names": ["Apple M1 Max"],
                "per_gpu_vram_gb": [64],
                "host": "",
            }
        ],
    )
    assert roster_cuda_gpus(roster) == []
    assert roster_cuda_vram_total_gb(roster) == 0


def test_backend_normalization():
    assert normalize_backend("CUDA0") == BACKEND_CUDA
    assert normalize_backend("cuda:1") == BACKEND_CUDA
    assert normalize_backend("Metal") == BACKEND_METAL
    assert normalize_backend("mps") == BACKEND_METAL
    assert normalize_backend("rocm") == BACKEND_OTHER
    assert normalize_backend("") == BACKEND_OTHER


def test_host_group_hints_ip24_and_regdom():
    assert host_group_hints("http://203.0.113.137:20002") == ("203.0.113", "")
    assert host_group_hints("203.0.113.9") == ("203.0.113", "")
    assert host_group_hints("miner.example.com:9000") == ("", "example.com")
    assert host_group_hints("https://a.b.example.com") == ("", "example.com")
    assert host_group_hints("") == ("", "")
    assert host_group_hints("localhost") == ("", "")
    # The raw host never enters the roster; only these hints do.
    roster = _roster()
    for row in roster["workers"]:
        assert "host" not in row


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


def test_sign_verify_recover_round_trip():
    roster = _roster()
    signature = sign_roster(roster, ACCT.key)
    assert verify_roster_signature(roster, signature, ACCT.address)
    assert recover_roster_signer(roster, signature).lower() == ADDRESS


def test_signature_rejects_wrong_address_and_tampering():
    roster = _roster()
    signature = sign_roster(roster, ACCT.key)
    assert not verify_roster_signature(roster, signature, "0x" + "00" * 20)
    tampered = dict(roster)
    tampered["roster_epoch"] = roster["roster_epoch"] + 1
    assert not verify_roster_signature(tampered, signature, ACCT.address)
    shrunk = dict(roster)
    shrunk["workers"] = roster["workers"][:1]
    assert not verify_roster_signature(shrunk, signature, ACCT.address)
    assert not verify_roster_signature(roster, "", ACCT.address)


# ---------------------------------------------------------------------------
# Validation bounds
# ---------------------------------------------------------------------------


def _mutated(roster: dict, **top) -> dict:
    out = dict(roster)
    out.update(top)
    return out


def test_validate_rejects_malformed_rosters():
    roster = _roster()
    validate_roster(roster)

    with pytest.raises(ValueError):
        validate_roster(_mutated(roster, version=2))
    with pytest.raises(ValueError):
        validate_roster(_mutated(roster, workers=[]))
    with pytest.raises(ValueError):
        validate_roster(_mutated(roster, roster_epoch=-1))
    with pytest.raises(ValueError):
        validate_roster(_mutated(roster, slot={"chain_id": 945}))

    checksummed = dict(roster)
    checksummed["slot"] = dict(roster["slot"], address=ACCT.address)
    if ACCT.address != ADDRESS:
        with pytest.raises(ValueError):
            validate_roster(checksummed)

    duplicate = dict(roster)
    duplicate["workers"] = roster["workers"] + [roster["workers"][0]]
    with pytest.raises(ValueError):
        validate_roster(duplicate)

    misaligned = dict(roster)
    row = dict(roster["workers"][0])
    row["per_gpu_vram_gb"] = row["per_gpu_vram_gb"] + [80]
    misaligned["workers"] = [row] + roster["workers"][1:]
    with pytest.raises(ValueError):
        validate_roster(misaligned)

    gpuless_cuda = dict(roster)
    row = dict(roster["workers"][0])
    row["gpu_names"] = []
    row["per_gpu_vram_gb"] = []
    gpuless_cuda["workers"] = [row] + roster["workers"][1:]
    with pytest.raises(ValueError):
        validate_roster(gpuless_cuda)

    raw_backend = dict(roster)
    row = dict(roster["workers"][0])
    row["backend"] = "CUDA0"
    raw_backend["workers"] = [row] + roster["workers"][1:]
    with pytest.raises(ValueError):
        validate_roster(raw_backend)


def test_validate_bounds_worker_and_gpu_counts():
    many_workers = [
        {
            "worker_id": f"w-{index:03d}",
            "backend": "cuda",
            "gpu_names": ["NVIDIA GeForce RTX 4090"],
            "per_gpu_vram_gb": [24],
            "host": "",
        }
        for index in range(65)
    ]
    with pytest.raises(ValueError):
        build_roster(
            chain_id=945,
            netuid=405,
            address=ADDRESS,
            model_index=0,
            roster_epoch=1,
            workers=many_workers,
        )
    with pytest.raises(ValueError):
        build_roster(
            chain_id=945,
            netuid=405,
            address=ADDRESS,
            model_index=0,
            roster_epoch=1,
            workers=[
                {
                    "worker_id": "w-many",
                    "backend": "cuda",
                    "gpu_names": ["NVIDIA GeForce RTX 4090"] * 17,
                    "per_gpu_vram_gb": [24] * 17,
                    "host": "",
                }
            ],
        )
