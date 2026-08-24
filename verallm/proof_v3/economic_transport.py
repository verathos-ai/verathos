"""Bounded compressed transport for economic proof-v3 hard audits.

The cryptographic transcript and canonical parser continue to operate on the
uncompressed ``V3EW`` proof bytes.  This outer envelope only reduces network
cost.  Its declared lengths, codec, digest, decompression output, and trailing
data are all checked before the canonical proof parser runs.
"""

from __future__ import annotations

import hashlib
import struct

import zstandard

from verallm.proof_v3.economic_wire import (
    MAX_ECONOMIC_WIRE_BYTES,
    EconomicRecomputeProofV3,
)
from verallm.proof_v3.errors import ProofV3Error

ECONOMIC_TRANSPORT_VERSION = 2
# Bounded operational envelopes for one hard-audit response. These are
# transport/resource limits, not cryptographic parameters. Dense proofs retain
# the established 32 MiB boundary byte-for-byte. Sparse-MoE proofs carry a
# distinct authenticated transport flag because complete selected gate/up
# matrices are intentionally incompressible at production Qwen3.6 geometry.
# Their 96 MiB boundary remains below the canonical 128 MiB wire limit while
# covering the measured 65.25 MiB Qwen3.6-35B proof without reducing audit
# coverage.
MAX_ECONOMIC_TRANSPORT_BYTES = 32 << 20
MAX_ECONOMIC_MOE_TRANSPORT_BYTES = 96 << 20

_MAGIC = b"V3EZ"
_CODEC_ZSTD = 2
_FLAGS_DENSE = 0
_FLAGS_SPARSE_MOE = 1
_HEADER = struct.Struct("<4sHBBII32s")
_COMPRESSION_LEVEL = 9
_COMPRESSION_WINDOW_LOG = 24

__all__ = [
    "ECONOMIC_TRANSPORT_VERSION",
    "MAX_ECONOMIC_TRANSPORT_BYTES",
    "MAX_ECONOMIC_MOE_TRANSPORT_BYTES",
    "encode_economic_proof_transport_v3",
    "decode_economic_proof_transport_v3",
    "economic_proof_transport_maximum_bytes_v3",
]


def economic_proof_transport_maximum_bytes_v3(
    encoded: bytes,
    *,
    allow_sparse_moe: bool = False,
) -> int:
    """Return the canonical bound selected by an encoded transport header."""

    if not isinstance(allow_sparse_moe, bool):
        raise ProofV3Error("sparse-MoE transport permission must be boolean")
    if not isinstance(encoded, bytes) or len(encoded) < _HEADER.size:
        raise ProofV3Error("economic proof transport is malformed")
    try:
        magic, version, codec, flags, _raw, _compressed, _digest = (
            _HEADER.unpack_from(encoded)
        )
    except struct.error as exc:
        raise ProofV3Error("economic proof transport header is malformed") from exc
    if (
        magic != _MAGIC
        or version != ECONOMIC_TRANSPORT_VERSION
        or codec != _CODEC_ZSTD
        or flags not in {_FLAGS_DENSE, _FLAGS_SPARSE_MOE}
    ):
        raise ProofV3Error("economic proof transport header is not supported")
    if flags == _FLAGS_SPARSE_MOE:
        if not allow_sparse_moe:
            raise ProofV3Error("sparse-MoE proof transport is not permitted")
        return MAX_ECONOMIC_MOE_TRANSPORT_BYTES
    return MAX_ECONOMIC_TRANSPORT_BYTES


def encode_economic_proof_transport_v3(
    proof: EconomicRecomputeProofV3,
) -> bytes:
    """Encode one canonical proof into the bounded network envelope."""

    if not isinstance(proof, EconomicRecomputeProofV3):
        raise ProofV3Error("economic transport proof has an unexpected type")
    raw = proof.canonical_bytes()
    compressed = _compress_canonical_proof_v3(raw)
    flags = _FLAGS_SPARSE_MOE if proof.moe_wire else _FLAGS_DENSE
    maximum = (
        MAX_ECONOMIC_MOE_TRANSPORT_BYTES
        if flags == _FLAGS_SPARSE_MOE
        else MAX_ECONOMIC_TRANSPORT_BYTES
    )
    encoded = _HEADER.pack(
        _MAGIC,
        ECONOMIC_TRANSPORT_VERSION,
        _CODEC_ZSTD,
        flags,
        len(raw),
        len(compressed),
        hashlib.sha256(raw).digest(),
    ) + compressed
    if len(encoded) > maximum:
        raise ProofV3Error(
            "compressed economic proof exceeds the transport byte limit "
            f"({len(encoded)} > {maximum})"
        )
    return encoded


def _fallback_compressor_v3():
    return zstandard.ZstdCompressor(
        level=_COMPRESSION_LEVEL,
        write_content_size=True,
        write_checksum=True,
    )


def _long_distance_compressor_v3():
    """Return the optional bounded-window compressor when supported."""

    parameters_type = getattr(
        zstandard,
        "ZstdCompressionParameters",
        None,
    )
    from_level = getattr(parameters_type, "from_level", None)
    if not callable(from_level):
        return None
    try:
        parameters = from_level(
            _COMPRESSION_LEVEL,
            enable_ldm=True,
            window_log=_COMPRESSION_WINDOW_LOG,
            write_content_size=True,
            write_checksum=True,
        )
        if (
            not bool(getattr(parameters, "enable_ldm", False))
            or int(getattr(parameters, "window_log", -1))
            != _COMPRESSION_WINDOW_LOG
            or not bool(getattr(parameters, "write_content_size", False))
            or not bool(getattr(parameters, "write_checksum", False))
        ):
            return None
        return zstandard.ZstdCompressor(compression_params=parameters)
    except (AttributeError, TypeError, ValueError, zstandard.ZstdError):
        return None


def _compress_canonical_proof_v3(raw: bytes) -> bytes:
    compressor = _long_distance_compressor_v3()
    if compressor is not None:
        try:
            return compressor.compress(raw)
        except zstandard.ZstdError:
            pass
    return _fallback_compressor_v3().compress(raw)


def decode_economic_proof_transport_v3(
    encoded: bytes,
    *,
    allow_sparse_moe: bool = False,
) -> EconomicRecomputeProofV3:
    """Boundedly decompress and canonical-parse one network proof."""

    if not isinstance(allow_sparse_moe, bool):
        raise ProofV3Error("sparse-MoE transport permission must be boolean")
    if (
        not isinstance(encoded, bytes)
        or len(encoded) < _HEADER.size
        or len(encoded) > MAX_ECONOMIC_MOE_TRANSPORT_BYTES
    ):
        raise ProofV3Error("economic proof transport is malformed")
    maximum = economic_proof_transport_maximum_bytes_v3(
        encoded,
        allow_sparse_moe=allow_sparse_moe,
    )
    try:
        (
            magic,
            version,
            codec,
            flags,
            raw_length,
            compressed_length,
            raw_digest,
        ) = _HEADER.unpack_from(encoded)
    except struct.error as exc:
        raise ProofV3Error("economic proof transport header is malformed") from exc
    if (
        magic != _MAGIC
        or version != ECONOMIC_TRANSPORT_VERSION
        or codec != _CODEC_ZSTD
        or flags not in {_FLAGS_DENSE, _FLAGS_SPARSE_MOE}
    ):
        raise ProofV3Error("economic proof transport header is not supported")
    sparse_moe = flags == _FLAGS_SPARSE_MOE
    if len(encoded) > maximum:
        raise ProofV3Error("economic proof transport exceeds the byte limit")
    if not 0 < raw_length <= MAX_ECONOMIC_WIRE_BYTES:
        raise ProofV3Error("economic proof transport raw length is out of range")
    if compressed_length != len(encoded) - _HEADER.size:
        raise ProofV3Error(
            "economic proof transport compressed length is inconsistent"
        )
    if compressed_length < 1:
        raise ProofV3Error("economic proof transport payload is empty")

    payload = encoded[_HEADER.size:]
    try:
        if zstandard.frame_content_size(payload) != raw_length:
            raise ProofV3Error(
                "economic proof transport frame length is inconsistent"
            )
        raw = zstandard.ZstdDecompressor().decompress(
            payload,
            max_output_size=raw_length,
            allow_extra_data=False,
        )
    except zstandard.ZstdError as exc:
        raise ProofV3Error("economic proof transport decompression failed") from exc
    if len(raw) != raw_length:
        raise ProofV3Error(
            "economic proof transport decompressed length is inconsistent"
        )
    if hashlib.sha256(raw).digest() != raw_digest:
        raise ProofV3Error("economic proof transport digest does not match")
    proof = EconomicRecomputeProofV3.from_canonical_bytes(raw)
    if bool(proof.moe_wire) != sparse_moe:
        raise ProofV3Error(
            "economic proof transport sparse-MoE flag is inconsistent"
        )
    return proof
