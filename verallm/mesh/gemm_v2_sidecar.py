"""PCS-bound GEMM v2 sidecar for mesh proofs (Tier 2, staged).

The v1 GGML proof authenticates spot openings against Merkle roots, which
bounds cheating economically but never authenticates the terminal
multilinear evaluations: the arithmetic relation itself is only sampled.
The v2 sidecar carries an exact batched sumcheck over the padded band
relation ``Y = X @ W`` for every challenged output block, plus IPA
openings of the X, W and Y multilinear extensions at the transcript-derived
points, so the sampled op's arithmetic is sound rather than spot-checked.

Band decomposition (this is what removes the vector-size cap): the X band
is committed **per row** and the W band **per column**, each a vector of
the padded inner length. The MLE of a row-major matrix at a split point
factorizes as

    X(row_pt ++ inner_pt) = sum_r eq(row_pt, r) * X_row_r(inner_pt)
    W(inner_pt ++ col_pt) = sum_c eq(col_pt, c) * W_col_c(inner_pt)

so one native linear-combination opening per band authenticates the
sumcheck's terminal claim against the per-vector commitments, whose
weighted sum the verifier recomputes with ``combine_commitments``. A
64-row band over inner=8192 needs only length-8192 vectors, far under the
native cap, and the sumcheck itself runs in the native Rust prover.

Requirement state (the mesh has not shipped, so the default is always the
latest implementation; the env gates below exist for unit fixtures that
run without native crypto, NOT as a rollout stage):

- Workers attach the sidecar whenever the native PCS library is present.
- Validators verify a present sidecar and hard-fail a wrong one, and
  require its presence by default (``VERATHOS_MESH_REQUIRE_GEMM_V2``
  defaults to 1).
- The statement is bound to the v1 transcript label (which already commits
  the beacon, receipt context and manifest root), the tensor identity and
  the challenged block indexes.

Composition with the base proof (what welds the sidecar to the audited op):

- **Y is bound exactly.** The v1 payload opens the full challenged output
  block (``leaf_data``, Merkle-authenticated against ``output_root``); the
  verifier recomputes the sidecar's Y commitment from those authenticated
  bytes and requires equality, so the sumcheck's product is the audited
  op's actual output block, not a prover-chosen one.
- **W is bound exactly** wherever the signed GGUF manifest carries the
  tensor's per-column PCS commitment root (``pcs_w_col_root``): the prover
  opens the challenged column groups against that root and the verifier
  requires the sidecar's per-column commitments to equal the opened ones
  byte for byte. Registration binds the manifest root on-chain, so the
  weights are anchored to the model the miner registered, not to bytes the
  prover chose. Tensors registered before the field existed fall back to
  the spot binding below.
- **X is bound exactly** through ``input_band_openings``: the challenged
  band's own chunks of the SAME flat tree that ``input_root`` commits (and
  the v1 engine already binds into the transcript) are shipped with their
  Merkle paths, so the verifier rebuilds the band from authenticated bytes
  and recomputes the sidecar's per-row X commitments. This closes the last
  place where a prover could choose values the sampled spot positions
  happened to miss. Cost tracks the band, not the matrix: a decode step
  challenges one row. The opening is MANDATORY, with no staging flag: the
  mesh has never shipped, so there is no older worker to tolerate and a
  payload without it is simply not a valid mesh proof.
- **Spot binding remains** as a redundant check: the per-row/per-column
  commitments must also open to the same values the v1 proof
  Merkle-verifies at the transcript-derived spot positions.

Note that restructuring the C++ stage-boundary capture into a Merkle tree
is NOT required for this: the boundary roots authenticate what crosses the
RPC edge between stages (a separate statement, and still flat stream
digests), while the sidecar's X anchor needs the audited op's own input,
which the v1 input tree already commits per chunk.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from zkllm.crypto.gemm_v2_batch import (
    GemmV2BatchProof,
    GemmV2BatchStatement,
    verify_gemm_v2_batch_sumcheck,
)
from zkllm.crypto.gemm_v2_reference import (
    GemmV2FormatError,
    GemmV2Statement,
    GemmV2VerificationError,
    PALLAS_SCALAR_MODULUS,
)

GEMM_V2_SIDECAR_VERSION = 3
MESH_REQUIRE_GEMM_V2_ENV = "VERATHOS_MESH_REQUIRE_GEMM_V2"

_OPERATION_IDENTITY_DOMAIN = b"verathos-mesh-gemm-v2-op-v1\x00"
_COMMITMENT_LIST_DOMAIN = b"verathos-mesh-gemm-v2-commitments-v1\x00"


class GemmV2SidecarError(RuntimeError):
    """A malformed or cryptographically invalid v2 sidecar."""


def gemm_v2_required() -> bool:
    # Default ON: every worker platform ships the native PCS library,
    # including the aarch64/macOS build for Metal workers, certified on an
    # M1 Max (crate tests green, python ctypes roundtrip against the arm64
    # dylib, matching ABI) with the x86-only halo2curves `asm` feature
    # split by target arch. Set VERATHOS_MESH_REQUIRE_GEMM_V2=0 to stage off.
    raw = os.environ.get(MESH_REQUIRE_GEMM_V2_ENV, "1")
    return raw.strip() != "0"


def gemm_v2_available() -> bool:
    """True when the native PCS library can be loaded on this host."""

    try:
        from zkllm.crypto.pcs_v2 import native_library_path

        native_library_path()
        return True
    except Exception:
        return False


def _pow2_at_least(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def gemm_v2_block_supported(
    *,
    rows: int,
    inner: int,
    columns: int,
    block_size: int,
) -> bool:
    """Whether one challenged block's bands fit the native PCS limits.

    With per-row/per-column commitments the binding vectors have the padded
    inner length, so the practical bound is ``MAX_VECTOR_LEN`` on the inner
    dimension (131072 covers every current model) and 256 combine terms on
    the padded block edge (the block size is 64).
    """

    from zkllm.crypto.pcs_v2 import MAX_COMBINE_TERMS, MAX_VECTOR_LEN

    band_rows = _pow2_at_least(min(int(block_size), int(rows)))
    band_cols = _pow2_at_least(min(int(block_size), int(columns)))
    padded_inner = _pow2_at_least(int(inner))
    return (
        padded_inner <= MAX_VECTOR_LEN
        and band_rows <= MAX_COMBINE_TERMS
        and band_cols <= MAX_COMBINE_TERMS
        and band_rows * band_cols <= MAX_VECTOR_LEN
    )


def _operation_identity(
    *,
    tensor_name: str,
    layer_index: int,
    op_index: int,
) -> bytes:
    name = tensor_name.encode("utf-8", "strict")
    return (
        _OPERATION_IDENTITY_DOMAIN
        + len(name).to_bytes(2, "little")
        + name
        + int(layer_index).to_bytes(8, "little", signed=True)
        + int(op_index).to_bytes(8, "little", signed=True)
    )


def _commitment_list_bytes(label: bytes, commitments: Sequence[bytes]) -> bytes:
    points = tuple(bytes(item) for item in commitments)
    if not points or len(points) > 256 or any(len(item) != 32 for item in points):
        raise GemmV2SidecarError("per-vector commitment list is invalid")
    return (
        _COMMITMENT_LIST_DOMAIN
        + struct.pack("<H", len(label))
        + label
        + struct.pack("<H", len(points))
        + b"".join(points)
    )


def _mle_coefficients(point: Sequence[int]) -> tuple[int, ...]:
    """Equality-polynomial coefficients over the point, lexicographic order.

    Mirrors the proof-v2 engine's expansion; correctness of the bit order is
    additionally enforced by the prove-time self-check against the
    sumcheck's terminal claims.
    """

    modulus = PALLAS_SCALAR_MODULUS
    coefficients = [1]
    for challenge in point:
        challenge = int(challenge)
        if not 0 <= challenge < modulus:
            raise GemmV2SidecarError("MLE challenge is outside the Pallas field")
        zero = (1 - challenge) % modulus
        expanded = []
        for coefficient in coefficients:
            expanded.append(coefficient * zero % modulus)
            expanded.append(coefficient * challenge % modulus)
        coefficients = expanded
    return tuple(coefficients)


def _padded_band(
    matrix: np.ndarray,
    rows: int,
    columns: int,
    dtype: np.dtype,
) -> np.ndarray:
    padded = np.zeros((rows, columns), dtype=dtype)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return padded


def _scalar_int(value: bytes) -> int:
    from zkllm.crypto.pcs_v2 import scalar_to_int

    return scalar_to_int(value)


def _statement(
    *,
    outer_label: bytes,
    tensor_name: str,
    layer_index: int,
    op_index: int,
    bi: int,
    bj: int,
    rows: int,
    inner: int,
    valid_inner: int,
    columns: int,
    x_row_commitments: Sequence[bytes],
    w_col_commitments: Sequence[bytes],
    y_commitment: bytes,
) -> GemmV2Statement:
    return GemmV2Statement(
        outer_digest=outer_label,
        operation_identity=_operation_identity(
            tensor_name=tensor_name,
            layer_index=layer_index,
            op_index=op_index,
        ),
        row_block=int(bi),
        column_block=int(bj),
        rows=rows,
        inner=inner,
        valid_inner=valid_inner,
        columns=columns,
        x_commitment=_commitment_list_bytes(b"x", x_row_commitments),
        w_commitment=_commitment_list_bytes(b"w", w_col_commitments),
        y_commitment=y_commitment,
    )


def build_gemm_v2_sidecar(
    *,
    x_matrix: np.ndarray,
    w_matrix: np.ndarray,
    y_matrix: np.ndarray,
    challenged_blocks: list[tuple[int, int]],
    block_size: int,
    transcript_label: bytes,
    tensor_name: str,
    layer_index: int,
    op_index: int,
    spot_positions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove every challenged block and return the serialized sidecar.

    ``x_matrix``/``w_matrix`` are the same int8 witnesses the v1 prover
    consumes; ``y_matrix`` is the int64 product. The X band for block
    ``(bi, bj)`` is the challenged output rows across the full inner
    dimension, the W band the full inner dimension across the challenged
    output columns, exactly the bands whose spot openings the v1 proof
    Merkle-verifies.
    """

    from zkllm.crypto import pcs_v2

    if x_matrix.ndim != 2 or w_matrix.ndim != 2 or y_matrix.ndim != 2:
        raise GemmV2SidecarError("sidecar witnesses must be 2-D matrices")
    m, k = x_matrix.shape
    k2, n = w_matrix.shape
    if k != k2 or y_matrix.shape != (m, n):
        raise GemmV2SidecarError("sidecar witness shapes are inconsistent")
    if not transcript_label:
        raise GemmV2SidecarError("transcript label must not be empty")
    if not challenged_blocks:
        raise GemmV2SidecarError("sidecar needs at least one challenged block")
    # The v1 transcript label is a domain prefix plus a digest; the v2
    # statement wants exactly 32 bytes, so bind its hash.
    outer_label = hashlib.sha256(bytes(transcript_label)).digest()

    # All blocks come from one operation, so the padded dimensions are
    # uniform, which is what the native batch prover requires.
    inner = _pow2_at_least(k)
    x_pads: list[np.ndarray] = []
    w_pads: list[np.ndarray] = []
    y_pads: list[np.ndarray] = []
    statements: list[GemmV2Statement] = []
    blocks: list[dict[str, Any]] = []
    rows = columns = 0
    for bi, bj in challenged_blocks:
        row_start = int(bi) * int(block_size)
        col_start = int(bj) * int(block_size)
        if row_start >= m or col_start >= n:
            raise GemmV2SidecarError("challenged block is outside the output")
        x_band = x_matrix[row_start : row_start + block_size, :]
        w_band = w_matrix[:, col_start : col_start + block_size]
        y_block = y_matrix[
            row_start : row_start + block_size,
            col_start : col_start + block_size,
        ]
        block_rows = _pow2_at_least(x_band.shape[0])
        block_cols = _pow2_at_least(w_band.shape[1])
        rows = max(rows, block_rows)
        columns = max(columns, block_cols)
        x_pads.append((x_band, block_rows))
        w_pads.append((w_band, block_cols))
        y_pads.append(y_block)
        blocks.append({"row_block": int(bi), "column_block": int(bj)})

    x_arrays: list[np.ndarray] = []
    w_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    for index, ((x_band, _), (w_band, _), y_block) in enumerate(
        zip(x_pads, w_pads, y_pads)
    ):
        x_pad = _padded_band(x_band, rows, inner, np.dtype(np.int8))
        w_pad = _padded_band(w_band, inner, columns, np.dtype(np.int8))
        y_pad = _padded_band(y_block, rows, columns, np.dtype(np.int64))
        x_arrays.append(x_pad)
        w_arrays.append(w_pad)
        y_arrays.append(y_pad)

        x_row_commitments = pcs_v2.commit_i8_batch(
            x_pad.tobytes(), vector_length=inner
        )
        w_col_commitments = pcs_v2.commit_i8_batch(
            np.ascontiguousarray(w_pad.T).tobytes(), vector_length=inner
        )
        y_commitment = pcs_v2.commit(
            [int(value) for value in y_pad.reshape(-1)],
            encoding=pcs_v2.ENCODING_SIGNED_I64,
        )
        statements.append(
            _statement(
                outer_label=outer_label,
                tensor_name=tensor_name,
                layer_index=layer_index,
                op_index=op_index,
                bi=blocks[index]["row_block"],
                bj=blocks[index]["column_block"],
                rows=rows,
                inner=inner,
                valid_inner=k,
                columns=columns,
                x_row_commitments=x_row_commitments,
                w_col_commitments=w_col_commitments,
                y_commitment=y_commitment,
            )
        )
        blocks[index].update(
            {
                "x_row_commitments": [c.hex() for c in x_row_commitments],
                "w_col_commitments": [c.hex() for c in w_col_commitments],
                "y_commitment": y_commitment.hex(),
            }
        )

    batch_statement = GemmV2BatchStatement(statements=tuple(statements))
    try:
        wire = pcs_v2.prove_gemm_batch_sumcheck_i8(
            batch_digest=batch_statement.digest(),
            x_values=b"".join(item.tobytes() for item in x_arrays),
            w_values=b"".join(item.tobytes() for item in w_arrays),
            y_values_i64_le=b"".join(
                item.astype("<i8").tobytes() for item in y_arrays
            ),
            valid_inner=tuple(k for _ in statements),
            block_count=len(statements),
            rows=rows,
            inner=inner,
            columns=columns,
        )
        proof = GemmV2BatchProof.from_canonical_bytes(
            wire,
            expected_blocks=len(statements),
            expected_rounds=batch_statement.inner_bits,
        )
        claims = verify_gemm_v2_batch_sumcheck(batch_statement, proof)
    except Exception as exc:
        # A witness that does not satisfy Y = X @ W fails its own sumcheck
        # right here, before anything is serialized.
        raise GemmV2SidecarError(f"gemm-v2 witness does not prove: {exc}") from exc

    row_coefficients = _mle_coefficients(claims.row_challenges)
    col_coefficients = _mle_coefficients(claims.column_challenges)

    for index, statement in enumerate(statements):
        statement_digest = statement.digest()
        x_opening = pcs_v2.prove_i8_linear_combination(
            x_arrays[index].tobytes(),
            vector_length=inner,
            coefficients=row_coefficients,
            point=claims.inner_challenges,
            outer_digest=statement_digest,
        )
        w_opening = pcs_v2.prove_i8_linear_combination(
            np.ascontiguousarray(w_arrays[index].T).tobytes(),
            vector_length=inner,
            coefficients=col_coefficients,
            point=claims.inner_challenges,
            outer_digest=statement_digest,
        )
        y_opening = pcs_v2.prove(
            [int(value) for value in y_arrays[index].reshape(-1)],
            claims.y_point,
            statement_digest,
            encoding=pcs_v2.ENCODING_SIGNED_I64,
        )
        # Fail closed at prove time: the opened evaluations must equal the
        # sumcheck's terminal claims. A mismatch here is a layout bug, and
        # shipping it would only surface as an unexplainable validator
        # strike.
        if _scalar_int(x_opening.evaluation) != claims.x_values[index]:
            raise GemmV2SidecarError("X opening does not match the sumcheck claim")
        if _scalar_int(w_opening.evaluation) != claims.w_values[index]:
            raise GemmV2SidecarError("W opening does not match the sumcheck claim")
        if _scalar_int(y_opening.evaluation) != claims.y_values[index]:
            raise GemmV2SidecarError("Y opening does not match the sumcheck claim")
        blocks[index].update(
            {
                "x_opening": _opening_to_dict(x_opening),
                "w_opening": _opening_to_dict(w_opening),
                "y_opening": _opening_to_dict(y_opening),
            }
        )

        # Witness binding: open the row/column commitments at the SAME
        # transcript-derived spot positions whose values the v1 proof
        # Merkle-verifies against the dumped witnesses. Both commitment
        # schemes must agree at every sampled cell, welding the sumcheck
        # statement to the operands the base proof checks. Challenge-time
        # cost only; nothing runs on the serve path.
        if spot_positions is not None:
            spots = spot_positions[index]
            bi = int(blocks[index]["row_block"])
            bj = int(blocks[index]["column_block"])
            nbits = max(1, (inner - 1).bit_length())
            x_spot_openings = []
            for row, col in spots.get("x", ()):
                local_row = int(row) - bi * int(block_size)
                if not 0 <= local_row < x_arrays[index].shape[0]:
                    raise GemmV2SidecarError("X spot row outside the block")
                if not 0 <= int(col) < k:
                    raise GemmV2SidecarError("X spot column outside the band")
                opening = pcs_v2.prove(
                    np.ascontiguousarray(x_arrays[index][local_row]).tobytes(),
                    _corner_point(int(col), nbits),
                    outer_label,
                    encoding=pcs_v2.ENCODING_SIGNED_I8,
                )
                x_spot_openings.append(
                    _spot_opening_to_dict(opening, int(row), int(col))
                )
            w_spot_openings = []
            for row, col in spots.get("w", ()):
                local_col = int(col) - bj * int(block_size)
                if not 0 <= local_col < w_arrays[index].shape[1]:
                    raise GemmV2SidecarError("W spot column outside the block")
                if not 0 <= int(row) < k:
                    raise GemmV2SidecarError("W spot row outside the band")
                opening = pcs_v2.prove(
                    np.ascontiguousarray(
                        w_arrays[index][:, local_col]
                    ).tobytes(),
                    _corner_point(int(row), nbits),
                    outer_label,
                    encoding=pcs_v2.ENCODING_SIGNED_I8,
                )
                w_spot_openings.append(
                    _spot_opening_to_dict(opening, int(row), int(col))
                )
            blocks[index]["x_spot_openings"] = x_spot_openings
            blocks[index]["w_spot_openings"] = w_spot_openings

    return {
        "version": GEMM_V2_SIDECAR_VERSION,
        "tensor_name": str(tensor_name),
        "layer_index": int(layer_index),
        "op_index": int(op_index),
        "block_size": int(block_size),
        "rows": rows,
        "inner": inner,
        "valid_inner": int(k),
        "columns": columns,
        "sumcheck_wire": wire.hex(),
        "blocks": blocks,
    }


def _opening_to_dict(opening: Any) -> dict[str, Any]:
    return {
        "commitment": opening.commitment.hex(),
        "evaluation": opening.evaluation.hex(),
        "proof": opening.proof.hex(),
        "vector_length": int(opening.vector_length),
        "padded_length": int(opening.padded_length),
        "encoding": int(opening.encoding),
    }


def _opening_from_dict(raw: Mapping[str, Any]) -> Any:
    from zkllm.crypto.pcs_v2 import PCSOpeningV2

    try:
        return PCSOpeningV2(
            commitment=bytes.fromhex(str(raw["commitment"])),
            evaluation=bytes.fromhex(str(raw["evaluation"])),
            proof=bytes.fromhex(str(raw["proof"])),
            vector_length=int(raw["vector_length"]),
            padded_length=int(raw["padded_length"]),
            encoding=int(raw["encoding"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise GemmV2SidecarError(f"malformed PCS opening: {exc}") from exc


def _corner_point(index: int, nbits: int) -> list[int]:
    """Boolean hypercube corner for flat index ``index``.

    The native MLE treats the FIRST supplied coordinate as the most
    significant selector bit (verified empirically against
    ``pcs_v2.evaluate``), so the point is emitted MSB-first.
    """

    if index < 0 or index >= (1 << nbits):
        raise GemmV2SidecarError("spot index outside the padded hypercube")
    return [(index >> (nbits - 1 - b)) & 1 for b in range(nbits)]


def _scalar_signed_int(evaluation: bytes) -> int:
    from zkllm.crypto import pcs_v2

    value = pcs_v2.scalar_to_int(evaluation)
    modulus = pcs_v2.PALLAS_SCALAR_MODULUS
    return value - modulus if value > modulus // 2 else value


def _spot_opening_to_dict(opening: Any, row: int, col: int) -> dict[str, Any]:
    # Deliberately NO commitment field: the verifier reconstructs the
    # opening with the commitment it expects (the sidecar's own row/column
    # commitment), so a prover cannot bind the spot proof to different data.
    return {
        "row": int(row),
        "col": int(col),
        "evaluation": opening.evaluation.hex(),
        "proof": opening.proof.hex(),
        "vector_length": int(opening.vector_length),
        "padded_length": int(opening.padded_length),
        "encoding": int(opening.encoding),
    }


def _spot_opening_from_dict(raw: Mapping[str, Any], commitment: bytes) -> Any:
    from zkllm.crypto.pcs_v2 import PCSOpeningV2

    try:
        return PCSOpeningV2(
            commitment=bytes(commitment),
            evaluation=bytes.fromhex(str(raw["evaluation"])),
            proof=bytes.fromhex(str(raw["proof"])),
            vector_length=int(raw["vector_length"]),
            padded_length=int(raw["padded_length"]),
            encoding=int(raw["encoding"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise GemmV2SidecarError(f"malformed PCS spot opening: {exc}") from exc


def verify_w_col_manifest_binding(
    sidecar: Mapping[str, Any],
    *,
    tensor_leaf: Mapping[str, Any],
    w_col_manifest: Any,
    challenged_blocks: list[tuple[int, int]],
    block_size: int,
    weight_shape: tuple[int, int],
) -> None:
    """Bind the sidecar's W column commitments to the signed manifest.

    ``tensor_leaf`` is the Merkle-authenticated manifest record (verified
    against the receipt-bound manifest root before this runs). When it
    carries a ``pcs_w_col_root``, the payload MUST open the challenged
    column groups against that root, and every real column's sidecar
    commitment must equal the opened manifest commitment byte-for-byte;
    pow2 pad columns must equal the all-zero column commitment. That makes
    the sumcheck's W operand EXACTLY the registered model's columns, not a
    prover-chosen band that merely matches at spot positions.
    """

    from verallm.mesh.gguf_manifest import (
        GGUF_PCS_W_COL_VERSION,
        pcs_w_col_groups_for_band,
        pcs_w_col_leaf_hash,
        pcs_w_col_zero_commitment,
        verify_pcs_w_col_membership,
    )

    root = str(tensor_leaf.get("pcs_w_col_root", "") or "")
    if not root:
        raise GemmV2SidecarError("tensor leaf has no pcs w-col root")
    if int(tensor_leaf.get("pcs_w_col_version", 0)) != GGUF_PCS_W_COL_VERSION:
        raise GemmV2SidecarError("unsupported pcs w-col version")
    group = int(tensor_leaf.get("pcs_w_col_group", 0))
    count = int(tensor_leaf.get("pcs_w_col_count", 0))
    vector_length = int(tensor_leaf.get("pcs_w_col_vector_length", 0))
    k, n = int(weight_shape[0]), int(weight_shape[1])
    if group <= 0 or count <= 0 or vector_length <= 0:
        raise GemmV2SidecarError("pcs w-col manifest record is malformed")
    if count != n or _pow2_at_least(k) != vector_length:
        raise GemmV2SidecarError(
            "pcs w-col record does not match the payload weight shape"
        )
    leaf_count = (count + group - 1) // group
    if not isinstance(w_col_manifest, Mapping):
        raise GemmV2SidecarError(
            "payload is missing the pcs w-col manifest openings"
        )
    if int(w_col_manifest.get("version", 0)) != GGUF_PCS_W_COL_VERSION:
        raise GemmV2SidecarError("unsupported pcs w-col opening version")
    raw_opening_blocks = w_col_manifest.get("blocks")
    if not isinstance(raw_opening_blocks, list) or len(raw_opening_blocks) != len(
        challenged_blocks
    ):
        raise GemmV2SidecarError("pcs w-col opening block count mismatch")
    raw_sidecar_blocks = sidecar.get("blocks")
    if not isinstance(raw_sidecar_blocks, list) or len(raw_sidecar_blocks) != len(
        challenged_blocks
    ):
        raise GemmV2SidecarError("sidecar block count mismatch")

    zero_commitment = pcs_w_col_zero_commitment(vector_length)
    for (bi, bj), opening_block, sidecar_block in zip(
        challenged_blocks, raw_opening_blocks, raw_sidecar_blocks
    ):
        if not isinstance(opening_block, Mapping) or not isinstance(
            sidecar_block, Mapping
        ):
            raise GemmV2SidecarError("pcs w-col opening block is not a mapping")
        if (
            int(opening_block.get("row_block", -1)) != int(bi)
            or int(opening_block.get("column_block", -1)) != int(bj)
        ):
            raise GemmV2SidecarError("pcs w-col opening block index mismatch")
        w_col_commitments = _commitment_list_from_hex(
            sidecar_block.get("w_col_commitments"), "W"
        )
        col_start = int(bj) * int(block_size)
        real_cols = min(int(block_size), n - col_start)
        if real_cols <= 0:
            raise GemmV2SidecarError("pcs w-col challenged block outside tensor")
        expected_groups = pcs_w_col_groups_for_band(
            col_start=col_start, real_cols=real_cols, group=group
        )
        raw_groups = opening_block.get("groups")
        if not isinstance(raw_groups, list) or len(raw_groups) != len(
            expected_groups
        ):
            raise GemmV2SidecarError("pcs w-col opening group count mismatch")
        opened: dict[int, bytes] = {}
        for expected_index, raw_group in zip(expected_groups, raw_groups):
            if not isinstance(raw_group, Mapping):
                raise GemmV2SidecarError("pcs w-col group is not a mapping")
            g = int(raw_group.get("group_index", -1))
            if g != int(expected_index):
                raise GemmV2SidecarError("pcs w-col group index mismatch")
            expected_size = min(group, count - g * group)
            commitments = _commitment_list_from_hex(
                raw_group.get("commitments"), "pcs w-col"
            )
            if len(commitments) != expected_size:
                raise GemmV2SidecarError("pcs w-col group size mismatch")
            if not verify_pcs_w_col_membership(
                leaf_hash=pcs_w_col_leaf_hash(g, commitments),
                root=root,
                leaf_count=leaf_count,
                index=g,
                path=list(raw_group.get("path", [])),
            ):
                raise GemmV2SidecarError(
                    "pcs w-col group membership proof failed "
                    f"(group={g} of {leaf_count}, group_size={group}, "
                    f"count={count}, k={k}, n={n}, "
                    f"root={root[:12]}..., "
                    f"path_len={len(list(raw_group.get('path', [])))})"
                )
            for offset, commitment in enumerate(commitments):
                opened[g * group + offset] = commitment
        for local, commitment in enumerate(w_col_commitments):
            column = col_start + local
            if local < real_cols:
                expected = opened.get(column)
                if expected is None:
                    raise GemmV2SidecarError(
                        "pcs w-col opening does not cover a challenged column"
                    )
                if commitment != expected:
                    raise GemmV2SidecarError(
                        "sidecar W column commitment disagrees with the "
                        "signed manifest"
                    )
            elif commitment != zero_commitment:
                raise GemmV2SidecarError(
                    "sidecar W pad column is not the zero commitment"
                )


def _commitment_list_from_hex(raw: Any, label: str) -> tuple[bytes, ...]:
    if not isinstance(raw, list) or not raw:
        raise GemmV2SidecarError(f"sidecar {label} commitment list is missing")
    try:
        items = tuple(bytes.fromhex(str(item)) for item in raw)
    except ValueError as exc:
        raise GemmV2SidecarError(f"malformed {label} commitment: {exc}") from exc
    if any(len(item) != 32 for item in items):
        raise GemmV2SidecarError(f"sidecar {label} commitment has a wrong length")
    return items


def verify_gemm_v2_sidecar(
    sidecar: Mapping[str, Any],
    *,
    transcript_label: bytes,
    challenged_blocks: list[tuple[int, int]],
    block_size: int,
    tensor_name: str,
    layer_index: int,
    op_index: int,
    input_shape: tuple[int, int],
    weight_shape: tuple[int, int],
    spot_x_values: Sequence[Sequence[tuple[int, int, int]]] | None = None,
    spot_w_values: Sequence[Sequence[tuple[int, int, int]]] | None = None,
    y_block_values: Sequence[np.ndarray | None] | None = None,
    x_band_values: Sequence[np.ndarray | None] | None = None,
    w_col_manifest: Any | None = None,
) -> None:
    """Verify the sidecar or raise :class:`GemmV2SidecarError`.

    The statement is reconstructed from validator-derived facts (the
    transcript label, the beacon-derived challenged blocks, the payload
    shapes); only the commitments and the proof material come from the
    sidecar, so a coordinator cannot move the statement.

    ``spot_x_values``/``spot_w_values`` carry, per challenged block, the
    transcript-derived ``(row, col, signed_value)`` spot checks whose values
    the v1 proof Merkle-verifies against the dumped witnesses. The sidecar
    must open its own row/column commitments to the same values at those
    positions, which binds the sumcheck statement to the operands the base
    proof checks instead of letting it stand over data of the prover's
    choosing.

    ``y_block_values`` carries, per challenged block, the int64 output
    block exactly as the v1 payload opens it against ``output_root``
    (actual edge shape, unpadded). When supplied, the verifier recomputes
    the sidecar's Y commitment from those authenticated values and requires
    byte equality, binding the sumcheck's product EXACTLY to the audited
    op's output block. A sidecar over any consistent-but-different triple
    (X', W', Y') is rejected outright instead of merely having to dodge
    the sampled spot positions.
    """

    from zkllm.crypto import pcs_v2

    if int(sidecar.get("version", 0)) != GEMM_V2_SIDECAR_VERSION:
        raise GemmV2SidecarError("unsupported gemm-v2 sidecar version")
    if not transcript_label:
        raise GemmV2SidecarError("transcript label must not be empty")
    outer_label = hashlib.sha256(bytes(transcript_label)).digest()
    if str(sidecar.get("tensor_name", "")) != str(tensor_name):
        raise GemmV2SidecarError("sidecar tensor name does not match the payload")
    if int(sidecar.get("op_index", -1)) != int(op_index):
        raise GemmV2SidecarError("sidecar op index does not match the payload")
    if int(sidecar.get("layer_index", -1)) != int(layer_index):
        raise GemmV2SidecarError("sidecar layer index does not match the payload")
    if int(sidecar.get("block_size", 0)) != int(block_size):
        raise GemmV2SidecarError("sidecar block size does not match the proof")

    m, k = (int(input_shape[0]), int(input_shape[1]))
    k2, n = (int(weight_shape[0]), int(weight_shape[1]))
    if k != k2:
        raise GemmV2SidecarError("payload shapes are not a valid GEMM")
    if not challenged_blocks:
        raise GemmV2SidecarError("no challenged blocks to verify against")

    inner = _pow2_at_least(k)
    expected_rows = max(
        _pow2_at_least(min(int(block_size), m - int(bi) * int(block_size)))
        for bi, _ in challenged_blocks
    )
    expected_cols = max(
        _pow2_at_least(min(int(block_size), n - int(bj) * int(block_size)))
        for _, bj in challenged_blocks
    )
    for field, value in (
        ("rows", expected_rows),
        ("inner", inner),
        ("valid_inner", k),
        ("columns", expected_cols),
    ):
        if int(sidecar.get(field, -1)) != value:
            raise GemmV2SidecarError(
                f"sidecar {field} does not match the payload shapes"
            )

    raw_blocks = sidecar.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise GemmV2SidecarError("sidecar carries no proven blocks")
    ordered_expected = [(int(bi), int(bj)) for bi, bj in challenged_blocks]
    ordered_actual = []
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            raise GemmV2SidecarError("sidecar block is not a mapping")
        ordered_actual.append(
            (int(raw.get("row_block", -1)), int(raw.get("column_block", -1)))
        )
    if ordered_actual != ordered_expected:
        raise GemmV2SidecarError(
            "sidecar blocks do not match the beacon-derived challenge set"
        )
    for bi, bj in ordered_expected:
        if bi * int(block_size) >= m or bj * int(block_size) >= n:
            raise GemmV2SidecarError("challenged block is outside the output")

    statements: list[GemmV2Statement] = []
    per_block: list[
        tuple[tuple[bytes, ...], tuple[bytes, ...], bytes, Mapping[str, Any]]
    ] = []
    for raw in raw_blocks:
        x_row_commitments = _commitment_list_from_hex(
            raw.get("x_row_commitments"), "X"
        )
        w_col_commitments = _commitment_list_from_hex(
            raw.get("w_col_commitments"), "W"
        )
        if len(x_row_commitments) != expected_rows:
            raise GemmV2SidecarError("X row commitment count mismatch")
        if len(w_col_commitments) != expected_cols:
            raise GemmV2SidecarError("W column commitment count mismatch")
        try:
            y_commitment = bytes.fromhex(str(raw.get("y_commitment", "")))
        except ValueError as exc:
            raise GemmV2SidecarError(f"malformed Y commitment: {exc}") from exc
        try:
            statements.append(
                _statement(
                    outer_label=outer_label,
                    tensor_name=tensor_name,
                    layer_index=layer_index,
                    op_index=op_index,
                    bi=int(raw["row_block"]),
                    bj=int(raw["column_block"]),
                    rows=expected_rows,
                    inner=inner,
                    valid_inner=k,
                    columns=expected_cols,
                    x_row_commitments=x_row_commitments,
                    w_col_commitments=w_col_commitments,
                    y_commitment=y_commitment,
                )
            )
        except (GemmV2FormatError, ValueError, TypeError, KeyError) as exc:
            raise GemmV2SidecarError(f"malformed gemm-v2 block: {exc}") from exc
        per_block.append((x_row_commitments, w_col_commitments, y_commitment, raw))
        block_index = len(per_block) - 1
        bi = int(raw["row_block"])
        bj = int(raw["column_block"])
        nbits = max(1, (inner - 1).bit_length())

        if x_band_values is not None:
            if len(x_band_values) != len(ordered_expected):
                raise GemmV2SidecarError("X band value count mismatch")
            x_values = x_band_values[block_index]
            if x_values is not None:
                x_values = np.asarray(x_values)
                rows_actual = min(int(block_size), m - bi * int(block_size))
                if x_values.shape != (rows_actual, k):
                    raise GemmV2SidecarError(
                        "X band values do not match the challenged band shape"
                    )
                if x_values.dtype != np.int8:
                    raise GemmV2SidecarError("X band values must be int8")
                x_pad = _padded_band(
                    x_values, expected_rows, inner, np.dtype(np.int8)
                )
                expected_x_commitments = pcs_v2.commit_i8_batch(
                    x_pad.tobytes(), vector_length=inner
                )
                if list(expected_x_commitments) != list(x_row_commitments):
                    raise GemmV2SidecarError(
                        "X commitments do not match the authenticated input band"
                    )

        if y_block_values is not None:
            if len(y_block_values) != len(ordered_expected):
                raise GemmV2SidecarError("Y block value count mismatch")
            y_values = y_block_values[block_index]
            if y_values is not None:
                y_values = np.asarray(y_values)
                rows_actual = min(int(block_size), m - bi * int(block_size))
                cols_actual = min(int(block_size), n - bj * int(block_size))
                if y_values.shape != (rows_actual, cols_actual):
                    raise GemmV2SidecarError(
                        "Y block values do not match the challenged block shape"
                    )
                if y_values.dtype != np.int64:
                    raise GemmV2SidecarError("Y block values must be int64")
                y_pad = _padded_band(
                    y_values, expected_rows, expected_cols, np.dtype(np.int64)
                )
                expected_y_commitment = pcs_v2.commit(
                    [int(value) for value in y_pad.reshape(-1)],
                    encoding=pcs_v2.ENCODING_SIGNED_I64,
                )
                if y_commitment != expected_y_commitment:
                    raise GemmV2SidecarError(
                        "Y commitment does not match the authenticated "
                        "output block opening"
                    )

        def _check_spot_openings(
            *,
            raw_openings: Any,
            expected_spots: Sequence[tuple[int, int, int]],
            commitments: tuple[bytes, ...],
            local_of: Any,
            point_of: Any,
            label: str,
        ) -> None:
            from zkllm.crypto import pcs_v2

            if not isinstance(raw_openings, list) or len(raw_openings) != len(
                expected_spots
            ):
                raise GemmV2SidecarError(
                    f"sidecar {label} spot opening count mismatch"
                )
            for raw_opening, (row, col, value) in zip(
                raw_openings, expected_spots
            ):
                if not isinstance(raw_opening, Mapping):
                    raise GemmV2SidecarError(
                        f"sidecar {label} spot opening is not a mapping"
                    )
                if (
                    int(raw_opening.get("row", -1)) != int(row)
                    or int(raw_opening.get("col", -1)) != int(col)
                ):
                    raise GemmV2SidecarError(
                        f"sidecar {label} spot opening position mismatch"
                    )
                local = local_of(int(row), int(col))
                if not 0 <= local < len(commitments):
                    raise GemmV2SidecarError(
                        f"sidecar {label} spot outside the committed block"
                    )
                opening = _spot_opening_from_dict(
                    raw_opening, commitments[local]
                )
                point = _corner_point(point_of(int(row), int(col)), nbits)
                if not pcs_v2.verify(opening, point, outer_label):
                    raise GemmV2SidecarError(
                        f"sidecar {label} spot opening failed to verify"
                    )
                if _scalar_signed_int(opening.evaluation) != int(value):
                    raise GemmV2SidecarError(
                        f"sidecar {label} spot value disagrees with the "
                        "Merkle-verified witness"
                    )

        if spot_x_values is not None:
            _check_spot_openings(
                raw_openings=raw.get("x_spot_openings"),
                expected_spots=spot_x_values[block_index],
                commitments=x_row_commitments,
                local_of=lambda row, col: row - bi * int(block_size),
                point_of=lambda row, col: col,
                label="X",
            )
        if spot_w_values is not None:
            _check_spot_openings(
                raw_openings=raw.get("w_spot_openings"),
                expected_spots=spot_w_values[block_index],
                commitments=w_col_commitments,
                local_of=lambda row, col: col - bj * int(block_size),
                point_of=lambda row, col: row,
                label="W",
            )

    try:
        batch_statement = GemmV2BatchStatement(statements=tuple(statements))
        wire = bytes.fromhex(str(sidecar.get("sumcheck_wire", "")))
        proof = GemmV2BatchProof.from_canonical_bytes(
            wire,
            expected_blocks=len(statements),
            expected_rounds=batch_statement.inner_bits,
        )
    except (GemmV2FormatError, ValueError) as exc:
        raise GemmV2SidecarError(f"malformed gemm-v2 sumcheck: {exc}") from exc
    try:
        claims = verify_gemm_v2_batch_sumcheck(batch_statement, proof)
    except (GemmV2FormatError, GemmV2VerificationError) as exc:
        raise GemmV2SidecarError(f"gemm-v2 sumcheck failed: {exc}") from exc

    row_coefficients = _mle_coefficients(claims.row_challenges)
    col_coefficients = _mle_coefficients(claims.column_challenges)

    for index, (x_rows, w_cols, y_commitment, raw) in enumerate(per_block):
        statement_digest = statements[index].digest()
        for label, opening_key, commitments, coefficients, point, value in (
            (
                "X",
                "x_opening",
                x_rows,
                row_coefficients,
                claims.inner_challenges,
                claims.x_values[index],
            ),
            (
                "W",
                "w_opening",
                w_cols,
                col_coefficients,
                claims.inner_challenges,
                claims.w_values[index],
            ),
            (
                "Y",
                "y_opening",
                None,
                None,
                claims.y_point,
                claims.y_values[index],
            ),
        ):
            opening_raw = raw.get(opening_key)
            if not isinstance(opening_raw, Mapping):
                raise GemmV2SidecarError(f"sidecar is missing the {label} opening")
            opening = _opening_from_dict(opening_raw)
            if commitments is None:
                expected_commitment = y_commitment
            else:
                try:
                    expected_commitment = pcs_v2.combine_commitments(
                        commitments, coefficients
                    )
                except pcs_v2.PCSFormatError as exc:
                    raise GemmV2SidecarError(
                        f"{label} commitments do not combine: {exc}"
                    ) from exc
            if opening.commitment != expected_commitment:
                raise GemmV2SidecarError(
                    f"{label} opening commitment does not match the statement"
                )
            if _scalar_int(opening.evaluation) != value:
                raise GemmV2SidecarError(
                    f"{label} opening does not equal the sumcheck claim"
                )
            try:
                valid = pcs_v2.verify(opening, point, statement_digest)
            except pcs_v2.PCSFormatError as exc:
                raise GemmV2SidecarError(
                    f"{label} opening is malformed: {exc}"
                ) from exc
            if not valid:
                raise GemmV2SidecarError(f"{label} IPA opening failed to verify")
