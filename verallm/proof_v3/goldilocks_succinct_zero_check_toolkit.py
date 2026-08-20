"""Toolkit for succinct tile wires: shared columns + eq-fold arguments.

Every per-cell tile relation decomposes, by linearity of the sum, into

    c_1 * (sum eq(z,i) A[i]) + c_2 * (sum eq(z,i) B[i]) + ...
        (+/- sum eq(z,i) A[i]*B[i]) + const == 0

so a tile proof is: PCS-commit each column ONCE, derive one eq point z
post-commit, prove each ``sum eq(z,.) column`` with an eq-fold argument
(degree-2 rounds + one PCS opening against the SHARED column
commitment), prove product terms with the succinct product argument,
and let the verifier check the linear couplings numerically. Padding
cells are chosen to satisfy every relation, so no indicator column is
needed and ``sum_i eq(z,i) == 1`` absorbs constants.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.goldilocks_linear_relation_reference import _fixed32
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
    GoldilocksMultilinearPcsStatementV3,
    commit_goldilocks_multilinear_v3,
    open_goldilocks_multilinear_v3,
    verify_goldilocks_multilinear_opening_v3,
)
from verallm.proof_v3.goldilocks_reference import GOLDILOCKS_MODULUS
from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
    _derive,
    _eq_eval,
    _eq_table,
    _lagrange_0123,
)

_COLUMN_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_TILE_COLUMN/V1"
_EQFOLD_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_TILE_EQFOLD/V1"
_PUBLIC_FOLD_LAYOUT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_PUBLIC_FOLD_LAYOUT/V1"
)
_VARIABLE_GROUP_LAYOUT_DOMAIN: Final = (
    b"VERATHOS/PROOF_V3/GOLDILOCKS_VARIABLE_COLUMN_GROUP/V1"
)


@dataclass(frozen=True, slots=True)
class SuccinctColumnV3:
    """One committed tile column (shared across all sub-arguments).

    When ``group_tag`` is set, the column is one BLOCK of a shared group
    tree: ``tree`` is the group tree, ``block_point`` the boolean
    coordinates (LSB-first, appended after the column point) that select
    the block inside the group cube.  Grouped columns require deferred
    (batched) openings."""

    tag: str
    pcs_statement: GoldilocksMultilinearPcsStatementV3
    tree: object
    values: tuple[int, ...]
    device_values: object = None
    group_tag: str | None = None
    block_point: tuple[int, ...] = ()
    # park slot: device values stashed on pinned host between the tile
    # phases and the claims phase (see park_column_device_values_v3)
    device_values_host: object = None


@dataclass(frozen=True, slots=True)
class VariableColumnMemberPlanV3:
    tag: str
    cell_count: int
    offset: int
    block_point: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VariableColumnGroupPlanV3:
    group_tag: str
    layout_digest: bytes
    cell_count: int
    members: tuple[VariableColumnMemberPlanV3, ...]


_COSET_PROFILE_STACK: list[str] = []


def current_pcs_coset_profile_v3() -> str:
    """Return the PCS coset profile pinned by the active protocol scope."""

    import os

    if _COSET_PROFILE_STACK:
        return _COSET_PROFILE_STACK[-1]
    if os.environ.get("VERATHOS_PCS_COSET_CHAIN") == "1":
        return "chain"
    return "v1"


class pcs_coset_profile_v3:
    """Pin the column coset profile for a protocol scope.

    The envelope-v5 capture-kv bundle runs its whole prove/verify under
    ``with pcs_coset_profile_v3("chain")`` so every column statement is
    constructed on the canonical shift chain -- explicitly versioned,
    not environment-dependent.  A mismatch between the two sides fails
    closed through the statement digest.
    """

    def __init__(self, profile: str) -> None:
        self._profile = profile

    def __enter__(self):
        _COSET_PROFILE_STACK.append(self._profile)
        return self

    def __exit__(self, *_exc) -> None:
        _COSET_PROFILE_STACK.pop()


def column_pcs_statement_v3(
    tile_digest: bytes, tag: str, variable_count: int
) -> GoldilocksMultilinearPcsStatementV3:
    # Profile priority: explicit protocol scope (envelope v5), then
    # VERATHOS_PCS_COSET_CHAIN=1 (bench A/B), else the shipped v1.
    profile = current_pcs_coset_profile_v3()
    return GoldilocksMultilinearPcsStatementV3(
        validator_binding_digest=hashlib.sha256(
            _COLUMN_DOMAIN + tile_digest + tag.encode()
        ).digest(),
        variable_count=variable_count,
        coset_profile=profile,
    )


def commit_succinct_column_v3(
    *,
    tile_digest: bytes,
    tag: str,
    values: tuple[int, ...],
    fused=None,
    canonical_input: bool = False,
) -> SuccinctColumnV3:
    """Commit one column; ``fused`` = (fold_ext, tree_ext) for GPU.

    ``canonical_input``: skip the O(N) canonicalization pass (caller
    guarantees every value is already a canonical field element)."""

    tensor_input = hasattr(values, "numel")
    n = int(values.numel()) if tensor_input else len(values)
    variable_count = n.bit_length() - 1
    if 1 << variable_count != n:
        raise ProofV3Error("tile column length must be a power of two")
    statement = column_pcs_statement_v3(tile_digest, tag, variable_count)
    if tensor_input:
        # encoded int64 device tensor (canonical mod-2^64 wrap): commit
        # directly, keep values device-resident (host copy only on demand)
        if fused is None:
            raise ProofV3Error(
                "tensor column commits need the fused backend")
        from verallm.proof_v3.native_pcs_backend import (
            fused_commit_multilinear_tree,
        )

        device = values.reshape(-1).contiguous().cuda()
        tree, _ = fused_commit_multilinear_tree(
            fused[0], fused[1], device, statement=statement)
        if hasattr(tree, "offload"):
            # levels + codeword move to host the moment the commit is
            # done: openings gather query paths CPU-side, and the
            # batched opening sumcheck runs on the RAW device values
            # kept in the column -- GB-scale tree buffers must not
            # stay resident (250k trees are the VRAM dominator)
            tree.offload()
        return SuccinctColumnV3(tag, statement, tree, None, device)
    if canonical_input:
        canonical = values
    else:
        canonical = tuple(v % GOLDILOCKS_MODULUS for v in values)
    if fused is not None:
        from verallm.proof_v3.native_goldilocks_backend import to_field_tensor
        from verallm.proof_v3.native_pcs_backend import (
            fused_commit_multilinear_tree,
        )

        device = to_field_tensor(canonical, "cuda")
        tree, _ = fused_commit_multilinear_tree(
            fused[0], fused[1], device, statement=statement)
        if hasattr(tree, "offload"):
            tree.offload()
        return SuccinctColumnV3(tag, statement, tree, canonical, device)
    tree = commit_goldilocks_multilinear_v3(
        statement=statement, evaluations=canonical)
    return SuccinctColumnV3(tag, statement, tree, canonical, None)


def commit_succinct_column_group_v3(
    *,
    tile_digest: bytes,
    group_tag: str,
    ordered: tuple[tuple[str, tuple[int, ...]], ...],
    fused=None,
    member_bindings=None,
) -> tuple[SuccinctColumnV3, dict[str, SuccinctColumnV3]]:
    """Commit many same-size columns as blocks of ONE shared tree.

    Returns ``(group_column, members)``: the group column (for the
    batch-opening argument) and per-tag member views whose ``tree`` is
    the group tree and whose terminal claims extend by ``block_point``.
    Missing blocks are zero (callers must make zero a valid cell for
    every relation that touches a pad block).
    """

    if not ordered:
        raise ProofV3Error("column group needs at least one column")

    def _length(values):
        return values.numel() if hasattr(values, "numel") else len(values)

    n = _length(ordered[0][1])
    variable_count = n.bit_length() - 1
    if 1 << variable_count != n:
        raise ProofV3Error("group column length must be a power of two")
    block_bits = (len(ordered) - 1).bit_length()
    block_count = 1 << block_bits
    for _tag, values in ordered:
        if _length(values) != n:
            raise ProofV3Error("group columns must share one cube size")
    tensor_members = any(hasattr(v, "numel") for _t, v in ordered)
    if fused is not None and tensor_members:
        # device-resident path: members may be encoded int64 tensors;
        # python tuples convert once, the concat crosses PCIe once, and
        # tensor members never materialize host values at all
        import torch

        from verallm.proof_v3.native_goldilocks_backend import (
            to_field_tensor,
        )
        from verallm.proof_v3.native_pcs_backend import (
            fused_commit_multilinear_tree,
        )

        parts = []
        for _tag, values in ordered:
            if hasattr(values, "numel"):
                parts.append(values.to("cuda"))
            else:
                parts.append(to_field_tensor(values, "cpu").to("cuda"))
        parts.append(torch.zeros(
            (block_count - len(ordered)) * n, dtype=torch.int64,
            device="cuda"))
        concat_dev = torch.cat(parts)
        statement = column_pcs_statement_v3(
            tile_digest, group_tag, variable_count + block_bits)
        tree, _ = fused_commit_multilinear_tree(
            fused[0], fused[1], concat_dev, statement=statement)
        if hasattr(tree, "offload"):
            tree.offload()
        group = SuccinctColumnV3(
            tag=group_tag, pcs_statement=statement, tree=tree,
            values=None, device_values=concat_dev)
    else:
        def _host(values):
            if hasattr(values, "numel"):
                return tuple(
                    v + (1 << 64) if v < 0 else v
                    for v in values.reshape(-1).tolist())
            return values

        ordered = tuple((tag, _host(values)) for tag, values in ordered)
        concat: list[int] = []
        for _tag, values in ordered:
            concat.extend(values)
        concat.extend([0] * ((block_count - len(ordered)) * n))
        group = commit_succinct_column_v3(
            tile_digest=tile_digest, tag=group_tag, values=tuple(concat),
            fused=fused, canonical_input=True)
    members: dict[str, SuccinctColumnV3] = {}
    for index, (tag, values) in enumerate(ordered):
        block_point = tuple(
            (index >> j) & 1 for j in range(block_bits))
        device_slice = None
        if group.device_values is not None:
            device_slice = group.device_values[index * n:(index + 1) * n]
        m_digest, m_tag = (
            member_bindings[tag] if member_bindings is not None
            else (tile_digest, tag))
        members[tag] = SuccinctColumnV3(
            tag=tag,
            pcs_statement=column_pcs_statement_v3(
                m_digest, m_tag, variable_count),
            tree=group.tree,
            values=None if hasattr(values, "numel") else values,
            device_values=device_slice,
            group_tag=group_tag,
            block_point=block_point,
        )
    return group, members


def plan_succinct_variable_column_groups_v3(
    *,
    tile_digest: bytes,
    group_tag_prefix: str,
    ordered_sizes: tuple[tuple[str, int], ...],
    max_group_cells: int,
) -> tuple[VariableColumnGroupPlanV3, ...]:
    """Pack power-of-two columns into canonical prefix-selected subcubes.

    Descending-size packing guarantees every member starts at an offset
    aligned to its own size. The member is therefore exactly the subcube
    selected by the high-bit ``block_point`` appended by the existing batch
    opening alias mechanism.
    """

    if (
        not isinstance(tile_digest, bytes)
        or len(tile_digest) != 32
        or not isinstance(group_tag_prefix, str)
        or not group_tag_prefix
        or len(group_tag_prefix.encode()) > 192
        or not isinstance(max_group_cells, int)
        or isinstance(max_group_cells, bool)
        or max_group_cells < 1
        or max_group_cells > 1 << 30
        or max_group_cells & (max_group_cells - 1)
    ):
        raise ProofV3Error("variable column group parameters are malformed")
    try:
        normalized = tuple(
            (str(tag), int(cell_count))
            for tag, cell_count in ordered_sizes
        )
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "variable column group inventory is malformed") from exc
    tags = tuple(tag for tag, _count in normalized)
    if (
        not normalized
        or any(not tag or len(tag.encode()) > 255 for tag in tags)
        or len(tags) != len(set(tags))
        or any(
            count < 2
            or count > max_group_cells
            or count & (count - 1)
            for _tag, count in normalized
        )
    ):
        raise ProofV3Error("variable column group inventory is malformed")
    pending = sorted(normalized, key=lambda item: (-item[1], item[0]))
    bins: list[list[tuple[str, int]]] = []
    used: list[int] = []
    for item in pending:
        for index, current in enumerate(used):
            if current + item[1] <= max_group_cells:
                bins[index].append(item)
                used[index] += item[1]
                break
        else:
            bins.append([item])
            used.append(item[1])

    plans = []
    for group_index, (items, active_cells) in enumerate(
        zip(bins, used, strict=True)
    ):
        group_cells = 1 << (active_cells - 1).bit_length()
        group_variables = group_cells.bit_length() - 1
        offset = 0
        members = []
        layout = bytearray(
            _VARIABLE_GROUP_LAYOUT_DOMAIN
            + tile_digest
            + struct.pack("<III", group_index, group_cells, len(items))
        )
        prefix = group_tag_prefix.encode()
        layout.extend(struct.pack("<H", len(prefix)))
        layout.extend(prefix)
        for tag, cell_count in items:
            if offset % cell_count:
                raise ProofV3Error(
                    "variable column packing lost subcube alignment")
            member_variables = cell_count.bit_length() - 1
            block_index = offset >> member_variables
            block_point = tuple(
                (block_index >> bit) & 1
                for bit in range(group_variables - member_variables)
            )
            encoded_tag = tag.encode()
            layout.extend(
                struct.pack("<HII", len(encoded_tag), cell_count, offset))
            layout.extend(encoded_tag)
            members.append(
                VariableColumnMemberPlanV3(
                    tag=tag,
                    cell_count=cell_count,
                    offset=offset,
                    block_point=block_point,
                )
            )
            offset += cell_count
        plans.append(
            VariableColumnGroupPlanV3(
                group_tag=f"{group_tag_prefix}/{group_index}",
                layout_digest=hashlib.sha256(bytes(layout)).digest(),
                cell_count=group_cells,
                members=tuple(members),
            )
        )
    return tuple(plans)


def commit_succinct_variable_column_groups_v3(
    *,
    tile_digest: bytes,
    group_tag_prefix: str,
    ordered: tuple[tuple[str, object], ...],
    max_group_cells: int,
    fused=None,
    member_bindings=None,
) -> tuple[
    tuple[SuccinctColumnV3, ...],
    dict[str, SuccinctColumnV3],
    tuple[VariableColumnGroupPlanV3, ...],
]:
    """Commit heterogeneous power-of-two members under few shared roots."""

    def _length(values):
        return int(values.numel()) if hasattr(values, "numel") else len(values)

    values_by_tag = {tag: values for tag, values in ordered}
    if len(values_by_tag) != len(ordered):
        raise ProofV3Error("variable column tags are duplicated")
    plans = plan_succinct_variable_column_groups_v3(
        tile_digest=tile_digest,
        group_tag_prefix=group_tag_prefix,
        ordered_sizes=tuple(
            (tag, _length(values)) for tag, values in ordered),
        max_group_cells=max_group_cells,
    )
    tensor_members = any(hasattr(values, "numel") for _tag, values in ordered)
    groups = []
    member_views: dict[str, SuccinctColumnV3] = {}
    for plan in plans:
        if fused is not None and tensor_members:
            import torch

            from verallm.proof_v3.native_goldilocks_backend import (
                to_field_tensor,
            )
            from verallm.proof_v3.native_pcs_backend import (
                fused_commit_multilinear_tree,
            )

            parts = []
            for member in plan.members:
                values = values_by_tag[member.tag]
                parts.append(
                    values.reshape(-1).to("cuda")
                    if hasattr(values, "numel")
                    else to_field_tensor(values, "cuda")
                )
            active = sum(member.cell_count for member in plan.members)
            if active < plan.cell_count:
                parts.append(torch.zeros(
                    plan.cell_count - active,
                    dtype=torch.int64,
                    device="cuda",
                ))
            concat = torch.cat(parts)
            statement = column_pcs_statement_v3(
                plan.layout_digest,
                plan.group_tag,
                plan.cell_count.bit_length() - 1,
            )
            tree, _ = fused_commit_multilinear_tree(
                fused[0], fused[1], concat, statement=statement)
            if hasattr(tree, "offload"):
                tree.offload()
            group = SuccinctColumnV3(
                tag=plan.group_tag,
                pcs_statement=statement,
                tree=tree,
                values=None,
                device_values=concat,
            )
        else:
            concat_values = []
            for member in plan.members:
                values = values_by_tag[member.tag]
                if hasattr(values, "numel"):
                    values = values.reshape(-1).tolist()
                concat_values.extend(
                    value + (1 << 64) if value < 0 else value
                    for value in values
                )
            concat_values.extend(
                [0] * (plan.cell_count - len(concat_values)))
            group = commit_succinct_column_v3(
                tile_digest=plan.layout_digest,
                tag=plan.group_tag,
                values=tuple(concat_values),
                fused=fused,
                canonical_input=True,
            )
        groups.append(group)
        for member in plan.members:
            values = values_by_tag[member.tag]
            binding_digest, binding_tag = (
                member_bindings[member.tag]
                if member_bindings is not None
                else (tile_digest, member.tag)
            )
            device_slice = (
                None
                if group.device_values is None
                else group.device_values[
                    member.offset : member.offset + member.cell_count
                ]
            )
            member_views[member.tag] = SuccinctColumnV3(
                tag=member.tag,
                pcs_statement=column_pcs_statement_v3(
                    binding_digest,
                    binding_tag,
                    member.cell_count.bit_length() - 1,
                ),
                tree=group.tree,
                values=(
                    None
                    if hasattr(values, "numel")
                    else tuple(values)
                ),
                device_values=device_slice,
                group_tag=plan.group_tag,
                block_point=member.block_point,
            )
    return tuple(groups), member_views, plans


@dataclass(frozen=True, slots=True)
class SuccinctEqFoldProofV3:
    """sum eq(z,.) * column == claimed, opened against the shared root."""

    claimed_sum: int
    round_polynomials: tuple[tuple[int, int, int, int], ...]
    opening: GoldilocksMultilinearOpeningProofV3


def _eqfold_seed(tile_digest, tag, commitment, z_point, validator_nonce):
    return hashlib.sha256(
        _EQFOLD_DOMAIN
        + tile_digest
        + tag.encode()
        + commitment
        + b"".join(v.to_bytes(8, "little") for v in z_point)
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()


def prove_succinct_eq_fold_v3(
    *,
    tile_digest: bytes,
    column: SuccinctColumnV3,
    z_point: tuple[int, ...],
    validator_nonce: bytes,
    fused=None,
    collector=None,
) -> SuccinctEqFoldProofV3:
    rounds: list[tuple[int, int, int, int]] = []
    challenges: list[int] = []
    if fused is not None and column.device_values is not None:
        import torch

        from verallm.proof_v3.native_goldilocks_backend import (
            gl_mul_t,
            gl_sum_t,
            to_field_tensor,
        )
        from verallm.proof_v3.native_pcs_backend import (
            _encode_challenge,
        )

        a = column.device_values
        f = to_field_tensor((1,), "cuda")
        for z in z_point:
            z_t = to_field_tensor((z % GOLDILOCKS_MODULUS,), "cuda")
            one_minus = to_field_tensor(
                ((1 - z) % GOLDILOCKS_MODULUS,), "cuda")
            f = torch.cat(
                (gl_mul_t(f, one_minus.expand_as(f)),
                 gl_mul_t(f, z_t.expand_as(f))))
        claimed = gl_sum_t(gl_mul_t(a, f))
        transcript = _eqfold_seed(
            tile_digest, column.tag, column.tree.commitment, z_point,
            validator_nonce)
        if hasattr(fused[0], "fs_round"):
            # device Fiat-Shamir; CUDA-graph replay per cube size when
            # available (near-zero launch overhead), inline enqueue else
            from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
                _CHALLENGE_DOMAIN,
            )

            n_rounds = a.numel().bit_length() - 1
            label = b"eqfold"
            graphed = None
            if hasattr(fused[0], "lerp_fold_ptr"):
                from verallm.proof_v3.native_pcs_backend import (
                    _eqfold_rounds_graphed,
                )

                try:
                    graphed = _eqfold_rounds_graphed(
                        fused[0], a, f, transcript,
                        _CHALLENGE_DOMAIN + label,
                        len(_CHALLENGE_DOMAIN), len(label))
                except RuntimeError:
                    graphed = None
            if graphed is not None:
                rounds_buf, chal_buf, a = graphed
                torch.cuda.synchronize()
            else:
                t_buf = torch.tensor(
                    list(transcript), dtype=torch.uint8, device="cuda")
                dom_label = torch.tensor(
                    list(_CHALLENGE_DOMAIN + label), dtype=torch.uint8,
                    device="cuda")
                rounds_buf = torch.zeros(
                    (n_rounds, 4), dtype=torch.int64, device="cuda")
                chal_buf = torch.zeros(
                    n_rounds, dtype=torch.int64, device="cuda")
                for r in range(n_rounds):
                    partials = fused[0].round_partials(a, f)
                    fused[0].fs_round(
                        partials, 3, t_buf, dom_label,
                        len(_CHALLENGE_DOMAIN), len(label), r + 1,
                        rounds_buf[r], chal_buf[r:r + 1])
                    a = fused[0].lerp_fold_ptr(a, chal_buf[r:r + 1])
                    f = fused[0].lerp_fold_ptr(f, chal_buf[r:r + 1])
                torch.cuda.synchronize()

            def _dec(v):
                return v + (1 << 64) if v < 0 else v

            rounds = [
                tuple(_dec(v) for v in row)
                for row in rounds_buf.cpu().tolist()
            ]
            challenges = [_dec(v) for v in chal_buf.cpu().tolist()]
        else:
            while a.numel() > 1:
                partials = fused[0].round_partials(a, f)
                torch.cuda.synchronize()
                from verallm.proof_v3.native_cuda_fold_backend import (
                    _sum_partials,
                )

                g0, g1, g2 = _sum_partials(partials)
                # degree-2 cell -> extend to the 4-eval wire exactly
                g3 = (3 * g2 - 3 * g1 + g0) % GOLDILOCKS_MODULUS
                evals = (g0, g1, g2, g3)
                rounds.append(evals)
                transcript = hashlib.sha256(
                    transcript
                    + b"".join(v.to_bytes(8, "little") for v in evals)
                ).digest()
                challenge = _derive(transcript, b"eqfold", len(rounds))
                challenges.append(challenge)
                a = fused[0].lerp_fold(a, _encode_challenge(challenge))
                f = fused[0].lerp_fold(f, _encode_challenge(challenge))
    else:
        eq = _eq_table(z_point)
        values = list(column.values)
        claimed = 0
        for v, q in zip(values, eq, strict=True):
            claimed = (claimed + v * q) % GOLDILOCKS_MODULUS
        transcript = _eqfold_seed(
            tile_digest, column.tag, column.tree.commitment, z_point,
            validator_nonce)
        work_v, work_q = values, list(eq)
        while len(work_v) > 1:
            half = len(work_v) // 2
            evals4 = [0, 0, 0, 0]
            for i in range(half):
                v_lo, v_hi = work_v[i], work_v[half + i]
                q_lo, q_hi = work_q[i], work_q[half + i]
                for z in range(4):
                    vv = (v_lo + z * (v_hi - v_lo)) % GOLDILOCKS_MODULUS
                    qq = (q_lo + z * (q_hi - q_lo)) % GOLDILOCKS_MODULUS
                    evals4[z] = (evals4[z] + vv * qq) % GOLDILOCKS_MODULUS
            evals = tuple(evals4)
            rounds.append(evals)
            transcript = hashlib.sha256(
                transcript
                + b"".join(v.to_bytes(8, "little") for v in evals)
            ).digest()
            challenge = _derive(transcript, b"eqfold", len(rounds))
            challenges.append(challenge)
            work_v = [
                (work_v[i] + challenge * (work_v[half + i] - work_v[i]))
                % GOLDILOCKS_MODULUS for i in range(half)
            ]
            work_q = [
                (work_q[i] + challenge * (work_q[half + i] - work_q[i]))
                % GOLDILOCKS_MODULUS for i in range(half)
            ]
    point = tuple(reversed(challenges))
    if collector is not None:
        if fused is not None and column.device_values is not None:
            terminal = int(a.cpu().item())
            if terminal < 0:
                terminal += 1 << 64
        else:
            terminal = work_v[0]
        opening = collector.defer(column.tag, point, terminal)
    elif fused is not None and column.device_values is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_open_goldilocks_multilinear_v3,
        )

        opening = fused_open_goldilocks_multilinear_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            statement=column.pcs_statement, tree=column.tree,
            evaluations_device=column.device_values, point=point,
            validator_nonce=validator_nonce)
    else:
        opening = open_goldilocks_multilinear_v3(
            statement=column.pcs_statement, tree=column.tree,
            evaluations=column.values, point=point,
            validator_nonce=validator_nonce)
    return SuccinctEqFoldProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        opening=opening,
    )


_BATCH_FOLD_GRAPHS: dict = {}


def prove_succinct_eq_folds_batched_v3(
    *,
    tile_digest: bytes,
    members: tuple[SuccinctColumnV3, ...],
    group_device,
    z_point: tuple[int, ...],
    validator_nonce: bytes,
    fused,
    collector,
) -> tuple[SuccinctEqFoldProofV3, ...]:
    """ALL same-point eq-folds of one group in ONE kernel sequence.

    ``group_device`` holds the member columns contiguously (rows of a
    matrix). Per-argument transcripts/challenges evolve in one batched
    FS kernel; byte-identical per-argument results.
    """

    import torch

    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        _CHALLENGE_DOMAIN,
        _eq_table,
    )
    from verallm.proof_v3.native_goldilocks_backend import to_field_tensor
    from verallm.proof_v3.native_pcs_backend import _u8_tensor_cached

    n_args = len(members)
    cells = members[0].values and len(members[0].values) or (
        members[0].device_values.numel())
    n_rounds = cells.bit_length() - 1
    label = b"eqfold"
    dom_label = _u8_tensor_cached(_CHALLENGE_DOMAIN + label)
    seeds = b"".join(
        _eqfold_seed(tile_digest, member.tag, member.tree.commitment,
                     z_point, validator_nonce)
        for member in members
    )
    from verallm.proof_v3.native_pcs_backend import _device_eq_table
    eq_dev = _device_eq_table(z_point)
    a_src = group_device[: n_args * cells].contiguous()
    ext = fused[0]
    entry_key = (n_args, cells, label)
    entry = _BATCH_FOLD_GRAPHS.get(entry_key)
    if entry is None and n_args * cells > (1 << 23):
        entry = False
        _BATCH_FOLD_GRAPHS[entry_key] = False
    if entry is None and len(_BATCH_FOLD_GRAPHS) < 24 and hasattr(
        torch.cuda, "CUDAGraph"
    ):
        a_s = torch.empty_like(a_src)
        f_s = torch.empty(n_args * cells, dtype=torch.int64, device="cuda")
        t_s = torch.empty(32 * n_args, dtype=torch.uint8, device="cuda")
        rounds_s = torch.zeros(
            (n_rounds, n_args, 4), dtype=torch.int64, device="cuda")
        chal_s = torch.zeros(
            (n_rounds, n_args), dtype=torch.int64, device="cuda")

        def _loop():
            am, fm = a_s, f_s
            half_ = cells // 2
            for r in range(n_rounds):
                blocks = max(1, min(32, (half_ + 255) // 256))
                partials = ext.round_partials_b(am, fm, n_args, blocks)
                ext.fs_round_b(
                    partials, blocks, t_s, dom_label,
                    len(_CHALLENGE_DOMAIN), len(label), r + 1, n_args,
                    rounds_s[r].reshape(-1), chal_s[r])
                am = ext.lerp_fold_b(am, chal_s[r], n_args, blocks)
                fm = ext.lerp_fold_b(fm, chal_s[r], n_args, blocks)
                half_ //= 2
            return am

        def _fill(a_val, f_val, t_val):
            a_s.copy_(a_val)
            f_s.copy_(f_val)
            t_s.copy_(t_val)

        try:
            _fill(a_src, eq_dev.repeat(n_args),
                  torch.tensor(list(seeds), dtype=torch.uint8,
                               device="cuda"))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _loop()
            torch.cuda.current_stream().wait_stream(stream)
            _fill(a_src, eq_dev.repeat(n_args),
                  torch.tensor(list(seeds), dtype=torch.uint8,
                               device="cuda"))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                a_out = _loop()
            graph.replay()
            entry = (graph, _fill, rounds_s, chal_s, a_out)
            _BATCH_FOLD_GRAPHS[entry_key] = entry
        except RuntimeError:
            entry = None
            _BATCH_FOLD_GRAPHS[entry_key] = False
    elif entry is False:
        entry = None
    if entry:
        graph, fill, rounds_buf, chal_buf, a_mat = entry
        fill(a_src, eq_dev.repeat(n_args),
             torch.tensor(list(seeds), dtype=torch.uint8, device="cuda"))
        graph.replay()
    else:
        transcripts = torch.tensor(
            list(seeds), dtype=torch.uint8, device="cuda")
        a_mat = a_src
        f_mat = eq_dev.repeat(n_args).contiguous()
        rounds_buf = torch.zeros(
            (n_rounds, n_args, 4), dtype=torch.int64, device="cuda")
        chal_buf = torch.zeros(
            (n_rounds, n_args), dtype=torch.int64, device="cuda")
        half = cells // 2
        for r in range(n_rounds):
            blocks = max(1, min(32, (half + 255) // 256))
            partials = ext.round_partials_b(a_mat, f_mat, n_args, blocks)
            ext.fs_round_b(
                partials, blocks, transcripts, dom_label,
                len(_CHALLENGE_DOMAIN), len(label), r + 1, n_args,
                rounds_buf[r].reshape(-1), chal_buf[r])
            a_mat = ext.lerp_fold_b(a_mat, chal_buf[r], n_args, blocks)
            f_mat = ext.lerp_fold_b(f_mat, chal_buf[r], n_args, blocks)
            half //= 2
    torch.cuda.synchronize()

    def _dec(v):
        return v + (1 << 64) if v < 0 else v

    rounds_host = rounds_buf.cpu().tolist()
    chal_host = chal_buf.cpu().tolist()
    terminals = a_mat.cpu().tolist()
    proofs = []
    for m, member in enumerate(members):
        rounds_m = tuple(
            tuple(_dec(v) for v in rounds_host[r][m])
            for r in range(n_rounds)
        )
        challenges_m = [_dec(chal_host[r][m]) for r in range(n_rounds)]
        claimed = (rounds_m[0][0] + rounds_m[0][1]) % GOLDILOCKS_MODULUS
        point = tuple(reversed(challenges_m))
        opening = collector.defer(member.tag, point, _dec(terminals[m]))
        proofs.append(SuccinctEqFoldProofV3(
            claimed_sum=claimed,
            round_polynomials=rounds_m,
            opening=opening,
        ))
    return tuple(proofs)


def verify_succinct_eq_fold_v3(
    proof: object,
    *,
    tile_digest: bytes,
    tag: str,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    z_point: tuple[int, ...],
    validator_nonce: bytes,
    checker=None,
) -> int:
    """Verify and return the claimed ``sum eq(z,.) column`` value."""

    try:
        if not isinstance(proof, SuccinctEqFoldProofV3):
            raise ProofV3VerificationError("eq-fold proof type is wrong")
        n = pcs_statement.variable_count
        if len(proof.round_polynomials) != n or len(z_point) != n:
            raise ProofV3VerificationError("eq-fold arity is wrong")
        transcript = _eqfold_seed(
            tile_digest, tag, _fixed32(commitment, "commitment"), z_point,
            validator_nonce)
        running = proof.claimed_sum % GOLDILOCKS_MODULUS
        challenges: list[int] = []
        compiled = None
        try:
            from verallm.proof_v3.c_multiopen import replay_rounds4
            from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
                _CHALLENGE_DOMAIN as _LOGUP_DOMAIN,
            )

            compiled = replay_rounds4(
                transcript, running,
                tuple(tuple(int(v) for v in row)
                      for row in proof.round_polynomials),
                _LOGUP_DOMAIN, b"eqfold", False, 1)
        except ImportError:
            compiled = None
        if isinstance(compiled, tuple):
            challenges_t, running, transcript = compiled
            challenges = list(challenges_t)
        elif isinstance(compiled, int):
            raise ProofV3VerificationError(
                "eq-fold round replay fails")
        else:
            for evals in proof.round_polynomials:
                evals = tuple(v % GOLDILOCKS_MODULUS for v in evals)
                if (evals[0] + evals[1]) % GOLDILOCKS_MODULUS != running:
                    raise ProofV3VerificationError(
                        "eq-fold round does not match the running sum")
                transcript = hashlib.sha256(
                    transcript
                    + b"".join(v.to_bytes(8, "little") for v in evals)
                ).digest()
                challenge = _derive(
                    transcript, b"eqfold", len(challenges) + 1)
                challenges.append(challenge)
                running = _lagrange_0123(evals, challenge)
        point = tuple(reversed(challenges))
        eq_value = _eq_eval(point, tuple(z_point))
        expected = (
            proof.opening.claimed_value * eq_value % GOLDILOCKS_MODULUS
        )
        if running != expected:
            raise ProofV3VerificationError("eq-fold terminal coupling fails")
        if checker is not None:
            checker.expect(tag, point, proof.opening.claimed_value)
        else:
            verify_goldilocks_multilinear_opening_v3(
                proof.opening, statement=pcs_statement, commitment=commitment,
                point=point, expected_value=proof.opening.claimed_value,
                validator_nonce=validator_nonce)
        return proof.claimed_sum % GOLDILOCKS_MODULUS
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError("eq-fold proof is malformed") from exc


def derive_tile_eq_point_v3(
    tile_digest: bytes,
    commitments: tuple[bytes, ...],
    validator_nonce: bytes,
    variable_count: int,
    label: bytes = b"z",
) -> tuple[int, ...]:
    seed = hashlib.sha256(
        tile_digest
        + b"".join(commitments)
        + _fixed32(validator_nonce, "validator_nonce")
    ).digest()
    return tuple(
        _derive(seed, label, j) for j in range(variable_count)
    )


def _public_fold_layout_v3(
    *,
    column_variable_count: int,
    product_variable_count: int | None,
    point_map: tuple[int, ...] | None,
    factor_digest: bytes,
) -> tuple[int, tuple[int, ...] | None, bytes]:
    """Validate and transcript-bind an optional broadcast layout.

    The legacy/default mode deliberately returns ``factor_digest`` unchanged,
    preserving its transcript byte-for-byte.  Broadcast mode proves over a
    larger product cube while opening the smaller committed column at the
    mapped terminal point, so both the cube arity and map are protocol data.
    """

    if point_map is None:
        if product_variable_count is not None:
            raise ProofV3Error(
                "public-fold product arity needs a point map")
        return column_variable_count, None, factor_digest
    if product_variable_count is None:
        raise ProofV3Error(
            "broadcast public fold needs the product arity")
    if (
        not isinstance(product_variable_count, int)
        or product_variable_count < column_variable_count
        or product_variable_count > 30
    ):
        raise ProofV3Error("public-fold product arity is invalid")
    mapped = tuple(point_map)
    if (
        len(mapped) != column_variable_count
        or any(not isinstance(index, int) for index in mapped)
        or len(set(mapped)) != len(mapped)
        or any(index < 0 or index >= product_variable_count
               for index in mapped)
    ):
        raise ProofV3Error("public-fold point map is invalid")
    layout_digest = hashlib.sha256(
        _PUBLIC_FOLD_LAYOUT_DOMAIN
        + struct.pack("<II", product_variable_count, len(mapped))
        + b"".join(struct.pack("<I", index) for index in mapped)
        + factor_digest
    ).digest()
    return product_variable_count, mapped, layout_digest


def prove_succinct_public_fold_v3(
    *,
    tile_digest: bytes,
    column: SuccinctColumnV3,
    factor: tuple[int, ...],
    label: str,
    validator_nonce: bytes,
    fused=None,
    collector=None,
    structured_binding: bytes | None = None,
    product_values=None,
    factor_device=None,
    product_variable_count: int | None = None,
    point_map: tuple[int, ...] | None = None,
) -> SuccinctEqFoldProofV3:
    """sum factor[i]*column[i] for an ARBITRARY public factor vector.

    Rounds are identical to the eq-fold; the verifier evaluates the
    factor MLE at the terminal point directly (O(N) -- use only on
    small cubes such as per-layer attention score grids).

    ``product_values`` + ``point_map`` enable broadcast-free composition:
    the rounds use the column's values broadcast onto a larger product cube,
    while the terminal claim opens the original smaller commitment at the
    mapped sub-point.  This mode requires a deferred batch-opening collector.
    """

    column_variables = column.pcs_statement.variable_count
    if product_values is not None and point_map is None:
        raise ProofV3Error(
            "public-fold product values need a point map")
    if point_map is not None and collector is None:
        raise ProofV3Error(
            "broadcast public folds require a deferred collector")
    if factor_device is not None and structured_binding is None:
        raise ProofV3Error(
            "device public-fold factors need a structured binding")
    source_values = (
        product_values if product_values is not None
        else (
            column.values if column.values is not None
            else column.device_values
        )
    )
    # Preserve the legacy fast path: committed columns commonly retain both
    # host values and a CUDA tensor.  Host values canonicalize the transcript
    # input, while rounds still run over the existing device tensor.
    round_device_values = (
        product_values
        if product_values is not None
        and hasattr(product_values, "is_cuda")
        and product_values.is_cuda
        else (
            column.device_values
            if product_values is None
            else None
        )
    )
    if source_values is None:
        raise ProofV3Error("public-fold column values are unavailable")

    n_cells = (
        int(source_values.numel())
        if hasattr(source_values, "numel")
        else len(source_values))
    preliminary_digest = (
        structured_binding if structured_binding is not None
        else None)
    round_variables, mapped_point, _unused = _public_fold_layout_v3(
        column_variable_count=column_variables,
        product_variable_count=product_variable_count,
        point_map=point_map,
        # The actual factor digest is supplied below after canonicalization.
        factor_digest=preliminary_digest or b"",
    )
    if n_cells != 1 << round_variables:
        raise ProofV3Error("public-fold product value arity is wrong")
    use_np = n_cells >= 2048
    if use_np:
        import numpy as _np

        from verallm.proof_v3.goldilocks_numpy import (
            gl_add_np, gl_mul_np, gl_sum_np, public_fold_round_np,
        )

        if factor_device is not None:
            # The device-factor path computes the claim and every round on
            # CUDA. Avoid an otherwise redundant full-cube D2H copy merely to
            # initialize NumPy arrays that are never consumed.
            work_vn = None
        elif not hasattr(source_values, "numel"):
            work_vn = _np.array(source_values, dtype=_np.uint64)
        else:
            # canonical mod-2^64 int64 encoding: the u64 REINTERPRET is
            # the decode (two's-complement bit pattern == canonical value)
            work_vn = _np.ascontiguousarray(
                source_values.cpu().numpy()).view(_np.uint64).copy()
        if factor_device is not None:
            if (
                not hasattr(factor_device, "numel")
                or not getattr(factor_device, "is_cuda", False)
                or int(factor_device.numel()) != n_cells
                or round_device_values is None
                or not getattr(round_device_values, "is_cuda", False)
            ):
                raise ProofV3Error(
                    "device public-fold factor shape is wrong")
            work_fn = None
        elif isinstance(factor, _np.ndarray):
            work_fn = _np.ascontiguousarray(factor, dtype=_np.uint64)
        else:
            work_fn = _np.array(
                [int(v) % GOLDILOCKS_MODULUS for v in factor],
                dtype=_np.uint64)
        if (
            work_fn is not None
            and (work_vn is None or work_vn.shape != work_fn.shape)
        ):
            raise ProofV3Error("public-fold factor length mismatch")
        claimed = (
            None
            if work_fn is None
            else gl_sum_np(gl_mul_np(work_vn.copy(), work_fn.copy()))
        )
        # same little-endian bytes the python path digests
        factor_digest_base = (
            structured_binding if structured_binding is not None
            else hashlib.sha256(
                work_fn.astype("<u8").tobytes()).digest())
        values = f_values = None
    else:
        if not hasattr(source_values, "numel"):
            values = list(source_values)
        else:
            values = [
                v + (1 << 64) if v < 0 else v
                for v in source_values.detach().cpu().tolist()]
        if factor_device is not None:
            if (
                not hasattr(factor_device, "numel")
                or not getattr(factor_device, "is_cuda", False)
                or int(factor_device.numel()) != n_cells
            ):
                raise ProofV3Error(
                    "device public-fold factor shape is wrong")
            f_values = [
                (
                    int(value) + (1 << 64)
                    if int(value) < 0
                    else int(value)
                )
                for value in factor_device.detach().cpu().tolist()
            ]
        else:
            f_values = [int(v) % GOLDILOCKS_MODULUS for v in factor]
        if len(values) != len(f_values):
            raise ProofV3Error("public-fold factor length mismatch")
        claimed = 0
        for v, f in zip(values, f_values, strict=True):
            claimed = (claimed + v * f) % GOLDILOCKS_MODULUS
        factor_digest_base = (
            structured_binding if structured_binding is not None
            else hashlib.sha256(
                b"".join(v.to_bytes(8, "little")
                         for v in f_values)).digest())
    round_variables, mapped_point, factor_digest = _public_fold_layout_v3(
        column_variable_count=column_variables,
        product_variable_count=product_variable_count,
        point_map=point_map,
        factor_digest=factor_digest_base,
    )
    if factor_device is not None:
        from verallm.proof_v3.native_goldilocks_backend import (
            gl_mul_t,
            gl_sum_t,
        )

        claimed = gl_sum_t(
            gl_mul_t(round_device_values, factor_device))
    transcript = hashlib.sha256(
        _EQFOLD_DOMAIN + b"PUB" + tile_digest + label.encode()
        + column.tree.commitment + factor_digest
        + _fixed32(validator_nonce, "validator_nonce")).digest()
    rounds: list[tuple[int, int, int, int]] = []
    challenges: list[int] = []
    if (
        use_np
        and fused is not None
        and round_device_values is not None
        and hasattr(round_device_values, "is_cuda")
        and round_device_values.is_cuda
    ):
        # DEVICE rounds, byte-identical to the numpy loop: the product
        # v(z)*f(z) is quadratic, so the 4th eval is the exact
        # finite-difference extrapolation (g3 = 3g2 - 3g1 + g0); the
        # transcript/challenge sequence is unchanged.  The numpy rounds
        # over multi-million-cell cubes were the prove hot spot
        # (~11s/layer at 250k across the seven public folds).
        import numpy as _np3
        import torch as _torch

        from verallm.proof_v3.native_cuda_fold_backend import (
            _sum_partials,
        )
        from verallm.proof_v3.native_pcs_backend import (
            _encode_challenge,
        )

        v_dev = round_device_values
        f_dev = (
            factor_device
            if factor_device is not None
            else _torch.from_numpy(
                work_fn.view(_np3.int64).copy()).cuda()
        )
        while v_dev.numel() > 1:
            partials = fused[0].round_partials(v_dev, f_dev)
            _torch.cuda.synchronize()
            g0, g1, g2 = _sum_partials(partials)
            g3 = (3 * g2 - 3 * g1 + g0) % GOLDILOCKS_MODULUS
            evals = (g0, g1, g2, g3)
            rounds.append(evals)
            transcript = hashlib.sha256(
                transcript
                + b"".join(v.to_bytes(8, "little") for v in evals)
            ).digest()
            challenge = _derive(transcript, b"pubfold", len(rounds))
            challenges.append(challenge)
            encoded = _encode_challenge(challenge)
            v_dev = fused[0].lerp_fold(v_dev, encoded)
            f_dev = fused[0].lerp_fold(f_dev, encoded)
        terminal = int(v_dev.cpu()[0].item())
        terminal = terminal + (1 << 64) if terminal < 0 else terminal
        point = tuple(reversed(challenges))
        opening_point = (
            point if mapped_point is None
            else tuple(point[index] for index in mapped_point)
        )
        if collector is not None:
            opening = collector.defer(column.tag, opening_point, terminal)
        else:
            from verallm.proof_v3.native_pcs_backend import (
                fused_open_goldilocks_multilinear_v3,
            )

            opening = fused_open_goldilocks_multilinear_v3(
                fold_extension=fused[0], tree_extension=fused[1],
                statement=column.pcs_statement, tree=column.tree,
                evaluations_device=column.device_values, point=point,
                validator_nonce=validator_nonce)
        return SuccinctEqFoldProofV3(
            claimed_sum=int(claimed),
            round_polynomials=tuple(rounds), opening=opening)
    work_v, work_f = values, f_values
    while (len(work_v) if not use_np else work_vn.shape[0]) > 1:
        if use_np:
            # exact uint64-numpy round (fuzz-identical to the python loop;
            # the host rounds over 2^20 cells were the chunk-prove hot spot)
            evals, v_lo, _v_hi, dv, f_lo, _f_hi, df = (
                public_fold_round_np(work_vn, work_fn))
        else:
            half = len(work_v) // 2
            evals4 = [0, 0, 0, 0]
            for i in range(half):
                v_lo, v_hi = work_v[i], work_v[half + i]
                f_lo, f_hi = work_f[i], work_f[half + i]
                for z in range(4):
                    vv = (v_lo + z * (v_hi - v_lo)) % GOLDILOCKS_MODULUS
                    ff = (f_lo + z * (f_hi - f_lo)) % GOLDILOCKS_MODULUS
                    evals4[z] = (
                        evals4[z] + vv * ff) % GOLDILOCKS_MODULUS
            evals = tuple(evals4)
        rounds.append(evals)
        transcript = hashlib.sha256(
            transcript
            + b"".join(v.to_bytes(8, "little") for v in evals)).digest()
        challenge = _derive(transcript, b"pubfold", len(rounds))
        challenges.append(challenge)
        if use_np:
            import numpy as _np2

            c = _np2.broadcast_to(
                _np2.uint64(challenge), dv.shape).copy()
            work_vn = gl_add_np(v_lo, gl_mul_np(dv, c))
            work_fn = gl_add_np(f_lo, gl_mul_np(df, c))
            continue
        work_v = [
            (work_v[i] + challenge * (work_v[half + i] - work_v[i]))
            % GOLDILOCKS_MODULUS for i in range(half)]
        work_f = [
            (work_f[i] + challenge * (work_f[half + i] - work_f[i]))
            % GOLDILOCKS_MODULUS for i in range(half)]
    if use_np:
        work_v = [int(work_vn[0])]
    point = tuple(reversed(challenges))
    opening_point = (
        point if mapped_point is None
        else tuple(point[index] for index in mapped_point)
    )
    if collector is not None:
        opening = collector.defer(column.tag, opening_point, work_v[0])
    elif fused is not None and column.device_values is not None:
        from verallm.proof_v3.native_pcs_backend import (
            fused_open_goldilocks_multilinear_v3,
        )

        opening = fused_open_goldilocks_multilinear_v3(
            fold_extension=fused[0], tree_extension=fused[1],
            statement=column.pcs_statement, tree=column.tree,
            evaluations_device=column.device_values, point=point,
            validator_nonce=validator_nonce)
    else:
        opening = open_goldilocks_multilinear_v3(
            statement=column.pcs_statement, tree=column.tree,
            evaluations=column.values, point=point,
            validator_nonce=validator_nonce)
    return SuccinctEqFoldProofV3(
        claimed_sum=claimed, round_polynomials=tuple(rounds),
        opening=opening)


def _mle_eval_msb_local(values, point) -> int:
    if len(values) >= 2048:
        try:
            from verallm.proof_v3.goldilocks_numpy import mle_eval_msb_np

            return mle_eval_msb_np(
                tuple(v % GOLDILOCKS_MODULUS for v in values), tuple(point))
        except ImportError:
            pass
    work = [v % GOLDILOCKS_MODULUS for v in values]
    for r in point:
        half = len(work) // 2
        work = [
            (work[i] + r * (work[half + i] - work[i])) % GOLDILOCKS_MODULUS
            for i in range(half)]
    return work[0]


def verify_succinct_public_fold_v3(
    proof: object,
    *,
    tile_digest: bytes,
    label: str,
    pcs_statement: GoldilocksMultilinearPcsStatementV3,
    commitment: bytes,
    factor: tuple[int, ...],
    validator_nonce: bytes,
    checker=None,
    tag: str | None = None,
    factor_eval=None,
    structured_binding: bytes | None = None,
    product_variable_count: int | None = None,
    point_map: tuple[int, ...] | None = None,
) -> int:
    try:
        if not isinstance(proof, SuccinctEqFoldProofV3):
            raise ProofV3VerificationError("public-fold proof type is wrong")
        n, mapped_point, _unused = _public_fold_layout_v3(
            column_variable_count=pcs_statement.variable_count,
            product_variable_count=product_variable_count,
            point_map=point_map,
            factor_digest=b"",
        )
        if mapped_point is not None and checker is None:
            raise ProofV3VerificationError(
                "broadcast public folds require a deferred checker")
        import numpy as _np

        if factor_eval is not None:
            if structured_binding is None:
                raise ProofV3VerificationError(
                    "closed-form public folds need the structured "
                    "factor binding")
            if len(proof.round_polynomials) != n:
                raise ProofV3VerificationError(
                    "public-fold arity is wrong")
            f_np = None
            f_values = None
            factor_digest_base = structured_binding
        elif isinstance(factor, _np.ndarray) or len(factor) >= 2048:
            f_np = (
                _np.ascontiguousarray(factor, dtype=_np.uint64)
                if isinstance(factor, _np.ndarray)
                else _np.array(
                    [int(v) % GOLDILOCKS_MODULUS for v in factor],
                    dtype=_np.uint64))
            f_values = None
            if len(proof.round_polynomials) != n or f_np.shape[0] != 1 << n:
                raise ProofV3VerificationError("public-fold arity is wrong")
            factor_digest_base = (
                structured_binding
                if structured_binding is not None
                else hashlib.sha256(
                    f_np.astype("<u8").tobytes()).digest()
            )
        else:
            f_np = None
            f_values = [int(v) % GOLDILOCKS_MODULUS for v in factor]
            if len(proof.round_polynomials) != n or len(f_values) != 1 << n:
                raise ProofV3VerificationError("public-fold arity is wrong")
            factor_digest_base = (
                structured_binding
                if structured_binding is not None
                else hashlib.sha256(
                    b"".join(
                        v.to_bytes(8, "little") for v in f_values
                    )
                ).digest()
            )
        n, mapped_point, factor_digest = _public_fold_layout_v3(
            column_variable_count=pcs_statement.variable_count,
            product_variable_count=product_variable_count,
            point_map=point_map,
            factor_digest=factor_digest_base,
        )
        transcript = hashlib.sha256(
            _EQFOLD_DOMAIN + b"PUB" + tile_digest + label.encode()
            + _fixed32(commitment, "commitment") + factor_digest
            + _fixed32(validator_nonce, "validator_nonce")).digest()
        running = proof.claimed_sum % GOLDILOCKS_MODULUS
        challenges: list[int] = []
        compiled = None
        try:
            from verallm.proof_v3.c_multiopen import replay_rounds4
            from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (  # noqa: E501
                _CHALLENGE_DOMAIN as _LOGUP_DOMAIN,
            )

            compiled = replay_rounds4(
                transcript, running,
                tuple(tuple(int(v) for v in row)
                      for row in proof.round_polynomials),
                _LOGUP_DOMAIN, b"pubfold", False, 1)
        except ImportError:
            compiled = None
        if isinstance(compiled, tuple):
            challenges_t, running, transcript = compiled
            challenges = list(challenges_t)
        elif isinstance(compiled, int):
            raise ProofV3VerificationError(
                f"public-fold {label!r} round replay fails")
        else:
            for evals in proof.round_polynomials:
                evals = tuple(v % GOLDILOCKS_MODULUS for v in evals)
                if (evals[0] + evals[1]) % GOLDILOCKS_MODULUS != running:
                    raise ProofV3VerificationError(
                        "public-fold round does not match the running sum")
                transcript = hashlib.sha256(
                    transcript
                    + b"".join(v.to_bytes(8, "little") for v in evals)
                ).digest()
                challenge = _derive(
                    transcript, b"pubfold", len(challenges) + 1)
                challenges.append(challenge)
                running = _lagrange_0123(evals, challenge)
        if factor_eval is not None:
            factor_at_point = int(factor_eval(challenges)) % (
                GOLDILOCKS_MODULUS)
        elif f_np is not None:
            # exact uint64-numpy MLE fold (the O(N) factor evaluation is the
            # public-fold verifier's hot spot at large cubes)
            from verallm.proof_v3.goldilocks_numpy import mle_eval_msb_np

            factor_at_point = mle_eval_msb_np(f_np, challenges)
        else:
            factor_at_point = _mle_eval_msb_local(f_values, challenges)
        expected = (
            proof.opening.claimed_value * factor_at_point % GOLDILOCKS_MODULUS)
        if running != expected:
            raise ProofV3VerificationError(
                "public-fold terminal coupling fails")
        point = tuple(reversed(challenges))
        opening_point = (
            point if mapped_point is None
            else tuple(point[index] for index in mapped_point)
        )
        if checker is not None:
            checker.expect(tag, opening_point, proof.opening.claimed_value)
        else:
            verify_goldilocks_multilinear_opening_v3(
                proof.opening, statement=pcs_statement, commitment=commitment,
                point=opening_point,
                expected_value=proof.opening.claimed_value,
                validator_nonce=validator_nonce)
        return proof.claimed_sum % GOLDILOCKS_MODULUS
    except ProofV3VerificationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError, ProofV3Error) as exc:
        raise ProofV3VerificationError(
            "public-fold proof is malformed") from exc
