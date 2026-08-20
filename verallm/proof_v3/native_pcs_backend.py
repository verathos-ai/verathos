"""Device-resident BaseFold PCS + succinct-fold prover (byte-identical).

The last prover-side gap between the 43 s Python qualification and the
~1 s kernel budget: the succinct fold argument's PCS commit/open still
ran the pure-Python NTT + Merkle + sumcheck. This module keeps the whole
prover on device:

* ``DeviceMerkleTreeW1`` builds every tree level with the fused SHA-256
  kernels, RETAINS the levels on device, and extracts only the queried
  authentication paths (O(q log N) 32-byte nodes cross PCIe, never the
  tree);
* ``fused_open_goldilocks_multilinear_v3`` runs the interleaved
  sumcheck/FRI opening with tensor field ops (LSB-adjacent pairing via
  strided views) and the fused NTT per fold layer;
* ``fused_prove_goldilocks_succinct_fold_v3`` composes the outer fold
  sumcheck (fused kernels) with the device PCS opening.

Byte-identical by construction: transcripts, challenges, roots, and
openings reproduce the reference exactly (asserted by conformance
tests), so proofs verify against the unmodified CPU verifier.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Final

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.goldilocks_reference import (
    GOLDILOCKS_MODULUS,
    goldilocks_principal_root_of_unity,
)
from verallm.proof_v3.goldilocks_merkle_reference import (
    GOLDILOCKS_MERKLE_REFERENCE_ABI_V3 as _MERKLE_ABI_V3,
    GoldilocksMerkleMultiOpeningReference,
    GoldilocksMerkleSiblingReference,
    _expected_sibling_coordinates,
)
from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
    GoldilocksMultilinearOpeningProofV3,
    GoldilocksMultilinearPcsStatementV3,
    _derive_field,
    _transcript_seed,
    _layer_query_indices,
    _QUERY_DOMAIN,
)

_LEAF_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/LEAF/SHA256"
_NODE_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/NODE/SHA256"
_ROOT_DOMAIN: Final = b"VERATHOS/PROOF_V3/GOLDILOCKS_MERKLE/V1/ROOT/SHA256"
_TWO63: Final = 1 << 63
_TWO64: Final = 1 << 64


def _to_int(value) -> int:
    number = int(value)
    return number + _TWO64 if number < 0 else number


class DeviceMerkleTreeW1:
    """Width-1 Merkle tree living on device with sparse path extraction."""

    __slots__ = ("binding_digest", "leaf_count", "leaf_width",
                 "commitment", "_values", "_levels", "_ext",
                 "_raw_root_dev", "_prefix", "_seg_bits")

    def __init__(self, extension, values_device, *,
                 binding_digest: bytes, lazy_commit: bool = False,
                 leaf_prefix=None, node_prefix=None):
        import torch

        n = values_device.numel()
        if n < 2 or n & (n - 1):
            raise ProofV3Error("device tree leaf count must be a power of two")
        header = binding_digest + struct.pack("<II", n, 1)
        if leaf_prefix is None:
            leaf_prefix = torch.tensor(
                list(_LEAF_DOMAIN + header), dtype=torch.uint8,
                device="cuda")
        if node_prefix is None:
            node_prefix = torch.tensor(
                list(_NODE_DOMAIN + header), dtype=torch.uint8,
                device="cuda")
        levels = [extension.leaf_hash_w1(leaf_prefix, values_device.contiguous())]
        count, plevel = n, 1
        while count > 1:
            parents = count // 2
            children = levels[-1].view(parents, 64)
            levels.append(
                extension.node_hash(node_prefix, plevel, children.reshape(-1)))
            count, plevel = parents, plevel + 1
        self.binding_digest = binding_digest
        self.leaf_count = n
        self.leaf_width = 1
        self._values = values_device
        self._levels = levels
        self._ext = extension
        self._raw_root_dev = levels[-1]
        self._prefix = _ROOT_DOMAIN + header
        self._seg_bits = None  # full level retention
        if lazy_commit:
            # commitment computed ON DEVICE by the caller (no sync here)
            self.commitment = None
        else:
            raw_root = bytes(levels[-1].cpu().tolist())
            self.commitment = hashlib.sha256(
                _ROOT_DOMAIN + header + raw_root).digest()

    def offload(self):
        """Move hash levels + leaf values to host RAM.

        Openings only gather a few hundred sibling nodes / rows, which
        open_prepare does directly from CPU tensors; the GB-scale level
        buffers stop occupying VRAM the moment the commit is done."""

        if self._values.is_cuda:
            self._values = self._values.cpu()
            self._levels = [level.cpu() for level in self._levels]
            self._raw_root_dev = self._levels[-1]
        return self

    @classmethod
    def build_streamed(cls, extension, values_device, *,
                       binding_digest: bytes, lazy_commit: bool = False,
                       segment_leaves: int = 1 << 22):
        """Byte-identical tree built segment-wise with BOUNDED device
        working set: each 2^k-leaf subtree hashes on device with the
        *_base kernels (leaf/node hashes bind GLOBAL indices), its level
        pieces stream straight to host, and only the sub-roots stay
        resident for the top levels.  Peak device usage is one segment's
        level chain (~2x32xsegment bytes) instead of 2x32xN.

        The values tensor is NOT copied to host here -- the caller keeps
        (and later offloads or frees) the codeword; ``_values`` is set to
        a host copy so openings gather rows CPU-side like offload()."""

        import torch

        n = values_device.numel()
        if n < 2 or n & (n - 1):
            raise ProofV3Error(
                "device tree leaf count must be a power of two")
        if n <= segment_leaves or not hasattr(
                extension, "leaf_hash_w1_base"):
            tree = cls(extension, values_device,
                       binding_digest=binding_digest,
                       lazy_commit=lazy_commit)
            values_host = tree._values.cpu()
            tree._levels = [level.cpu() for level in tree._levels]
            tree._values = values_host
            if not lazy_commit:
                tree._raw_root_dev = tree._levels[-1]
            return tree
        header = binding_digest + struct.pack("<II", n, 1)
        leaf_prefix = torch.tensor(
            list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda")
        node_prefix = torch.tensor(
            list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda")
        seg = segment_leaves
        seg_bits = seg.bit_length() - 1
        segments = n // seg
        # VALUES-ONLY retention: levels below seg_bits cost 64B/leaf on
        # host (32GB-class at 250k) and are fully rederivable from the
        # retained values -- openings rebuild ONLY the queried segments'
        # chains (~0.02s each).  Only the top (sub-roots upward) is kept.
        host_levels: list = [None] * seg_bits + [
            torch.empty(32 * (n >> lvl), dtype=torch.uint8)
            for lvl in range(seg_bits, n.bit_length())
        ]
        sub_roots = torch.empty(32 * segments, dtype=torch.uint8,
                                device="cuda")
        for s in range(segments):
            level = extension.leaf_hash_w1_base(
                leaf_prefix,
                values_device[s * seg:(s + 1) * seg].contiguous(),
                s * seg)
            count, plevel = seg, 1
            while count > 1:
                parents = count // 2
                children = level.view(parents, 64)
                level = extension.node_hash_base(
                    node_prefix, plevel, children.reshape(-1),
                    s * (seg >> plevel))
                count, plevel = parents, plevel + 1
            host_levels[seg_bits][32 * s:32 * (s + 1)].copy_(level)
            sub_roots[32 * s:32 * (s + 1)].copy_(level)
        level = sub_roots
        count, plevel = segments, seg_bits + 1
        while count > 1:
            parents = count // 2
            children = level.view(parents, 64)
            level = extension.node_hash(
                node_prefix, plevel, children.reshape(-1))
            host_levels[plevel][:32 * parents].copy_(level)
            count, plevel = parents, plevel + 1
        tree = cls.__new__(cls)
        tree.binding_digest = binding_digest
        tree.leaf_count = n
        tree.leaf_width = 1
        tree._values = values_device.cpu()
        tree._levels = host_levels
        tree._ext = extension
        tree._seg_bits = seg_bits
        # 32-byte device root retained for zero-sync commit absorption
        tree._raw_root_dev = level
        tree._prefix = _ROOT_DOMAIN + header
        if lazy_commit:
            tree.commitment = None
        else:
            raw_root = bytes(host_levels[-1].tolist())
            tree.commitment = hashlib.sha256(
                _ROOT_DOMAIN + header + raw_root).digest()
        return tree

    @property
    def rows(self):
        return tuple((_to_int(v),) for v in self._values.cpu().tolist())

    def _segment_chain(self, seg_id: int) -> dict:
        """Rebuild one retained-values segment's level chain on device
        (byte-identical: the *_base kernels bind the GLOBAL indices)."""

        import torch

        seg_bits = self._seg_bits
        seg = 1 << seg_bits
        header = self.binding_digest + struct.pack(
            "<II", self.leaf_count, 1)
        leaf_prefix = torch.tensor(
            list(_LEAF_DOMAIN + header), dtype=torch.uint8, device="cuda")
        node_prefix = torch.tensor(
            list(_NODE_DOMAIN + header), dtype=torch.uint8, device="cuda")
        vals = self._values[seg_id * seg:(seg_id + 1) * seg]
        if not vals.is_cuda:
            vals = vals.to("cuda")
        level = self._ext.leaf_hash_w1_base(
            leaf_prefix, vals.contiguous(), seg_id * seg)
        chain = {0: level}
        count, plevel = seg, 1
        while count > 1:
            parents = count // 2
            level = self._ext.node_hash_base(
                node_prefix, plevel, level.view(parents, 64).reshape(-1),
                seg_id * (seg >> plevel))
            chain[plevel] = level
            count, plevel = parents, plevel + 1
        return chain

    def open_prepare(self, indices):
        """Gather for a multi-open (values-only trees rebuild the queried
        segments, ONE chain resident at a time); nodes/rows come back as
        CPU tensors."""

        import torch

        selected = tuple(sorted(set(int(i) for i in indices)))
        coordinates = tuple(_expected_sibling_coordinates(
            leaf_count=self.leaf_count, indices=selected))
        by_level: dict[int, list[tuple[int, int]]] = {}
        for position, (level, index) in enumerate(coordinates):
            by_level.setdefault(level, []).append((position, index))
        order: list[int] = []
        gathers = []
        # rebuild plan: segment id -> [(piece, row, level, index)]
        seg_plan: dict[int, list] = {}
        for level, items in by_level.items():
            order.extend(position for position, _ in items)
            stored = self._levels[level]
            if stored is not None:
                idx = torch.tensor(
                    [index for _, index in items], dtype=torch.long,
                    device=stored.device)
                gathers.append(
                    stored.view(-1, 32).index_select(0, idx).cpu())
                continue
            piece = torch.empty((len(items), 32), dtype=torch.uint8)
            gathers.append(piece)
            rel = self._seg_bits - level
            for row, (_position, index) in enumerate(items):
                seg_plan.setdefault(index >> rel, []).append(
                    (piece, row, level, index - ((index >> rel) << rel)))
        for seg_id, needs in seg_plan.items():
            chain = self._segment_chain(seg_id)
            for piece, row, level, local in needs:
                piece[row] = chain[level].view(-1, 32)[local].cpu()
            del chain
        nodes = (
            torch.cat(gathers, dim=0) if gathers
            else torch.zeros((0, 32), dtype=torch.uint8))
        rows = self._values.index_select(
            0, torch.tensor(
                selected, dtype=torch.long,
                device=self._values.device)).cpu()
        return (selected, coordinates, order, nodes, rows)

    def open_finish(self, prepared, nodes_host, rows_host):
        # Fast marshalling: all inputs are canonical by construction
        # (coordinates from _expected_sibling_coordinates, digests from our
        # own tree levels), so the reference dataclass validation is skipped
        # via object.__new__; verifiers revalidate the full layout.
        selected, coordinates, order, _nd, _rd = prepared
        buf = nodes_host.contiguous().numpy().tobytes()
        digests: list[bytes | None] = [None] * len(coordinates)
        for row, position in enumerate(order):
            digests[position] = buf[row << 5:(row + 1) << 5]
        new = object.__new__
        set_ = object.__setattr__
        siblings = []
        for position, (level, index) in enumerate(coordinates):
            node = new(GoldilocksMerkleSiblingReference)
            set_(node, "level", level)
            set_(node, "index", index)
            set_(node, "digest", digests[position])
            siblings.append(node)
        rows = tuple((_to_int(v),) for v in rows_host.tolist())
        opening = new(GoldilocksMerkleMultiOpeningReference)
        set_(opening, "binding_digest", self.binding_digest)
        set_(opening, "leaf_count", self.leaf_count)
        set_(opening, "leaf_width", 1)
        set_(opening, "indices", selected)
        set_(opening, "rows", rows)
        set_(opening, "siblings", tuple(siblings))
        set_(opening, "abi_id", _MERKLE_ABI_V3)
        return opening

    def open(self, indices) -> GoldilocksMerkleMultiOpeningReference:
        selected, coordinates, order, nodes, rows = self.open_prepare(
            indices)
        digests: list[bytes | None] = [None] * len(coordinates)
        buf = nodes.contiguous().numpy().tobytes()
        for row, position in enumerate(order):
            digests[position] = buf[row << 5:(row + 1) << 5]
        siblings = tuple(
            GoldilocksMerkleSiblingReference(
                level=level, index=index, digest=digests[position])
            for position, (level, index) in enumerate(coordinates))
        return GoldilocksMerkleMultiOpeningReference(
            binding_digest=self.binding_digest,
            leaf_count=self.leaf_count,
            leaf_width=1,
            indices=selected,
            rows=tuple(
                (_to_int(rows[i].item()),)
                for i in range(len(selected))),
            siblings=siblings,
        )


def _scalar_tensor(value: int, like):
    import torch

    encoded = value - _TWO64 if value >= _TWO63 else value
    return torch.full_like(like, encoded)


def _encode_challenge(value: int) -> int:
    return value - _TWO64 if value >= _TWO63 else value


_BITREV_CACHE: dict[int, object] = {}


def _bitrev_indices(bits: int):
    """Device bit-reversal permutation for size 2^bits (cached)."""

    import torch

    cached = _BITREV_CACHE.get(bits)
    if cached is None:
        n = 1 << bits
        forward = torch.arange(n, dtype=torch.long, device="cuda")
        rev = torch.zeros_like(forward)
        for _ in range(bits):
            rev = (rev << 1) | (forward & 1)
            forward = forward >> 1
        cached = rev
        _BITREV_CACHE[bits] = cached
    return cached


_INV2: Final = (GOLDILOCKS_MODULUS + 1) // 2


def _direct_fold_enabled() -> bool:
    """Default OFF: fold layer codewords directly instead of re-encoding.

    The FRI fold maps a codeword on ``s * <g>`` to one on ``s^2 * <g^2>``
    (see ``GoldilocksMultilinearPcsStatementV3.domain_shift``), so the
    layer-(j+1) codeword equals the elementwise lerp of the layer-j
    even/odd parts -- byte-identical to ``encode(fold_pairs(values))``
    without the per-round NTT (pinned by test_proof_v3_pcs_direct_fold).

    Measured A40 @250k: per-tree opens are LAYER-TREE-HASH dominated, so
    replacing the NTTs is a wash on time and costs transient VRAM.  The
    identity exists for the batched D-chain opening (opening-v2 Part 1),
    where ONE folded chain serves every column; VERATHOS_PCS_DIRECT_FOLD=1
    enables it per-tree for A/B only.
    """

    import os

    return os.environ.get("VERATHOS_PCS_DIRECT_FOLD", "0") == "1"


def _gl_scalar_powers(base: int, count: int):
    """Device table [base^0, base^1, ..., base^(count-1)] by doubling."""

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        to_field_tensor,
    )

    table = to_field_tensor((1,), "cuda")
    power = base % GOLDILOCKS_MODULUS
    while table.numel() < count:
        shifted = gl_mul_t(table, _scalar_tensor(power, table))
        table = torch.cat((table, shifted))
        power = power * power % GOLDILOCKS_MODULUS
    return table[:count].contiguous()


def _prepare_direct_fold(fold_extension, statement, tree, values_natural):
    """Return (base codeword on device, inverse-generator table) or Nones.

    The base codeword prefers the tree's committed leaves (exact bytes);
    a lazily-valued tree falls back to one full-size re-encode.  The
    inverse-generator table holds g_M^(-i) for the base domain; each fold
    round keeps its even entries, matching g -> g^2 per layer.
    """

    import torch

    if not _direct_fold_enabled():
        return None, None
    from verallm.proof_v3.goldilocks_reference import goldilocks_inv
    from verallm.proof_v3.native_cuda_fold_backend import (
        fused_ntt_goldilocks,
    )

    size = statement.codeword_size
    stored = getattr(tree, "_values", None)
    if (
        isinstance(stored, torch.Tensor)
        and stored.numel() == size
        and stored.dtype == torch.int64
    ):
        codeword = stored.cuda() if not stored.is_cuda else stored
    else:
        padded = torch.zeros(size, dtype=torch.int64, device="cuda")
        padded[: values_natural.numel()] = values_natural
        codeword = fused_ntt_goldilocks(
            fold_extension, padded,
            shift=statement.domain_shift(0),
            generator=goldilocks_principal_root_of_unity(size),
            mutable=True)
    inverse_generator = goldilocks_inv(
        goldilocks_principal_root_of_unity(size))
    return codeword, _gl_scalar_powers(inverse_generator, size // 2)


def _direct_fold_step(fold_extension, codeword, inverse_table, statement,
                      round_index, *, chal_ptr=None, chal_host=None):
    """One FRI fold on the codeword; returns (folded, halved table).

    Matches the verifier equation exactly: even = (lo + hi) / 2,
    odd = (lo - hi) / (2 * s_j * g^i), folded = (1-c) * even + c * odd,
    evaluated via the half-split lerp kernel on cat(even, odd).
    """

    import torch

    from verallm.proof_v3.goldilocks_reference import goldilocks_inv
    from verallm.proof_v3.native_goldilocks_backend import (
        gl_add_t,
        gl_mul_t,
        gl_scale_t,
        gl_sub_t,
    )

    half = codeword.numel() // 2
    low = codeword[:half].contiguous()
    high = codeword[half:].contiguous()
    # halve constants late: lerp(cat(lo+hi, t*(lo-hi))) then one scaled
    # pass on the HALVED result -- fewer full-size transients.
    total = gl_add_t(low, high)
    scale = goldilocks_inv(statement.domain_shift(round_index))
    twiddles = gl_scale_t(inverse_table, scale)
    odd_raw = gl_mul_t(gl_sub_t(low, high), twiddles)
    del low, high, twiddles
    paired = torch.cat((total, odd_raw))
    del total, odd_raw
    if chal_ptr is not None:
        folded = fold_extension.lerp_fold_ptr(paired, chal_ptr)
    else:
        folded = fold_extension.lerp_fold(paired, chal_host)
    del paired
    folded = gl_scale_t(folded, _INV2)
    if half > 1:
        inverse_table = inverse_table[::2].contiguous()
    return folded, inverse_table


def fused_open_goldilocks_multilinear_v3(
    *,
    fold_extension,
    tree_extension,
    statement: GoldilocksMultilinearPcsStatementV3,
    tree: DeviceMerkleTreeW1,
    evaluations_device,
    point: tuple[int, ...],
    validator_nonce: bytes,
    claimed_value: int | None = None,
) -> GoldilocksMultilinearOpeningProofV3:
    """Byte-identical device replication of open_goldilocks_multilinear_v3."""

    import torch

    from verallm.proof_v3.native_cuda_fold_backend import (
        _sum_partials,
        fused_ntt_goldilocks,
    )
    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        gl_sum_t,
    )

    n = statement.variable_count
    if len(point) != n:
        raise ProofV3Error("mlpcs point arity does not match the statement")
    values = evaluations_device
    if values.numel() != statement.leaf_count:
        raise ProofV3Error("mlpcs evaluation count does not match the statement")
    # eq weights: iterative tensor product, LSB variable first.
    weights = torch.ones(1, dtype=torch.int64, device="cuda")
    for r in point:
        one_minus = (1 - r) % GOLDILOCKS_MODULUS
        low = gl_mul_t(weights, _scalar_tensor(one_minus, weights))
        high = gl_mul_t(weights, _scalar_tensor(r, weights))
        weights = torch.cat([low, high])
    if claimed_value is None:
        claimed = gl_sum_t(gl_mul_t(values, weights))
    else:
        claimed = claimed_value
    transcript = _transcript_seed(
        statement, tree.commitment, tuple(point), claimed, validator_nonce)
    # Adjacent-pair (LSB) folding == half-split (MSB) folding of the
    # bit-reversed vector: one permutation, then the fused round kernels.
    rev = _bitrev_indices(n)
    v_rev = values.index_select(0, rev)
    w_rev = weights.index_select(0, rev)
    rounds: list[tuple[int, int, int]] = []
    layer_trees: list = [tree]
    layer_roots: list[bytes] = []
    use_fs = hasattr(fold_extension, "fs_round_v2") and hasattr(
        fold_extension, "root_commit_absorb")
    if use_fs:
        # zero-sync fold rounds: transcript, challenges, and layer-root
        # commitments all evolve on device; ONE sync at the end.
        # When available, the entire sequence replays as ONE CUDA graph.
        from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
            _CHALLENGE_DOMAIN as _PCS_DOMAIN,
        )

        label = b"fold"
        graphed = None
        # graph capture pins every layer tree on device for the whole
        # open: big trees take the streamed loop instead (bounded VRAM)
        if v_rev.numel() <= _STREAM_TREE_MIN_LEAVES or not _stream_trees():
            try:
                graphed = _open_fold_graphed(
                    fold_extension, tree_extension, statement, v_rev, w_rev,
                    transcript, _PCS_DOMAIN + label, len(_PCS_DOMAIN),
                    len(label))
            except RuntimeError:
                graphed = None
        if graphed is not None:
            (rounds_buf, chal_buf, graph_trees, commit_bufs, t_buf_dev,
             v_out) = graphed
            torch.cuda.synchronize()
            rounds_host = rounds_buf.cpu().tolist()
            rounds = [
                tuple(v + _TWO64 if v < 0 else v for v in row[:3])
                for row in rounds_host
            ]
            if commit_bufs:
                cat_bytes = torch.cat(commit_bufs).cpu().numpy().tobytes()
            else:
                cat_bytes = b""
            if len(graph_trees) != len(commit_bufs):
                raise ProofV3Error("graphed fold tree/commit count mismatch")
            for pos, folded_tree in enumerate(graph_trees):
                folded_tree.commitment = cat_bytes[32 * pos:32 * pos + 32]
                layer_roots.append(folded_tree.commitment)
                layer_trees.append(folded_tree)
            transcript = bytes(t_buf_dev.cpu().tolist())
            v_rev = v_out
            final_value = _to_int(v_rev.cpu()[0].item())
            query_seed = hashlib.sha256(transcript + b"queries").digest()
            return _finish_opening(
                statement, claimed, rounds, layer_roots, layer_trees,
                final_value, query_seed, n)
        t_buf = torch.tensor(
            list(transcript), dtype=torch.uint8, device="cuda")
        dom_label = _u8_tensor_cached(_PCS_DOMAIN + label)
        rounds_buf = torch.zeros((n, 4), dtype=torch.int64, device="cuda")
        chal_buf = torch.zeros(n, dtype=torch.int64, device="cuda")
        empty = torch.zeros(0, dtype=torch.int64, device="cuda")
        commit_bufs = []
        direct_cw, inv_table = _prepare_direct_fold(
            fold_extension, statement, tree, values)
        for round_index in range(n):
            partials = fold_extension.round_partials(v_rev, w_rev)
            fold_extension.fs_round_v2(
                partials, empty, 6, t_buf, dom_label, len(_PCS_DOMAIN),
                len(label), 0, round_index,
                rounds_buf[round_index],
                chal_buf[round_index:round_index + 1])
            v_rev = fold_extension.lerp_fold_ptr(
                v_rev, chal_buf[round_index:round_index + 1])
            w_rev = fold_extension.lerp_fold_ptr(
                w_rev, chal_buf[round_index:round_index + 1])
            if direct_cw is not None:
                direct_cw, inv_table = _direct_fold_step(
                    fold_extension, direct_cw, inv_table, statement,
                    round_index,
                    chal_ptr=chal_buf[round_index:round_index + 1])
                folded_codeword = direct_cw
            else:
                remaining = n - round_index - 1
                natural = v_rev.index_select(0, _bitrev_indices(remaining))
                size = statement.codeword_size >> (round_index + 1)
                padded = torch.zeros(size, dtype=torch.int64, device="cuda")
                padded[: natural.numel()] = natural
                folded_codeword = fused_ntt_goldilocks(
                    fold_extension, padded,
                    shift=statement.domain_shift(round_index + 1),
                    generator=goldilocks_principal_root_of_unity(size),
                    mutable=True)
            folded_tree = _build_layer_tree(
                tree_extension, folded_codeword,
                binding_digest=statement.layer_binding_digest(
                    round_index + 1),
                lazy_commit=True)
            cbuf = torch.zeros(32, dtype=torch.uint8, device="cuda")
            fold_extension.root_commit_absorb(
                _u8_tensor_cached(folded_tree._prefix),
                folded_tree._raw_root_dev, cbuf, t_buf)
            commit_bufs.append(cbuf)
            layer_trees.append(folded_tree)
        torch.cuda.synchronize()
        rounds_host = rounds_buf.cpu().tolist()
        rounds = [
            tuple(v + _TWO64 if v < 0 else v for v in row[:3])
            for row in rounds_host
        ]
        for folded_tree, cbuf in zip(
            layer_trees[1:], commit_bufs, strict=True
        ):
            folded_tree.commitment = bytes(cbuf.cpu().tolist())
            layer_roots.append(folded_tree.commitment)
        transcript = bytes(t_buf.cpu().tolist())
    else:
        direct_cw, inv_table = _prepare_direct_fold(
            fold_extension, statement, tree, values)
        for round_index in range(n):
            g0, g1, g2 = _sum_partials(
                fold_extension.round_partials(v_rev, w_rev))
            rounds.append((g0, g1, g2))
            transcript = hashlib.sha256(
                transcript
                + g0.to_bytes(8, "little")
                + g1.to_bytes(8, "little")
                + g2.to_bytes(8, "little")
            ).digest()
            challenge = _derive_field(transcript, b"fold", round_index)
            encoded = _encode_challenge(challenge)
            v_rev = fold_extension.lerp_fold(v_rev, encoded)
            w_rev = fold_extension.lerp_fold(w_rev, encoded)
            if direct_cw is not None:
                direct_cw, inv_table = _direct_fold_step(
                    fold_extension, direct_cw, inv_table, statement,
                    round_index, chal_host=encoded)
                folded_codeword = direct_cw
            else:
                remaining = n - round_index - 1
                # natural (coefficient) order for the NTT input
                natural = v_rev.index_select(0, _bitrev_indices(remaining))
                size = statement.codeword_size >> (round_index + 1)
                padded = torch.zeros(size, dtype=torch.int64, device="cuda")
                padded[: natural.numel()] = natural
                folded_codeword = fused_ntt_goldilocks(
                    fold_extension, padded,
                    shift=statement.domain_shift(round_index + 1),
                    generator=goldilocks_principal_root_of_unity(size),
                    mutable=True)
            folded_tree = _build_layer_tree(
                tree_extension, folded_codeword,
                binding_digest=statement.layer_binding_digest(
                    round_index + 1),
                lazy_commit=False)
            layer_trees.append(folded_tree)
            layer_roots.append(folded_tree.commitment)
            transcript = hashlib.sha256(
                transcript + folded_tree.commitment).digest()
    final_value = _to_int(v_rev.cpu()[0].item())
    query_seed = hashlib.sha256(transcript + b"queries").digest()
    return _finish_opening(
        statement, claimed, rounds, layer_roots, layer_trees,
        final_value, query_seed, n)


def _finish_opening(statement, claimed, rounds, layer_roots, layer_trees,
                    final_value, query_seed, n):
    import torch

    base_size = statement.codeword_size
    positions = [
        int.from_bytes(
            hashlib.sha256(
                _QUERY_DOMAIN + query_seed + struct.pack("<I", query_index)
            ).digest()[:8], "little") % (base_size // 2)
        for query_index in range(statement.query_count)
    ]
    prepared_list = []
    for layer_index, layer_tree in enumerate(layer_trees):
        size = base_size >> layer_index
        prepared_list.append(layer_tree.open_prepare(_layer_query_indices(
            positions=positions, size=size, is_final=layer_index == n)))
    all_nodes = torch.cat([pr[3].cpu() for pr in prepared_list], dim=0)
    all_rows = torch.cat([pr[4].cpu() for pr in prepared_list], dim=0)
    layer_openings = []
    node_off = row_off = 0
    for layer_tree, prepared in zip(layer_trees, prepared_list, strict=True):
        n_nodes = prepared[3].shape[0]
        n_rows = prepared[4].shape[0]
        layer_openings.append(layer_tree.open_finish(
            prepared,
            all_nodes[node_off:node_off + n_nodes],
            all_rows[row_off:row_off + n_rows]))
        node_off += n_nodes
        row_off += n_rows
    return GoldilocksMultilinearOpeningProofV3(
        claimed_value=claimed,
        round_polynomials=tuple(rounds),
        layer_commitments=tuple(layer_roots),
        final_value=final_value,
        layer_openings=tuple(layer_openings),
    )


# streamed builds kick in above this leaf count when offload mode is on:
# below it the eager build's transient levels are small enough
_STREAM_TREE_MIN_LEAVES: Final = 1 << 22


def _stream_trees() -> bool:
    import os

    return _offload_trees() and os.environ.get(
        "VERATHOS_PROOF_V3_STREAM_TREES", "1") != "0"


def _build_layer_tree(tree_extension, codeword, *, binding_digest: bytes,
                      lazy_commit: bool):
    """Fold-layer tree: streamed (bounded VRAM, levels straight to host)
    for big codewords, eager for small ones."""

    if _stream_trees() and codeword.numel() > _STREAM_TREE_MIN_LEAVES \
            and hasattr(tree_extension, "leaf_hash_w1_base"):
        return DeviceMerkleTreeW1.build_streamed(
            tree_extension, codeword, binding_digest=binding_digest,
            lazy_commit=lazy_commit)
    return DeviceMerkleTreeW1(
        tree_extension, codeword, binding_digest=binding_digest,
        lazy_commit=lazy_commit)


def fused_commit_multilinear_tree(
    fold_extension, tree_extension, evaluations_device, *, statement
):
    """Device commit returning (DeviceMerkleTreeW1, codeword) byte-exact."""

    import torch

    from verallm.proof_v3.native_cuda_fold_backend import (
        _four_step_enabled,
        fused_ntt_goldilocks_consume,
        fused_ntt_goldilocks_four_step,
    )

    n = evaluations_device.numel()
    if n != statement.leaf_count:
        raise ProofV3Error("fused commit eval count does not match the statement")
    codeword_size = statement.codeword_size
    use_four_step = False
    if _four_step_enabled(codeword_size):
        bits = (codeword_size - 1).bit_length()
        n2 = 1 << (bits - (bits + 1) // 2)
        use_four_step = n >= n2 and n % n2 == 0
    if use_four_step:
        codeword = fused_ntt_goldilocks_four_step(
            fold_extension, evaluations_device, codeword_size,
            shift=statement.domain_shift(0),
            generator=goldilocks_principal_root_of_unity(codeword_size))
    else:
        padded = torch.zeros(codeword_size, dtype=torch.int64, device="cuda")
        padded[:n] = evaluations_device
        holder = [padded]
        del padded
        codeword = fused_ntt_goldilocks_consume(
            fold_extension, holder,
            shift=statement.domain_shift(0),
            generator=goldilocks_principal_root_of_unity(codeword_size))
    if _stream_trees() and codeword.numel() > _STREAM_TREE_MIN_LEAVES \
            and hasattr(tree_extension, "leaf_hash_w1_base"):
        tree = DeviceMerkleTreeW1.build_streamed(
            tree_extension, codeword,
            binding_digest=statement.layer_binding_digest(0))
        return tree, codeword
    tree = DeviceMerkleTreeW1(
        tree_extension, codeword,
        binding_digest=statement.layer_binding_digest(0),
    )
    if _offload_trees():
        tree.offload()
    return tree, codeword


def fused_prove_goldilocks_succinct_fold_v3(
    *,
    fold_extension,
    tree_extension,
    statement,
    tree: DeviceMerkleTreeW1,
    x_device,
    factor_device,
    factor_components: tuple[tuple[int, ...], ...],
    validator_nonce: bytes,
):
    """Device outer fold sumcheck + device PCS opening (byte-identical)."""

    from verallm.proof_v3.goldilocks_fold_sumcheck_reference import _challenge
    from verallm.proof_v3.goldilocks_succinct_fold_argument_reference import (
        GoldilocksSuccinctFoldProofV3,
        _factor_digest,
        _outer_seed,
    )
    from verallm.proof_v3.native_cuda_fold_backend import _sum_partials
    from verallm.proof_v3.native_goldilocks_backend import gl_mul_t, gl_sum_t

    components = tuple(
        tuple(v % GOLDILOCKS_MODULUS for v in component)
        for component in factor_components
    )
    x = x_device
    f = factor_device
    claimed = gl_sum_t(gl_mul_t(x, f))
    transcript = _outer_seed(
        statement, tree.commitment, _factor_digest(components), claimed,
        validator_nonce)
    rounds: list[tuple[int, int, int]] = []
    challenges: list[int] = []
    while x.numel() > 1:
        g0, g1, g2 = _sum_partials(fold_extension.round_partials(x, f))
        rounds.append((g0, g1, g2))
        transcript = hashlib.sha256(
            transcript
            + g0.to_bytes(8, "little")
            + g1.to_bytes(8, "little")
            + g2.to_bytes(8, "little")
        ).digest()
        challenge = _challenge(transcript, len(rounds))
        challenges.append(challenge)
        encoded = _encode_challenge(challenge)
        x = fold_extension.lerp_fold(x, encoded)
        f = fold_extension.lerp_fold(f, encoded)
    pcs_point = tuple(reversed(challenges))
    opening = fused_open_goldilocks_multilinear_v3(
        fold_extension=fold_extension,
        tree_extension=tree_extension,
        statement=statement.pcs_statement(),
        tree=tree,
        evaluations_device=x_device,
        point=pcs_point,
        validator_nonce=validator_nonce,
    )
    return GoldilocksSuccinctFoldProofV3(
        claimed_sum=claimed,
        outer_rounds=tuple(rounds),
        opening=opening,
    )


__all__ = [
    "DeviceMerkleTreeW1",
    "fused_commit_multilinear_tree",
    "fused_open_goldilocks_multilinear_v3",
    "fused_prove_goldilocks_succinct_fold_v3",
]


def _batch_inverse(values: list[int]) -> list[int]:
    """Montgomery batch inversion: one field inverse total."""

    from verallm.proof_v3.goldilocks_reference import goldilocks_inv

    prefix = [1] * (len(values) + 1)
    for index, value in enumerate(values):
        if value == 0:
            raise ProofV3Error("succinct-logup denominator is zero")
        prefix[index + 1] = prefix[index] * value % GOLDILOCKS_MODULUS
    inverse_total = goldilocks_inv(prefix[-1])
    out = [0] * len(values)
    running = inverse_total
    for index in range(len(values) - 1, -1, -1):
        out[index] = running * prefix[index] % GOLDILOCKS_MODULUS
        running = running * values[index] % GOLDILOCKS_MODULUS
    return out


_U8_TENSOR_CACHE: dict = {}


def _u8_tensor_cached(data: bytes):
    import torch

    t = _U8_TENSOR_CACHE.get(data)
    if t is None:
        t = torch.tensor(list(data), dtype=torch.uint8, device="cuda")
        if len(_U8_TENSOR_CACHE) < 4096:
            _U8_TENSOR_CACHE[data] = t
    return t


_EQFOLD_GRAPHS: dict = {}
_EQFOLD_GRAPHS_MAX = 24


def _eqfold_rounds_graphed(fold_extension, a, f, transcript: bytes,
                           dom_label_bytes: bytes, dom_len: int,
                           label_len: int):
    """CUDA-graph-captured eq-fold round loop (one replay per fold).

    All kernels are stream-routed and CPU-free, so the whole round
    sequence records once per cube size and replays with near-zero
    launch overhead. Byte-identical: the captured code IS the loop.
    """

    import torch

    n = a.numel().bit_length() - 1
    key = (a.numel(), dom_label_bytes)

    def _loop(a_t, f_t, t_t, dl_t, rounds_t, chal_t):
        empty = torch.zeros(0, dtype=torch.int64, device="cuda")
        aa, ff = a_t, f_t
        for r in range(n):
            partials = fold_extension.round_partials(aa, ff)
            fold_extension.fs_round_v2(
                partials, empty, 3, t_t, dl_t, dom_len, label_len, 0,
                r + 1, rounds_t[r], chal_t[r:r + 1])
            aa = fold_extension.lerp_fold_ptr(aa, chal_t[r:r + 1])
            ff = fold_extension.lerp_fold_ptr(ff, chal_t[r:r + 1])
        return aa

    entry = _EQFOLD_GRAPHS.get(key)
    if entry is None:
        if a.numel() > _GRAPH_MAX_CELLS:
            entry = False
        elif len(_EQFOLD_GRAPHS) >= _EQFOLD_GRAPHS_MAX:
            entry = False
        else:
            a_s = torch.empty_like(a)
            f_s = torch.empty_like(f)
            t_s = torch.empty(32, dtype=torch.uint8, device="cuda")
            dl_s = torch.empty(
                len(dom_label_bytes), dtype=torch.uint8, device="cuda")
            rounds_s = torch.zeros(
                (n, 4), dtype=torch.int64, device="cuda")
            chal_s = torch.zeros(n, dtype=torch.int64, device="cuda")
            a_s.copy_(a)
            f_s.copy_(f)
            t_s.copy_(_u8_tensor_cached(transcript))
            dl_s.copy_(_u8_tensor_cached(dom_label_bytes))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _loop(a_s, f_s, t_s, dl_s, rounds_s, chal_s)  # warmup
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                a_out = _loop(a_s, f_s, t_s, dl_s, rounds_s, chal_s)
            entry = (graph, a_s, f_s, t_s, dl_s, rounds_s, chal_s, a_out)
        _EQFOLD_GRAPHS[key] = entry
    if entry is False:
        return None
    graph, a_s, f_s, t_s, dl_s, rounds_s, chal_s, a_out = entry
    a_s.copy_(a)
    f_s.copy_(f)
    t_s.copy_(_u8_tensor_cached(transcript))
    graph.replay()
    return rounds_s, chal_s, a_out


def _offload_trees() -> bool:
    """Default ON: committed tree levels leave VRAM immediately (openings
    gather sibling paths from host). The hard-audit prover runs BESIDE a
    serving engine at gpu_util 0.8-0.85, so bounded-minimal VRAM is a hard
    production requirement; set VERATHOS_PROOF_V3_OFFLOAD_TREES=0 only for
    isolated benchmarks."""
    import os

    return os.environ.get("VERATHOS_PROOF_V3_OFFLOAD_TREES", "1") != "0"


_OPEN_GRAPHS: dict = {}
_OPEN_GRAPHS_MAX = 16
# shapes above this skip CUDA-graph capture: kernel time dominates the
# launch overhead there, and each captured entry pins GB-scale buffers
_GRAPH_MAX_CELLS = 1 << 22


def _open_fold_graphed(fold_extension, tree_extension, statement,
                       v_rev, w_rev, transcript: bytes, dom_label: bytes,
                       dom_len: int, label_len: int):
    """CUDA-graph captured fused_open fold sequence: rounds, NTTs,
    layer trees and the on-device transcript all replay per statement
    (statement-specific bindings/cosets are static input buffers).
    Returns (rounds_s, chal_s, trees, commit_bufs, t_s, v_out) or None.
    """

    import torch

    from verallm.proof_v3.native_cuda_fold_backend import (
        _ntt_coset_powers,
        _ntt_stage_tables,
        fused_ntt_goldilocks,
    )

    n = v_rev.numel().bit_length() - 1
    key = (v_rev.numel(), statement.codeword_size, dom_label)
    metas = []
    for round_index in range(n):
        size = statement.codeword_size >> (round_index + 1)
        shift = statement.domain_shift(round_index + 1)
        binding = statement.layer_binding_digest(round_index + 1)
        header = binding + struct.pack("<II", size, 1)
        metas.append((size, shift, binding, header))

    def _fill(entry):
        entry["v_s"].copy_(v_rev)
        entry["w_s"].copy_(w_rev)
        entry["t_s"].copy_(_u8_tensor_cached(transcript))
        entry["dl_s"].copy_(_u8_tensor_cached(dom_label))
        for i, (size, shift, _binding, header) in enumerate(metas):
            entry["leaf_pre"][i].copy_(
                _u8_tensor_cached(_LEAF_DOMAIN + header))
            entry["node_pre"][i].copy_(
                _u8_tensor_cached(_NODE_DOMAIN + header))
            entry["root_pre"][i].copy_(
                _u8_tensor_cached(_ROOT_DOMAIN + header))
            entry["coset_s"][i].copy_(_ntt_coset_powers(shift, size))

    def _refresh_tree_meta(entry):
        for tree, (_size, _shift, binding, header) in zip(
            entry["trees"], metas, strict=True
        ):
            tree.binding_digest = binding
            tree._prefix = _ROOT_DOMAIN + header

    entry = _OPEN_GRAPHS.get(key)
    if entry is None:
        if v_rev.numel() > _GRAPH_MAX_CELLS:
            return None
        if len(_OPEN_GRAPHS) >= _OPEN_GRAPHS_MAX:
            return None
        entry = {
            "v_s": torch.empty_like(v_rev),
            "w_s": torch.empty_like(w_rev),
            "t_s": torch.empty(32, dtype=torch.uint8, device="cuda"),
            "dl_s": torch.empty(
                len(dom_label), dtype=torch.uint8, device="cuda"),
            "rounds_s": torch.zeros(
                (n, 4), dtype=torch.int64, device="cuda"),
            "chal_s": torch.zeros(n, dtype=torch.int64, device="cuda"),
            "leaf_pre": [
                torch.empty(90, dtype=torch.uint8, device="cuda")
                for _ in range(n)],
            "node_pre": [
                torch.empty(90, dtype=torch.uint8, device="cuda")
                for _ in range(n)],
            "root_pre": [
                torch.empty(90, dtype=torch.uint8, device="cuda")
                for _ in range(n)],
            "coset_s": [
                torch.empty(size, dtype=torch.int64, device="cuda")
                for size, *_ in metas],
            "commit_bufs": [
                torch.zeros(32, dtype=torch.uint8, device="cuda")
                for _ in range(n)],
        }
        empty = torch.zeros(0, dtype=torch.int64, device="cuda")
        stage_tables = [
            _ntt_stage_tables(
                goldilocks_principal_root_of_unity(size), size)
            for size, *_ in metas]

        def _loop():
            trees = []
            vv, ww = entry["v_s"], entry["w_s"]
            for round_index in range(n):
                partials = fold_extension.round_partials(vv, ww)
                fold_extension.fs_round_v2(
                    partials, empty, 6, entry["t_s"], entry["dl_s"],
                    dom_len, label_len, 0, round_index,
                    entry["rounds_s"][round_index],
                    entry["chal_s"][round_index:round_index + 1])
                vv = fold_extension.lerp_fold_ptr(
                    vv, entry["chal_s"][round_index:round_index + 1])
                ww = fold_extension.lerp_fold_ptr(
                    ww, entry["chal_s"][round_index:round_index + 1])
                remaining = n - round_index - 1
                natural = vv.index_select(0, _bitrev_indices(remaining))
                size = metas[round_index][0]
                padded = torch.zeros(
                    size, dtype=torch.int64, device="cuda")
                padded[: natural.numel()] = natural
                codeword = fused_ntt_goldilocks(
                    fold_extension, padded, shift=0, generator=0,
                    tables=(entry["coset_s"][round_index],
                            stage_tables[round_index]),
                    mutable=True)
                folded_tree = DeviceMerkleTreeW1(
                    tree_extension, codeword,
                    binding_digest=metas[round_index][2],
                    lazy_commit=True,
                    leaf_prefix=entry["leaf_pre"][round_index],
                    node_prefix=entry["node_pre"][round_index])
                fold_extension.root_commit_absorb(
                    entry["root_pre"][round_index],
                    folded_tree._raw_root_dev,
                    entry["commit_bufs"][round_index], entry["t_s"])
                trees.append(folded_tree)
            return vv, trees

        _fill(entry)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _loop()  # warm every cache/allocator path
        torch.cuda.current_stream().wait_stream(stream)
        _fill(entry)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            v_out, trees = _loop()
        entry["graph"] = graph
        entry["trees"] = trees
        entry["v_out"] = v_out
        # the graph RECORDED these tensors' device addresses: they must
        # stay alive as long as the graph replays (the NTT cache used
        # to retain them by accident; its size cap makes big tables
        # per-call, so the entry must hold its own references)
        entry["stage_tables"] = stage_tables
        _OPEN_GRAPHS[key] = entry
        # capture only RECORDS; produce results for the current inputs
        graph.replay()
        _refresh_tree_meta(entry)
        return (entry["rounds_s"], entry["chal_s"], entry["trees"],
                entry["commit_bufs"], entry["t_s"], entry["v_out"])
    _fill(entry)
    entry["graph"].replay()
    _refresh_tree_meta(entry)
    return (entry["rounds_s"], entry["chal_s"], entry["trees"],
            entry["commit_bufs"], entry["t_s"], entry["v_out"])


_LOGUP_GRAPHS: dict = {}
_LOGUP_GRAPHS_MAX = 48


def _fs_rounds_graphed(fold_extension, kind: str, cols, transcript: bytes,
                       tag: bytes):
    """CUDA-graph captured LogUp round loops (linear/triple/etf).

    Returns (rounds_s, chal_s, out_tensors) or None. Static column
    buffers are copied per replay; the FS chain runs in-graph.
    """

    import torch

    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        _CHALLENGE_DOMAIN,
    )

    n = cols[0].numel().bit_length() - 1
    key = (kind, cols[0].numel(), tag)
    entry = _LOGUP_GRAPHS.get(key)
    if entry is None and (
        cols[0].numel() > _GRAPH_MAX_CELLS
        or len(_LOGUP_GRAPHS) >= _LOGUP_GRAPHS_MAX
    ):
        return None
    dom_label = _CHALLENGE_DOMAIN + tag
    if entry is None:
        col_s = [torch.empty_like(c) for c in cols]
        t_s = torch.empty(32, dtype=torch.uint8, device="cuda")
        dl_s = torch.empty(len(dom_label), dtype=torch.uint8, device="cuda")
        rounds_s = torch.zeros((n, 4), dtype=torch.int64, device="cuda")
        chal_s = torch.zeros(n, dtype=torch.int64, device="cuda")
        empty = torch.zeros(0, dtype=torch.int64, device="cuda")

        def _fill():
            for cs, c in zip(col_s, cols, strict=True):
                cs.copy_(c)
            t_s.copy_(_u8_tensor_cached(transcript))
            dl_s.copy_(_u8_tensor_cached(dom_label))

        def _loop():
            work = list(col_s)
            for r in range(n):
                if kind == "linear":
                    partials = fold_extension.round_partials(
                        work[0], _fs_ones(work[0].numel()))
                    fold_extension.fs_round_v2(
                        partials, empty, 2, t_s, dl_s,
                        len(_CHALLENGE_DOMAIN), len(tag), 1, r + 1,
                        rounds_s[r], chal_s[r:r + 1])
                elif kind == "triple":
                    partials = fold_extension.product_round_partials(
                        work[0], work[1], work[2])
                    fold_extension.fs_round_v2(
                        partials, empty, 4, t_s, dl_s,
                        len(_CHALLENGE_DOMAIN), len(tag), 1, r + 1,
                        rounds_s[r], chal_s[r:r + 1])
                else:  # etf
                    p_part = fold_extension.product_round_partials(
                        work[0], work[1], work[3])
                    m_part = fold_extension.round_partials(
                        work[2], work[3])
                    fold_extension.fs_round_v2(
                        p_part, m_part, 5, t_s, dl_s,
                        len(_CHALLENGE_DOMAIN), len(tag), 1, r + 1,
                        rounds_s[r], chal_s[r:r + 1])
                work = [
                    fold_extension.lerp_fold_ptr(w, chal_s[r:r + 1])
                    for w in work
                ]
            return work

        _fill()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _loop()
        torch.cuda.current_stream().wait_stream(stream)
        _fill()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _loop()
        entry = (graph, col_s, t_s, rounds_s, chal_s, out, _fill)
        _LOGUP_GRAPHS[key] = entry
        graph.replay()
        return rounds_s, chal_s, out, t_s
    graph, col_s, t_s, rounds_s, chal_s, out, _old_fill = entry
    for cs, c in zip(col_s, cols, strict=True):
        cs.copy_(c)
    t_s.copy_(_u8_tensor_cached(transcript))
    graph.replay()
    return rounds_s, chal_s, out, t_s


_FS_ONES_CACHE: dict = {}


def _fs_ones(n: int):
    import torch

    t = _FS_ONES_CACHE.get(n)
    if t is None:
        t = torch.ones(n, dtype=torch.int64, device="cuda")
        _FS_ONES_CACHE[n] = t
    return t


def _fs_buffers(transcript: bytes, tag: bytes, n_rounds: int):
    import torch

    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        _CHALLENGE_DOMAIN,
    )

    t_buf = torch.tensor(list(transcript), dtype=torch.uint8, device="cuda")
    dom_label = _u8_tensor_cached(_CHALLENGE_DOMAIN + tag)
    rounds_buf = torch.zeros((n_rounds, 4), dtype=torch.int64, device="cuda")
    chal_buf = torch.zeros(n_rounds, dtype=torch.int64, device="cuda")
    return t_buf, dom_label, len(_CHALLENGE_DOMAIN), rounds_buf, chal_buf


def _fs_finish(t_buf, rounds_buf, chal_buf):
    import torch

    torch.cuda.synchronize()

    def _dec(v):
        return v + _TWO64 if v < 0 else v

    rounds = [tuple(_dec(v) for v in row) for row in rounds_buf.cpu().tolist()]
    challenges = [_dec(v) for v in chal_buf.cpu().tolist()]
    transcript = bytes(t_buf.cpu().tolist())
    return rounds, challenges, transcript


def _device_linear_rounds(column, transcript: bytes, tag: bytes,
                          fold_extension):
    """Zero-sync device rounds for sum_i column[i] (4-eval wire)."""

    graphed = None
    try:
        graphed = _fs_rounds_graphed(
            fold_extension, "linear", (column,), transcript, tag)
    except RuntimeError:
        graphed = None
    if graphed is not None:
        import torch

        rounds_s, chal_s, out, t_s = graphed
        torch.cuda.synchronize()

        def _dec(v):
            return v + _TWO64 if v < 0 else v

        rounds = [
            tuple(_dec(v) for v in row)
            for row in rounds_s.cpu().tolist()
        ]
        challenges = [_dec(v) for v in chal_s.cpu().tolist()]
        transcript = bytes(t_s.cpu().tolist())
        return rounds, challenges, transcript, out[0]
    import torch

    empty = torch.zeros(0, dtype=torch.int64, device="cuda")
    n_rounds = column.numel().bit_length() - 1
    t_buf, dom_label, dom_len, rounds_buf, chal_buf = _fs_buffers(
        transcript, tag, n_rounds)
    work = column
    for r in range(n_rounds):
        partials = fold_extension.round_partials(
            work, _fs_ones(work.numel()))
        fold_extension.fs_round_v2(
            partials, empty, 2, t_buf, dom_label, dom_len, len(tag), 1,
            r + 1, rounds_buf[r], chal_buf[r:r + 1])
        work = fold_extension.lerp_fold_ptr(work, chal_buf[r:r + 1])
    rounds, challenges, transcript = _fs_finish(t_buf, rounds_buf, chal_buf)
    return rounds, challenges, transcript, work


def _device_triple_rounds(col_a, col_b, col_f, transcript: bytes, tag: bytes,
                          fold_extension):
    """Zero-sync device rounds for sum_i a*b*f (degree-3, 4-eval wire)."""

    graphed = None
    try:
        graphed = _fs_rounds_graphed(
            fold_extension, "triple", (col_a, col_b, col_f), transcript, tag)
    except RuntimeError:
        graphed = None
    if graphed is not None:
        import torch

        rounds_s, chal_s, out, t_s = graphed
        torch.cuda.synchronize()

        def _dec(v):
            return v + _TWO64 if v < 0 else v

        rounds = [
            tuple(_dec(v) for v in row)
            for row in rounds_s.cpu().tolist()
        ]
        challenges = [_dec(v) for v in chal_s.cpu().tolist()]
        transcript = bytes(t_s.cpu().tolist())
        return rounds, challenges, transcript
    import torch

    empty = torch.zeros(0, dtype=torch.int64, device="cuda")
    n_rounds = col_a.numel().bit_length() - 1
    t_buf, dom_label, dom_len, rounds_buf, chal_buf = _fs_buffers(
        transcript, tag, n_rounds)
    a, b, f = col_a, col_b, col_f
    for r in range(n_rounds):
        partials = fold_extension.product_round_partials(a, b, f)
        fold_extension.fs_round_v2(
            partials, empty, 4, t_buf, dom_label, dom_len, len(tag), 1,
            r + 1, rounds_buf[r], chal_buf[r:r + 1])
        a = fold_extension.lerp_fold_ptr(a, chal_buf[r:r + 1])
        b = fold_extension.lerp_fold_ptr(b, chal_buf[r:r + 1])
        f = fold_extension.lerp_fold_ptr(f, chal_buf[r:r + 1])
    rounds, challenges, transcript = _fs_finish(t_buf, rounds_buf, chal_buf)
    return rounds, challenges, transcript


def _device_eq_table(point):
    """eq(point, .) built on device: CPU prefix + log-doubling (exact)."""

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        to_field_tensor,
    )

    prefix_vars = min(12, len(point))
    prefix = [1]
    for z in point[:prefix_vars]:
        z_c = z % GOLDILOCKS_MODULUS
        one_minus = (1 - z) % GOLDILOCKS_MODULUS
        prefix = [
            v * f % GOLDILOCKS_MODULUS
            for f in (one_minus, z_c) for v in prefix
        ]
    eq_dev = to_field_tensor(tuple(prefix), "cuda")
    tail = tuple(point[prefix_vars:])
    if tail:
        pairs = to_field_tensor(
            tuple((1 - z) % GOLDILOCKS_MODULUS for z in tail)
            + tuple(z % GOLDILOCKS_MODULUS for z in tail), "cuda")
        n_tail = len(tail)
        for j in range(n_tail):
            eq_dev = torch.cat(
                (gl_mul_t(eq_dev, pairs[j].expand_as(eq_dev)),
                 gl_mul_t(eq_dev, pairs[n_tail + j].expand_as(eq_dev))))
    return eq_dev


def _device_mle_at(fold_extension, values_device, challenges):
    a = values_device
    for challenge in challenges:
        a = fold_extension.lerp_fold(a, _encode_challenge(challenge))
    value = int(a.cpu().item())
    return value + _TWO64 if value < 0 else value


def _register_batch_column(collector, tag, pcs_statement, tree, values,
                           device_values):
    if tag not in collector.columns:
        import types as _types

        collector.register_column(tag, _types.SimpleNamespace(
            pcs_statement=pcs_statement, tree=tree, values=values,
            device_values=device_values))


def fused_prove_goldilocks_succinct_logup_v3(
    *,
    fold_extension,
    tree_extension,
    statement,
    looked_up_values: tuple[int, ...],
    validator_nonce: bytes,
    witness_column=None,
    collector=None,
    tag_prefix: str | None = None,
    witness_tag: str | None = None,
    aux_ctx=None,
):
    """Device-resident succinct-LogUp prover (reference-verifiable).

    Witness-side columns (size 2^w) commit, run rounds, and open on
    device; table-side columns (table size, typically tiny) use the
    reference path. Transcript evolution mirrors the reference prover
    exactly, so the unmodified CPU verifier accepts.
    """

    import hashlib as _h
    import torch

    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        GoldilocksSuccinctLogupProofV3,
        GoldilocksSuccinctLogupSubProofV3,
        _derive,
        _eq_table,
        _mle_eval_msb,
        _seed_transcript,
        _sumcheck_rounds,
        logup_batch_tag_v3,
    )
    from verallm.proof_v3.goldilocks_multilinear_pcs_reference import (
        commit_goldilocks_multilinear_v3,
        open_goldilocks_multilinear_v3,
    )
    from verallm.proof_v3.native_goldilocks_backend import to_field_tensor

    table = statement.padded_table()
    device_only = aux_ctx is not None and "w_dev" in aux_ctx
    if device_only:
        witness = None
        multiplicities = None
    else:
        witness = tuple(
            v % GOLDILOCKS_MODULUS for v in looked_up_values
        )
        if len(witness) < statement.witness_size:
            witness = witness + (statement.table[0],) * (
                statement.witness_size - len(witness))
        table_index = {
            value: index for index, value in enumerate(statement.table)}
        counts: dict[int, int] = {}
        for value in witness:
            if value not in table_index:
                raise ProofV3Error(
                    "succinct-logup witness value is not in the table")
            counts[table_index[value]] = counts.get(
                table_index[value], 0) + 1
        multiplicities = tuple(
            counts.get(index, 0) for index in range(statement.table_size))

    if witness_column is not None:
        w_dev = witness_column.device_values
        w_tree = witness_column.tree
        if w_dev is None:
            w_dev = to_field_tensor(witness, "cuda")
    else:
        w_dev = to_field_tensor(witness, "cuda")
        w_tree, _ = fused_commit_multilinear_tree(
            fold_extension, tree_extension, w_dev,
            statement=statement.column_pcs_statement("W"))
    if aux_ctx is not None:
        m_commit_dev = aux_ctx["m_dev"]
        m_tree = aux_ctx["m_tree"]
    elif statement.table_size >= GOLDILOCKS_FUSED_LOGUP_TABLE_DEVICE_MIN_V3:
        m_commit_dev = to_field_tensor(multiplicities, "cuda")
        m_tree, _ = fused_commit_multilinear_tree(
            fold_extension, tree_extension, m_commit_dev,
            statement=statement.column_pcs_statement("M"))
    else:
        m_commit_dev = None
        m_tree = commit_goldilocks_multilinear_v3(
            statement=statement.column_pcs_statement("M"),
            evaluations=multiplicities)
    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        _CHALLENGE_COUNT,
    )

    transcript = _seed_transcript(
        statement, w_tree.commitment, m_tree.commitment, validator_nonce)
    betas = tuple(
        _derive(transcript, b"beta", c) for c in range(_CHALLENGE_COUNT))

    from verallm.proof_v3.native_goldilocks_backend import (
        from_field_tensor,
        gl_inv_t,
        gl_mul_t,
    )

    inverse_data = []
    for c, beta in enumerate(betas):
        if aux_ctx is not None:
            d_dev = aux_ctx["d_dev"]
            d_column = aux_ctx["d_column"]
            e_dev_full = aux_ctx["e_dev"]
            e_column = aux_ctx["e_column"]
            d_tree = aux_ctx["d_tree"]
        else:
            # exact device batch inversion (one fused Fermat chain for
            # the witness-side AND table-side shifts together)
            shift_dev = to_field_tensor(
                tuple((beta + value) % GOLDILOCKS_MODULUS
                      for value in witness)
                + tuple((beta + table[t]) % GOLDILOCKS_MODULUS
                        for t in range(statement.table_size)), "cuda")
            inverted = gl_inv_t(shift_dev)
            d_dev = inverted[:len(witness)].contiguous()
            d_column = from_field_tensor(d_dev)
            m_dev_tmp = (
                m_commit_dev if m_commit_dev is not None
                else to_field_tensor(multiplicities, "cuda"))
            e_dev_full = gl_mul_t(
                m_dev_tmp, inverted[len(witness):].contiguous())
            e_column = from_field_tensor(e_dev_full)
            d_tree, _ = fused_commit_multilinear_tree(
                fold_extension, tree_extension, d_dev,
                statement=statement.column_pcs_statement(f"D{c}"))
        if aux_ctx is not None:
            e_dev = e_dev_full
            e_tree = aux_ctx["e_tree"]
        elif statement.table_size >= (
            GOLDILOCKS_FUSED_LOGUP_TABLE_DEVICE_MIN_V3
        ):
            e_dev = e_dev_full
            e_tree, _ = fused_commit_multilinear_tree(
                fold_extension, tree_extension, e_dev,
                statement=statement.column_pcs_statement(f"E{c}"))
        else:
            e_dev = None
            e_tree = commit_goldilocks_multilinear_v3(
                statement=statement.column_pcs_statement(f"E{c}"),
                evaluations=e_column)
        inverse_data.append((d_column, d_dev, d_tree, e_column, e_tree, e_dev))
        transcript = _h.sha256(
            transcript + d_tree.commitment + e_tree.commitment).digest()

    sums = []
    subproofs = []
    for c, beta in enumerate(betas):
        d_column, d_dev, d_tree, e_column, e_tree, e_dev = inverse_data[c]
        if device_only:
            from verallm.proof_v3.native_goldilocks_backend import (
                gl_add_t,
                gl_sum_t,
            )

            s_c = gl_sum_t(d_dev)
            beta_enc = beta - (1 << 64) if beta >= (1 << 63) else beta
            beta_t = torch.full(
                (1,), beta_enc, dtype=torch.int64, device="cuda")
            bw_dev = gl_add_t(
                aux_ctx["w_dev"], beta_t.expand_as(aux_ctx["w_dev"]))
            table_dev = _device_padded_table_cached(statement)
            bt_dev_pre = gl_add_t(
                table_dev, beta_t.expand_as(table_dev))
            bt = None
        else:
            s_c = sum(d_column) % GOLDILOCKS_MODULUS
            bw_dev = to_field_tensor(
                tuple((beta + value) % GOLDILOCKS_MODULUS
                      for value in witness),
                "cuda")
            bt = [(beta + value) % GOLDILOCKS_MODULUS for value in table]
            bt_dev_pre = None
        sums.append(s_c)
        z_w = tuple(_derive(transcript, b"zw", c * 64 + j)
                    for j in range(statement.witness_variable_count))
        z_t = tuple(_derive(transcript, b"zt", c * 64 + j)
                    for j in range(statement.table_variable_count))
        eq_dev = _device_eq_table(z_w)
        eq_t = None  # small-table CPU branch builds it lazily

        def _open_device(tree, values_dev, tag, point):
            return fused_open_goldilocks_multilinear_v3(
                fold_extension=fold_extension,
                tree_extension=tree_extension,
                statement=statement.column_pcs_statement(tag),
                tree=tree,
                evaluations_device=values_dev,
                point=point,
                validator_nonce=validator_nonce,
            )

        # dsum (device)
        rounds, challenges, transcript, d_folded = _device_linear_rounds(
            d_dev, transcript, b"dsum", fold_extension)
        point = tuple(reversed(challenges))
        if collector is not None:
            d_batch_tag = logup_batch_tag_v3("D", c, tag_prefix, witness_tag)
            _register_batch_column(
                collector, d_batch_tag,
                statement.column_pcs_statement(f"D{c}"), d_tree, d_column,
                d_dev)
            terminal = int(d_folded.cpu().item())
            terminal += _TWO64 if terminal < 0 else 0
            subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                claimed_sum=s_c, round_polynomials=tuple(rounds),
                openings=(collector.defer(d_batch_tag, point, terminal),)))
        else:
            subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                claimed_sum=s_c, round_polynomials=tuple(rounds),
                openings=(_open_device(d_tree, d_dev, f"D{c}", point),)))
        # esum (device when the table is large)
        e_batch_tag = logup_batch_tag_v3("E", c, tag_prefix, witness_tag)
        if collector is not None:
            _register_batch_column(
                collector, e_batch_tag,
                statement.column_pcs_statement(f"E{c}"), e_tree, e_column,
                e_dev)
        if e_dev is not None:
            rounds, challenges, transcript, e_folded = _device_linear_rounds(
                e_dev, transcript, b"esum", fold_extension)
            point = tuple(reversed(challenges))
            if collector is not None:
                terminal = int(e_folded.cpu().item())
                terminal += _TWO64 if terminal < 0 else 0
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=s_c, round_polynomials=tuple(rounds),
                    openings=(collector.defer(
                        e_batch_tag, point, terminal),)))
            else:
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=s_c, round_polynomials=tuple(rounds),
                    openings=(fused_open_goldilocks_multilinear_v3(
                        fold_extension=fold_extension,
                        tree_extension=tree_extension,
                        statement=statement.column_pcs_statement(f"E{c}"),
                        tree=e_tree, evaluations_device=e_dev, point=point,
                        validator_nonce=validator_nonce),)))
        else:
            rounds, challenges, transcript = _sumcheck_rounds(
                [list(e_column)], lambda e: e, transcript, b"esum")
            point = tuple(reversed(challenges))
            if collector is not None:
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=s_c, round_polynomials=tuple(rounds),
                    openings=(collector.defer(
                        e_batch_tag, point,
                        _mle_eval_msb(tuple(e_column),
                                      tuple(challenges))),)))
            else:
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=s_c, round_polynomials=tuple(rounds),
                    openings=(open_goldilocks_multilinear_v3(
                        statement=statement.column_pcs_statement(f"E{c}"),
                        tree=e_tree, evaluations=e_column, point=point,
                        validator_nonce=validator_nonce),)))
        # dwf (device triple)
        rounds, challenges, transcript = _device_triple_rounds(
            d_dev, bw_dev, eq_dev, transcript, b"dwf", fold_extension)
        point = tuple(reversed(challenges))
        if collector is not None:
            w_batch_tag = logup_batch_tag_v3("W", c, tag_prefix, witness_tag)
            if w_batch_tag not in collector.columns:
                _register_batch_column(
                    collector, w_batch_tag,
                    statement.column_pcs_statement("W"), w_tree, witness,
                    w_dev)
            subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                claimed_sum=1, round_polynomials=tuple(rounds),
                openings=(
                    collector.defer(
                        logup_batch_tag_v3("D", c, tag_prefix, witness_tag),
                        point,
                        _device_mle_at(fold_extension, d_dev, challenges)),
                    collector.defer(
                        w_batch_tag, point,
                        _device_mle_at(fold_extension, w_dev, challenges)),
                )))
        else:
            subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                claimed_sum=1, round_polynomials=tuple(rounds),
                openings=(
                    _open_device(d_tree, d_dev, f"D{c}", point),
                    _open_device(w_tree, w_dev, "W", point),
                )))
        # etf (device when the table is large)
        if e_dev is not None:
            bt_dev = (
                bt_dev_pre if bt_dev_pre is not None
                else to_field_tensor(tuple(bt), "cuda"))
            m_col_dev = (
                m_commit_dev if m_commit_dev is not None
                else to_field_tensor(multiplicities, "cuda"))
            eq_t_dev = _device_eq_table(z_t)
            rounds, challenges, transcript = _device_etf_rounds(
                e_dev, bt_dev, m_col_dev, eq_t_dev, transcript,
                fold_extension)
            point = tuple(reversed(challenges))
            if collector is not None:
                m_batch_tag = logup_batch_tag_v3(
                    "M", c, tag_prefix, witness_tag)
                _register_batch_column(
                    collector, m_batch_tag,
                    statement.column_pcs_statement("M"), m_tree,
                    multiplicities, m_col_dev)
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=0, round_polynomials=tuple(rounds),
                    openings=(
                        collector.defer(
                            e_batch_tag, point,
                            _device_mle_at(
                                fold_extension, e_dev, challenges)),
                        collector.defer(
                            m_batch_tag, point,
                            _device_mle_at(
                                fold_extension, m_col_dev, challenges)),
                    )))
            else:
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=0, round_polynomials=tuple(rounds),
                    openings=(
                        fused_open_goldilocks_multilinear_v3(
                            fold_extension=fold_extension,
                            tree_extension=tree_extension,
                            statement=statement.column_pcs_statement(f"E{c}"),
                            tree=e_tree, evaluations_device=e_dev, point=point,
                            validator_nonce=validator_nonce),
                        fused_open_goldilocks_multilinear_v3(
                            fold_extension=fold_extension,
                            tree_extension=tree_extension,
                            statement=statement.column_pcs_statement("M"),
                            tree=m_tree, evaluations_device=m_col_dev,
                            point=point, validator_nonce=validator_nonce),
                    )))
        else:
            if eq_t is None:
                eq_t = _eq_table(z_t)
            rounds, challenges, transcript = _sumcheck_rounds(
                [list(e_column), bt, list(multiplicities), list(eq_t)],
                lambda e, b_, m_, q: (e * b_ - m_) % GOLDILOCKS_MODULUS * q
                % GOLDILOCKS_MODULUS,
                transcript, b"etf")
            point = tuple(reversed(challenges))
            if collector is not None:
                m_batch_tag = logup_batch_tag_v3(
                    "M", c, tag_prefix, witness_tag)
                _register_batch_column(
                    collector, m_batch_tag,
                    statement.column_pcs_statement("M"), m_tree,
                    multiplicities, m_commit_dev)
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=0, round_polynomials=tuple(rounds),
                    openings=(
                        collector.defer(
                            e_batch_tag, point,
                            _mle_eval_msb(tuple(e_column),
                                          tuple(challenges))),
                        collector.defer(
                            m_batch_tag, point,
                            _mle_eval_msb(tuple(multiplicities),
                                          tuple(challenges))),
                    )))
            else:
                subproofs.append(GoldilocksSuccinctLogupSubProofV3(
                    claimed_sum=0, round_polynomials=tuple(rounds),
                    openings=(
                        open_goldilocks_multilinear_v3(
                            statement=statement.column_pcs_statement(f"E{c}"),
                            tree=e_tree, evaluations=e_column, point=point,
                            validator_nonce=validator_nonce),
                        open_goldilocks_multilinear_v3(
                            statement=statement.column_pcs_statement("M"),
                            tree=m_tree, evaluations=multiplicities,
                            point=point,
                            validator_nonce=validator_nonce),
                    )))
    return GoldilocksSuccinctLogupProofV3(
        witness_commitment=w_tree.commitment,
        multiplicity_commitment=m_tree.commitment,
        inverse_commitments=tuple(
            root for _dc, _dd, d_tree, _ec, e_tree, _ed in inverse_data
            for root in (d_tree.commitment, e_tree.commitment)),
        sums=tuple(sums),
        subproofs=tuple(subproofs),
    )


__all__.append("fused_prove_goldilocks_succinct_logup_v3")


def _device_etf_rounds(col_e, col_bt, col_m, col_eq, transcript: bytes,
                       fold_extension):
    """Zero-sync device rounds for sum eq*(E*BT - M) (4-eval wire)."""

    n_rounds = col_e.numel().bit_length() - 1
    tag = b"etf"
    graphed = None
    try:
        graphed = _fs_rounds_graphed(
            fold_extension, "etf", (col_e, col_bt, col_m, col_eq), transcript, tag)
    except RuntimeError:
        graphed = None
    if graphed is not None:
        import torch

        rounds_s, chal_s, out, t_s = graphed
        torch.cuda.synchronize()

        def _dec(v):
            return v + _TWO64 if v < 0 else v

        rounds = [
            tuple(_dec(v) for v in row)
            for row in rounds_s.cpu().tolist()
        ]
        challenges = [_dec(v) for v in chal_s.cpu().tolist()]
        transcript = bytes(t_s.cpu().tolist())
        return rounds, challenges, transcript
    t_buf, dom_label, dom_len, rounds_buf, chal_buf = _fs_buffers(
        transcript, tag, n_rounds)
    e, bt, m, q = col_e, col_bt, col_m, col_eq
    for r in range(n_rounds):
        p_part = fold_extension.product_round_partials(e, bt, q)
        m_part = fold_extension.round_partials(m, q)
        fold_extension.fs_round_v2(
            p_part, m_part, 5, t_buf, dom_label, dom_len, len(tag), 1,
            r + 1, rounds_buf[r], chal_buf[r:r + 1])
        e = fold_extension.lerp_fold_ptr(e, chal_buf[r:r + 1])
        bt = fold_extension.lerp_fold_ptr(bt, chal_buf[r:r + 1])
        m = fold_extension.lerp_fold_ptr(m, chal_buf[r:r + 1])
        q = fold_extension.lerp_fold_ptr(q, chal_buf[r:r + 1])
    rounds, challenges, transcript = _fs_finish(t_buf, rounds_buf, chal_buf)
    return rounds, challenges, transcript

GOLDILOCKS_FUSED_LOGUP_TABLE_DEVICE_MIN_V3 = 1 << 11


def _streamed_product_rounds(fold_extension, gen, total_cells: int,
                             resident_cells: int, transcript: bytes,
                             _prod_derive):
    """Out-of-core prefix of the product sumcheck: the base cube is
    GENERATED chunk-wise (never materialized), each round's folded a/b/f
    spill to host, and rounds run until the vectors fit
    ``resident_cells``.  Byte-identical transcript: the per-round evals
    are the same exact field sums, just accumulated chunk-wise.

    ``gen(lo, hi)`` returns the encoded int64 CUDA chunks
    ``(a, b, f)`` of the base cube for flat index range [lo, hi).
    Returns ``(a, b, f, transcript, rounds, challenges)`` with the
    residents on device.
    """

    import hashlib as _h

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_add_t,
        gl_mul_t,
        gl_sub_t,
    )

    chunk = resident_cells
    spill = None  # (a_host, b_host, f_host) after the first fold
    rounds: list = []
    challenges: list = []
    length = total_cells

    def _load(offset, count):
        if spill is None:
            return gen(offset, offset + count)
        return tuple(
            t[offset:offset + count].to("cuda", non_blocking=True)
            for t in spill)

    while length > resident_cells:
        half = length >> 1
        sums = [0, 0, 0, 0]
        for lo in range(0, half, chunk):
            count = min(chunk, half - lo)
            lo_abf = _load(lo, count)
            hi_abf = _load(half + lo, count)
            cat = tuple(
                torch.cat((lo_t, hi_t))
                for lo_t, hi_t in zip(lo_abf, hi_abf))
            partials = fold_extension.product_round_partials(*cat)
            torch.cuda.synchronize()
            for index, value in enumerate(partials.cpu().tolist()):
                value = value + _TWO64 if value < 0 else value
                sums[index % 4] = (
                    sums[index % 4] + value) % GOLDILOCKS_MODULUS
        evals = tuple(sums)
        rounds.append(evals)
        transcript = _h.sha256(
            transcript
            + b"".join(v.to_bytes(8, "little") for v in evals)
        ).digest()
        challenge = _prod_derive(transcript, len(rounds))
        challenges.append(challenge)
        c_enc = _encode_challenge(challenge)
        # PINNED spill: pageable host buffers make every D2H/H2D copy
        # synchronous and ~2x slower; the spill traffic dominates the
        # out-of-core rounds at long context
        new_spill = tuple(
            torch.empty(half, dtype=torch.int64).pin_memory()
            for _ in range(3))
        c_full = torch.full(
            (min(chunk, half),), c_enc, dtype=torch.int64, device="cuda")
        for lo in range(0, half, chunk):
            count = min(chunk, half - lo)
            lo_abf = _load(lo, count)
            hi_abf = _load(half + lo, count)
            for slot, (lo_t, hi_t) in enumerate(zip(lo_abf, hi_abf)):
                folded = gl_add_t(
                    lo_t,
                    gl_mul_t(c_full[:count], gl_sub_t(hi_t, lo_t)))
                new_spill[slot][lo:lo + count].copy_(folded)
        spill = new_spill
        length = half
    a, b, f = (t.to("cuda") for t in spill)
    return a, b, f, transcript, rounds, challenges


def fused_prove_goldilocks_succinct_product_v3(
    *,
    fold_extension,
    tree_extension,
    statement,
    a_column,
    b_column,
    factor_components,
    validator_nonce: bytes,
    collector=None,
    a_tag: str | None = None,
    b_tag: str | None = None,
    a_point_map=None,
    b_point_map=None,
    a_fold_device=None,
    b_fold_device=None,
    stream_gen=None,
    stream_cells: int | None = None,
    compact_spec=None,
    stream_resident: int = 1 << 24,
):
    """Device prover for the succinct product argument.

    Broadcast-free mode: ``a_fold_device``/``b_fold_device`` supply the
    TRANSIENT broadcast tensor the rounds fold over (the committed
    column stays small), and ``a_point_map``/``b_point_map`` select the
    terminal sub-point (LSB-first indices) the small column opens at.

    Byte-identical transcript to the CPU reference prover: same seed,
    same 4-eval degree-3 rounds, same terminal PCS openings against the
    shared column trees.  The public tensor factor is built exactly on
    device (MSB-first components, new component = least significant).
    """

    import hashlib as _h

    import torch

    from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (
        GoldilocksSuccinctProductProofV3,
        _derive as _prod_derive,
        _factor_digest,
        _field,
        _seed as _prod_seed,
    )
    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        gl_sum_t,
        to_field_tensor,
    )

    components = tuple(
        tuple(_field(v, "factor value") for v in component)
        for component in factor_components
    )
    if tuple(len(c) for c in components) != statement.factor_component_sizes:
        raise ProofV3Error("succinct-product factor shapes are wrong")
    import os as _os

    stream_resident = int(_os.environ.get(
        "VERATHOS_PROD_STREAM_RESIDENT", stream_resident))
    if (compact_spec is not None
            and _os.environ.get("VERATHOS_PROD_COMPACT", "1") != "0"):
        # COMPACT mode: all prefix rounds contract on the un-broadcast
        # operands (kinds td/sd/row + separable factor); the cube is
        # materialized only once it fits ``stream_resident``.
        # Byte-identical transcript to the streamed path.
        import time as _time

        _trace = _os.environ.get("VERATHOS_ATTN_TRACE") == "1"
        _t0 = _time.perf_counter()
        ck_a, ce_a, ck_b, ce_b, ax_tables, dims = compact_spec
        full_sizes = dict(zip(_AXIS_ORDER, dims))
        tab_map = {
            ax: tab.reshape(-1).contiguous().cuda()
            for ax, tab in zip(_AXIS_ORDER, ax_tables)}
        claimed = _compact_contract(
            _KIND_AXES[ck_a], ce_a.reshape(-1).contiguous().cuda(),
            _KIND_AXES[ck_b], ce_b.reshape(-1).contiguous().cuda(),
            tab_map, full_sizes)
        transcript = _prod_seed(
            statement, a_column.tree.commitment,
            b_column.tree.commitment, _factor_digest(components),
            claimed, validator_nonce,
        )
        a, b, f, transcript, rounds, challenges = (
            _compact_product_prefix(
                ck_a, ce_a, ck_b, ce_b, ax_tables, dims,
                stream_resident, transcript, _prod_derive))
        if _trace:
            print(f"PROD compact-prefix: "
                  f"+{_time.perf_counter() - _t0:.2f}s", flush=True)
    elif stream_gen is not None:
        # OUT-OF-CORE mode: the broadcast cube is never materialized --
        # `claimed` accumulates over generated chunks, the prefix rounds
        # fold via host spill, and the resident tail below finishes.
        import time as _time

        _trace = _os.environ.get("VERATHOS_ATTN_TRACE") == "1"
        _t0 = _time.perf_counter()
        total = int(stream_cells)
        if total & (total - 1):
            raise ProofV3Error("streamed product size must be a power of two")
        sum_acc = 0
        for lo in range(0, total, stream_resident):
            ga, gb, gf = stream_gen(lo, min(lo + stream_resident, total))
            sum_acc = (sum_acc + gl_sum_t(
                gl_mul_t(gl_mul_t(ga, gb), gf))) % GOLDILOCKS_MODULUS
        claimed = sum_acc
        if _trace:
            print(f"PROD claimed: +{_time.perf_counter() - _t0:.2f}s",
                  flush=True)
            _t0 = _time.perf_counter()
        transcript = _prod_seed(
            statement, a_column.tree.commitment, b_column.tree.commitment,
            _factor_digest(components), claimed, validator_nonce,
        )
        a, b, f, transcript, rounds, challenges = _streamed_product_rounds(
            fold_extension, stream_gen, total, stream_resident,
            transcript, _prod_derive)
        if _trace:
            print(f"PROD stream-rounds: "
                  f"+{_time.perf_counter() - _t0:.2f}s", flush=True)
    else:
        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            _resume_device_values,
        )

        a = a_fold_device if a_fold_device is not None else (
            _resume_device_values(a_column))
        b = b_fold_device if b_fold_device is not None else (
            _resume_device_values(b_column))
        factor = to_field_tensor((1,), "cuda")
        for component in components:
            comp_t = to_field_tensor(component, "cuda")
            factor = gl_mul_t(
                factor.repeat_interleave(len(component)),
                comp_t.repeat(factor.numel()),
            )
        if factor.numel() != a.numel() or a.numel() != b.numel():
            raise ProofV3Error("succinct-product column lengths mismatch")
        claimed = gl_sum_t(gl_mul_t(gl_mul_t(a, b), factor))
        transcript = _prod_seed(
            statement, a_column.tree.commitment, b_column.tree.commitment,
            _factor_digest(components), claimed, validator_nonce,
        )
        rounds = []
        challenges = []
        f = factor
    if (stream_gen is None and compact_spec is None
            and hasattr(fold_extension, "fs_round_v2")):
        # zero-sync product rounds: the product-module derive has NO
        # label (domain + seed + <II>), absorb has no tag
        from verallm.proof_v3.goldilocks_succinct_product_argument_reference import (  # noqa: E501
            _CHALLENGE_DOMAIN as _PROD_DOMAIN,
        )

        n_rounds = a.numel().bit_length() - 1
        t_buf = torch.tensor(
            list(transcript), dtype=torch.uint8, device="cuda")
        dom_label = _u8_tensor_cached(_PROD_DOMAIN)
        rounds_buf = torch.zeros(
            (n_rounds, 4), dtype=torch.int64, device="cuda")
        chal_buf = torch.zeros(n_rounds, dtype=torch.int64, device="cuda")
        empty = torch.zeros(0, dtype=torch.int64, device="cuda")
        for r in range(n_rounds):
            partials = fold_extension.product_round_partials(a, b, f)
            fold_extension.fs_round_v2(
                partials, empty, 4, t_buf, dom_label, len(_PROD_DOMAIN),
                0, 0, r + 1, rounds_buf[r], chal_buf[r:r + 1])
            a = fold_extension.lerp_fold_ptr(a, chal_buf[r:r + 1])
            b = fold_extension.lerp_fold_ptr(b, chal_buf[r:r + 1])
            f = fold_extension.lerp_fold_ptr(f, chal_buf[r:r + 1])
        torch.cuda.synchronize()

        def _dec(v):
            return v + _TWO64 if v < 0 else v

        rounds = [
            tuple(_dec(v) for v in row)
            for row in rounds_buf.cpu().tolist()
        ]
        challenges = [_dec(v) for v in chal_buf.cpu().tolist()]
    while a.numel() > 1:
        partials = fold_extension.product_round_partials(a, b, f)
        torch.cuda.synchronize()
        values = [v + _TWO64 if v < 0 else v for v in partials.cpu().tolist()]
        sums = [0, 0, 0, 0]
        for index, value in enumerate(values):
            sums[index % 4] = (sums[index % 4] + value) % GOLDILOCKS_MODULUS
        evals = tuple(sums)
        rounds.append(evals)
        transcript = _h.sha256(
            transcript
            + b"".join(v.to_bytes(8, "little") for v in evals)
        ).digest()
        challenge = _prod_derive(transcript, len(rounds))
        challenges.append(challenge)
        encoded = _encode_challenge(challenge)
        a = fold_extension.lerp_fold(a, encoded)
        b = fold_extension.lerp_fold(b, encoded)
        f = fold_extension.lerp_fold(f, encoded)
    point = tuple(reversed(challenges))
    a_point = (
        point if a_point_map is None
        else tuple(point[i] for i in a_point_map))
    b_point = (
        point if b_point_map is None
        else tuple(point[i] for i in b_point_map))
    if collector is not None:
        def _terminal(t):
            v = int(t.cpu().item())
            return v + _TWO64 if v < 0 else v

        return GoldilocksSuccinctProductProofV3(
            claimed_sum=claimed,
            round_polynomials=tuple(rounds),
            a_opening=collector.defer(a_tag, a_point, _terminal(a)),
            b_opening=collector.defer(b_tag, b_point, _terminal(b)),
        )
    if a_point_map is not None or b_point_map is not None:
        raise ProofV3Error(
            "broadcast-free product claims require a deferred collector")
    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        _resume_device_values,
    )

    a_opening = fused_open_goldilocks_multilinear_v3(
        fold_extension=fold_extension, tree_extension=tree_extension,
        statement=a_column.pcs_statement, tree=a_column.tree,
        evaluations_device=_resume_device_values(a_column), point=point,
        validator_nonce=validator_nonce)
    b_opening = fused_open_goldilocks_multilinear_v3(
        fold_extension=fold_extension, tree_extension=tree_extension,
        statement=b_column.pcs_statement, tree=b_column.tree,
        evaluations_device=_resume_device_values(b_column), point=point,
        validator_nonce=validator_nonce)
    return GoldilocksSuccinctProductProofV3(
        claimed_sum=claimed,
        round_polynomials=tuple(rounds),
        a_opening=a_opening,
        b_opening=b_opening,
    )


_DEVICE_TABLE_CACHE: dict = {}


def _device_table_cached(table: tuple):
    from verallm.proof_v3.native_goldilocks_backend import to_field_tensor

    key = id(table)
    hit = _DEVICE_TABLE_CACHE.get(key)
    if hit is not None and hit[0] is table:
        return hit[1]
    dev = to_field_tensor(table, "cuda")
    if len(_DEVICE_TABLE_CACHE) < 256:
        _DEVICE_TABLE_CACHE[key] = (table, dev)
    return dev


def _device_padded_table_cached(statement):
    """Device tensor of the PADDED table (etf runs on the table cube)."""

    from verallm.proof_v3.native_goldilocks_backend import to_field_tensor

    key = (id(statement.table), "padded")
    hit = _DEVICE_TABLE_CACHE.get(key)
    if hit is not None and hit[0] is statement.table:
        return hit[1]
    dev = to_field_tensor(statement.padded_table(), "cuda")
    if len(_DEVICE_TABLE_CACHE) < 256:
        _DEVICE_TABLE_CACHE[key] = (statement.table, dev)
    return dev


def logup_aux_group_plan_v3(shapes):
    """Deterministic aux grouping: shapes = ordered (tag_prefix,
    witness_vars, table_vars). Returns per-kind {tag_prefix: (group_tag,
    block_point)} maps plus {group_tag: (vars, block_count)}."""

    plans = {}
    group_meta = {}
    for kind, var_index in (("M", 2), ("D", 1), ("E", 2)):
        buckets: dict[int, list[str]] = {}
        for shape in shapes:
            buckets.setdefault(shape[var_index], []).append(shape[0])
        kind_map = {}
        for vars_, prefixes in sorted(buckets.items()):
            group_tag = f"logup_aux/{kind}{vars_}"
            bits = max(1, (len(prefixes) - 1).bit_length())
            group_meta[group_tag] = (vars_ + bits, len(prefixes))
            for index, prefix in enumerate(prefixes):
                kind_map[prefix] = (
                    group_tag,
                    tuple((index >> j) & 1 for j in range(bits)))
        plans[kind] = kind_map
    return plans, group_meta


def fused_prove_logup_batch_v3(
    *,
    fold_extension,
    tree_extension,
    tile_digest: bytes,
    instances,
    validator_nonce: bytes,
    collector,
):
    """Phase-structured LogUp batch: aux columns of ALL instances commit
    into shared block trees (one batched opening per group).

    instances: ordered (statement, witness_column, tag_prefix,
    witness_tag). Returns the per-instance proofs in order.
    """

    import torch

    from verallm.proof_v3.goldilocks_succinct_logup_argument_reference import (
        _derive,
        _seed_transcript,
    )
    from verallm.proof_v3.goldilocks_succinct_zero_check_toolkit import (
        column_pcs_statement_v3,
    )
    from verallm.proof_v3.native_goldilocks_backend import (
        from_field_tensor,
        gl_inv_t,
        gl_mul_t,
        to_field_tensor,
    )

    shapes = tuple(
        (tag_prefix, statement.witness_variable_count,
         statement.table_variable_count)
        for statement, _column, tag_prefix, _wtag in instances)
    plans, group_meta = logup_aux_group_plan_v3(shapes)

    def _commit_group(group_tag, member_tensors):
        vars_total, used = group_meta[group_tag]
        cell = member_tensors[0].numel()
        blocks = 1 << (vars_total - (cell.bit_length() - 1))
        concat = torch.zeros(
            blocks * cell, dtype=torch.int64, device="cuda")
        for index, tensor in enumerate(member_tensors):
            concat[index * cell:(index + 1) * cell] = tensor
        pcs_statement = column_pcs_statement_v3(
            tile_digest, group_tag, vars_total)
        tree, _ = fused_commit_multilinear_tree(
            fold_extension, tree_extension, concat,
            statement=pcs_statement)
        import types as _types

        from verallm.proof_v3.goldilocks_succinct_batch_opening import (
            park_column_device_values_v3,
        )

        column_ns = _types.SimpleNamespace(
            pcs_statement=pcs_statement, tree=tree, values=None,
            device_values=concat, group_tag=None, block_point=())
        # nothing reads the group concat between commit and claims:
        # park it immediately (3 aux groups = a codeword's worth of
        # values stacked under the NEXT group's commit otherwise)
        park_column_device_values_v3(column_ns)
        collector.register_column(group_tag, column_ns)
        return tree, concat

    # ---- phase 1+2: DEVICE multiplicities and shared M trees ----
    import torch

    from verallm.proof_v3.goldilocks_succinct_batch_opening import (
        _resume_device_values,
    )

    prepped = []
    m_members: dict[str, list] = {}
    mask32 = torch.tensor(0xFFFFFFFF, dtype=torch.int64, device="cuda")
    for statement, column, tag_prefix, witness_tag in instances:
        w_dev = _resume_device_values(column)
        if w_dev is None or w_dev.numel() != statement.witness_size:
            # reference path (CPU columns or padded witnesses)
            witness = tuple(
                v % GOLDILOCKS_MODULUS for v in column.values)
            if len(witness) < statement.witness_size:
                witness = witness + (statement.table[0],) * (
                    statement.witness_size - len(witness))
            w_dev = to_field_tensor(witness, "cuda")
        # every wire table indexes by its low 32 bits (ranges are their
        # own index; packed tables use pack = 2^32)
        idx = torch.bitwise_and(w_dev, mask32)
        if int(idx.max()) >= statement.table_size or int(idx.min()) < 0:
            raise ProofV3Error(
                "succinct-logup witness value is not in the table")
        table_dev = _device_padded_table_cached(statement)
        if not bool(torch.equal(table_dev.index_select(0, idx), w_dev)):
            raise ProofV3Error(
                "succinct-logup witness value is not in the table")
        m_dev = torch.bincount(
            idx, minlength=statement.table_size).to(torch.int64)
        group_tag, block_point = plans["M"][tag_prefix]
        m_members.setdefault(group_tag, []).append(m_dev)
        collector.aliases[f"{tag_prefix}/M"] = (group_tag, block_point)
        prepped.append(dict(
            statement=statement, column=column, tag_prefix=tag_prefix,
            witness_tag=witness_tag, w_dev=w_dev, m_dev=m_dev))
    m_trees = {
        group_tag: _commit_group(group_tag, members)[0]
        for group_tag, members in m_members.items()
    }

    # ---- phase 3+4: betas, D/E columns, shared D/E trees ----
    d_members: dict[str, list] = {}
    e_members: dict[str, list] = {}
    from verallm.proof_v3.native_goldilocks_backend import gl_add_t

    for item in prepped:
        statement = item["statement"]
        tag_prefix = item["tag_prefix"]
        m_group_tag, _ = plans["M"][tag_prefix]
        seed = _seed_transcript(
            statement, item["column"].tree.commitment,
            m_trees[m_group_tag].commitment, validator_nonce)
        beta = _derive(seed, b"beta", 0)
        item["beta"] = beta
        w_dev = item["w_dev"]
        table_dev = _device_padded_table_cached(statement)
        beta_enc = beta - (1 << 64) if beta >= (1 << 63) else beta
        beta_t = torch.full(
            (1,), beta_enc, dtype=torch.int64, device="cuda")
        shift_dev = torch.cat((
            gl_add_t(w_dev, beta_t.expand_as(w_dev)),
            gl_add_t(table_dev, beta_t.expand_as(table_dev))))
        inverted = gl_inv_t(shift_dev)
        n_w = w_dev.numel()
        d_dev = inverted[:n_w].contiguous()
        e_dev = gl_mul_t(item["m_dev"], inverted[n_w:].contiguous())
        item["d_dev"] = d_dev
        item["d_column"] = None
        item["e_dev"] = e_dev
        item["e_column"] = None
        d_tag, d_point = plans["D"][tag_prefix]
        e_tag, e_point = plans["E"][tag_prefix]
        d_members.setdefault(d_tag, []).append(d_dev)
        e_members.setdefault(e_tag, []).append(e_dev)
        collector.aliases[f"{tag_prefix}/D0"] = (d_tag, d_point)
        collector.aliases[f"{tag_prefix}/E0"] = (e_tag, e_point)
    d_trees = {
        tag: _commit_group(tag, members)[0]
        for tag, members in d_members.items()
    }
    e_trees = {
        tag: _commit_group(tag, members)[0]
        for tag, members in e_members.items()
    }

    # ---- phase 5: per-instance sumchecks with the shared roots ----
    proofs = []
    for item in prepped:
        tag_prefix = item["tag_prefix"]
        aux_ctx = dict(
            m_tree=m_trees[plans["M"][tag_prefix][0]],
            m_dev=item["m_dev"],
            d_tree=d_trees[plans["D"][tag_prefix][0]],
            d_dev=item["d_dev"], d_column=item["d_column"],
            e_tree=e_trees[plans["E"][tag_prefix][0]],
            e_dev=item["e_dev"], e_column=item["e_column"],
            w_dev=item["w_dev"],
        )
        proofs.append(fused_prove_goldilocks_succinct_logup_v3(
            fold_extension=fold_extension, tree_extension=tree_extension,
            statement=item["statement"],
            looked_up_values=None,
            validator_nonce=validator_nonce,
            witness_column=item["column"],
            collector=collector, tag_prefix=tag_prefix,
            witness_tag=item["witness_tag"],
            aux_ctx=aux_ctx))
    return proofs


# ---------------------------------------------------------------------------
# COMPACT product prefix: fold the broadcast cube WITHOUT materializing it.
#
# The flat cube index is (h, t, s, d) MSB-first, so the sumcheck folds
# h, then t, then s, then d.  Through that whole prefix every operand
# is a broadcast of a COMPACT tensor over its own axes ("td" = (h,t,d),
# "sd" = (h,s,d), "row" = (h,t,s)) and the public factor is
# axis-separable -- so each round's evals contract on the compacts
# (exclusive axes first, then the shared axes) and every intermediate
# stays compact-sized.  Byte-identical eval/transcript sequence to the
# generated-chunk path: the sums are the same field values.
# ---------------------------------------------------------------------------

_KIND_AXES: dict = {
    "td": ("h", "t", "d"), "sd": ("h", "s", "d"),
    "row": ("h", "t", "s"),
}
_AXIS_ORDER = ("h", "t", "s", "d")


def _gl_mul_b(a, b):
    """Broadcast-safe exact mul: the CUDA elementwise kernel takes
    raw same-shape contiguous tensors, so materialize the broadcast
    first (compact-sized here)."""

    import torch

    from verallm.proof_v3.native_goldilocks_backend import gl_mul_t

    if a.shape != b.shape:
        shape = torch.broadcast_shapes(a.shape, b.shape)
        a = a.expand(shape)
        b = b.expand(shape)
    return gl_mul_t(a.contiguous(), b.contiguous())


def _gl_axis_sum(t, dim: int):
    """Exact modular reduction along ``dim`` (power-of-two size)."""

    from verallm.proof_v3.native_goldilocks_backend import gl_add_t

    while t.shape[dim] > 1:
        n = t.shape[dim] // 2
        t = gl_add_t(t.narrow(dim, 0, n).contiguous(),
                     t.narrow(dim, n, n).contiguous())
    return t


def _compact_contract(a_axes, a_t, b_axes, b_t, tables, sizes):
    """sum over the full cube of a * b * factor, all compact-sized."""

    from verallm.proof_v3.native_goldilocks_backend import gl_mul_t

    def _shaped(axes, t):
        shape = tuple(
            sizes[ax] if ax in axes else 1 for ax in _AXIS_ORDER)
        return t.reshape(shape)

    def _tab(ax):
        shape = tuple(
            sizes[ax] if ax2 == ax else 1 for ax2 in _AXIS_ORDER)
        return tables[ax].reshape(shape)

    shared = tuple(ax for ax in _AXIS_ORDER
                   if ax in a_axes and ax in b_axes)
    av = _shaped(a_axes, a_t)
    bv = _shaped(b_axes, b_t)
    for ax in a_axes:
        if ax not in shared:
            av = _gl_axis_sum(
                _gl_mul_b(av, _tab(ax)), _AXIS_ORDER.index(ax))
    for ax in b_axes:
        if ax not in shared:
            bv = _gl_axis_sum(
                _gl_mul_b(bv, _tab(ax)), _AXIS_ORDER.index(ax))
    out = _gl_mul_b(av, bv)
    for ax in shared:
        out = _gl_mul_b(out, _tab(ax))
    for ax in shared:
        out = _gl_axis_sum(out, _AXIS_ORDER.index(ax))
    # axes absent from BOTH operands still carry their factor mass
    for ax in _AXIS_ORDER:
        if ax not in a_axes and ax not in b_axes:
            out = _gl_mul_b(
                out, _gl_axis_sum(_tab(ax), _AXIS_ORDER.index(ax)))
    value = int(out.reshape(()).cpu().item())
    # encoded int64 -> canonical u64 (two's-complement bit pattern)
    return value + _TWO64 if value < 0 else value


def _lerp_top(t, axes, axis, z_enc, sizes):
    """Fold the TOP bit of ``axis``: lo + z * (hi - lo), compact."""

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_add_t,
        gl_mul_t,
        gl_sub_t,
    )

    if axis not in axes:
        return t
    dim = axes.index(axis)
    shape = tuple(sizes[ax] for ax in axes)
    tv = t.reshape(shape)
    n = shape[dim] // 2
    lo = tv.narrow(dim, 0, n).contiguous()
    hi = tv.narrow(dim, n, n).contiguous()
    return gl_add_t(
        lo, _gl_mul_b(
            z_enc.reshape(tuple(1 for _ in shape)),
            gl_sub_t(hi, lo))).reshape(-1)


def _compact_product_prefix(a_kind, a_enc, b_kind, b_enc, tables,
                            dims, resident_cells: int,
                            transcript: bytes, _prod_derive):
    """All prefix rounds on compacts; returns the materialized
    residual (a, b, f) + transcript/rounds/challenges -- drop-in for
    _streamed_product_rounds."""

    import hashlib as _h

    import torch

    from verallm.proof_v3.native_goldilocks_backend import (
        gl_mul_t,
        to_field_tensor,
    )

    hp, tp, sp, d = dims
    sizes = {"h": hp, "t": tp, "s": sp, "d": d}
    a_axes = _KIND_AXES[a_kind]
    b_axes = _KIND_AXES[b_kind]
    a_t = a_enc.reshape(-1).contiguous().cuda()
    b_t = b_enc.reshape(-1).contiguous().cuda()
    tables = {
        ax: tab.reshape(-1).contiguous().cuda()
        for ax, tab in zip(_AXIS_ORDER, tables)}
    z_encs = [
        to_field_tensor((z,), "cuda").reshape(
            tuple(1 for _ in _AXIS_ORDER))
        for z in range(4)]
    rounds: list = []
    challenges: list = []

    def _total():
        out = 1
        for ax in _AXIS_ORDER:
            out *= sizes[ax]
        return out

    while _total() > resident_cells:
        axis = next(ax for ax in _AXIS_ORDER if sizes[ax] > 1)
        evals = []
        half_sizes = dict(sizes)
        half_sizes[axis] //= 2
        for z in range(4):
            az = _lerp_top(a_t, a_axes, axis, z_encs[z], sizes)
            bz = _lerp_top(b_t, b_axes, axis, z_encs[z], sizes)
            fz = dict(tables)
            fz[axis] = _lerp_top(
                tables[axis], (axis,), axis, z_encs[z], sizes)
            evals.append(_compact_contract(
                a_axes, az, b_axes, bz, fz, half_sizes))
        evals = tuple(evals)
        rounds.append(evals)
        transcript = _h.sha256(
            transcript
            + b"".join(v.to_bytes(8, "little") for v in evals)
        ).digest()
        challenge = _prod_derive(transcript, len(rounds))
        challenges.append(challenge)
        c_enc = torch.tensor(
            [_encode_challenge(challenge)], dtype=torch.int64,
            device="cuda").reshape(tuple(1 for _ in _AXIS_ORDER))
        a_t = _lerp_top(a_t, a_axes, axis, c_enc, sizes)
        b_t = _lerp_top(b_t, b_axes, axis, c_enc, sizes)
        tables[axis] = _lerp_top(
            tables[axis], (axis,), axis, c_enc, sizes)
        sizes[axis] //= 2

    def _expand(axes, t):
        shape = tuple(
            sizes[ax] if ax in axes else 1 for ax in _AXIS_ORDER)
        full = tuple(sizes[ax] for ax in _AXIS_ORDER)
        return t.reshape(shape).expand(full).reshape(-1).contiguous()

    a_res = _expand(a_axes, a_t)
    b_res = _expand(b_axes, b_t)
    f_res = to_field_tensor((1,), "cuda")
    for ax in _AXIS_ORDER:
        f_res = gl_mul_t(
            f_res.repeat_interleave(sizes[ax]),
            tables[ax].repeat(f_res.numel()))
    return a_res, b_res, f_res, transcript, rounds, challenges
