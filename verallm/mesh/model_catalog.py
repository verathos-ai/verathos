"""What a pool can serve: the subnet's model catalog matched to its hardware.

The source of truth for servable models is the subnet owner's on-chain
ModelSpec — the operator picks from it, never registers anything. Local
GGUF files and the pool's own registry only ANNOTATE that list ("on this
machine, no download") or extend it for local-only pools. Fit against the
pool's actual workers uses the same VRAM headroom rule the manager's
placement recommender applies, so the quick annotation in a picker never
contradicts the full recommendation that follows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


# The chain list changes when the subnet owner registers a model, i.e.
# rarely; a short cache keeps interactive boards instant without going
# stale in any way that matters. monotonic-keyed, per network.
_SUBNET_CATALOG_TTL_S = 300.0
_subnet_catalog_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
# Disk layer under the in-memory one: a fresh CLI process (manage is
# reopened many times a day) otherwise re-reads the chain every start,
# which is the visible seconds the operator called annoying. Refresh
# ("board" in manage) bypasses; an unreachable chain falls back to the
# disk copy at ANY age, since a stale catalog beats an empty one.
# 15 minutes, NOT hours: as a primary source this cache hides the subnet
# owner's own registry changes — a model retired on-chain kept showing
# as launchable for the rest of a 6h window.
_SUBNET_CATALOG_DISK_TTL_S = 15 * 60.0
# Bound for the ONLY remaining foreground chain wait (a box with no disk
# cache at all): the sequential per-model spec reads took minutes against
# a congested node and hung interactive boards.
_SUBNET_CATALOG_FETCH_TIMEOUT_S = 45.0
_subnet_catalog_refresh_inflight: set[str] = set()


def _spawn_subnet_catalog_refresh(
    cache_key: str, network: str, repo_root: Path | None
) -> None:
    """Single-flight background refresh; the next redraw sees the result."""

    import threading

    if cache_key in _subnet_catalog_refresh_inflight:
        return
    _subnet_catalog_refresh_inflight.add(cache_key)

    def _run() -> None:
        try:
            subnet_model_catalog(network, repo_root, refresh=True)
        except Exception as exc:  # never let a refresh kill anything
            logger.debug("background catalog refresh failed: %s", exc)
        finally:
            _subnet_catalog_refresh_inflight.discard(cache_key)

    threading.Thread(
        target=_run, daemon=True, name=f"subnet-catalog-refresh-{cache_key}"
    ).start()


def _subnet_catalog_cache_path(network: str) -> Path:
    import re as _re

    safe = _re.sub(
        r"[^a-z0-9_-]+", "-", str(network or "default").lower()
    ).strip("-") or "default"
    return (
        Path.home() / ".verathos" / "cache" / f"subnet-catalog-{safe}.json"
    )


def _read_subnet_catalog_disk(
    network: str,
) -> tuple[float, list[dict[str, Any]]] | None:
    import json as _json

    try:
        payload = _json.loads(
            _subnet_catalog_cache_path(network).read_text(encoding="utf-8")
        )
        rows = payload.get("rows")
        fetched = float(payload.get("fetched_at_unix", 0) or 0)
        if isinstance(rows, list) and fetched > 0:
            return fetched, [dict(row) for row in rows]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def _write_subnet_catalog_disk(
    network: str, rows: list[dict[str, Any]]
) -> None:
    import json as _json
    import time as _time

    path = _subnet_catalog_cache_path(network)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(
                {"fetched_at_unix": _time.time(), "rows": rows}
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("subnet catalog disk cache not written: %s", exc)


def subnet_model_catalog(
    network: str,
    repo_root: Path | None = None,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """GGUF models the subnet owner registered on-chain; [] when unreachable.

    A read-only RPC call through the shipped chain config — no wallet is
    involved. Stale-while-revalidate: every call answers INSTANTLY from
    cache (memory 5 min, disk 15 min; any age when the chain is away) and
    an expired cache triggers a single-flight BACKGROUND refresh, visible
    on the next redraw. The chain read is one list call plus one spec
    call per model, so running it inline against a congested node can hang
    interactive boards for minutes. Only a box with no cache reads the chain in the
    foreground, and that read is hard-bounded. ``refresh=True`` forces
    the inline bounded read.
    """

    import time as _time

    cache_key = str(network or "")
    if not refresh:
        cached = _subnet_catalog_cache.get(cache_key)
        if cached and _time.monotonic() - cached[0] < _SUBNET_CATALOG_TTL_S:
            return [dict(row) for row in cached[1]]
        disk = _read_subnet_catalog_disk(cache_key)
        if disk is not None:
            fresh = _time.time() - disk[0] < _SUBNET_CATALOG_DISK_TTL_S
            if not fresh:
                # Serve stale NOW, refresh in the background: the next
                # redraw gets the fresh list without any load ever paying
                # the multi-second (worst case: hanging) chain read.
                _spawn_subnet_catalog_refresh(cache_key, network, repo_root)
            _subnet_catalog_cache[cache_key] = (
                _time.monotonic(),
                [dict(row) for row in disk[1]],
            )
            return [dict(row) for row in disk[1]]
        # No cache at all (first run on this box): kick the refresh and
        # wait for it BOUNDED; an empty catalog beats a hung board.
        _spawn_subnet_catalog_refresh(cache_key, network, repo_root)
        deadline = _time.monotonic() + _SUBNET_CATALOG_FETCH_TIMEOUT_S
        while _time.monotonic() < deadline:
            cached = _subnet_catalog_cache.get(cache_key)
            if cached:
                return [dict(row) for row in cached[1]]
            _time.sleep(0.25)
        logger.warning(
            "subnet model catalog read exceeded %.0fs; serving an empty "
            "chain list (the background refresh keeps running)",
            _SUBNET_CATALOG_FETCH_TIMEOUT_S,
        )
        return []

    from verallm.mesh.onboarding import resolve_network

    def _stale_disk_fallback() -> list[dict[str, Any]]:
        disk = _read_subnet_catalog_disk(cache_key)
        if disk is None:
            return []
        logger.debug(
            "subnet model catalog unreachable; serving the disk copy "
            "(age %.0fs)",
            _time.time() - disk[0],
        )
        return [dict(row) for row in disk[1]]

    try:
        binding = resolve_network(network, repo_root)
    except SystemExit:
        return _stale_disk_fallback()
    if binding is None:
        return []
    try:
        import json

        from verallm.chain.config import ChainConfig
        from verallm.chain.model_registry import ModelRegistryClient

        config_path = binding["chain_config"]
        client = ModelRegistryClient(ChainConfig.from_json(config_path))
        try:
            manifest_urls = tuple(
                json.loads(Path(config_path).read_text(encoding="utf-8")).get(
                    "mesh_manifest_base_urls"
                )
                or ()
            )
        except (OSError, ValueError):
            manifest_urls = ()
        rows: list[dict[str, Any]] = []
        for model_id in client.get_model_list():
            spec = client.get_model_spec(str(model_id))
            if spec is None:
                continue
            quant_mode = str(getattr(spec, "quant_mode", "") or "").lower()
            if not quant_mode.startswith("gguf_"):
                continue
            root = bytes(
                getattr(spec, "weight_merkle_root", b"") or b""
            ).hex()

            def _hex(name: str) -> str:
                raw = bytes(getattr(spec, name, b"") or b"")
                value = raw.hex()
                return value if any(raw) else ""

            rows.append(
                {
                    "model_id": str(model_id),
                    "model_package_hash": _hex("weight_file_hash"),
                    "tokenizer_hash": _hex("tokenizer_hash"),
                    "layers": int(getattr(spec, "num_layers", 0) or 0),
                    "quant": quant_mode[len("gguf_"):],
                    "tensor_manifest_root": root,
                    "manifest_urls": manifest_urls,
                    "hidden_dim": int(getattr(spec, "hidden_dim", 0) or 0),
                    "intermediate_dim": int(
                        getattr(spec, "intermediate_dim", 0) or 0
                    ),
                    "vocab_size": int(getattr(spec, "vocab_size", 0) or 0),
                    "num_experts": int(getattr(spec, "num_experts", 0) or 0),
                    "router_top_k": int(
                        getattr(spec, "router_top_k", 0) or 0
                    ),
                }
            )
        _subnet_catalog_cache[cache_key] = (
            _time.monotonic(),
            [dict(row) for row in rows],
        )
        _write_subnet_catalog_disk(cache_key, [dict(row) for row in rows])
        return rows
    except Exception as exc:
        logger.debug("subnet model catalog unavailable: %s", exc)
        return _stale_disk_fallback()


def base_score(model_id: str) -> float:
    """Base earning weight from the mesh model catalogue; 0.0 = unknown.

    Never invented: the number comes from verallm/registry/models.py
    (MESH_MODELS + GGUF quant ladder). A registered model that scores 0
    here is missing from the catalogue, which the UI surfaces as exactly
    that.
    """

    try:
        from verallm.registry.models import mesh_model_base_score

        score = mesh_model_base_score(str(model_id))
        return float(score) if score else 0.0
    except Exception as exc:
        logger.debug("mesh base score unavailable for %s: %s", model_id, exc)
        return 0.0


def _store_manifest_files(
    model_id: str, root: str, manifest_urls: tuple[str, ...]
) -> tuple[set[tuple[str, int]], str] | None:
    """The (filename, size) set the owner's manifest records, plus its path.

    Fetched from the gleipnir store (cached, root-verified); None when the
    store is unreachable or the model is not published there.
    """

    try:
        from verallm.mesh.gguf_manifest import load_gguf_tensor_manifest
        from verallm.mesh.manifest_store import (
            configured_mesh_manifest_base_urls,
            fetch_mesh_tensor_manifest,
        )

        base_urls = configured_mesh_manifest_base_urls(manifest_urls)
        if not base_urls or not root:
            return None
        fetched = fetch_mesh_tensor_manifest(
            model_id,
            expected_tensor_manifest_root=root,
            base_urls=base_urls,
        )
        manifest = load_gguf_tensor_manifest(fetched)
        files = {
            (Path(str(f.get("path", ""))).name, int(f.get("n_bytes", 0) or 0))
            for f in manifest.get("model_files") or []
        }
        return (files, str(fetched)) if files else None
    except Exception as exc:
        logger.debug("store manifest for %s unavailable: %s", model_id, exc)
        return None


def fit_annotation(
    model_bytes: int, workers: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, str]:
    """Whether (and how) a model fits this pool's GPUs, quick math only.

    Mirrors the manager's sizing rule (model bytes x VRAM_HEADROOM) so the
    picker's annotation and the full placement recommendation agree; RTT,
    busyness, and driver capability stay with the real recommender.
    """

    from verallm.mesh.pool import VRAM_HEADROOM

    # Per MACHINE, the VRAM of each of its GPUs. A worker owning several GPUs
    # advertises their sum in vram_gb, so sizing against that alone reported
    # "fits on one GPU" for a model far larger than any single GPU. The split
    # a model actually needs is what the operator has to know.
    machines: list[list[float]] = []
    for worker in workers.values():
        cap = worker.get("capability") or {}
        per_gpu = [
            float(v)
            for v in (cap.get("per_gpu_vram_gb") or [])
            if float(v or 0) > 0
        ]
        if not per_gpu:
            # Pre-multi-GPU workers advertise only the total; treat it as one.
            total = float(cap.get("vram_gb", 0) or 0)
            per_gpu = [total] if total > 0 else []
        if per_gpu:
            machines.append(sorted(per_gpu, reverse=True))
    if model_bytes <= 0:
        return False, "size unknown"
    if not machines:
        return False, "no workers joined"
    need_gb = model_bytes / 1e9 * VRAM_HEADROOM
    machines.sort(key=sum, reverse=True)

    if max(gpus[0] for gpus in machines) >= need_gb:
        return True, "fits on one GPU"
    # Prefer staying on ONE machine: local tensor-split beats RPC between
    # boxes, so a model that fits in a single machine's GPUs is placed there.
    for gpus in machines:
        if sum(gpus) >= need_gb:
            running = 0.0
            for count, vram in enumerate(gpus, start=1):
                running += vram
                if running >= need_gb:
                    return True, f"fits split across {count} GPUs on one machine"
    running = 0.0
    gpu_count = 0
    for machine_count, gpus in enumerate(machines, start=1):
        for vram in gpus:
            running += vram
            gpu_count += 1
            if running >= need_gb:
                return True, (
                    f"fits split across {gpu_count} GPUs on "
                    f"{machine_count} machines"
                )
    return False, (
        f"needs ~{need_gb:.0f} GB VRAM; this pool has {running:.0f} GB"
    )


def assemble_candidates(
    view: Mapping[str, Any],
    *,
    network: str = "",
    repo_root: Path | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Everything this pool could serve, one row per model.

    Merged from: the subnet's on-chain catalog (when a network is named
    and reachable), GGUF files already on this machine, and models the
    pool already knows (worker catalogs + download sources). Each row
    carries where the model would come from and whether it is launchable
    right now.
    """

    from verallm.mesh.units import discover_local_gguf_models

    candidates: dict[str, dict[str, Any]] = {}

    def _row(model_id: str) -> dict[str, Any]:
        return candidates.setdefault(
            model_id,
            {
                "model_id": model_id,
                "origins": set(),
                "model_bytes": 0,
                "layers": 0,
                "disk_files": [],
                "hf_repo": "",
                "hf_files": [],
                "in_pool": False,
                "quant": "",
                "base_score": 0.0,
                "mesh_status": "",
                "mesh_detail": "",
            },
        )

    subnet_rows = (
        subnet_model_catalog(network, repo_root, refresh=refresh)
        if network
        else []
    )
    for entry in subnet_rows:
        row = _row(entry["model_id"])
        row["origins"].add("subnet")
        row["layers"] = row["layers"] or int(entry.get("layers", 0) or 0)
        row["tensor_manifest_root"] = str(
            entry.get("tensor_manifest_root", "") or ""
        )
        row["manifest_urls"] = tuple(entry.get("manifest_urls") or ())
        row["quant"] = str(entry.get("quant", "") or "")
        row["model_package_hash"] = str(
            entry.get("model_package_hash", "") or ""
        )
        row["tokenizer_hash"] = str(entry.get("tokenizer_hash", "") or "")
        row["base_score"] = base_score(entry["model_id"])
        # The Python catalogue is the canonical source for every
        # ModelSpec-registered model's download source; a chain model
        # missing there is a catalogue bug, not a normal state.
        source = None
        try:
            from verallm.registry.models import mesh_model_source

            source = mesh_model_source(entry["model_id"])
        except Exception as exc:
            logger.debug(
                "mesh catalogue source unavailable for %s: %s",
                entry["model_id"],
                exc,
            )
        if source is not None:
            hf_repo, hf_files, model_bytes, layers = source
            row["hf_repo"] = row["hf_repo"] or hf_repo
            row["hf_files"] = row["hf_files"] or list(hf_files)
            row["model_bytes"] = max(
                int(row["model_bytes"]), int(model_bytes)
            )
            row["layers"] = row["layers"] or int(layers)

    # Disk shard sets matching a catalogue variant exactly (same file
    # basenames, same total bytes) fold under the canonical mesh id:
    # filename-derived ids rarely agree with the registered id
    # ("DeepSeek-V4-Flash-0731-UD-IQ1_M" vs
    # "deepseek-v4-flash-0731-iq1-m"), and the unmatched row would show
    # an unscored, quantless duplicate of a model the catalogue fully
    # describes.
    catalogue_by_files: dict[tuple[frozenset[str], int], str] = {}
    try:
        from verallm.registry.models import MESH_GGUF_MODELS

        for mesh_id, (_entry, variant) in MESH_GGUF_MODELS.items():
            catalogue_by_files[
                (
                    frozenset(Path(f).name for f in variant.hf_files),
                    int(variant.model_bytes),
                )
            ] = mesh_id
    except Exception as exc:
        logger.debug("mesh catalogue unavailable for disk match: %s", exc)

    for entry in discover_local_gguf_models():
        names = frozenset(Path(str(p)).name for p in entry["files"])
        model_id = catalogue_by_files.get(
            (names, int(entry["bytes"])), entry["model_id"]
        )
        row = _row(model_id)
        row["origins"].add("disk")
        row["disk_files"] = [str(path) for path in entry["files"]]
        row["model_bytes"] = max(
            int(row["model_bytes"]), int(entry["bytes"])
        )

    for worker in (view.get("workers") or {}).values():
        for item in worker.get("catalog") or []:
            model_id = str(item.get("model_id", "") or "")
            if not model_id:
                continue
            row = _row(model_id)
            row["origins"].add("pool")
            row["in_pool"] = True
            row["model_bytes"] = max(
                int(row["model_bytes"]),
                int(item.get("model_bytes", 0) or 0),
            )
            row["layers"] = row["layers"] or int(item.get("layers", 0) or 0)

    for model_id, entry in (view.get("models_detail") or {}).items():
        row = _row(str(model_id))
        row["origins"].add("pool")
        # A registry entry means the pool KNOWS the model (usually just
        # its download source) — it does NOT mean any worker holds the
        # file. "in_pool"/"ready" is reserved for worker-catalog entries;
        # conflating the two once showed a model as ready while its
        # first download was still at 6%.
        row["model_bytes"] = max(
            int(row["model_bytes"]),
            int((entry or {}).get("model_bytes", 0) or 0),
        )
        row["layers"] = row["layers"] or int(
            (entry or {}).get("layers", 0) or 0
        )
        if (entry or {}).get("hf_repo"):
            row["hf_repo"] = str(entry["hf_repo"])
            row["hf_files"] = [str(f) for f in entry.get("hf_files") or []]

    # Live mesh state per model: a mesh mid-formation (fetching, driving,
    # joining) or serving overrides any static availability text.
    workers_view = view.get("workers") or {}
    for mesh in (view.get("meshes") or {}).values():
        model_id = str((mesh or {}).get("model_id", "") or "")
        status = str((mesh or {}).get("status", "") or "")
        if not model_id or status in ("stopped", "stopping", "error"):
            continue
        row = _row(model_id)
        row["mesh_status"] = status
        driver = str((mesh or {}).get("driver", "") or "")
        driver_status = str(
            (workers_view.get(driver) or {}).get("status", "") or ""
        )
        # The driver's own status carries live progress ("fetching 6%",
        # "driving (warming proofs 30%)").
        row["mesh_detail"] = driver_status or status

    # Match disk files to subnet models through the OWNER's published
    # manifest (exact file names + sizes), never by guessing from names:
    # the chain id and a filename-derived id rarely agree ("qwen2.5-7b-
    # q4-k-m" vs "qwen2.5-7b-instruct-q4_k_m"), and a wrong guess would
    # launch a mesh whose digests fail much later. A matched disk copy
    # folds into the subnet row and the standalone disk row disappears.
    disk_only = [
        row
        for row in candidates.values()
        if row["disk_files"] and not row["in_pool"]
        and "subnet" not in row["origins"]
    ]
    if disk_only:
        disk_sets = {
            row["model_id"]: {
                (Path(f).name, Path(f).stat().st_size)
                for f in row["disk_files"]
                if Path(f).is_file()
            }
            for row in disk_only
        }
        for row in list(candidates.values()):
            if (
                "subnet" not in row["origins"]
                or row["disk_files"]
                or row["in_pool"]
            ):
                continue
            published = _store_manifest_files(
                row["model_id"],
                str(row.get("tensor_manifest_root", "") or ""),
                tuple(row.get("manifest_urls") or ()),
            )
            if published is None:
                continue
            want, manifest_path = published
            for disk_row in disk_only:
                if disk_sets.get(disk_row["model_id"]) == want:
                    row["disk_files"] = list(disk_row["disk_files"])
                    row["model_bytes"] = int(disk_row["model_bytes"])
                    row["store_manifest"] = manifest_path
                    candidates.pop(disk_row["model_id"], None)
                    disk_only.remove(disk_row)
                    break

    # Drop duplicate artifacts under legacy ids: a non-subnet row whose
    # exact bytes (and layers, when both know them) equal a subnet row's
    # is the same file added before the chain id was known. Only folded
    # when the subnet row is servable on its own, so nothing launchable
    # ever disappears.
    subnet_ids = [
        model_id
        for model_id, row in candidates.items()
        if "subnet" in row["origins"]
    ]
    for model_id in list(candidates):
        row = candidates[model_id]
        if "subnet" in row["origins"] or row["model_bytes"] <= 0:
            continue
        for subnet_id in subnet_ids:
            subnet_row = candidates.get(subnet_id)
            if subnet_row is None:
                continue
            same_bytes = subnet_row["model_bytes"] == row["model_bytes"]
            layers_agree = (
                not subnet_row["layers"]
                or not row["layers"]
                or subnet_row["layers"] == row["layers"]
            )
            standalone = bool(
                subnet_row["in_pool"]
                or subnet_row["disk_files"]
                or subnet_row["hf_repo"]
            )
            if same_bytes and layers_agree and standalone:
                candidates.pop(model_id)
                break

    workers = view.get("workers") or {}
    try:
        from verallm.registry.models import MESH_GGUF_MODELS

        retired_ids = {
            model_id
            for model_id, (entry, _variant) in MESH_GGUF_MODELS.items()
            if entry.retired
        }
    except Exception:
        retired_ids = set()
    rows: list[dict[str, Any]] = []
    for row in candidates.values():
        # Model onboarding is the subnet owner's process, no one
        # else's: the owner builds the tensor manifest, registers the
        # ModelSpec on-chain, and publishes the manifest to the store.
        # An artifact without that registration (a stray local GGUF, an
        # unregistered id) is not an operator-facing option at all -
        # only what the pool already holds or serves stays visible.
        if not (
            "subnet" in row["origins"]
            or row["in_pool"]
            or row["mesh_status"]
        ):
            continue
        # Registry backstop: a catalogue model reachable without the chain
        # (disk fold above, dev pools) still shows its quant, score, and
        # download source instead of a row of question marks.
        catalogue = None
        try:
            from verallm.registry.models import MESH_GGUF_MODELS

            catalogue = MESH_GGUF_MODELS.get(row["model_id"])
        except Exception:
            catalogue = None
        if catalogue is not None:
            entry, variant = catalogue
            row["quant"] = row["quant"] or variant.gguf_scheme
            row["base_score"] = row["base_score"] or base_score(
                row["model_id"]
            )
            row["hf_repo"] = row["hf_repo"] or (
                variant.hf_repo or entry.hf_repo
            )
            row["hf_files"] = row["hf_files"] or list(variant.hf_files)
            row["layers"] = row["layers"] or int(entry.layers)
            row["model_bytes"] = max(
                int(row["model_bytes"]), int(variant.model_bytes)
            )
        if row["model_id"] in retired_ids:
            # Superseded release: never offered for new launches. Still
            # shown (unlaunchable) while a mesh runs it or a worker holds
            # it, so an operator sees what they are actually serving.
            if not (row["in_pool"] or row["mesh_status"]):
                continue
            row["retired"] = True
        fits, fit_text = fit_annotation(int(row["model_bytes"]), workers)
        row["fits"] = fits
        row["fit"] = fit_text
        # The same sizing against IDLE machines only: what could launch
        # RIGHT NOW without stopping anything. The full-pool fit stays the
        # broad overview; picking from it alone let operators select models
        # whose capacity was entirely occupied and learn only from the
        # placement failure (live user feedback).
        idle_workers = {
            worker_id: worker
            for worker_id, worker in workers.items()
            if str(worker.get("status", "") or "") == "idle"
            and not worker.get("stale")
        }
        fits_now, fit_now_text = fit_annotation(
            int(row["model_bytes"]), idle_workers
        )
        row["fits_now"] = fits_now
        row["fit_now"] = fit_now_text
        row["unpublished"] = not (
            row["in_pool"] or row["disk_files"] or row["hf_repo"]
        )
        if row["in_pool"]:
            row["where"] = "in the pool"
        elif row["disk_files"]:
            row["where"] = "on this machine, no download"
        elif row["hf_repo"]:
            row["where"] = (
                f"download ~{row['model_bytes'] / 1e9:.1f} GB"
                if row["model_bytes"]
                else "download (size unknown)"
            )
        else:
            row["where"] = (
                "missing from the model catalogue "
                "(verallm/registry/models.py)"
            )
        if row.get("retired"):
            row["where"] = "retired · superseded release"
        row["launchable"] = bool(
            row["fits"] and not row["unpublished"] and not row.get("retired")
        )
        rows.append(row)
    # Launchable first, best earner on top, then by name.
    return sorted(
        rows,
        key=lambda r: (
            not r["launchable"],
            -float(r.get("base_score", 0) or 0),
            "subnet" not in r["origins"],
            r["model_id"],
        ),
    )
