"""Rich terminal panels for pool state, model catalog, and placement.

The web operator dashboard renders these concepts already (RTT map,
colored link chips, BEST badges, status dots); these helpers mirror the
same visual language for the terminal, through render.py's TTY gate so
piped/agent output stays plain and parseable. Shared by the setup wizard
today and any future manage surfaces.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from verallm.mesh import render


def short_worker_names(worker_ids: Sequence[str]) -> dict[str, str]:
    """Collapse a common host prefix: bad9511112bf-gpu0 -> gpu0.

    Only when EVERY id shares the prefix and the short names stay unique;
    otherwise ids pass through unchanged so nothing ambiguous is shown.
    GPU-group ids (host-gpu2-3) keep the whole group as the short name:
    the generic last-dash split would cut them to "3" and, worse, give
    the group a different "host" than its siblings, defeating the
    collapse for exactly the pools that need it most.
    """

    import re

    ids = list(dict.fromkeys(str(w) for w in worker_ids))
    if len(ids) < 2:
        return {w: w for w in ids}
    matches = [re.fullmatch(r"(.+?)-(gpu\d+(?:-\d+)?)", w) for w in ids]
    if all(matches) and len({m.group(1) for m in matches}) == 1:
        shorts = [m.group(2) for m in matches]
        if len(set(shorts)) == len(shorts):
            return dict(zip(ids, shorts))
    split = [w.rpartition("-") for w in ids]
    prefixes = {head for head, sep, _tail in split if sep}
    if len(prefixes) == 1 and all(sep for _h, sep, _t in split):
        shorts = [tail for _h, _s, tail in split]
        if len(set(shorts)) == len(shorts) and all(shorts):
            return dict(zip(ids, shorts))
    return {w: w for w in ids}


def _matrix_cell(ms: float, *, styled: bool) -> str:
    """One peer-rtt matrix cell, graded on mesh link quality."""

    text = f"{ms:.1f}ms" if ms < 10 else f"{ms:.0f}ms"
    if ms < 10:
        return render.green(text, styled=styled)
    if ms < 40:
        return render.yellow(text, styled=styled)
    return render.red(text, styled=styled)


def _rtt_chip(ms: float | None, *, styled: bool) -> str:
    if ms is None:
        return render.dim("rtt –", styled=styled)
    text = f"rtt {ms:.1f}ms" if ms < 10 else f"rtt {ms:.0f}ms"
    if ms <= 50:
        return render.green(text, styled=styled)
    if ms <= 150:
        return render.yellow(text, styled=styled)
    return render.red(text, styled=styled)


def _status_dot(status: str, stale: bool, *, styled: bool) -> str:
    if stale or status in ("error", "failed", "dead"):
        return render.red("●", styled=styled)
    if status == "serving":
        return render.green("●", styled=styled)
    if status in ("driving", "joining", "fetching", "assigned", "stopping"):
        return render.yellow("●", styled=styled)
    return render.dim("●", styled=styled)


def link_chip(link_class: str, rtt_ms: float, *, styled: bool) -> str:
    ms = f"{rtt_ms:.1f}ms" if rtt_ms < 10 else f"{rtt_ms:.0f}ms"
    text = f"{link_class} · {ms}"
    if link_class in ("local", "lan"):
        return render.green("● " + text, styled=styled)
    if link_class == "regional":
        return render.yellow("● " + text, styled=styled)
    return render.red("● " + text, styled=styled)


def _peer_ms(
    a: Mapping[str, Any], b: Mapping[str, Any], a_id: str, b_id: str
) -> float | None:
    ab = (a.get("peer_rtt_ms") or {}).get(b_id)
    ba = (b.get("peer_rtt_ms") or {}).get(a_id)
    values = [float(v) for v in (ab, ba) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _worker_host(worker_id: str) -> str:
    """The machine behind a worker id: <host>-gpu<group> -> <host>.

    Ids without the -gpu suffix (single-unit workers like a Mac) are their
    own host.
    """

    head, sep, tail = str(worker_id).rpartition("-gpu")
    if sep and tail and tail[0].isdigit():
        return head
    return str(worker_id)


def _link_class(ms: float) -> str:
    if ms < 2:
        return "local"
    if ms < 10:
        return "lan"
    if ms < 40:
        return "regional"
    if ms < 90:
        return "far"
    return "cross-region"


def pool_panel(
    view: Mapping[str, Any],
    *,
    pool_id: str = "",
    mode: str = "",
    styled: bool | None = None,
    links: bool = True,
) -> list[str]:
    """Machines + latency links, the terminal cousin of the dashboard map.

    ``links=False`` renders the machine rows only - the between-action
    overview on the manage board wants current state at a glance, not the
    latency map (that stays on the full board redraw).
    """

    if styled is None:
        styled = render.use_style()
    workers = dict(view.get("workers") or {})
    names = short_worker_names(list(workers))
    lines: list[str] = []
    total_vram = sum(
        float((w.get("capability") or {}).get("vram_gb", 0) or 0)
        for w in workers.values()
    )
    title = f"machines ({len(workers)} · {total_vram:.0f}G VRAM)"
    if pool_id:
        title = f"{pool_id}{' · ' + mode if mode else ''} · " + title
    lines.append("  " + render.bold(title, styled=styled))
    if not workers:
        lines.append(render.dim("    none joined yet", styled=styled))
        return lines
    if any(names[w] != w for w in workers):
        host = _worker_host(next(iter(workers)))
        lines.append(render.dim(f"    host {host}", styled=styled))
    rows = []
    for worker_id in sorted(workers):
        worker = workers[worker_id]
        capability = worker.get("capability") or {}
        status = str(worker.get("status", "") or "idle")
        stale = bool(worker.get("stale"))
        manager_ms = (worker.get("rtt_ms") or {}).get("manager")
        rows.append(
            [
                "    "
                + _status_dot(status, stale, styled=styled)
                + " "
                + render.bold(names[worker_id], styled=styled),
                str(capability.get("gpu_name", "?") or "?"),
                f"{float(capability.get('vram_gb', 0) or 0):.0f}G",
                _rtt_chip(
                    float(manager_ms) if manager_ms is not None else None,
                    styled=styled,
                ),
                render.badge(status + (" (stale)" if stale else ""), styled=styled),
            ]
        )
    lines.extend(render.table(rows, indent="", styled=styled))
    if not links:
        return lines
    # Peer links: pairwise lines for small pools, matrix for larger ones.
    ids = sorted(workers)
    pairs = []
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            ms = _peer_ms(workers[a_id], workers[b_id], a_id, b_id)
            if ms is not None:
                pairs.append((a_id, b_id, ms))
    if pairs and len(ids) <= 3:
        lines.append("")
        # Through the table renderer so the chips line up: pair names have
        # different lengths and plain concatenation left the link chips
        # ragged next to the aligned machine rows above.
        link_rows = [
            [
                "    " + render.dim("links", styled=styled),
                f"{names[a_id]} ⇄ {names[b_id]}",
                link_chip(_link_class(ms), ms, styled=styled),
            ]
            for a_id, b_id, ms in pairs
        ]
        lines.extend(render.table(link_rows, indent="", styled=styled))
    elif pairs or len(ids) > 3:
        # Larger pools collapse to MACHINES: same-host links are always
        # ~0.1ms and tell an operator nothing, while an N-worker matrix
        # stops fitting a terminal around 6 workers. Placement cares about
        # cross-machine links only, so the host matrix (best measured
        # worker link per host pair) is the decision-relevant view.
        hosts: list[str] = []
        host_workers: dict[str, list[str]] = {}
        for worker_id in ids:
            host = _worker_host(worker_id)
            if host not in host_workers:
                hosts.append(host)
                host_workers[host] = []
            host_workers[host].append(worker_id)

        def _host_ms(a_host: str, b_host: str) -> float | None:
            best: float | None = None
            for a_id in host_workers[a_host]:
                for b_id in host_workers[b_host]:
                    ms = _peer_ms(workers[a_id], workers[b_id], a_id, b_id)
                    if ms is not None and (best is None or ms < best):
                        best = ms
            return best

        if len(hosts) == 1:
            lines.append("")
            lines.append(
                render.dim(
                    f"    links  all {len(ids)} workers on one machine "
                    "(local)",
                    styled=styled,
                )
            )
        elif len(hosts) <= 8:
            lines.append("")
            lines.append(
                "    "
                + render.bold("machine ↔ machine rtt", styled=styled)
                + render.dim(
                    "   best worker link · – = no direct route between "
                    "the machines (or not probed yet; probes run every 10s)",
                    styled=styled,
                )
            )
            header = [""] + hosts
            matrix_rows = []
            for a_host in hosts:
                cells = ["    " + a_host]
                for b_host in hosts:
                    if a_host == b_host:
                        cells.append(render.dim("·", styled=styled))
                        continue
                    ms = _host_ms(a_host, b_host)
                    cells.append(
                        _matrix_cell(ms, styled=styled)
                        if ms is not None
                        else render.dim("–", styled=styled)
                    )
                matrix_rows.append(cells)
            lines.extend(
                render.table(
                    matrix_rows, header=header, indent="", styled=styled
                )
            )
        else:
            # Too many machines for a matrix: one digest line per host with
            # its best/worst measured peer and how many peers have no route.
            lines.append("")
            lines.append(
                "    "
                + render.bold("machine links", styled=styled)
                + render.dim(
                    "   best/worst measured peer per machine",
                    styled=styled,
                )
            )
            for a_host in hosts:
                measured: list[tuple[float, str]] = []
                unreachable = 0
                for b_host in hosts:
                    if b_host == a_host:
                        continue
                    ms = _host_ms(a_host, b_host)
                    if ms is None:
                        unreachable += 1
                    else:
                        measured.append((ms, b_host))
                if measured:
                    best_ms, best_host = min(measured)
                    worst_ms, worst_host = max(measured)
                    text = (
                        f"    {a_host}  best "
                        + link_chip(
                            _link_class(best_ms), best_ms, styled=styled
                        )
                        + f" ({best_host})"
                    )
                    if worst_host != best_host:
                        text += (
                            "  worst "
                            + link_chip(
                                _link_class(worst_ms), worst_ms, styled=styled
                            )
                            + f" ({worst_host})"
                        )
                else:
                    text = f"    {a_host}  " + render.dim(
                        "no measured peers", styled=styled
                    )
                if unreachable:
                    text += render.dim(
                        f"  · {unreachable} no route", styled=styled
                    )
                lines.append(text)
    elif len(ids) > 1:
        lines.append("")
        lines.append(
            render.dim(
                "    links  no peer measurements yet (workers probe each "
                "other every 10s; loopback-advertised workers are skipped)",
                styled=styled,
            )
        )
    return lines


_NETWORK_LABELS = {"test": "testnet", "finney": "mainnet"}


def _hours_text(seconds: float) -> str:
    if seconds <= 0:
        return "expired"
    hours = seconds / 3600.0
    if hours < 1:
        return f"{int(seconds // 60)}m"
    if hours < 48:
        return f"{hours:.0f}h"
    return f"{hours / 24:.0f}d"


def subnet_status_lines(
    view: Mapping[str, Any],
    *,
    chain: Mapping[str, Any] | None = None,
    styled: bool | None = None,
) -> list[str]:
    """The pool's subnet identity + on-chain standing, one or two lines.

    ``view`` carries the persisted identity (network, wallet, binding,
    stored registration) from ``flows._pool_view``; ``chain`` is the
    manager's cached metagraph read (``flows.pool_operator_score``), {}
    or None when unavailable. Dev pools render nothing: the mode label in
    the panel title already says "local only".
    """

    if styled is None:
        styled = render.use_style()
    if str(view.get("serving_mode", "") or "") != "subnet":
        return []
    binding = dict(view.get("validator_binding") or {})
    network = str(view.get("subtensor_network", "") or "")
    network_label = _NETWORK_LABELS.get(network, network or "?")
    netuid = binding.get("netuid")
    parts = [f"subnet: {network_label}"]
    if netuid is not None:
        parts[0] += f" (netuid {netuid})"
    wallet = str(view.get("wallet_name", "") or "")
    hotkey = str(view.get("wallet_hotkey", "") or "")
    if wallet:
        parts.append(f"hotkey {wallet}/{hotkey or 'default'}")
    elif view.get("coordinator_hotkey_ss58"):
        ss58 = str(view["coordinator_hotkey_ss58"])
        parts.append(f"hotkey {ss58[:8]}…{ss58[-4:]}")
    chain = dict(chain or {})
    lines: list[str] = []
    warn = ""
    if chain.get("available"):
        if chain.get("registered"):
            parts.append(f"UID {chain.get('uid')}")
            if chain.get("incentive") is not None:
                parts.append(f"incentive {chain['incentive']}")
            if chain.get("emission") is not None:
                parts.append(f"emission {chain['emission']}")
            if chain.get("stake") is not None:
                parts.append(f"stake {chain['stake']}τ")
            if chain.get("stale"):
                parts.append("cached")
        else:
            warn = (
                f"hotkey NOT registered on netuid {netuid}: local serving "
                "continues, subnet earnings stopped"
            )
    elif chain.get("pending"):
        # First chain read since the manager started is still in flight;
        # the warmer fills the cache and the next redraw shows it.
        parts.append("chain status: fetching…")
    elif chain.get("error"):
        parts.append("chain unreachable")
    else:
        parts.append("chain status unavailable")
    lines.append("  " + render.dim(" · ".join(parts), styled=styled))
    if warn:
        lines.append("  " + render.red(warn, styled=styled))
    return lines


def _registration_chip(
    view: Mapping[str, Any],
    mesh_key: str,
    mesh: Mapping[str, Any],
    *,
    styled: bool,
) -> str:
    """Per-mesh on-chain standing: registered index + lease, or local only.

    A mesh either carries the chain binding it was launched with
    (``model_index``, chain-bound lane) or serves unregistered (the
    measurement lane / dev pools). The stored registration adds the lease
    countdown when it points at this mesh.
    """

    if str(view.get("serving_mode", "") or "") != "subnet":
        return ""
    model_index = mesh.get("model_index")
    if model_index is None:
        # Unregistered lane: show what the hardware MEASURED (the value a
        # deploy would register), when a launch has measured it. The raw
        # status view carries "models" as the registry MAPPING, but the
        # board's assembled view uses "models" for a plain id LIST (with
        # the mapping under "models_detail") - dict() on that list
        # crashed the whole board.
        models = view.get("models")
        if not isinstance(models, Mapping):
            models = view.get("models_detail")
        entry: Mapping[str, Any] = {}
        if isinstance(models, Mapping):
            candidate = models.get(str(mesh.get("model_id", "") or ""))
            if isinstance(candidate, Mapping):
                entry = candidate
        measured = int(entry.get("measured_ctx_budget", 0) or 0)
        text = "local only"
        if measured > 0:
            text += f" · tested {_ctx_text(measured)}"
        return render.dim(text, styled=styled)
    import time

    registrations = dict(view.get("mesh_registrations") or {})
    if not registrations and view.get("mesh_registration"):
        # pre-multi-model managers publish only the single slot
        single = dict(view.get("mesh_registration") or {})
        registrations = {str(single.get("model_id", "?")): single}
    # The lease belongs to the MODEL at this chain index, not to one mesh
    # instance: a relaunch produces a new mesh_key while the registration
    # stays valid, so matching by mesh_key alone showed a freshly
    # relaunched registered model as "not registered".
    model_id = str(mesh.get("model_id", "") or "")
    registration = dict(registrations.get(model_id) or {})
    if registration and registration.get("index") is not None and int(
        registration["index"]
    ) != int(model_index):
        registration = {}
    if not registration:
        registration = next(
            (
                entry
                for entry in registrations.values()
                if str(entry.get("mesh_key", "") or "") == str(mesh_key)
            ),
            {},
        )
    # The contract this mesh serves under (registered context); canaries
    # arrive at exactly this size, so the operator should see it.
    contract = int(mesh.get("max_context_len", 0) or 0)
    ctx = f" · {_ctx_text(contract)}" if contract > 0 else ""
    if registration:
        text = f"on-chain #{model_index}{ctx}"
        expires_at = registration.get("expires_at")
        if expires_at:
            text += " · lease " + _hours_text(
                float(expires_at) - time.time()
            )
        return render.green(text, styled=styled)
    # Chain-bound (its snapshots sign this index) but no stored
    # registration: a deploy that stopped before registerModel, or a
    # retired entry. Saying "on-chain" here would overclaim.
    return render.yellow(
        f"bound #{model_index}{ctx} · not registered", styled=styled
    )


def _ctx_text(tokens: int) -> str:
    """Context length for humans: 98304 -> '96k ctx', 8192 -> '8k ctx'.

    Binary k, matching how model contexts are named (131072 = 128k)."""

    if tokens >= 1024:
        return f"{round(tokens / 1024)}k ctx"
    return f"{tokens} ctx"


def _age_text(created_at_unix: int) -> str:
    import time

    seconds = max(0, int(time.time()) - int(created_at_unix))
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h {minutes % 60:02d}m"
    return f"{hours // 24}d"


def mesh_board(
    view: Mapping[str, Any],
    *,
    styled: bool | None = None,
    score_of: Any = None,
) -> list[str]:
    """One row per mesh: model, colored state, placement, age, score.

    ``score_of`` (model_id -> base earning weight) orders rows by what
    each mesh earns, best first; without it (surfaces that must not pay
    the registry import, like ``mesh fleet``) rows group by model.
    """

    if styled is None:
        styled = render.use_style()
    meshes = dict(view.get("meshes") or {})
    workers = dict(view.get("workers") or {})
    names = short_worker_names(list(workers))
    lines: list[str] = []
    lines.append("  " + render.bold(f"meshes ({len(meshes)})", styled=styled))
    if not meshes:
        lines.append(render.dim("    none running", styled=styled))
        return lines

    def _order(mesh_key: str) -> tuple:
        model_id = str((meshes[mesh_key] or {}).get("model_id", "") or "")
        score = 0.0
        if score_of is not None:
            try:
                score = float(score_of(model_id) or 0.0)
            except Exception:
                score = 0.0
        return (-score, model_id, mesh_key)

    ordered_keys = sorted(meshes, key=_order)
    rows = []
    for mesh_key in ordered_keys:
        mesh = meshes[mesh_key] or {}
        status = str(mesh.get("status", "?") or "?")
        driver = str(mesh.get("driver", "") or "")
        if status == "serving" and not mesh.get("routing_ready", True):
            state = render.red("serving (NOT routable)", styled=styled)
        elif status == "serving":
            state = render.green("● serving", styled=styled)
        elif status in ("fetching", "driving", "joining"):
            detail = str(
                (workers.get(driver) or {}).get("status", "") or status
            )
            if detail == "driving":
                # The bare state between fetch and proof warmup is the
                # llama-server weight load, which reports no percentage;
                # name the phase instead of looking stalled.
                detail = "driving · loading the model"
            state = render.yellow(f"⟳ {detail}", styled=styled)
        elif status in ("stopping",):
            state = render.yellow("stopping", styled=styled)
        else:
            state = render.red(status, styled=styled)
        members = [
            names.get(str(m), str(m)) for m in (mesh.get("members") or [])
        ]
        score = mesh.get("validator_score")
        score_text = ""
        if isinstance(score, Mapping):
            if score.get("available") and score.get("score") is not None:
                score_text = f"score {score['score']}"
        elif score not in (None, ""):
            score_text = f"score {score}"
        created = int(mesh.get("created_at_unix", 0) or 0)
        # "up" claims a working mesh; a forming one is merely elapsing.
        age = ""
        if created:
            age = (
                f"up {_age_text(created)}"
                if status == "serving"
                else f"forming {_age_text(created)}"
                if status in ("fetching", "driving", "joining")
                else _age_text(created)
            )
        rows.append(
            [
                "    " + render.bold(str(mesh_key), styled=styled),
                str(mesh.get("model_id", "?") or "?"),
                state,
                _registration_chip(view, str(mesh_key), mesh, styled=styled),
                "on " + "+".join(members) if members else "",
                age,
                score_text,
            ]
        )
    lines.extend(render.table(rows, indent="", styled=styled))
    for mesh_key in ordered_keys:
        error = (meshes[mesh_key] or {}).get("error")
        if error:
            lines.append(
                render.dim(
                    f"    {mesh_key}: {error}", styled=styled
                )
            )
    return lines


def mesh_hardware_label(
    view: Mapping[str, Any], mesh: Mapping[str, Any]
) -> str:
    """Where a mesh runs, for pickers: "on gpu1 · A100-SXM4-80GB".

    Members collapse through short_worker_names; the GPU name comes from
    the first member's capability with the vendor prefix dropped (every
    row saying NVIDIA says nothing).
    """

    workers = dict(view.get("workers") or {})
    names = short_worker_names(list(workers))
    members = [str(m) for m in (mesh.get("members") or [])]
    if not members:
        return ""
    where = "+".join(names.get(m, m) for m in members)
    gpu = str(
        ((workers.get(members[0]) or {}).get("capability") or {}).get(
            "gpu_name", ""
        )
        or ""
    )
    gpu = gpu.removeprefix("NVIDIA ").strip()
    return f"on {where} · {gpu}" if gpu else f"on {where}"


def catalog_table(
    candidates: Sequence[Mapping[str, Any]], *, styled: bool | None = None
) -> tuple[list[str], list[Mapping[str, Any]]]:
    """The "what can this pool serve?" table.

    Returns (lines, launchable_rows_in_display_order); the caller numbers
    its prompt off the returned rows so table and selection always agree.
    """

    if styled is None:
        styled = render.use_style()
    launchable = [c for c in candidates if c.get("launchable")]
    hidden = [c for c in candidates if not c.get("launchable")]
    scores = [float(c.get("base_score", 0) or 0) for c in launchable]
    best = max(scores) if any(scores) else 0.0
    rows = []
    for index, cand in enumerate(launchable, start=1):
        size = (
            f"{cand.get('model_bytes', 0) / 1e9:.1f} GB"
            if cand.get("model_bytes")
            else "?"
        )
        score = float(cand.get("base_score", 0) or 0)
        if best > 0 and score > 0:
            rel = score / best * 100
            if rel >= 99.5:
                score_cell = render.c(f"★ {rel:.0f}%", "1;32", styled=styled)
            else:
                score_cell = f"{rel:.0f}%"
        else:
            score_cell = render.dim("?", styled=styled)
        mesh_status = str(cand.get("mesh_status", "") or "")
        if mesh_status == "serving":
            avail = render.green("● serving now", styled=styled)
        elif mesh_status:
            # A mesh is forming for this model right now; show its LIVE
            # progress ("fetching 6%") instead of a static claim.
            detail = str(cand.get("mesh_detail", "") or mesh_status)
            avail = render.yellow(f"⟳ {detail}", styled=styled)
        elif cand.get("in_pool"):
            avail = render.green("● ready · in the pool", styled=styled)
        elif cand.get("disk_files"):
            avail = render.green("● ready · local file", styled=styled)
        else:
            avail = render.cyan(
                f"↓ download ~{cand.get('model_bytes', 0) / 1e9:.0f} GB"
                if cand.get("model_bytes")
                else "↓ download",
                styled=styled,
            )
        fit_text = str(cand.get("fit", "") or "")
        fit = render.green(
            "✓ " + fit_text.removeprefix("fits ").replace("on one GPU", "one GPU"),
            styled=styled,
        )
        rows.append(
            [
                str(index),
                render.bold(str(cand["model_id"]), styled=styled),
                str(cand.get("quant", "") or "?").upper(),
                size,
                score_cell,
                avail,
                fit,
            ]
        )
    lines = render.table(
        rows,
        header=["#", "model", "quant", "size", "score", "availability", "fit"],
        align="rllrr",
        styled=styled,
    )
    unfit = [c for c in hidden if not c.get("unpublished")]
    for cand in unfit:
        lines.append(
            render.dim(
                f"    {cand['model_id']}: {cand.get('fit', '')}",
                styled=styled,
            )
        )
    unpublished = [c for c in hidden if c.get("unpublished")]
    if unpublished:
        # A ModelSpec-registered model always belongs in the Python
        # catalogue; this line appearing at all means the catalogue is
        # incomplete, and it says so.
        names = " · ".join(str(c["model_id"]) for c in unpublished)
        lines.append(
            render.warn(
                "missing from the model catalogue "
                f"(verallm/registry/models.py): {names}",
                styled=styled,
            )
        )
    if any(scores):
        lines.append(
            render.dim(
                "  score = relative base earning weight (validator demand "
                "bonuses come on top)",
                styled=styled,
            )
        )
    return lines, launchable


def placement_table(
    model_id: str,
    advice: Mapping[str, Any],
    workers: Mapping[str, Any],
    *,
    styled: bool | None = None,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    """The full placement recommendation, dashboard-style."""

    if styled is None:
        styled = render.use_style()
    suggestions = list(advice.get("suggestions") or [])
    reasons = dict(advice.get("reasons") or {})
    names = short_worker_names(
        list(workers)
        + [w for s in suggestions for w in (s.get("workers") or [])]
    )

    def vram_of(worker_ids: Sequence[str]) -> float:
        total = 0.0
        for worker_id in worker_ids:
            worker = workers.get(worker_id) or {}
            total += float(
                (worker.get("capability") or {}).get("vram_gb", 0) or 0
            )
        return total

    def gpu_of(worker_ids: Sequence[str]) -> str:
        # Worker ids are operator-chosen labels; the card type is a fact the
        # placement decision hinges on, so it gets its own column instead of
        # hoping the label mentions it.
        parts = []
        for worker_id in worker_ids:
            worker = workers.get(worker_id) or {}
            name = str(
                (worker.get("capability") or {}).get("gpu_name", "") or ""
            )
            for noise in ("NVIDIA ", "GeForce ", "Tesla "):
                name = name.replace(noise, "")
            parts.append(name.strip() or "?")
        return "+".join(parts)

    rows = []
    for index, suggestion in enumerate(suggestions, start=1):
        members = [str(w) for w in suggestion.get("workers") or []]
        star = (
            render.c("★", "1;32", styled=styled) if index == 1 else " "
        )
        note = ""
        if suggestion.get("fetch"):
            note = render.cyan(
                f"↓ ~{float(suggestion.get('download_gb', 0)):.0f} GB",
                styled=styled,
            )
        rows.append(
            [
                str(index),
                star,
                render.bold(
                    "+".join(names.get(w, w) for w in members),
                    styled=styled,
                ),
                gpu_of(members),
                names.get(
                    str(suggestion.get("driver", "") or ""),
                    str(suggestion.get("driver", "") or "?"),
                ),
                link_chip(
                    str(suggestion.get("link_class", "?") or "?"),
                    float(suggestion.get("max_rtt_ms", 0.0) or 0.0),
                    styled=styled,
                ),
                f"{vram_of(members):.0f}G",
                note,
            ]
        )
    lines = render.table(
        rows,
        header=["#", "", "workers", "gpu", "driver", "link", "vram", ""],
        align="rlllllr",
        styled=styled,
    )
    for index, suggestion in enumerate(suggestions, start=1):
        if suggestion.get("warn"):
            lines.append(
                render.warn(f"#{index}: {suggestion['warn']}", styled=styled)
            )
    # Only LIMITS are worth a line ("busy", "offline", "member only: …",
    # "pairs only: solo needs ~28 GB VRAM"). The recommender also emits an
    # "available (…)" reason for every healthy worker; printing those under
    # a "not part of these placements" header contradicted the table right
    # above it (the worker WAS row #2) and buried the real answer to "why
    # is my 4090 not offered solo?".
    limits = {
        worker_id: text
        for worker_id, text in reasons.items()
        if not str(text).startswith("available")
    }
    if limits:
        lines.append(render.dim("  worker limits:", styled=styled))
        for worker_id in sorted(limits):
            lines.append(
                render.dim(
                    f"    {names.get(worker_id, worker_id)}: "
                    f"{limits[worker_id]}",
                    styled=styled,
                )
            )
    return lines, suggestions
