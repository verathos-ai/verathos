"""The terminal panels: dashboard-style pool/catalog/placement views.

Piped output must stay plain and parseable; styled output must align
exactly like plain (widths use visible length, not byte length).
"""

from __future__ import annotations

import re

from verallm.mesh import panels, render

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _view():
    return {
        "workers": {
            "bad9511112bf-gpu0": {
                "status": "idle",
                "stale": False,
                "capability": {"gpu_name": "RTX 5090", "vram_gb": 31},
                "rtt_ms": {"manager": 0.3},
                "peer_rtt_ms": {"bad9511112bf-gpu1": 0.4},
                "catalog": [],
            },
            "bad9511112bf-gpu1": {
                "status": "serving",
                "stale": False,
                "capability": {"gpu_name": "RTX 5090", "vram_gb": 31},
                "rtt_ms": {"manager": 0.2},
                "peer_rtt_ms": {"bad9511112bf-gpu0": 0.4},
                "catalog": [],
            },
        }
    }


def test_short_worker_names_collapse_common_prefix():
    names = panels.short_worker_names(
        ["bad9511112bf-gpu0", "bad9511112bf-gpu1"]
    )
    assert names == {
        "bad9511112bf-gpu0": "gpu0",
        "bad9511112bf-gpu1": "gpu1",
    }
    # Different hosts: never collapse into ambiguity.
    names = panels.short_worker_names(["host-a-gpu0", "host-b-gpu0"])
    assert names == {"host-a-gpu0": "host-a-gpu0", "host-b-gpu0": "host-b-gpu0"}


def test_short_worker_names_keep_gpu_groups_whole():
    # A GPU-group worker (gpu2-3) must shorten to the whole group, not to
    # "3": the generic last-dash split also gave it a different "host"
    # than its siblings, so the pools that most need collapsing (one box,
    # mixed single/group workers) got none at all.
    names = panels.short_worker_names(
        [
            "792664a7aaf8-gpu0",
            "792664a7aaf8-gpu1",
            "792664a7aaf8-gpu2-3",
        ]
    )
    assert names == {
        "792664a7aaf8-gpu0": "gpu0",
        "792664a7aaf8-gpu1": "gpu1",
        "792664a7aaf8-gpu2-3": "gpu2-3",
    }


def test_pool_panel_link_chips_align(monkeypatch):
    view = {
        "workers": {
            "host-gpu0": {
                "capability": {"gpu_name": "A100", "vram_gb": 80},
                "status": "serving",
                "peer_rtt_ms": {"host-gpu1": 0.1, "host-gpu2-3": 0.1},
            },
            "host-gpu1": {
                "capability": {"gpu_name": "A100", "vram_gb": 80},
                "status": "serving",
                "peer_rtt_ms": {"host-gpu2-3": 0.1},
            },
            "host-gpu2-3": {
                "capability": {"gpu_name": "A100 x2", "vram_gb": 160},
                "status": "serving",
            },
        },
        "meshes": {},
    }
    lines = panels.pool_panel(view, styled=False)
    link_lines = [line for line in lines if "⇄" in line]
    assert len(link_lines) == 3
    # Chips (the "● local ..." cell) start at the same column on every
    # row even though the pair names differ in width.
    chip_columns = {line.index("● local") for line in link_lines}
    assert len(chip_columns) == 1


def test_mesh_board_orders_by_score_then_model():
    view = {
        "workers": {},
        "meshes": {
            "m-aaa": {"model_id": "alpha", "status": "serving"},
            "m-bbb": {"model_id": "beta", "status": "serving"},
            "m-ccc": {"model_id": "gamma", "status": "serving"},
        },
    }

    def order_of(lines):
        return [
            line.split()[0]
            for line in lines
            if line.strip().startswith("m-")
        ]

    scores = {"beta": 2.0, "gamma": 1.0, "alpha": 0.5}
    lines = panels.mesh_board(
        view, styled=False, score_of=lambda m: scores.get(m, 0)
    )
    # Best earner on top, exactly like the catalog table.
    assert order_of(lines) == ["m-bbb", "m-ccc", "m-aaa"]
    # Without a score source (mesh fleet must not pay the registry
    # import) rows group by model id instead.
    assert order_of(panels.mesh_board(view, styled=False)) == [
        "m-aaa",
        "m-bbb",
        "m-ccc",
    ]


def test_brand_banner_uses_the_artwork_and_degrades_plain():
    from verallm.mesh import render

    lines = ["VERATHOS · mesh manage", "protocols", "v0.1.0"]
    styled = render.brand_banner(lines, styled=True)
    joined = "\n".join(styled)
    # The logo is RENDERED FROM assets/logo.png (braille dots + a
    # 24-bit color sampled from the artwork), not hand-drawn box
    # characters. The exact RGB shifts with the render size, so pin the
    # truecolor-escape form rather than one sample.
    assert "⣿" in joined or "⣠" in joined
    assert re.search(r"\x1b\[38;2;\d+;\d+;\d+m", joined)
    for line in lines:
        assert line in joined
    # Piped output carries no art and no escapes: parseable text only.
    plain = render.brand_banner(lines, styled=False)
    assert plain == ["  " + line for line in lines]
    assert all("\x1b" not in line for line in plain)


def test_mesh_hardware_label_names_machine_and_gpu():
    view = {
        "workers": {
            "792664a7aaf8-gpu2-3": {
                "capability": {"gpu_name": "NVIDIA A100-SXM4-80GB x2"}
            },
            "792664a7aaf8-gpu0": {
                "capability": {"gpu_name": "NVIDIA A100-SXM4-80GB"}
            },
        }
    }
    label = panels.mesh_hardware_label(
        view, {"members": ["792664a7aaf8-gpu2-3"]}
    )
    assert label == "on gpu2-3 · A100-SXM4-80GB x2"
    # A worker with no capability entry still names the machine.
    label = panels.mesh_hardware_label(view, {"members": ["macbook.home"]})
    assert label == "on macbook.home"
    assert panels.mesh_hardware_label(view, {"members": []}) == ""


def test_pool_panel_plain_and_styled_align(monkeypatch):
    plain = panels.pool_panel(_view(), pool_id="pool-x", styled=False)
    text = "\n".join(plain)
    assert "\x1b" not in text
    assert "gpu0" in text and "gpu1" in text
    assert "RTX 5090" in text
    assert "gpu0 ⇄ gpu1" in text  # the latency link line
    assert "0.4ms" in text
    styled = panels.pool_panel(_view(), pool_id="pool-x", styled=True)
    assert any("\x1b" in line for line in styled)
    # Stripped styled output matches plain byte for byte.
    assert [_ANSI.sub("", line) for line in styled] == plain


def test_catalog_table_hides_unpublished_and_ranks_scores():
    candidates = [
        {
            "model_id": "qwen3.6-27b-q4-k-m",
            "quant": "q4_k_m",
            "model_bytes": 16_000_000_000,
            "base_score": 30.0,
            "in_pool": False,
            "disk_files": [],
            "hf_repo": "o/r",
            "launchable": True,
            "unpublished": False,
            "fit": "fits split across 2 GPUs",
        },
        {
            "model_id": "qwen2.5-7b-q4-k-m",
            "quant": "q4_k_m",
            "model_bytes": 4_700_000_000,
            "base_score": 20.0,
            "in_pool": False,
            "disk_files": ["/models/a.gguf"],
            "hf_repo": "",
            "launchable": True,
            "unpublished": False,
            "fit": "fits on one GPU",
        },
        {
            "model_id": "deepseek-v4-flash-iq3-xxs",
            "quant": "iq3_xxs",
            "model_bytes": 0,
            "base_score": 0.0,
            "in_pool": False,
            "disk_files": [],
            "hf_repo": "",
            "launchable": False,
            "unpublished": True,
            "fit": "size unknown",
        },
    ]
    lines, launchable = panels.catalog_table(candidates, styled=False)
    text = "\n".join(lines)
    assert [c["model_id"] for c in launchable] == [
        "qwen3.6-27b-q4-k-m",
        "qwen2.5-7b-q4-k-m",
    ]
    assert "★ 100%" in text  # best score starred
    assert "67%" in text  # 20/30 relative
    assert "● ready · local file" in text
    assert "↓ download ~16 GB" in text
    # Unpublished models take ONE dim line, not table rows.
    assert text.count("deepseek-v4-flash-iq3-xxs") == 1
    assert "missing from the model catalogue" in text
    assert "no download source published" not in text
    assert "\x1b" not in text


def test_placement_table_stars_best_and_colors_links():
    view = _view()
    advice = {
        "suggestions": [
            {
                "workers": ["bad9511112bf-gpu0", "bad9511112bf-gpu1"],
                "driver": "bad9511112bf-gpu0",
                "link_class": "local",
                "max_rtt_ms": 2.0,
            },
            {
                "workers": ["bad9511112bf-gpu1"],
                "driver": "bad9511112bf-gpu1",
                "link_class": "local",
                "max_rtt_ms": 0.0,
                "fetch": True,
                "download_gb": 5.0,
            },
        ],
        "reasons": {
            "bad9511112bf-gpu0": "available (can drive)",
            "bad9511112bf-gpu1": (
                "pairs only: solo needs ~28 GB VRAM (advertises 23 GB)"
            ),
        },
    }
    lines, suggestions = panels.placement_table(
        "m", advice, view["workers"], styled=False
    )
    text = "\n".join(lines)
    assert len(suggestions) == 2
    assert "★" in lines[1]  # row #1 carries the best badge
    assert "gpu0+gpu1" in text  # short names
    assert "RTX 5090+RTX 5090" in text  # card type column (vendor noise cut)
    assert "62G" in text  # summed VRAM
    assert "local · 2.0ms" in text
    assert "↓ ~5 GB" in text
    # Only LIMITS print; "available" reasons describe workers already in the
    # table and printing them under a limits header contradicted it.
    assert "worker limits:" in text
    assert "pairs only: solo needs ~28 GB VRAM" in text
    assert "available (can drive)" not in text
    assert "\x1b" not in text
    styled_lines, _ = panels.placement_table(
        "m", advice, view["workers"], styled=True
    )
    assert [_ANSI.sub("", line) for line in styled_lines] == lines


def test_table_aligns_colored_cells_like_plain():
    plain = render.table(
        [["1", "aaa", "9G"], ["2", "b", "31G"]],
        header=["#", "name", "vram"],
        align="rlr",
        styled=False,
    )
    colored = render.table(
        [
            ["1", render.green("aaa", styled=True), "9G"],
            ["2", "b", render.red("31G", styled=True)],
        ],
        header=["#", "name", "vram"],
        align="rlr",
        styled=False,
    )
    assert [_ANSI.sub("", line) for line in colored] == plain


def test_base_score_comes_from_the_registry_catalogue():
    from verallm.mesh import model_catalog
    from verallm.registry.models import mesh_model_base_score

    small = model_catalog.base_score("qwen2.5-7b-q4-k-m")
    big = model_catalog.base_score("qwen3.6-27b-q4-k-m")
    assert 0 < small < big
    assert small == mesh_model_base_score("qwen2.5-7b-q4-k-m")
    # Never invented: unknown models score 0 (rendered as "?"), they do
    # not get a made-up number.
    assert model_catalog.base_score("not-in-the-catalogue") == 0.0


def test_mesh_catalogue_covers_every_registered_model():
    """Every ModelSpec-registered mesh model needs catalogue facts: the
    download source AND a positive base score (REGISTER_NEW_MODEL.md)."""

    from verallm.registry.models import (
        MESH_GGUF_MODELS,
        mesh_model_base_score,
        mesh_model_source,
    )

    registered = (
        "qwen2.5-7b-q4-k-m",
        "qwen3.6-27b-q4-k-m",
        "qwen3.6-35b-a3b-q4-k-m",
        "deepseek-v4-flash-iq3-xxs",
    )
    for mesh_id in registered:
        assert mesh_id in MESH_GGUF_MODELS, f"{mesh_id} missing"
        repo, files, model_bytes, layers = mesh_model_source(mesh_id)
        assert repo and files and model_bytes > 0 and layers > 0
        score = mesh_model_base_score(mesh_id)
        assert score is not None and score > 0


def test_gguf_quant_ladder_is_monotonic():
    from verallm.registry.models import gguf_quant_quality

    ladder = ["q8_0", "q6_k", "q4_k_m", "q3_k_m", "iq3_xxs", "q2_k", "iq1_s"]
    values = [gguf_quant_quality(s) for s in ladder]
    assert values == sorted(values, reverse=True)
    assert gguf_quant_quality("unknown-scheme") == 0.80


def test_catalog_shows_live_mesh_progress_not_ready(monkeypatch):
    """A registry download source is NOT 'ready', and a mesh mid-fetch
    shows its live progress (the 'ready while downloading at 6%' bug)."""

    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "subnet_model_catalog", lambda *a, **k: []
    )
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module, "discover_local_gguf_models", lambda roots=None: []
    )
    view = {
        "workers": {
            "w-a": {
                "status": "fetching 6%",
                "capability": {"vram_gb": 31},
                "catalog": [],
            },
        },
        "models_detail": {
            "qwen3.6-27b-q4-k-m": {
                "hf_repo": "unsloth/Qwen3.6-27B-GGUF",
                "hf_files": ["Qwen3.6-27B-Q4_K_M.gguf"],
                "model_bytes": 16_817_244_384,
                "layers": 64,
            }
        },
        "meshes": {
            "m-1": {
                "model_id": "qwen3.6-27b-q4-k-m",
                "status": "fetching",
                "driver": "w-a",
            }
        },
    }
    rows = model_catalog.assemble_candidates(view, network="")
    row = next(r for r in rows if r["model_id"] == "qwen3.6-27b-q4-k-m")
    assert row["in_pool"] is False  # source known != file present
    assert row["mesh_status"] == "fetching"
    assert row["mesh_detail"] == "fetching 6%"
    lines, _ = panels.catalog_table(rows, styled=False)
    text = "\n".join(lines)
    assert "⟳ fetching 6%" in text
    assert "ready" not in text


def test_pool_panel_collapses_links_to_machines():
    from verallm.mesh import panels

    def worker(peer=None):
        return {
            "status": "idle",
            "capability": {"gpu_name": "A100", "vram_gb": 80},
            "rtt_ms": {"manager": 0.2},
            "peer_rtt_ms": peer or {},
        }

    view = {
        "workers": {
            "hosta-gpu0": worker({"hosta-gpu1": 0.1, "hostb-gpu0": 12.0}),
            "hosta-gpu1": worker({"hosta-gpu0": 0.1}),
            "hosta-gpu2-3": worker(),
            "hostb-gpu0": worker({"hosta-gpu0": 14.0}),
            "macbook-pro.home": worker(),
        }
    }
    lines = panels.pool_panel(view, styled=False)
    text = "\n".join(lines)
    # Machine-level matrix, not a 5x5 worker matrix.
    assert "machine ↔ machine rtt" in text
    assert "hosta" in text and "hostb" in text
    assert "macbook-pro.home" in text
    # Same-host worker ids never appear as matrix axes.
    matrix_part = text[text.index("machine ↔ machine rtt"):]
    assert "hosta-gpu0" not in matrix_part
    # Cross-machine link is the BEST measured worker pair (12 and 14 avg
    # per direction pair -> single pair 13.0), rendered as a regional link.
    assert "13" in matrix_part


def test_pool_panel_single_machine_links_line():
    from verallm.mesh import panels

    def worker(peer=None):
        return {
            "status": "idle",
            "capability": {"gpu_name": "A100", "vram_gb": 80},
            "rtt_ms": {"manager": 0.2},
            "peer_rtt_ms": peer or {},
        }

    view = {
        "workers": {
            "host-gpu0": worker({"host-gpu1": 0.1}),
            "host-gpu1": worker(),
            "host-gpu2": worker(),
            "host-gpu3": worker(),
        }
    }
    lines = panels.pool_panel(view, styled=False)
    text = "\n".join(lines)
    assert "all 4 workers on one machine" in text
    assert "machine ↔ machine rtt" not in text


def test_subnet_status_lines_render_identity_and_standing():
    view = {
        "serving_mode": "subnet",
        "subtensor_network": "test",
        "wallet_name": "test_miner96",
        "wallet_hotkey": "default",
        "validator_binding": {"netuid": 405, "chain_id": 945},
    }
    chain = {
        "available": True,
        "registered": True,
        "uid": 1,
        "incentive": 0.0123,
        "stake": 107.0,
        "emission": 0.5,
    }
    text = "\n".join(
        panels.subnet_status_lines(view, chain=chain, styled=False)
    )
    assert "subnet: testnet (netuid 405)" in text
    assert "hotkey test_miner96/default" in text
    assert "UID 1" in text
    assert "incentive 0.0123" in text
    assert "emission 0.5" in text
    # A dev pool renders nothing: the title mode label already says it.
    assert panels.subnet_status_lines({"serving_mode": "dev"}) == []


def test_subnet_status_lines_warn_on_deregistered_hotkey():
    """Chain answered, no UID: serving continues but earnings stopped.

    Distinct from "chain status unavailable" (RPC down), which must not
    scare the operator with a deregistration warning."""
    view = {
        "serving_mode": "subnet",
        "subtensor_network": "test",
        "wallet_name": "m",
        "wallet_hotkey": "default",
        "validator_binding": {"netuid": 405},
    }
    dereg = "\n".join(
        panels.subnet_status_lines(
            view, chain={"available": True, "registered": False}, styled=False
        )
    )
    assert "NOT registered on netuid 405" in dereg
    assert "local serving continues" in dereg
    down = "\n".join(
        panels.subnet_status_lines(view, chain={}, styled=False)
    )
    assert "chain status unavailable" in down
    assert "NOT registered" not in down


def test_subnet_status_lines_distinguish_fetching_and_stale():
    """The first chain read after a manager (re)start takes 10-30s; the
    manager answers ``pending`` instantly and the board must say
    "fetching", never the scary "unavailable". A value served from the
    restart-persisted cache is marked ``stale`` and renders "cached"."""
    view = {
        "serving_mode": "subnet",
        "subtensor_network": "test",
        "wallet_name": "m",
        "wallet_hotkey": "default",
        "validator_binding": {"netuid": 405},
    }
    fetching = "\n".join(
        panels.subnet_status_lines(
            view,
            chain={"available": False, "pending": True},
            styled=False,
        )
    )
    assert "chain status: fetching" in fetching
    assert "unavailable" not in fetching
    unreachable = "\n".join(
        panels.subnet_status_lines(
            view,
            chain={"available": False, "error": "RuntimeError: 429"},
            styled=False,
        )
    )
    assert "chain unreachable" in unreachable
    cached = "\n".join(
        panels.subnet_status_lines(
            view,
            chain={
                "available": True,
                "registered": True,
                "uid": 1,
                "incentive": 0.0123,
                "stale": True,
            },
            styled=False,
        )
    )
    assert "UID 1" in cached
    assert "cached" in cached


def test_mesh_board_shows_on_chain_standing_per_mesh():
    import time as time_module

    view = {
        "serving_mode": "subnet",
        "workers": {},
        "meshes": {
            "m-reg": {
                "model_id": "glm-5.2-iq2-m",
                "status": "serving",
                "routing_ready": True,
                "model_index": 40,
                "max_context_len": 98_304,
            },
            "m-loc": {"model_id": "qwen2.5-7b-q4-k-m", "status": "serving",
                      "routing_ready": True},
        },
        "models": {
            # A measurement launch already sized this one.
            "qwen2.5-7b-q4-k-m": {"measured_ctx_budget": 131_072},
        },
        "mesh_registration": {
            "model_id": "glm-5.2-iq2-m",
            # Stale mesh_key from BEFORE a relaunch: the lease belongs to
            # the model+index, so the badge must still say on-chain.
            "mesh_key": "m-before-relaunch",
            "index": 40,
            "expires_at": time_module.time() + 22 * 3600,
        },
    }
    text = "\n".join(panels.mesh_board(view, styled=False))
    assert "on-chain #40" in text
    # The registered contract, humanized: canaries arrive at this size.
    assert "96k ctx" in text
    assert "lease 22h" in text
    assert "local only" in text
    # Unregistered lane shows what the hardware measured instead.
    assert "tested 128k ctx" in text
    # The BOARD-shaped view carries "models" as a plain id LIST (the
    # mapping lives under "models_detail"); dict() on that list crashed
    # the whole board . It must render, measured chip
    # included when the detail mapping has it.
    board_view = {
        **view,
        "models": ["glm-5.2-iq2-m", "qwen2.5-7b-q4-k-m"],
        "models_detail": {
            "qwen2.5-7b-q4-k-m": {"measured_ctx_budget": 131_072},
        },
    }
    board_text = "\n".join(panels.mesh_board(board_view, styled=False))
    assert "on-chain #40" in board_text
    assert "tested 128k ctx" in board_text
    # Dev pools show neither chip: everything is local by definition.
    dev_text = "\n".join(
        panels.mesh_board({**view, "serving_mode": "dev"}, styled=False)
    )
    assert "on-chain" not in dev_text and "local only" not in dev_text


def test_fits_now_sized_against_idle_machines_only():
    """The broad fit column sizes against ALL machines; the launch picker
    sizes against IDLE ones. A model whose only fitting machine is busy
    serving is fits=True but fits_now=False (live user feedback: picking
    it produced nothing but a placement failure)."""
    from verallm.mesh import model_catalog

    catalog = [
        {
            "model_id": "big-model",
            "hf_repo": "org/big",
            "hf_files": ["big.gguf"],
            "model_bytes": 200_000_000_000,
            "layers": 64,
        },
        {
            "model_id": "small-model",
            "hf_repo": "org/small",
            "hf_files": ["small.gguf"],
            "model_bytes": 4_700_000_000,
            "layers": 28,
        },
    ]
    view = {
        "workers": {
            "big": {
                "status": "serving",
                "capability": {
                    "vram_gb": 320,
                    "per_gpu_vram_gb": [80, 80, 80, 80],
                },
                "catalog": catalog,
            },
            "small": {
                "status": "idle",
                "capability": {"vram_gb": 23, "per_gpu_vram_gb": [23]},
                "catalog": catalog,
            },
        },
        "models_detail": {},
        "meshes": {},
    }
    rows = model_catalog.assemble_candidates(view, network="")
    big = next(r for r in rows if r["model_id"] == "big-model")
    small = next(r for r in rows if r["model_id"] == "small-model")
    assert big["fits"] is True  # the pool CAN serve it (broad overview)
    assert big["fits_now"] is False  # but its machine is busy right now
    assert small["fits"] is True
    assert small["fits_now"] is True  # the idle 4090-class machine takes it

    # A stale "idle" worker must not count as free capacity.
    view["workers"]["small"]["stale"] = True
    rows = model_catalog.assemble_candidates(view, network="")
    small = next(r for r in rows if r["model_id"] == "small-model")
    assert small["fits_now"] is False
