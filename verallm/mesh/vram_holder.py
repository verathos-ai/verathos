"""Resident VRAM holder child for the hot-capacity audit workspace.

Runs as ``python -m verallm.mesh.vram_holder <hold_mb> <parent_pid>`` with
``CUDA_VISIBLE_DEVICES`` masked to exactly one GPU. Pins the audit
workspace from daemon start so the serving stack (llama KV auto-fit, lazy
graph growth) sizes itself around it.

Zero-gap window handoff: at an audit window the bench does NOT run in a
fresh subprocess. The parent sends a ``bench`` command on stdin and the
combined bench executes INSIDE this process. Dropping the hold buffer
returns its pages to this process's torch caching allocator — never to
CUDA — so the llama server can never claim them while the bench spins up,
and the re-hold after the bench is served from the same cache with no
window in which the VRAM is free. The old kill-holder/spawn-bench swap
left the full workspace unowned for the seconds a fresh interpreter needs
to import torch, and llama's opportunistic growth ratcheted into it a
little more at every window until benches OOM'd.

Protocol, line-based and parent-driven:

* stdout: ``held`` once the buffer is resident, then
  ``VRAM_HOLDER_EVT {json}`` event lines (``bench_started``,
  ``bench_done``, ``pong``, ``protocol_error``). Anything else on stdout
  is diagnostic noise and ignored by the parent.
* stdin: one JSON object per line. ``{"cmd": "bench", "argv": [...],
  "sys_path": [...], "lease_id": ...}`` runs the combined bench
  in-process; ``{"cmd": "ping"}`` answers ``pong``.

Exits when the original parent goes away (ppid comparison — orphans
reparent to a subreaper on desktop systems, never pid 1) or stdin hits
EOF, and with status 3 when a post-bench re-hold fails so the parent's
``ensure()`` replaces it.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

EVENT_PREFIX = "VRAM_HOLDER_EVT "

#: Exit status when the post-bench re-hold could not allocate: the hold is
#: gone, so dying loudly lets the parent respawn a fresh holder.
REHOLD_FAILED_EXIT = 3


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(EVENT_PREFIX + json.dumps(event, sort_keys=True) + "\n")
    sys.stdout.flush()


def _import_bench(sys_path: list[str]):
    """Import the combined bench harness, tree layout first.

    ``sys_path`` carries the parent's script directory so a holder spawned
    with a minimal env (daemon-start path) still resolves the harness.
    """
    for entry in reversed([str(p) for p in sys_path if str(p).strip()]):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        import bench_combined  # type: ignore  # source-tree layout

        return bench_combined
    except ImportError:
        from hot_capacity_workspace import (  # type: ignore  # release wheel
            bench_combined,
        )

        return bench_combined


def _write_final_summary(args, summary: dict[str, Any]) -> None:
    """Replicates ``bench_combined.main()``'s artifact writes.

    The stdout summary print is intentionally skipped: this process's
    stdout is the holder protocol channel and the collector consumes the
    file artifacts only.
    """
    if getattr(args, "child", False) and getattr(args, "out_dir", None):
        args.out_dir.mkdir(parents=True, exist_ok=True)
        child_text = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        (args.out_dir / f"{args.lease_id}_final.json").write_text(
            child_text + "\n"
        )
    out = getattr(args, "out", None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _rehold(state: dict[str, Any], hold_mb: int, torch) -> str:
    """Re-pin the hold buffer; returns an error string when it failed.

    Normally served straight from this process's caching allocator (the
    bench freed its tensors back into it), so no VRAM is ever exposed.
    The empty_cache retry only matters when fragmentation forced torch to
    grow fresh segments during the bench.
    """
    for attempt in range(2):
        try:
            state["buf"] = torch.empty(
                int(hold_mb) * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            return ""
        except Exception as exc:
            if attempt == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                return str(exc)
    return "unreachable"


def _run_bench(command: dict[str, Any], state: dict[str, Any], hold_mb: int, torch) -> None:
    lease = str(command.get("lease_id", "") or "")
    try:
        bench = _import_bench(list(command.get("sys_path") or []))
        argv = [str(a) for a in (command.get("argv") or [])]
        args = bench.build_parser().parse_args(argv)
    except BaseException as exc:  # noqa: BLE001 - report, keep holding
        _emit(
            {
                "event": "bench_done",
                "lease_id": lease,
                "rc": 1,
                "held": state.get("buf") is not None,
                "error": f"bench import/argv failed: {exc}"[:2000],
            }
        )
        return
    # Hand the workspace to the bench INSIDE this process: dropping the
    # reference returns the pages to the torch caching allocator, never
    # to CUDA, so nothing outside this process can claim them.
    state["buf"] = None
    _emit({"event": "bench_started", "lease_id": lease})
    rc = 0
    error = ""
    try:
        summary = bench.run(args)
        _write_final_summary(args, summary)
        del summary
    except BaseException as exc:  # noqa: BLE001 - bench failures are data
        rc = 1
        error = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
    # The bench frame is gone; sweep cycles so every workspace tensor is
    # back in the cache before the re-hold claims it.
    import gc

    gc.collect()
    rehold_error = _rehold(state, hold_mb, torch)
    _emit(
        {
            "event": "bench_done",
            "lease_id": lease,
            "rc": rc,
            "held": state.get("buf") is not None,
            "error": error[:2000],
            "rehold_error": rehold_error[:500],
        }
    )
    if state.get("buf") is None:
        # The hold is gone; die loudly so the parent replaces this holder.
        sys.exit(REHOLD_FAILED_EXIT)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hold_mb = int(argv[0])
    parent = int(argv[1])

    import select

    import torch

    torch.zeros(1, device="cuda")
    state: dict[str, Any] = {
        "buf": torch.empty(
            hold_mb * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
    }
    print("held", flush=True)

    while True:
        # Die with the daemon: an orphaned holder survives a process-manager
        # restart and its pinned VRAM can OOM the fresh serve launch.
        if os.getppid() != parent:
            return 0
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 2.0)
        except Exception:
            return 0
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            # stdin EOF: the parent closed the pipe or died.
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except Exception:
            _emit({"event": "protocol_error", "error": "bad json line"})
            continue
        if not isinstance(command, dict):
            _emit({"event": "protocol_error", "error": "not an object"})
            continue
        cmd = str(command.get("cmd", "") or "")
        if cmd == "bench":
            _run_bench(command, state, hold_mb, torch)
        elif cmd == "ping":
            _emit({"event": "pong"})
        else:
            _emit(
                {"event": "protocol_error", "error": f"unknown cmd {cmd!r}"}
            )


if __name__ == "__main__":
    sys.exit(main())
