# Verathos llama.cpp patches

The mesh serves inference through a patched llama.cpp: the coordinator runs a
patched `llama-server`, and each proving stage runs a patched
`verathos-rpc-server` (copied from the built `rpc-server`). The patches add
proof-capture instrumentation — every sampled GGEMM's witness (int8 weights,
inputs, output) is dumped during serve so the sidecar can prove it, gated by an
enable-file and a per-capture budget so steady-state cost stays low.

## Base

Upstream llama.cpp commit `e79e4bf66` (see UPSTREAM_BASE.txt).

## Patch series

- `0001-verathos-proof-capture-cuda-cpu-rpc-server.patch` — CPU, CUDA, RPC, and
  server changes. Apply for a CUDA build (any sm_XX; arch is a cmake flag).
- `0002-verathos-proof-capture-metal.patch` — Metal backend instrumentation
  ONLY (`ggml-metal/*`). A pure add-on: it applies on top of the full
  0001-rooted series, which already provides the CPU/RPC/server changes.
  Apply last for an Apple Silicon build.
- `0003-verathos-streaming-execution-anchors.patch` — streaming prefill
  execution anchors (inert unless `VERATHOS_GGML_ANCHOR_TENSORS` is set).
- `0005-verathos-mmid-decode-intra-parity.patch` — MUL_MAT_ID intra parity
  between eager and graph-captured decode.
- `0007-verathos-rpc-foreign-view-serialization.patch` — RPC serialization of
  foreign-buffer views (multi-device split serving).
- `0008-verathos-wildcard-op-arming.patch` — `*:intra` ord-agnostic dump
  arming (small-row instances only).
- `0009-verathos-name-keyed-op-arming.patch` — `n:<weight name>` dump arming:
  graph- and intra-agnostic, small-row only. The identity that survives
  architectures with length-dependent op streams (glm-dsa).
- `0010-verathos-decode-priority-interleave.patch` — decode-priority batch
  interleaving in the server scheduler (see section below). Server-only C++,
  platform-independent; applied last on both platforms.
- `0011-verathos-prefill-fairness.patch` — smallest-remaining-first prompt
  fill order in the server scheduler (see section below). Server-only C++,
  platform-independent; applied after 0010 on both platforms.

CUDA builds apply 0001 then 0003, 0005, 0007, 0008, 0009, 0010, 0011 in order;
Metal builds apply the same series and then 0002 before 0010. Both platforms therefore
share identical cpu/rpc/tools-server code; 0002 adds only the Metal backend
hooks (its former standalone cpu/rpc/server variants were older forks of the
same design and were dropped in favor of the series versions).

## Build

```bash
bash patches/llama.cpp/build.sh --backend cuda    # or: metal
```

CUDA builds target the first detected local GPU. Cross-builds without an
accessible GPU must pass the architecture explicitly, for example
`--cuda-architectures 89` for an RTX 4090.

Build parallelism is capped at 8 jobs by default to avoid exhausting a desktop
operator host. Set `VERATHOS_BUILD_JOBS` to a positive integer to choose an
explicit limit.

The script owns and resets its cache checkout. A custom `--src` must therefore
be an empty path or a checkout previously created by this script; it refuses to
force-reset an unmanaged llama.cpp worktree.

Produces `llama-server` and `verathos-rpc-server` under the build dir.

## Streaming return_tokens extension (in 0001)

0001 makes the OpenAI-compat streaming routes honor
`return_tokens`: each chunk carries its sampled token ids as a top-level
`"tokens"` extension field (the same non-standard top-level family as
`"timings"`). Upstream ignores `return_tokens` while streaming, so the
mesh worker previously requested `logprobs` purely to learn the sampled
ids, and llama-server sorts the full vocab per generated token for
logprobs. Measured on a 4090 mesh at 8-way concurrency, 512-token
streamed outputs: that logprobs path alone was -30% aggregate
throughput; with this extension the worker never requests logprobs on
organic serves. Server-only C++ (`tools/server`), platform-independent;
both platforms get it through 0001.

## Eager-on-selected-dump guard (in 0001)

`0001-*.patch` carries an eager-execution guard in
`ggml_backend_cuda_graph_compute`: when a SELECTED-dump window is armed
(exclusive audit replay), CUDA graph replay is bypassed for that call,
because graph replay runs no capture hooks and the audit would find no
witness instance (`verathos_cuda_selected_dump_pending`). The metal patch
(0002) is unaffected: the guard is CUDA-only, and metal serving does not
use CUDA graph replay.

The placement fix ensures the
guard's original shape vetoed `use_cuda_graph` AFTER upstream's
warmup/update bookkeeping. `ggml_cuda_graph_update_required()` refreshes
`graph->uid` and `graph->node_props` as a side effect and the warmup
branch sets `warmup_complete` while scheduling a capture for that same
pass; vetoing the capture afterwards left the previously captured
instance in place while the bookkeeping recorded the current properties
as captured. The first disarmed pass then replayed the stale instance.
When the graph geometry moved across the window (a 142k-token canary
prefill co-batched with live serves re-reserves the compute pool and
relocates graph allocations), the stale instance's recorded device
pointers referenced freed memory: CUDA "illegal memory access" at the
next stream sync (surfacing from `common_sampler_sample`), aborting the
whole server. The guard now gates the entire bookkeeping block on
`!verathos_cuda_selected_dump_pending()` (on the `graph->is_enabled()`
condition), so an armed window is a pure eager pass with zero
graph-state writes; the next disarmed pass re-checks properties against
the last consistent snapshot and recaptures if anything changed.
Regression tripwire: `tests/verallm/test_llama_cpp_patch_guard.py`.

## Decode-priority interleave (0010)

Problem: during a full-context canary (104k-token prefill on glm at ~320
tok/s prefill), interactive chat slots starved. Upstream's `update_slots`
builds one batch per pass: the generating slots' single sampled tokens plus
up to `n_batch` prompt tokens, and sampling happens once per pass, after the
whole batch decodes. With `n_ubatch = 4096` each pass carried a 4096-token
prefill chunk taking ~13 s, so every live chat emitted one token per ~13 s
for the duration of the prefill.

0010 adds decode-priority scheduling to `pre_decode` in
`tools/server/server-context.cpp`. While at least one slot is generating AND
another slot is mid-prompt-processing:

1. Prompt-carrying (mixed) passes cap the prompt tokens taken into the batch
   at `VERATHOS_DECODE_PRIORITY_PREFILL_CAP` (default 512). This bounds the
   worst-case inter-token stall of a chat at one small prefill chunk.
2. Between prompt-carrying passes the scheduler interposes decode-only
   passes (the generating slots' sampled tokens, zero prompt tokens). Their
   count is derived from the measured wall-clock duration of the previous
   prompt-carrying pass divided by `VERATHOS_DECODE_PRIORITY_INTERVAL_MS`
   (default 250), clamped to `VERATHOS_DECODE_PRIORITY_MAX_DECODE_PASSES`
   (default 32). A decode-only pass at batch size 1-8 costs tens of
   milliseconds, so this self-tunes: slow prefill chunks (big models, RPC
   mesh) get many decode opportunities in between, fast chunks get few or
   none, and the added prefill overhead stays a roughly constant fraction
   (about decode-pass-cost / interval, ~10-15%) regardless of model speed.

Measured (ornith-1.0-35b Q4_K_M, M1 Max Metal, `-b 8192 -ub 4096
--parallel 4`, ~9.4k-token prefill co-batched with a live streamed chat):
upstream behavior starves the chat to ~0.7-1.2 tok/s with 6-9 s
inter-token stalls; with 0010 the chat holds ~5 tok/s with a 0.76 s
worst gap (one capped chunk), while prefill throughput stays within run
variance (858 -> 695 tok/s worst pair, ~0-19%).

`VERATHOS_DECODE_PRIORITY=0` disables the whole mechanism (default on).
When no slot is generating, prompt processing runs at full `n_batch` exactly
as upstream. Non-splittable prompts (embedding/rerank style tasks that must
land in a single batch) are exempt from the cap.

Proof-capture safety: the patch changes only how tokens are GROUPED into
batches, never which tokens are computed or their order within a slot.
Decode results and prompt-processing outcomes are identical apart from
batching boundaries (ubatch composition). Name-keyed arming (0009) already
tolerates length-dependent op streams, so capture identities are unaffected.

Regression tripwire: `tests/verallm/test_llama_cpp_patch_guard.py`
(`test_decode_priority_*`).

## Prefill fairness (0011)

Problem (`-np 8`, chunked prefill): while one slot
prefilled a 100k+ token prompt, a newly-arrived small chat prompt emitted
zero tokens until the entire giant prefill finished; the client canceled
~60 s later (log signature: "processing task", then cancel with zero
tokens). Cause: the prompt-fill section of each `update_slots` pass
iterates `slots` in fixed vector order, so whenever the giant-prefill slot
precedes the new request's slot, it consumes the whole prompt budget
(`n_batch`, or 0010's capped budget) on every pass. 0010 does not help
here: it protects slots that are already DECODING, not ones still waiting
to start their prompt.

0011 fills the prompt section from prompt-pending slots in ascending
order of REMAINING prompt tokens (`std::stable_sort` over slot pointers;
the fill body is unchanged). A small prompt fits inside its first pass
with budget to spare and reaches decode immediately; the remainder of
every pass's budget still flows to the giant prefill, so it keeps
progressing (no starvation in either direction). Which tokens are
computed and their order within each slot are unchanged; only the fill
order across slots changes, so proof capture is unaffected for the same
reason as 0010. Non-splittable prompts keep upstream's own can-fit guard.

Measured (Qwen2.5-7B-Instruct Q4_K_M, RTX 4090, `-c 262144 -np 8 -b 4096
-ub 2048`, 29k-token giant on slot 0, 15-token chat on slot 5 arriving
2 s in, plus a live decoding stream on slot 7):

- without 0011: chat TTFT 1.74 s == the giant's remaining prefill time
  (scales with prompt size: two 29k giants -> 5.62 s); with a production
  100k+ prompt this is the full multi-ten-second starvation.
- with 0011: chat TTFT 0.111 s in both scenarios; giant prefill
  3.74 -> 3.87 s (+3.5%, two-giant total +2.0%). The 0010 decode stream
  stayed at a <= 0.10 s worst inter-token gap throughout.

`VERATHOS_PREFILL_FAIRNESS=0` disables it (default on; verified to
restore the upstream fill order bit-for-bit: TTFT 1.737 s vs the 1.739 s
baseline). Slot iteration order is only ever changed inside the
prompt-fill section; decode batching, slot selection, and task dispatch
are untouched.

Regression tripwire: `tests/verallm/test_llama_cpp_patch_guard.py`
(`test_prefill_fairness_*`).
