# Mesh operator flow: who does what, and which command is which

This is the authoritative map from **roles** to **commands** to **code**.
Read this before touching the pool CLI. The user-facing walkthrough lives
in [docs/public/mesh_quickstart.md](mesh_quickstart.md); this
document exists because the low-level commands *look* symmetrical while
belonging to completely different roles, and using the wrong one builds a
mesh that resembles production while binding none of its trust anchors.

## The three roles

| Role | Owns | On-chain writes |
|---|---|---|
| **Subnet owner** | The model catalog: `ModelRegistry` ModelSpecs with tensor-manifest roots, tokenizer hashes, quant schemes | `registerModel` (ModelSpec) |
| **Mesh operator (miner)** | A pool: manager, workers, meshes; one coordinator UID | `registerEvm`, MinerRegistry endpoint `registerModel`/`renewModel` (a *lease on an endpoint*, not a model spec) |
| **Validator** | Canary schedule, scoring, weights | `set_weights` |

The word "register" appears in two unrelated senses. The subnet owner
registers **ModelSpecs** (what a model *is*: weights root, tokenizer,
quant). The miner registers an **endpoint lease** (where a serving mesh
*lives*). A miner never creates catalog entries; `verathos mesh deploy`
derives every trust anchor *from* the catalog (deploy.py step 1,
`resolve_mesh_chain_anchors`) and refuses to run against a model the
owner has not registered.

## One word, three meanings: "coordinator"

The single largest source of operator confusion. Disambiguate before
reading anything else:

1. **The coordinator box** runs the pool *manager* (`mesh pool serve`,
   port 9500): worker registry, placement, dashboard. Control plane only,
   never in the token data path, needs no GPU.
2. **The mesh coordinator process** is what the *driver worker* runs for
   one launched mesh (port 9443): the llama client plus the
   validator-facing inference endpoint. This is the endpoint `deploy`
   registers on-chain, and in a multi-box pool it usually lives on a
   DIFFERENT machine than the pool manager.
3. **`--coordinator-address` / `--coordinator-uid`** are the miner's
   on-chain EVM address and subnet UID: chain identity, not a machine.

So `--endpoint https://<host>:9443` in a deploy takes the **driver
worker's** host (sense 2), not the pool-manager box (sense 1).

## Ports, and who dials whom

| Port | Bound by | Dialed by | Notes |
|---|---|---|---|
| 9500 | pool manager (coordinator box) | workers + operator commands, outbound | the one endpoint every worker must reach; embedded in the join token |
| 9443 (+2 per GPU unit) | driver worker's mesh coordinator | validators / clients, inbound | the registered inference endpoint |
| 9444 (mesh+1) | driver's llama-server | local only | |
| 9402 (+1 per unit) | every worker's proof HTTP | the driver, **inbound on the worker** | |
| 50052 (+1 per unit) | every worker's verathos-rpc-server | the driver, **inbound on the worker** | |

"Workers dial out, so NAT is fine" is true for **pool membership** (9500)
only. The moment a mesh spans machines, every non-driver worker must
accept inbound 50052 and 9402 from the driver, and each worker's
`--advertise-host` must be an address the driver can actually dial
(loopback or a NAT-internal IP there breaks the mesh with no local
symptom on the worker).

## The operator's real flow (five steps)

Guided doors over the same machinery: `verathos mesh setup` (interactive,
human) and `verathos setup mesh-coordinator` / `mesh-worker --token
vtpool_...` (flag-driven, scriptable). Both delegate to
verallm/mesh/onboarding.py, refuse to silently mint a second pool on a
configured machine, and print the worker join one-liner
(`verathos mesh pool join-token` re-prints it any time). The primitives
underneath:

```
setup            bash scripts/setup_mesh.sh                     (each machine;
                 scripts/setup_mesh.sh)
coordinator      verathos mesh pool create + serve              (one machine)
workers join     bash scripts/join_pool.sh --token vtpool_...   (each GPU box;
                 wraps `mesh pool worker`, one PM2 unit per GPU)
placement        verathos mesh pool recommend --model-id <id>   (advice, free)
go live + earn   verathos mesh deploy <model-id> ...            (the one command)
```

(`mesh pool worker` by hand additionally needs `--workdir`,
`--advertise-host`, and the runtime binary flags; `join_pool.sh` derives
all of them, which is why the flow points there.)

`deploy` is the whole miner lifecycle in one pipeline (deploy.py,
stages numbered in the source). It supports testnet and mainnet
identically: `--subtensor-network test|finney` selects the shipped chain
config and turns on the metagraph hotkey check.

1. **Chain preconditions**: fetch the owner's ModelSpec, derive all six
   digests from it. Fail-closed, no writes, nothing hand-typed.
2. **Hotkey binding**: the coordinator hotkey must hold a UID on the
   target netuid (metagraph read); `--uid` and the pool binding are
   cross-checked against it.
3. **Pool preconditions**: subnet serving mode, coordinator address
   == signing wallet, chain/netuid/uid binding.
4. **Model source**: flags > pool registry > the shipped catalogue.
5. **Measurement launch**: the model runs UNREGISTERED first (operator
   lane only, no snapshot, no chain slot), because the registered
   `max_context_len` must be the KV auto-fit's MEASURED value. A mesh
   already serving this model is reused instead.
6. **Probe gate**: the ready-to-register confirmation. Light-proof
   probes, an explicit hard-tier probe (upgrade-only request; validator
   audits assert the hard relation), a full-context probe at 0.8x the
   context about to be registered, throughput floor, TTFT report, and
   the measured-context honesty check (registered <= measured).
7. **Registered context**: the measurement by default;
   `--max-context-len` is accepted only within 10% of it (the same
   restart-jitter rule as the vLLM miner's `_ctx_close_enough`).
8. **EVM binding** (`registerEvm`, first write, no reputation risk).
9. **Index prediction** (a chain-bound launch needs `model_index`
   before the chain write can confirm it).
10. **Pool model registration**, with the chain-derived anchors and the
    measured context. This is the ONLY legitimate caller of
    `pool register-model` in production.
11. **Relaunch chain-bound**: the measurement mesh is replaced by one
    whose signed snapshot binds the predicted index (skipped when the
    serving mesh is already bound to that index).
12. **Final verification**: one hard probe through the chain-bound mesh
    (snapshot-bound), public reachability (health 200, validator route
    exactly 403). Reachability is never forceable.
13. **MinerRegistry endpoint registration**, index-guarded, only on
    pass. On a failed gate: no chain write, the mesh keeps serving
    unregistered.

Lease renewal afterwards is the pool manager's job (subnet-mode serve
with wallet flags), or `verathos mesh renew --once` by hand.

The standalone confirmation exists too: `verathos mesh pool probe
--gate` runs the same gate against any serving mesh and prints a
READY TO REGISTER / NOT READY verdict with the measured context, light
and hard proof results, tok/s and TTFT, so an operator can check
readiness before ever touching a wallet.

Humans do not have to type the deploy line: after a verified probe on a
subnet-mode pool, `verathos mesh setup` offers the same pipeline
interactively (model from the serving mesh, network from the pool's
chain binding, endpoint defaulted to the driver worker's public mesh
port). Agents get placement advice from
`verathos mesh fleet --model-id <id> --json` (or `pool recommend`)
before calling `deploy` with flags.

Three consistency rules the tooling now enforces (each was a real
incident):

- **Live view over state file.** `mesh setup` and `mesh status` ask the
  running manager (`/v1/pool/status`) and only fall back to
  pool-state.json when no manager answers; a "serving" mesh is only
  reported when it is `routing_ready` (all stages heartbeating).
- **No double-join.** The join-local-GPUs offer checks live pool
  membership first: GPUs whose workers already serve in the pool are
  never enrolled a second time.
- **No orphaned runtimes.** A worker told `unknown-worker` (manager was
  reset) tears down its mesh processes before rejoining as idle, and a
  restarting manager drops meshes no surviving worker could ever serve
  or stop.

## What the low-level commands are FOR

`pool create/serve/worker/status/launch/stop/remove-worker/probe` and
`pool register-model` are the primitives deploy is built from. They stay
public because operators debug with them and tests drive them, but the
catalog entry a bare `register-model` writes is only as trustworthy as
the hands that typed it; in subnet mode the snapshot signs what the
manager was told, so wrong anchors mean every canary fails later with no
local symptom. That is why deploy derives them and why a human should
not type them.

`verathos mesh chat` / `fleet` / `logs` / `status` are operator
conveniences over a serving pool and make no chain writes at all.

## Proof tiers in one paragraph

There are exactly two proof tiers and nothing in between: **light**
(openings-only against serve-time commitments, milliseconds, no
weight-execution claim) and **hard** (the full GEMM relation, seconds).
Organic traffic is ALWAYS light; it carries no hard-audit bps at all.
Validator canaries are mostly light too: exactly one canary slot per
miner per epoch is forced hard (drawn from the validator seed + epoch
salt, `neurons/canary.py`), and every other canary resolves the ambient
post-receipt nonce draw (`postcommit_audit_decision`), which is light in
the common case with a Bernoulli tail that upgrades a random extra
fraction to hard. A signed reveal may demand stricter-only hard.
Operator chat and `mesh infer` pick a tier per-session (`--proof-tier` /
the interactive picker / `/tier`); an explicit `hard` upgrades on every
lane, and probes refuse a downgrade to light because the gate must
assert the hard relation.

**Sampling rates are not tiers.** `--proof-sample-bps` and
`--decode-audit-bps` (and their serve-time counterparts) control how
OFTEN a request is drawn for its lane's proof or for the decode/logit
audit; they never select which relation runs. Raising
`proof_sample_bps` to 10000 makes every request carry its lane's proof,
which on an organic lane is still light. If a knob is meant to change
the relation, it is `proof_tier`; if it is measured in bps, it is a
sampling rate.

## Dev mode vs subnet mode (local-only vs on-the-subnet)

A pool is always MINER-side infrastructure. `--serving-mode subnet` means
this miner's pool is registered on the subnet and signs the verification
snapshots that **remote validators verify**; the operator is and stays the
miner and never runs a validator. The mode used to be spelled `validator`
(it named the audience, not the operator, and read as if a miner ran a
validator); that value is still accepted everywhere as a deprecated alias.
The wizards therefore present the choice as "local only" vs "on the
subnet". Which network (testnet netuid 405, mainnet netuid 96) is a
separate, orthogonal choice: `--network testnet|mainnet` fills
`--chain-id`/`--netuid` from the shipped chain config.

`--serving-mode dev` exists to exercise serving and proofs without chain
bindings: no signed snapshot, no coordinator UID, chat rides the organic
light default. It is a test harness, not a small production mode:
`deploy` refuses it (stage 2), and nothing it serves is scoreable.

### Switching modes in place

`verathos mesh pool upgrade` (or `verathos mesh manage`, "go live on the
subnet") flips a dev pool to subnet mode IN PLACE: the pool keeps its id,
both tokens, worker records, model registry, and every on-disk artifact.
What actually changes is the chain binding in the state file, the manager
restarting with the signing wallet (lease renewer), and workers rejoining
so their stage identities get pinned. Recreating the pool is never
required. The one hard constraint: a remote (non-loopback) subnet manager
endpoint must be HTTPS, and because the worker token embeds the endpoint,
changing the scheme re-mints the token files (same secret, new endpoint),
so remote workers need the re-printed token.

### Registration lifecycle, and what deregistration does NOT break

The hotkey's subnet registration is checked exactly once, at deploy time
(stage 2). Nothing in the serving path re-checks it: the manager, the
driver coordinator, the allowlist refresher (it tracks OTHER validators'
hotkeys), the lease renewer, and snapshot signing all keep working if the
hotkey later loses its UID. Local serving, chat, and the private pool API
therefore continue through a deregistration; what stops is subnet
earnings (no UID, no weights) and, once the UID is recycled to another
hotkey, validator canaries fail snapshot identity checks. The manage
board surfaces this state explicitly ("hotkey NOT registered on netuid
N: local serving continues, subnet earnings stopped") via the manager's
metagraph-backed operator score.

Leaving the subnet deliberately is `verathos mesh retire`: it stops the
mesh (unless `--keep-serving`), calls `deactivateModel` on chain (which
releases the endpoint claim, so the URL can be reused without an
endpoint-claim collision), verifies the entry reads inactive, and clears
the manager's stored registration so the lease renewer stops renewing
it. `mesh pool stop` alone only lapses the lease passively over 24h and
keeps the endpoint claimed.

## Private pool API (operator keys, OpenAI compatible)

The pool manager serves the operator's OWN meshes to their own tools:
`GET /v1/models` and `POST /v1/chat/completions` (streaming and
non-streaming) on the manager port (default 9500, TLS-capable),
authenticated with keys minted by `verathos mesh apikey create` (also in
`mesh manage`, "api keys"). Keys are stored as sha256 hashes in the
owner-only pool state; the cleartext is printed once at mint time.

This is deliberately DISTINCT from the validator-hosted subnet API: the
validator's proxy fronts EVERY registered mesh on the subnet with
payment and scoring; the pool API fronts one pool with the operator's
keys and works with or without subnet registration (and keeps working
through a hotkey deregistration). It never touches the coordinator port:
the registered endpoint's validator-auth posture stays exactly as the
deploy pipeline enforces it.

Internet exposure follows the operator-console TLS shape (the manager
itself stays plain http because workers dial it over http): put the
https shim in front (`sudo bash scripts/setup_https.sh --port 9543
--backend-port 9500 --append`) and reach
`https://<public-ip-or-domain>:9543/v1/...`. Keys must never cross plain
http off-machine. The surface is rate limited server-side (fixed
one-minute windows: 120 requests/min per key, 30 failed auth attempts
per source address) on top of the existing body caps and header
hardening, so an exposed port is neither brute-forceable nor floodable
through this lane. A cloud relay through verathos.ai was considered and
rejected for the same reasons the operator console rejects a cloud
control page (operator_console_plan.md): the subnet API on the validator
proxy IS the hosted product; this lane stays operator-direct.

Semantics: the OpenAI `model` field is the pool model id (each id names
a quant, so picking a quant IS picking a model; `auto` works when one
model serves). Proofs are always on, riding the same organic light tier
as subnet traffic. Sampler params (`temperature`, `top_k`, `top_p`,
`min_p`, `seed`) are honored through the committed light sampled
profile: the support clamps to the decode-audit width and the applied
controls are bound into the signed receipt, so a sampled response is
just as verified as a greedy one. `logit_bias`, `grammar`,
`json_schema`, and non-text `response_format` are rejected (they
transform logits after the proved LM-head computation). Tool calling
runs the same engine-agnostic decision pass as the validator proxy
(non-streaming only; streaming plus tools is a 400). Proof facts ride a
`verathos` extension object on responses and final stream chunks.

## Operational gotchas that cost real time

- Workers need `verathos-rpc-server` (from `patches/llama.cpp/build.sh`),
  never raw `ggml-rpc-server`; the launch error says so explicitly.
- Runtime processes hold their loaded `.so`s; after rebuilding the
  llama.cpp tree in place, RESTART workers, or replay-side numbering can
  disagree with freshly-built verifiers ("slot view replay intra-graph
  index mismatch" with a stale member was exactly this).
- A SIGKILLed worker orphans its `mesh serve`/rpc/llama children, and a
  later launch refuses to steal their ports ("refusing to kill unowned
  listener"). Kill children by anchored full path
  (`pkill -f '^/path/to/bin/llama-server'`), never by bare binary name:
  the worker's own command line embeds those names in its arguments.
- `recommend()` is only as good as the capability adverts. Workers now
  auto-detect GPU name/VRAM when flags are absent; before that fix every
  unflagged worker advertised 0 GB and recommend refused solo placement.
