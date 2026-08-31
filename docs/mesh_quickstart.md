# GGUF Mesh Quickstart

Serve verified GGUF inference across one or many GPU machines. The mesh
path is independent of the vLLM miner path: separate venv (`.venv-mesh`),
separate PM2 units, and one machine can run both on different GPUs.

Two roles:

- **Coordinator**: runs the pool manager (control plane, no GPU needed),
  holds the wallet, registers the mesh on-chain.
- **Worker**: any GPU machine; serves model stages. Pool membership dials
  OUT to the coordinator, so joining works behind NAT. Serving a
  multi-machine mesh does need two inbound ports per worker from the
  driver machine (RPC 50052+, proof 9402+), and each worker's
  `--advertise-host` must be an address the driver can dial.

## 1. Coordinator setup

```bash
# Testnet release (branch: feature/sleipnir) — one command:
curl -fsSL https://raw.githubusercontent.com/verathos-ai/verathos/feature/sleipnir/install.sh \
  | bash -s -- --branch feature/sleipnir --mesh-coordinator

# or from a verathos checkout on that branch:
verathos setup mesh-coordinator
# (hosted installer at https://verathos.ai/install.sh follows the
#  mainnet public release)
```

The wizard asks for the serving mode (`dev` for local experiments,
`validator` for production), the address workers dial, and TLS. It creates
the pool, starts the manager under PM2 as `verathos-pool-manager`, and
prints the worker join one-liner. Every prompt has a flag; `--yes` runs
fully non-interactive.

The output shows two credentials. The **worker token** (`vtpool_...`) only
lets a machine join and serve; it embeds the manager address, so it is the
entire worker configuration. The **admin token** stays in the pool state
directory and controls the pool. Never share the admin token.

The coordinator itself needs no GPU, but if the machine has GPUs the
wizard offers to enroll them as worker units too (default yes), so a
single GPU box becomes a complete one-machine pool in one command. Both
roles share `.venv-mesh` and run as separate PM2 units. Non-interactive
runs choose with `--join-local-gpus` / `--no-join-local-gpus`.

Dashboard: `https://<coordinator>:<port>/operator`.

## 2. Add GPU machines

On the coordinator, print the exact join command for this pool:

```bash
verathos mesh pool join-token --endpoint https://<coordinator-public-ip>:<api-tls-port>
```

It prints a copy-paste one-liner of this shape — run it verbatim on each
GPU machine:

```bash
curl -fsSk --pinnedpubkey 'sha256//<pin>' \
  https://<coordinator>:<port>/install.sh | bash -s -- --token vtpool_...
```

The token is the entire worker configuration: it carries the manager
address, the pool credential, and the TLS certificate pin. The machine
needs `curl`, `python3`, and `tar` — bare container images (for example
`nvidia/cuda:*`) often lack curl, so run
`apt-get update && apt-get install -y curl` first.

### Machine requirements

- **RAM**: at least the model's GGUF file size plus ~8 GB headroom (a
  27B q4 model is a ~17 GB file, so 32 GB RAM works; 24 GB is too
  tight). Memory-capped containers are supported: serve processes run
  under a kernel memory limit a margin below the container's cap, so a
  model load throttles instead of getting killed. `free` inside a
  container often shows the HOST's RAM — check
  `/sys/fs/cgroup/memory.max` for the real budget.
- **Disk**: twice the model file size plus ~40 GB for the one-time
  runtime build and caches.
- **CPU**: the runtime compiles on the machine it runs on, so any
  x86-64 CPU the box boots with is fine; expect the first join to spend
  20-30 minutes building.
- **Port-mapped containers** (cloud GPU rentals behind NAT): append the
  published ports to the one-liner so validators can dial the machine —
  `--advertise-host <public-ip> --mesh-port N --proof-port N+2
  --rpc-port N+3`, using ports from your provider's mapped range. The
  join refuses undialable addresses instead of joining a worker that
  can only fail.

The installer fetches the source bundle from the coordinator, installs
dependencies, builds the patched llama.cpp for that machine's GPU, and
starts **one PM2 worker unit per GPU** (`verathos-mesh-<host>-gpu0`,
`-gpu1`, ...), each pinned to its device. Options after `--` pass
through, e.g. `-- --gpus 0,1` to use a subset or `-- --member-only` for
machines that should serve stages but never drive. A single-GPU machine
and an 8-GPU machine are the same command.

**Port-mapped containers** (RunPod, Vast, ...): the auto-detected
address is the container's private IP, which the pool refuses because no
validator can dial it. Append the public address and your provider's
published ports:

```bash
... | bash -s -- --token vtpool_... \
  --advertise-host <public-ip> \
  --mesh-port N --proof-port N+2 --rpc-port N+3
```

Mesh serving also binds `mesh_port + 1` (the local llama endpoint), so
with a consecutive published range `N..N+k` give the mesh port two slots:
mesh `N`, proof `N+2`, rpc `N+3`.

The installer waits for the pool's verdict and reports **JOINED** or the
manager's refusal reason per unit — it exits non-zero until every unit is
accepted, and re-running it is safe.

## 3. Operate

```bash
verathos mesh fleet                     # workers, GPUs, meshes, models
verathos mesh fleet --model-id <id>     # plus ranked placement advice
verathos mesh status                    # this machine's units (worker box)
verathos mesh logs [unit] / stop / start
verathos mesh pool remove-worker --worker-id <id>   # drop a DEAD box's record
```

Dead workers stay listed (marked stale) until removed — deliberate, so a
transient outage never erases a machine's record. `remove-worker` refuses
live workers.

Placement advice explains itself per worker ("available (can drive)",
"member only: downloading needs ~5 GB free disk, has 2 GB", ...) and warns
when a slow link would cap decode throughput.

## 4. Deploy and register (earn)

One command launches a mesh, proves it works, and only then registers the
endpoint on-chain:

```bash
verathos mesh deploy <model-id> \
  --pool-token-file <pool>/pool-admin-token.txt \
  --chain-config chain_config.json \
  --endpoint https://<driver-worker-host>:9443 \
  --wallet <name> --hotkey <hotkey>
```

The pipeline: verify the model's on-chain ModelSpec (all trust anchors are
derived from chain, never typed), pick a placement, launch, wait for
serving, then run the **probe gate**:

- every probe must come back proof-verified across all compute stages,
  bound to the mesh's signed verification snapshot (hard: any proof failure
  zeroes an epoch score),
- one canary-shaped full-context probe must finish inside the gate budget
  (hard: repeated full-context canary failures mean probation),
- a throughput floor (default 3 tok/s; ~1 tok/s is where canaries start
  timing out),
- public reachability: `/health` answers and the validator routes return
  exactly 403, proving the endpoint is up and in validator-auth posture.

On pass, the endpoint is registered (24h lease) and the pool manager takes
over lease renewal. On fail, **nothing is sent on-chain**: the mesh keeps
serving unregistered, and the report shows the failing checks with numbers
and the best alternative placement. `--force` overrides the gate (never
the chain preconditions or the reachability posture). `--dry-run` checks
everything up to placement without launching.

Choose `--max-context-len` honestly: scoring credits registered context
(log-scale), but full-context canaries regularly test the FULL advertised
window — register more than the mesh can serve inside the canary budget
and the failed canaries cost far more than the context credit earns.

## 5. Lease renewal

If the pool manager was started with a wallet
(`verathos mesh pool serve ... --wallet-name <w> --chain-config <cfg>`; the
coordinator wizard does not add these, restart the manager with them after
your first deploy), it renews the 24h lease automatically
whenever less than 12h remains, and deliberately lets the lease lapse if
the mesh stops serving. Without a wallet in the manager, run the renewal
from cron:

```bash
verathos mesh renew --pool-token-file ... --chain-config ... --wallet <name>
verathos mesh registration-status --pool-token-file ...
```

## 6. Your own API (optional)

Every pool doubles as a private OpenAI-compatible API for your own
tools — served by the pool manager, proofs always on, completely
independent of subnet registration. Keys are minted on the coordinator
(no flags needed there):

```bash
verathos mesh apikey create --name my-app   # cleartext key shown ONCE
verathos mesh apikey list                   # ids, names, last-used
verathos mesh apikey revoke --key-id <id>
```

Point any OpenAI client at the manager. Its TLS certificate is
self-signed; authenticate it by pinning the public key that
`verathos mesh apikey expose` prints:

```bash
curl --insecure --pinnedpubkey 'sha256//<pin-from-expose>' \
  -H 'Authorization: Bearer <key>' \
  https://<coordinator>:9543/v1/chat/completions \
  -d '{"model": "<model-id>", "messages": [{"role": "user", "content": "hi"}]}'
```

Responses carry a `verathos` block with `verified`, `receipt_verified`,
and the proof receipt root — the same verification your subnet traffic
gets. `/v1/models` lists what the pool serves. Keys are stored hashed in
the owner-only pool state. If the manager only listens on loopback,
publish the https listener with `verathos mesh apikey expose --port 9543`.

## 7. Verify your install (optional)

The repository ships the mesh test subset the maintainers run. On any
machine with the repo checked out:

```bash
pip install -e ".[test]" "torch>=2.10,<2.12"
pip install --no-index --find-links dist/ \
  zkllm hot-capacity-workspace-cuda verathos-proof-v3-cuda
python -m pytest -q \
  tests/verallm/test_mesh_capacity_roster.py \
  tests/neurons/test_mesh_repin_successor.py \
  tests/verallm/test_mesh_control.py \
  tests/verallm/test_mesh_registration.py \
  tests/verallm/test_mesh_worker_http_security.py \
  tests/verallm/test_mesh_capacity_audit_worker.py \
  tests/verallm/test_mesh_pool.py
```

All green means your Python environment, the shipped wheels, and the mesh
code agree. (GPU proof kernels are built during the pool join; the
prebuilt wheel kernels cover torch 2.10/2.11 + cu128.)
