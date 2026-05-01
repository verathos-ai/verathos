# Setup Guide

How to set up and run a Verathos miner or validator on Bittensor Subnet 96.

## Prerequisites

| Requirement | Miner | Validator |
|-------------|-------|-----------|
| Python | 3.10+ | 3.10+ |
| GPU | 24 GB+ VRAM (RTX 4090, A100, H100, etc.) | Not needed |
| CUDA | 12.8+ | Not needed |
| RAM | 32 GB+ | 16 GB+ |
| Storage | 100 GB+ SSD (model weights + Merkle cache) | 50 GB+ SSD |
| Network | 100 Mbps up/down | 100 Mbps up/down |
| OS | Ubuntu 22.04+ (or equivalent Linux) | Ubuntu 22.04+ (or equivalent Linux) |
| Bittensor CLI | `pip install bittensor-cli` | `pip install bittensor-cli` |

---

## Miner Setup

A miner serves a single model on a single GPU with cryptographic proofs and earns emissions based on throughput, latency, and model utility (parameters, context length, quantization). To serve multiple models, run separate miner instances on separate GPUs. Multi-GPU inference for models that exceed single-GPU VRAM is on the roadmap.

### Quick Setup (recommended)

One-command install + interactive wizard that handles wallet, registration, EVM funding, HTTPS, PM2, and config:

```bash
# Install (one command)
curl -fsSL https://verathos.ai/install.sh | bash

# Or manually:
git clone https://github.com/verathos-ai/verathos.git && cd verathos
bash scripts/setup_miner.sh

# Then complete setup interactively
verathos setup                 # guided wizard
verathos start                 # start mining via PM2
```

Check readiness anytime: `verathos status` | Network overview: `verathos network`

For manual step-by-step control, follow the sections below.

### 1. Install

The setup script creates a venv, installs all dependencies (vLLM, CUDA kernels, etc.), checks GPU compatibility, and builds the zkllm CUDA extension:

```bash
git clone https://github.com/verathos-ai/verathos.git
cd verathos
bash scripts/setup_miner.sh
```

Flags: `--skip-install` (skip deps, just verify).

After install completes the script prints model recommendations for your GPU and the next steps below.

### 2. Create wallet and register on subnet

```bash
# Create a new Bittensor wallet
btcli wallet create --wallet.name miner

# Register on Subnet 96 (requires staking TAO)
btcli subnet register --wallet.name miner --netuid 96 --subtensor.network finney
```

### 3. Fund your EVM address

Your miner needs a small amount of TAO on the Bittensor EVM for on-chain registration (gas fees). The EVM key is derived automatically from your hotkey, so no separate key management is needed.

```bash
# Show your EVM address and SS58 mirror
python scripts/show_evm_info.py --wallet miner --hotkey default

# Transfer a small amount for gas (use the SS58 mirror from above)
btcli wallet transfer --dest <SS58_MIRROR> --amount 0.1 --subtensor.network finney
```

### 4. Start the miner

EVM registration is automatic on startup: the miner proves hotkey ownership
via SR25519 signature, verified on-chain by the Sr25519Verify precompile.

```bash
python -m neurons.miner \
    --wallet miner --hotkey default \
    --model-id auto \
    --netuid 96 \
    --subtensor-network finney \
    --endpoint https://YOUR-PUBLIC-IP
```

HTTPS is required on mainnet. Self-signed certs work (see [Endpoint Security](#endpoint-security) below for a 4-command setup).

**With a local subtensor** (recommended for production, no rate limits):
```bash
python -m neurons.miner \
    --wallet miner --hotkey default \
    --model-id auto \
    --netuid 96 \
    --subtensor-network finney \
    --subtensor-chain-endpoint http://localhost:9944 \
    --endpoint https://YOUR-PUBLIC-IP
```

**What happens on startup:**
1. GPU detected, best model selected automatically (or use `--model-id` to specify)
2. EVM key derived from your hotkey (no config files)
3. Inference server starts, loads model into vLLM
4. Registers model + endpoint on MinerRegistry contract
5. Enters heartbeat loop (renews lease every 12h)

### Model selection

The `--model-id auto` (or `--auto` flag) detects your GPU and picks the optimal model. You can also filter or override:

```bash
# Auto-select best model (these are equivalent)
python -m neurons.miner ... --model-id auto
python -m neurons.miner ... --auto

# Auto-select a coding model
python -m neurons.miner ... --model-id auto --category coding

# Specific model, auto quant + context
python -m neurons.miner ... --model-id "Qwen/Qwen3-8B"

# Everything explicit
python -m neurons.miner ... --model-id "Qwen/Qwen3-8B" --quant fp16 --max-context-len 8192

# See what models fit your GPU
python -m verallm.registry --recommend
```

Available categories: `general`, `coding`, `agent_swe`, `reasoning`, `multimodal`.

### Auto-update

Enable automatic updates so your miner stays current without manual intervention:

```bash
python -m neurons.miner ... --auto-update
```

When enabled, the miner checks the git remote every 30 minutes. If a new miner version is available, it pulls the code, reinstalls, and restarts automatically. Validator-only updates do **not** trigger a miner restart.

Customize the check interval (in seconds):
```bash
python -m neurons.miner ... --auto-update --auto-update-interval 900  # every 15 min
```

### Pass extra args to vLLM

Anything after `--` is forwarded to the inference server:

```bash
python -m neurons.miner ... -- --gpu-memory-utilization 0.95
```

> **Production:** Use PM2 to keep your miner running reliably. See [Production Deployment (PM2)](#production-deployment-pm2) below.

### Endpoint Security

The miner server authenticates all inference requests: only validators registered on the subnet can call `/chat` and `/inference` (Sr25519 signature verification via the metagraph allowlist). Random requests from the public are rejected with 401/403.

**HTTPS is required on mainnet.** Miners must register an `https://` endpoint. Self-signed certificates work, no domain or certificate authority needed (see Option B below). A reverse proxy in front of your miner provides:

- **TLS encryption**: encrypts traffic between validators and your miner
- **DDoS protection**: prevents connection flooding
- **IP hiding**: a Cloudflare tunnel hides your server IP (optional)

**Option A: Cloudflare Tunnel (recommended, free, no ports to open)**

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# Create tunnel (one-time)
cloudflared tunnel login
cloudflared tunnel create miner
cloudflared tunnel route dns miner miner.yourdomain.com

# Run (use PM2 in production)
cloudflared tunnel --url http://localhost:8000 run miner
```

Then register with `--endpoint https://miner.yourdomain.com`.

**Option B: nginx with self-signed cert (no domain needed)**

Works on any server with just an IP address. Traffic is encrypted (TLSv1.3).

**Quick setup** (one command):
```bash
bash scripts/setup_https.sh                  # port 443 → localhost:8000
bash scripts/setup_https.sh --port 13998     # custom port (container environments)
```

**Manual setup:**

```bash
# Install nginx and generate self-signed cert
apt install -y nginx
mkdir -p /etc/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/miner.key \
  -out /etc/nginx/ssl/miner.crt \
  -subj "/CN=$(curl -s ifconfig.me)"
```

```nginx
# /etc/nginx/sites-available/miner
server {
    listen 443 ssl;                              # Change port if 443 is not available
    ssl_certificate /etc/nginx/ssl/miner.crt;
    ssl_certificate_key /etc/nginx/ssl/miner.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_read_timeout 120s;
    }
}
```

```bash
ln -sf /etc/nginx/sites-available/miner /etc/nginx/sites-enabled/miner
rm -f /etc/nginx/sites-enabled/default
nginx -t && nginx -s reload    # or: systemctl reload nginx
```

Then register with `--endpoint https://YOUR-IP` (add `:PORT` if not using 443).

> **Container / cloud GPU environments:** Many providers expose specific mapped ports
> rather than the standard 443. Check which ports are available on your instance and
> use that port in the nginx `listen` directive instead of 443. If nginx is already
> installed with its own configuration, check for conflicts in `/etc/nginx/nginx.conf`
> and `/etc/nginx/conf.d/`, and you may need to remove or comment out pre-existing server
> blocks that bind to ports already in use by other services.

**Option C: nginx with Let's Encrypt (requires domain)**

```nginx
server {
    listen 443 ssl;
    server_name miner.yourdomain.com;
    ssl_certificate /etc/letsencrypt/live/miner.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/miner.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_read_timeout 120s;
    }
}
```

---

## Validator Setup

A validator tests miners, verifies proofs, scores performance, and sets weights on Bittensor. No GPU needed.

### Quick Setup (recommended)

```bash
# Install
curl -fsSL https://verathos.ai/install.sh | bash -s -- --validator

# Or manually:
git clone https://github.com/verathos-ai/verathos.git && cd verathos
bash scripts/setup_validator.sh

# Then complete setup interactively
verathos setup validator              # guided wizard
verathos start validator              # start validating via PM2
```

For manual step-by-step control, follow the sections below.

### 1. Install

```bash
git clone https://github.com/verathos-ai/verathos.git
cd verathos
bash scripts/setup_validator.sh
```

The script creates a venv, installs dependencies, and prints next steps. Flags: `--skip-install`.

### 2. Create wallet and register on subnet

```bash
btcli wallet create --wallet.name validator
btcli subnet register --wallet.name validator --netuid 96 --subtensor.network finney
```

### 3. Fund your EVM address (optional)

Optional. With funding, your validator contributes to on-chain `reportOffline` votes (faster dead-miner cleanup). Without it, the network self-cleans via 24h lease expiry and other validators' votes; your validator still sets weights normally.

To fund (recommended):

```bash
python scripts/show_evm_info.py --wallet validator --hotkey default
btcli wallet transfer --dest <SS58_MIRROR> --amount 0.05 --subtensor.network finney
```

To skip, pass `--no-evm` to the validator (step 4). If unfunded, the validator auto-degrades with a startup warning.

### 4. Start the validator

EVM registration is automatic on startup (SR25519 hotkey proof).

**Recommended:** Set a HuggingFace token to avoid rate limits when downloading
tokenizers for proof verification. Get one at https://huggingface.co/settings/tokens:

```bash
export HF_TOKEN="hf_..."
```

```bash
python -m neurons.validator \
    --wallet validator --hotkey default \
    --netuid 96 \
    --subtensor-network finney
```

**With a local subtensor:**
```bash
python -m neurons.validator \
    --wallet validator --hotkey default \
    --netuid 96 \
    --subtensor-network finney \
    --subtensor-chain-endpoint http://localhost:9944
```

**What the validator does each epoch (~72 min):**
1. Discovers active miners from MinerRegistry
2. Schedules canary tests (indistinguishable from real traffic)
3. Verifies cryptographic proofs for each test
4. Scores miners on throughput, latency, and model utility (parameters, context length, quantization)
5. Sets weights on Bittensor

### Auto-update (recommended)

Enable automatic updates so your validator stays current:

```bash
python -m neurons.validator ... --auto-update
```

The validator checks the git remote every 30 minutes. If a new validator version is available, it pulls, reinstalls, and restarts. The restart is deferred if the validator is in the middle of epoch close or weight setting. Miner-only updates do **not** trigger a validator restart.

**Important:** The subnet uses on-chain version gating. When a protocol upgrade is released, the subnet owner updates the on-chain `weights_version`, and validators running old code will be unable to set weights until they upgrade. With `--auto-update` enabled, this happens automatically.

### Resource requirements

| Resource | Requirement |
|----------|-------------|
| GPU | None |
| RAM | 16 GB+ (tokenizers loaded for proof verification) |
| CPU | 4+ cores, 2.0 GHz+ (proof verification takes ~4ms) |
| Storage | 50 GB+ SSD |
| Network | 100 Mbps up/down (HTTP to miners + WebSocket to Substrate) |

> **Production:** Use PM2 to keep your validator and gateway running reliably. See [Production Deployment (PM2)](#production-deployment-pm2) below.

---

## Production Deployment (PM2) *(recommended)*

For production, use [PM2](https://pm2.keymetrics.io/) to manage your miner, validator, and gateway processes. PM2 handles auto-restart on crash, log rotation, and persistence across reboots.

### Install PM2

```bash
# Requires Node.js 18+
npm install -g pm2
```

### Configure

The repo includes an example PM2 config with all options documented:

```bash
cp ecosystem.config.example.js ecosystem.config.js
```

Edit `ecosystem.config.js` and replace all `<PLACEHOLDER>` values with your actual wallet, hotkey, netuid, endpoints, etc. Optional features (USDC deposits, x402 payments, cold wallet) are included as commented-out flags. Uncomment what you need.

### Start services

```bash
# Miner
pm2 start ecosystem.config.js --only miner

# Validator + gateway (run on same machine)
pm2 start ecosystem.config.js --only validator
pm2 start ecosystem.config.js --only gateway
```

### Persist across reboots

```bash
pm2 save && pm2 startup
```

### Useful commands

```bash
pm2 status                       # process table
pm2 logs miner --lines 50        # tail logs
pm2 monit                        # live dashboard (CPU/mem/logs)
pm2 restart validator             # restart after transient error
```

### Important notes

- **Miner has `autorestart: false`**: GPU processes should not auto-restart without investigation. A crash loop wastes GPU memory. Check logs with `pm2 logs miner`, fix the issue, then `pm2 restart miner`.
- **Validator and gateway have `autorestart: true`**: they recover automatically from transient errors (RPC timeouts, etc.).
- **Changing flags requires delete + re-create.** PM2 caches args on first start. A simple `pm2 restart` uses the old args:
  ```bash
  pm2 delete gateway && pm2 start ecosystem.config.js --only gateway
  ```
- **Gateway uses the same wallet as the validator**: they share the hotkey for receipt signing and EVM key derivation.

---

## Testnet

To test on Bittensor testnet before going to mainnet, use `--subtensor-network test` and `--netuid 405`:

### Testnet miner

```bash
# Register on testnet (get test TAO from faucet)
btcli subnet register --wallet.name miner --netuid 405 --subtensor.network test

# Fund EVM address
python scripts/show_evm_info.py --wallet miner --hotkey default
btcli wallet transfer --dest <SS58_MIRROR> --amount 0.1 --subtensor.network test

# Start miner
python -m neurons.miner \
    --wallet miner --hotkey default \
    --model-id auto \
    --netuid 405 \
    --subtensor-network test \
    --endpoint https://YOUR-PUBLIC-URL:8000
```

### Testnet validator

```bash
btcli subnet register --wallet.name validator --netuid 405 --subtensor.network test

python scripts/show_evm_info.py --wallet validator --hotkey default
btcli wallet transfer --dest <SS58_MIRROR> --amount 0.1 --subtensor.network test

python -m neurons.validator \
    --wallet validator --hotkey default \
    --netuid 405 \
    --subtensor-network test
```

---

## Chain Config

The `chain_config.json` specifies the chain ID and contract addresses. The RPC URL is derived automatically from `--subtensor-network` (finney or test).

```json
{
  "chain_id": 964,
  "netuid": 96,
  "model_registry_address": "0x...",
  "miner_registry_address": "0x...",
  "payment_gateway_address": "0x...",
  "validator_registry_address": "0x...",
  "checkpoint_registry_address": "0x...",
  "mock": false
}
```

The official config files are included in the repository:
- `chain_config.json`: mainnet (Subnet 96, chain ID 964)
- `chain_config_testnet.json`: testnet (Subnet 405, chain ID 945)

For local development without a chain, set `"mock": true`.

---

## Troubleshooting

### "Invalid SR25519 signature"

The hotkey doesn't match the claimed UID. Ensure your wallet is registered on the subnet (`btcli subnet metagraph`).

### "No models fit GPU tier"

Your GPU doesn't have enough VRAM. Check available models:

```bash
python -m verallm.registry --recommend
```

Minimum GPU: **24 GB VRAM** (RTX 4090, RTX 3090/Ti, L4, A10, A30).

### "Model not found on-chain ModelRegistry"

The model spec hasn't been registered on-chain yet. Only the subnet owner can register new model specs. Check which models are available via the registry.

### EVM address derivation

The EVM address is derived locally from your hotkey seed (no network call needed):

```bash
python scripts/show_evm_info.py --wallet miner --hotkey default
```

---

## Auto-Update

All neuron processes support `--auto-update`:

```bash
python -m neurons.validator ... --auto-update
python -m neurons.miner ... --auto-update
```

A background thread checks the git remote every 30 minutes. If a new version is available for your role, it pulls the code, reinstalls, and restarts automatically.

**Role-specific updates**: a validator-only fix doesn't restart miners, and vice versa:

| Version | Who restarts |
|---------|-------------|
| `miner_version` | Miners only |
| `validator_version` | Validators + gateways only |
| `spec_version` | On-chain enforcement (everyone) |

### On-chain version gating

The subnet uses Bittensor's `weights_version` hyperparameter. The validator passes its `spec_version` as `version_key` in every `set_weights()` call. If the subnet owner updates the on-chain `weights_version` to a newer value, validators running old code are **blocked from setting weights** until they upgrade.

With `--auto-update` enabled, upgrades happen automatically within 30 minutes of a push.

## Logging

All processes use bittensor's native `bt.logging` with consistent format and colors.

- **Log level flags**: `--logging.debug` (verbose), `--logging.trace` (maximum detail), `--logging.info` (default)
- **Startup banner**: branded Verathos banner with network, wallet, model, version info
- **Metagraph stats**: periodic display of block, incentive/vtrust, emission, stake (~1 min)
- **Per-request summary**: `Served req-xxx | 83→248 tokens | 80.0 tok/s | 1030ms`

Debug mode shows canary dispatch details, per-layer Merkle tree progress, proof verification timing, and validator identity on each request.

## Analytics Database

The validator stores analytics locally for debugging and network analysis:

- **Enable**: `--analytics` on validator or gateway to enable DB writes (disabled by default for lighter operation)
- **Path**: `~/.verathos/verathos_validator.db` (SQLite, configurable via `VERALLM_DATA_DIR`)
- **Tables**: `canary_results` (test outcomes), `epoch_scores` (per-miner scoring), `network_receipts` (other validators' traffic)
- **Gateway**: `request_log` table alongside the credits DB (PostgreSQL or SQLite)
- **Backup**: weekly cycle exports old rows to `~/.verathos/backups/*.jsonl.gz`, then cleans live DB

## Network Status

View a live dashboard of all miners, models, validators, and scores across the network:

```bash
verathos network                        # one-shot overview
verathos network --watch                # auto-refresh every 30s
verathos network --watch 10             # auto-refresh every 10s
verathos network --json                 # JSON output (scriptable)
verathos network --proxy-url <URL>      # use a specific proxy endpoint
```

The dashboard shows miner slots (model, quant, score, health, GPU, context length), active validators with proxy endpoints, and per-model aggregates. Data is fetched from the nearest gateway (it tries localhost first, then `api.verathos.ai`, then on-chain validator discovery).

## Next Steps

- **Miners**: Your miner is now earning emissions based on throughput, latency, and model utility (parameters, context length, quantization). See [Bittensor Integration](bittensor_integration.md) for scoring details.
- **Validators**: The validator gateway is not yet publicly available. It will be released soon so every validator can run their own gateway and earn from inference revenue. See the [User Guide](user_guide.md) for the user-facing API.
- **Users**: See the [User Guide](user_guide.md) to start making API requests via [api.verathos.ai](https://api.verathos.ai).
