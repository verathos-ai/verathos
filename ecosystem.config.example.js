// PM2 ecosystem config for Verathos.
// Auto-generate with: verathos setup
//
// Copy and fill in your values:
//   cp ecosystem.config.example.js ecosystem.config.js
//
// Network (pick one):
//   --subtensor-network finney               # mainnet (netuid 96)
//   --subtensor-network test                 # testnet (netuid 405)
//   --subtensor-chain-endpoint wss://...     # local or custom subtensor
//
// IMPORTANT: After changing args, delete + recreate the PM2 process:
//   pm2 delete miner && pm2 start ecosystem.config.js --only miner
//
// Usage:
//   pm2 start ecosystem.config.js          # start all
//   pm2 start ecosystem.config.js --only miner
//   pm2 logs validator --lines 50
//   pm2 stop all
//   pm2 monit                              # live dashboard

module.exports = {
  apps: [
    // ── Miner ────────────────────────────────────────────────────
    // Required: --wallet, --hotkey, --netuid, --endpoint
    // Model: --model-id auto (auto-select best for GPU) or --model-id <id> --quant <quant>
    // Logging: INFO by default. Add --logging.debug for verbose output.
    // TEE mode: add --tee-enabled --tee-platform tdx|sev-snp
    // Auto-update: restarts the GPU server on new versions — use with caution
    //   on expensive GPU instances. Consider manual updates for miners.
    {
      name: "miner",
      script: ".venv-vllm/bin/python",
      args: "-u -m neurons.miner --wallet <WALLET> --hotkey <HOTKEY> --netuid 96 --subtensor-network finney --model-id auto --endpoint https://<YOUR_PUBLIC_IP_OR_DOMAIN> --auto-update",
      cwd: "<REPO_ROOT>",
      // GPU-bound — do NOT auto-restart. Crash loops waste VRAM.
      // Investigate before restarting manually.
      autorestart: false,
      max_restarts: 0,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    },

    // ── Validator ─────────────────────────────────────────────────
    // Required: --wallet, --hotkey, --netuid
    // Optional: --analytics (canary results, epoch scores, network receipts)
    //           --retain-backups (keep analytics backup files; default: auto-deleted after 7 days)
    // Logging: INFO by default. Add --logging.debug for verbose output.
    {
      name: "validator",
      script: ".venv-validator/bin/python",
      args: "-u -m neurons.validator --wallet <WALLET> --hotkey <HOTKEY> --netuid 96 --subtensor-network finney --auto-update --analytics --retain-backups",
      cwd: "<REPO_ROOT>",
      env: {
        // HuggingFace token — avoids rate limits when downloading tokenizers
        // for input commitment verification. Get one at https://huggingface.co/settings/tokens
        HF_TOKEN: "",
      },
      // Auto-restart on crash (transient RPC errors, etc.)
      autorestart: true,
      max_restarts: 5,
      min_uptime: "60s",
      restart_delay: 10000,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      max_size: "50M",
      retain: 3,
    },
  ],
};
