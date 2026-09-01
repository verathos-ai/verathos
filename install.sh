#!/bin/bash
# =============================================================================
# Verathos One-Command Installer
# =============================================================================
#
# Downloads and sets up a Verathos miner, validator, or GGUF mesh node.
#
# Usage:
#   curl -fsSL https://verathos.ai/install.sh | bash
#   curl -fsSL https://verathos.ai/install.sh | bash -s -- --validator
#   curl -fsSL https://verathos.ai/install.sh | bash -s -- --mesh-worker --token vtpool_...
#   curl -fsSL https://verathos.ai/install.sh | bash -s -- --mesh-coordinator
#
# The vLLM miner path and the GGUF mesh path are independent: they use
# separate venvs and separate PM2 units, and one machine can run both.
# Flags after "--" are passed through to scripts/join_pool.sh (mesh-worker),
# e.g. -- --gpus 0,1 --member-only.
#
# After install completes, the wizard launches automatically if running
# interactively. If piped (curl | bash), it prints instructions instead.
# The mesh-worker role never needs a wizard: the token is the whole config.
#
# =============================================================================

set -e

ROLE="miner"
REPO_URL="${VERATHOS_REPO_URL:-https://github.com/verathos-ai/verathos.git}"
# Release branch defaults to main; an alternate branch can be selected
# explicitly through the environment or command-line flag.
REPO_BRANCH="${VERATHOS_BRANCH:-}"
MESH_TOKEN=""
MESH_TOKEN_FILE=""
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --validator) ROLE="validator"; shift ;;
        --mesh-worker) ROLE="mesh-worker"; shift ;;
        --mesh-coordinator) ROLE="mesh-coordinator"; shift ;;
        --token) MESH_TOKEN="$2"; shift 2 ;;
        --token-file) MESH_TOKEN_FILE="$2"; shift 2 ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        --branch) REPO_BRANCH="$2"; shift 2 ;;
        --) shift; PASSTHRU=("$@"); break ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [ "$ROLE" = "mesh-worker" ] && [ -z "$MESH_TOKEN" ] && [ -z "$MESH_TOKEN_FILE" ]; then
    echo "ERROR: --mesh-worker needs --token vtpool_... (from your coordinator)"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Verathos Installer ($ROLE)"
echo "============================================================"
echo ""

# A minimal cloud image may not include git.  The one-command installer must
# establish its own clone prerequisite instead of failing before the role
# setup script can install the remaining system packages.
if ! command -v git >/dev/null 2>&1; then
    echo "  Installing bootstrap dependency: git..."
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git ca-certificates
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo apt-get update -qq
        sudo env DEBIAN_FRONTEND=noninteractive \
            apt-get install -y -qq git ca-certificates
    else
        echo "ERROR: git is required and automatic installation needs root or passwordless sudo." >&2
        exit 1
    fi
fi

# ── Detect workspace ──────────────────────────────────────────────────────
if [ -d /workspace ] && [ -w /workspace ]; then
    WORKSPACE=/workspace
else
    WORKSPACE="$HOME"
fi

REPO_DIR="${WORKSPACE}/verathos"

# ── Find or clone repo ────────────────────────────────────────────────────
# If we're already inside a verathos repo, use it directly
if [ -f "pyproject.toml" ] && grep -q "verathos" pyproject.toml 2>/dev/null; then
    REPO_DIR="$(pwd)"
    echo "  Using current directory: $REPO_DIR"
elif [ -d "$REPO_DIR" ] && [ -f "$REPO_DIR/pyproject.toml" ]; then
    echo "  Updating existing repo at $REPO_DIR..."
    cd "$REPO_DIR" || true
    if [ -n "$REPO_BRANCH" ]; then
        git fetch origin "$REPO_BRANCH" 2>/dev/null || true
        git checkout "$REPO_BRANCH" 2>/dev/null || true
    fi
    git pull --ff-only 2>/dev/null || true
else
    echo "  Cloning verathos to $REPO_DIR..."
    if [ -n "$REPO_BRANCH" ]; then
        git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    else
        git clone "$REPO_URL" "$REPO_DIR"
    fi
fi

cd "$REPO_DIR"

# ── Run setup script ─────────────────────────────────────────────────────
case "$ROLE" in
    validator)
        echo ""
        echo "  Running validator setup..."
        bash scripts/setup_validator.sh
        ENV_FILE="$REPO_DIR/.env-validator.sh"
        ;;
    mesh-worker)
        echo ""
        echo "  Running mesh worker setup..."
        bash scripts/setup_mesh.sh --role worker
        JOIN=(bash scripts/join_pool.sh)
        [ -z "$MESH_TOKEN" ] || JOIN+=(--token "$MESH_TOKEN")
        [ -z "$MESH_TOKEN_FILE" ] || JOIN+=(--token-file "$MESH_TOKEN_FILE")
        "${JOIN[@]}" "${PASSTHRU[@]}"
        echo ""
        echo "============================================================"
        echo "  Mesh worker joined. Watch it with: pm2 logs"
        echo "============================================================"
        exit 0
        ;;
    mesh-coordinator)
        echo ""
        echo "  Running mesh coordinator setup..."
        bash scripts/setup_mesh.sh --role coordinator
        ENV_FILE=""
        ;;
    *)
        echo ""
        echo "  Running miner setup..."
        bash scripts/setup_miner.sh
        ENV_FILE="$REPO_DIR/.env.sh"
        ;;
esac

# ── Source the environment ────────────────────────────────────────────────
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# ── Launch wizard or print instructions ───────────────────────────────────
# When piped (curl | bash), stdin is the script — can't do interactive prompts.
# When run directly (bash install.sh), stdin is the terminal — wizard works.
if [ -t 0 ]; then
    echo ""
    echo "  Launching setup wizard..."
    echo ""
    if [ "$ROLE" = "mesh-coordinator" ]; then
        "$REPO_DIR/.venv-mesh/bin/python" neurons/mesh_wizard.py mesh-coordinator
    elif [ "$ROLE" = "validator" ]; then
        python neurons/wizard.py validator
    else
        python neurons/wizard.py miner
    fi
else
    # Setup script already printed "Next steps" with cd + activate.
    echo ""
    if [ "$ROLE" = "mesh-coordinator" ]; then
        echo "  Non-interactive run. Create the pool with:"
        echo "    cd $REPO_DIR && .venv-mesh/bin/python neurons/mesh_wizard.py mesh-coordinator --yes"
    fi
    echo "============================================================"
fi
