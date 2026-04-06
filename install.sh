#!/bin/bash
# =============================================================================
# Verathos One-Command Installer
# =============================================================================
#
# Downloads and sets up a Verathos miner or validator environment.
#
# Usage:
#   curl -fsSL https://verathos.ai/install.sh | bash
#   curl -fsSL https://verathos.ai/install.sh | bash -s -- --validator
#
# After install completes, the wizard launches automatically if running
# interactively. If piped (curl | bash), it prints instructions instead.
#
# =============================================================================

set -e

ROLE="miner"
REPO_URL="${VERATHOS_REPO_URL:-https://github.com/verathos-ai/verathos.git}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --validator) ROLE="validator"; shift ;;
        --repo) REPO_URL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  Verathos Installer ($ROLE)"
echo "============================================================"
echo ""

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
    cd "$REPO_DIR" && git pull --ff-only 2>/dev/null || true
else
    echo "  Cloning verathos to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# ── Run setup script ─────────────────────────────────────────────────────
if [ "$ROLE" = "validator" ]; then
    echo ""
    echo "  Running validator setup..."
    bash scripts/setup_validator.sh
    ENV_FILE="$REPO_DIR/.env-validator.sh"
else
    echo ""
    echo "  Running miner setup..."
    bash scripts/setup_miner.sh
    ENV_FILE="$REPO_DIR/.env.sh"
fi

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
    if [ "$ROLE" = "validator" ]; then
        python neurons/wizard.py validator
    else
        python neurons/wizard.py miner
    fi
else
    # Setup script already printed "Next steps" with cd + activate.
    echo ""
    echo "============================================================"
fi
