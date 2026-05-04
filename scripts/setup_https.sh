#!/bin/bash
# =============================================================================
# Setup HTTPS reverse proxy for Verathos miner (self-signed cert).
#
# Installs nginx, generates a self-signed TLS certificate, and configures
# a reverse proxy from the specified HTTPS port to the miner's local port.
#
# Usage:
#   bash scripts/setup_https.sh                 # port 443 → localhost:8000
#   bash scripts/setup_https.sh --port 13998    # port 13998 → localhost:8000
#   bash scripts/setup_https.sh --port 13998 --backend-port 9000
#
# After setup, register your miner with:
#   --endpoint https://<YOUR_IP>:<PORT>
# =============================================================================

set -e

HTTPS_PORT=443
BACKEND_PORT=8000
APPEND=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) HTTPS_PORT="$2"; shift 2 ;;
        --backend-port) BACKEND_PORT="$2"; shift 2 ;;
        --append) APPEND=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_IP")

echo ""
echo "============================================================"
echo "  Verathos HTTPS Setup (self-signed)"
echo "============================================================"
echo "  HTTPS port:    $HTTPS_PORT"
echo "  Backend port:  $BACKEND_PORT (miner server)"
echo "  Public IP:     $PUBLIC_IP"
echo "============================================================"

# ── Install nginx ────────────────────────────────────────────────

if ! command -v nginx &>/dev/null; then
    echo ""
    echo "  Installing nginx..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq nginx openssl 2>&1 | tail -3
    elif command -v dnf &>/dev/null; then
        dnf install -y nginx openssl 2>&1 | tail -3
    else
        echo "  ERROR: Cannot install nginx automatically. Install it manually."
        exit 1
    fi
else
    echo "  nginx already installed"
fi

# ── Generate self-signed cert ────────────────────────────────────

CERT_DIR="/etc/nginx/ssl"
mkdir -p "$CERT_DIR"

if [ ! -f "$CERT_DIR/miner.key" ]; then
    echo ""
    echo "  Generating self-signed certificate..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$CERT_DIR/miner.key" \
        -out "$CERT_DIR/miner.crt" \
        -subj "/CN=$PUBLIC_IP" 2>/dev/null
    echo "  Certificate generated: $CERT_DIR/miner.crt"
else
    echo "  Certificate already exists: $CERT_DIR/miner.crt"
fi

# ── Configure nginx ──────────────────────────────────────────────

# Check for existing nginx config that might conflict
CONFLICTS=""
if [ -f /etc/nginx/nginx.conf ]; then
    if grep -q "listen.*$HTTPS_PORT" /etc/nginx/nginx.conf 2>/dev/null; then
        CONFLICTS="nginx.conf"
    fi
fi
for f in /etc/nginx/sites-enabled/*; do
    if [ -f "$f" ] && grep -q "listen.*$HTTPS_PORT" "$f" 2>/dev/null; then
        CONFLICTS="$CONFLICTS $f"
    fi
done
if [ -n "$CONFLICTS" ]; then
    echo ""
    echo "  WARNING: Port $HTTPS_PORT already configured in: $CONFLICTS"
    echo "  Remove or edit the conflicting config before continuing."
fi

# Reusable server-block snippet (writes to stdout).  Used by both fresh
# write and append paths; --append wraps it in a one-line awk insertion.
write_server_block() {
    cat << CONFEOF
server {
    listen $HTTPS_PORT ssl;
    ssl_certificate $CERT_DIR/miner.crt;
    ssl_certificate_key $CERT_DIR/miner.key;

    client_max_body_size 10m;

    location / {
        proxy_pass http://127.0.0.1:$BACKEND_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
    }
}
CONFEOF
}

# Check if nginx.conf includes sites-enabled (standard distro setup).
# Many cloud GPU images ship a custom nginx.conf without sites-enabled —
# in that case, write our server block directly into nginx.conf.
USES_SITES=0
if grep -q "include.*sites-enabled" /etc/nginx/nginx.conf 2>/dev/null; then
    USES_SITES=1
fi

if [ "$APPEND" = "1" ]; then
    # ── Append a second/Nth server block, preserve existing ones ──
    if [ "$USES_SITES" = "1" ]; then
        TARGET="/etc/nginx/sites-available/verathos-miner"
        if [ ! -f "$TARGET" ]; then
            echo "  ERROR: --append used but $TARGET doesn't exist."
            echo "  Run setup_https.sh without --append first to create the base config."
            exit 1
        fi
        # Skip if a server block for this port already exists.
        if grep -qE "listen[[:space:]]+$HTTPS_PORT[[:space:]]+ssl" "$TARGET"; then
            echo "  Port $HTTPS_PORT already configured in $TARGET — nothing to append."
        else
            echo "" >> "$TARGET"
            write_server_block >> "$TARGET"
            echo "  Appended server block (port $HTTPS_PORT → :$BACKEND_PORT) to $TARGET"
        fi
        ln -sf "$TARGET" /etc/nginx/sites-enabled/verathos-miner
    else
        # Custom nginx.conf — append a server block inside the http {} body.
        TARGET="/etc/nginx/nginx.conf"
        if grep -qE "listen[[:space:]]+$HTTPS_PORT[[:space:]]+ssl" "$TARGET"; then
            echo "  Port $HTTPS_PORT already configured in $TARGET — nothing to append."
        else
            cp "$TARGET" "$TARGET.bak.$(date +%s)"
            BLOCK="$(write_server_block | sed 's/^/    /')"
            # Insert before the FINAL closing brace of the http {} block
            # (last "}" line in the file).  Escape special chars for awk.
            tmp="$(mktemp)"
            awk -v block="$BLOCK" '
                {
                    lines[NR] = $0
                }
                END {
                    last_brace = 0
                    for (i = NR; i >= 1; i--) {
                        if (lines[i] ~ /^[[:space:]]*}[[:space:]]*$/) {
                            last_brace = i
                            break
                        }
                    }
                    for (i = 1; i <= NR; i++) {
                        if (i == last_brace) print block
                        print lines[i]
                    }
                }
            ' "$TARGET" > "$tmp"
            mv "$tmp" "$TARGET"
            echo "  Appended server block (port $HTTPS_PORT → :$BACKEND_PORT) to $TARGET"
        fi
    fi
elif [ "$USES_SITES" = "1" ]; then
    # ── Fresh write, sites-enabled layout ──
    NGINX_CONF="/etc/nginx/sites-available/verathos-miner"
    write_server_block > "$NGINX_CONF"
    mkdir -p /etc/nginx/sites-enabled
    ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/verathos-miner
    if [ -f /etc/nginx/sites-enabled/default ]; then
        rm -f /etc/nginx/sites-enabled/default
    fi
else
    # ── Fresh write, custom nginx.conf layout ──
    cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak
    cat > /etc/nginx/nginx.conf << CONFEOF
events { worker_connections 2048; }
http {
$(write_server_block | sed 's/^/    /')
}
CONFEOF
    echo "  Note: replaced nginx.conf (backup at nginx.conf.bak)"
fi

# Test and reload
if nginx -t 2>&1 | grep -q "syntax is ok"; then
    # If nginx is running, reload gracefully; otherwise start fresh.
    # `nginx -s quit` fails when no nginx is running — `|| true` keeps
    # us going under `set -e`.
    if pgrep -x nginx >/dev/null 2>&1; then
        nginx -s reload 2>/dev/null || nginx -s quit 2>/dev/null || true
        sleep 1
    fi
    if ! pgrep -x nginx >/dev/null 2>&1; then
        nginx 2>/dev/null || {
            echo "  WARNING: nginx failed to start. Check for port conflicts:"
            echo "    ss -tlnp | grep $HTTPS_PORT"
            exit 1
        }
    fi
    echo ""
    echo "  nginx configured and running on port $HTTPS_PORT"
else
    echo ""
    echo "  ERROR: nginx config test failed:"
    nginx -t 2>&1
    exit 1
fi

# ── Verify ───────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  HTTPS reverse proxy ready!"
echo "============================================================"
echo ""
echo "  Register your miner with:"
echo "    --endpoint https://$PUBLIC_IP:$HTTPS_PORT"
echo ""
echo "  Test locally:"
echo "    curl -sk https://localhost:$HTTPS_PORT/health"
echo ""
echo "============================================================"
