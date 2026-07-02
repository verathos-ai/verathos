#!/bin/bash
# =============================================================================
# Setup nginx reverse proxy for validator hot-capacity audit ingest.
#
# The validator process should bind the ingest server to localhost. This script
# exposes only the capacity-audit ingest paths on a public HTTP port and forwards
# them to the local validator ingest port.
#
# Usage:
#   bash scripts/setup_capacity_audit_ingest_nginx.sh
#   bash scripts/setup_capacity_audit_ingest_nginx.sh --public-port 8091 --backend-port 8092
#
# Validator environment for the default setup:
#   VERATHOS_CAPACITY_AUDIT_INGEST_HOST=127.0.0.1
#   VERATHOS_CAPACITY_AUDIT_INGEST_PORT=8092
#   VERATHOS_CAPACITY_AUDIT_PUBLIC_URL=http://<PUBLIC_IP>:8091
# =============================================================================

set -e

PUBLIC_PORT=8091
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8092
SITE_NAME="verathos-capacity-audit-ingest"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --public-port) PUBLIC_PORT="$2"; shift 2 ;;
        --backend-host) BACKEND_HOST="$2"; shift 2 ;;
        --backend-port) BACKEND_PORT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

PUBLIC_IP=$(curl -fsS --max-time 5 ifconfig.me 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "YOUR_IP")

echo ""
echo "============================================================"
echo "  Verathos Capacity Audit Ingest Proxy"
echo "============================================================"
echo "  Public port:   $PUBLIC_PORT"
echo "  Backend:       http://$BACKEND_HOST:$BACKEND_PORT"
echo "  Public URL:    http://$PUBLIC_IP:$PUBLIC_PORT"
echo "============================================================"

if ! command -v nginx >/dev/null 2>&1; then
    echo ""
    echo "  Installing nginx..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y -qq nginx 2>&1 | tail -3
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y nginx 2>&1 | tail -3
    else
        echo "  ERROR: Cannot install nginx automatically. Install nginx manually."
        exit 1
    fi
else
    echo "  nginx already installed"
fi

if ss -tlnp 2>/dev/null | grep -qE ":${PUBLIC_PORT}[[:space:]]"; then
    if ! ss -tlnp 2>/dev/null | grep -E ":${PUBLIC_PORT}[[:space:]]" | grep -q nginx; then
        echo "  ERROR: port $PUBLIC_PORT is already used by a non-nginx process."
        ss -tlnp | grep -E ":${PUBLIC_PORT}[[:space:]]" || true
        exit 1
    fi
fi

write_server_block() {
    cat << CONFEOF
server {
    listen $PUBLIC_PORT;
    server_tokens off;
    client_max_body_size 64m;

    access_log /var/log/nginx/verathos-capacity-audit-access.log;
    error_log /var/log/nginx/verathos-capacity-audit-error.log warn;

    location = /capacity/audit/v1/health {
        limit_except GET { deny all; }
        proxy_pass http://$BACKEND_HOST:$BACKEND_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 10s;
        proxy_connect_timeout 3s;
        proxy_send_timeout 10s;
        proxy_buffering off;
    }

    location = /capacity/audit/v1/receipt {
        limit_except POST { deny all; }
        proxy_pass http://$BACKEND_HOST:$BACKEND_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 30s;
        proxy_connect_timeout 3s;
        proxy_send_timeout 30s;
        proxy_buffering off;
    }

    location = /capacity/audit/v1/proof {
        limit_except POST { deny all; }
        proxy_pass http://$BACKEND_HOST:$BACKEND_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        proxy_connect_timeout 3s;
        proxy_send_timeout 120s;
        proxy_buffering off;
    }

    location / {
        return 404;
    }
}
CONFEOF
}

USES_SITES=0
if grep -q "include.*sites-enabled" /etc/nginx/nginx.conf 2>/dev/null; then
    USES_SITES=1
fi

if [ "$USES_SITES" = "1" ]; then
    TARGET="/etc/nginx/sites-available/$SITE_NAME"
    write_server_block > "$TARGET"
    mkdir -p /etc/nginx/sites-enabled
    ln -sf "$TARGET" "/etc/nginx/sites-enabled/$SITE_NAME"
else
    TARGET="/etc/nginx/nginx.conf"
    cp "$TARGET" "$TARGET.bak.$(date +%s)"
    tmp="$(mktemp)"
    block="$(write_server_block | sed 's/^/    /')"
    awk -v block="$block" '
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
fi

nginx -t
if pgrep -x nginx >/dev/null 2>&1; then
    nginx -s reload
else
    nginx
fi

echo ""
echo "============================================================"
echo "  Capacity audit ingest proxy ready"
echo "============================================================"
echo ""
echo "  Set validator capacity-audit environment:"
echo "    VERATHOS_CAPACITY_AUDIT_INGEST_HOST=$BACKEND_HOST"
echo "    VERATHOS_CAPACITY_AUDIT_INGEST_PORT=$BACKEND_PORT"
echo "    VERATHOS_CAPACITY_AUDIT_PUBLIC_URL=http://$PUBLIC_IP:$PUBLIC_PORT"
echo ""
echo "  Test:"
echo "    curl -fsS http://$PUBLIC_IP:$PUBLIC_PORT/capacity/audit/v1/health"
echo ""
echo "============================================================"
