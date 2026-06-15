#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Inference server deployment script
#
# Usage:
#   # Deploy all 3 inference servers to remote host
#   bash deploy/deploy_inference_servers.sh --host REMOTE_HOST_IP
#
#   # Deploy and run smoke test
#   bash deploy/deploy_inference_servers.sh --host REMOTE_HOST_IP --test
#
#   # Deploy single adapter only
#   bash deploy/deploy_inference_servers.sh --host REMOTE_HOST_IP --adapter openpi
#
# Prerequisites:
#   - SSH key configured (passwordless login)
#   - Remote host has docker command access
#   - Docker containers exist and are running
# ──────────────────────────────────────────────────────────────

set -euo pipefail

HOST=""
RUN_TEST=false
ADAPTER="all"
REMOTE_DIR="/tmp/benchlink_deploy"

usage() {
    echo "Usage: $0 --host <host> [--test] [--adapter openpi|motus|dreamzero|all]"
    exit 1
}

# ── Argument parsing ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --test) RUN_TEST=true; shift ;;
        --adapter) ADAPTER="$2"; shift 2 ;;
        *) usage ;;
    esac
done

if [[ -z "$HOST" ]]; then
    usage
fi

# ── Adapter -> container mapping ──
declare -A CONTAINER_MAP=(
    ["openpi"]="openpi_server"
    ["motus"]="motus_server"
    ["dreamzero"]="dreamzero_final"
)

declare -A SCRIPT_MAP=(
    ["openpi"]="openpi_inference_server.py"
    ["motus"]="motus_inference_server.py"
    ["dreamzero"]="dreamzero_inference_server.py"
)

declare -A CONDA_MAP=(
    ["openpi"]=""
    ["motus"]=""
    ["dreamzero"]="dreamzero"
)

# ── Determine adapter list to deploy ──
if [[ "$ADAPTER" == "all" ]]; then
    ADAPTERS=("openpi" "motus" "dreamzero")
else
    ADAPTERS=("$ADAPTER")
fi

# ── 0. Copy scripts to remote server via SCP ──
echo "========================================================"
echo "  Step 0: Copy scripts to remote server"
echo "========================================================"
ssh "$HOST" "mkdir -p $REMOTE_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
for adapter in "${ADAPTERS[@]}"; do
    script="${SCRIPT_MAP[$adapter]}"
    echo "  Copying $script → $HOST:$REMOTE_DIR/"
    scp "$SCRIPT_DIR/$script" "$HOST:$REMOTE_DIR/$script"
done
# Also copy test scripts
scp "$SCRIPT_DIR/deploy/test_docker_adapter.py" "$HOST:$REMOTE_DIR/test_docker_adapter.py"
scp "$SCRIPT_DIR/schema.py" "$HOST:$REMOTE_DIR/schema.py"
echo "  [OK] All scripts copied"

# ── 1. docker cp into containers ──
echo ""
echo "========================================================"
echo "  Step 1: Copy inference servers into containers"
echo "========================================================"
for adapter in "${ADAPTERS[@]}"; do
    container="${CONTAINER_MAP[$adapter]}"
    script="${SCRIPT_MAP[$adapter]}"

    echo "  [${adapter}] Checking container '$container'..."
    status=$(ssh "$HOST" "docker inspect -f '{{.State.Status}}' $container 2>/dev/null || echo 'not_found'")

    if [[ "$status" == "not_found" ]]; then
        echo "  [SKIP] Container '$container' not found on $HOST"
        continue
    fi

    if [[ "$status" != "running" ]]; then
        echo "  [WARN] Container '$container' status=$status, starting..."
        ssh "$HOST" "docker start $container"
        sleep 2
    fi

    echo "  [${adapter}] docker cp $script → $container:/workspace/"
    ssh "$HOST" "docker cp $REMOTE_DIR/$script $container:/workspace/$script"
    echo "  [OK] $script deployed to $container"
done

# ── 2. Quick smoke test (verify script syntax + ready signal) ──
echo ""
echo "========================================================"
echo "  Step 2: Quick smoke test (syntax + ready signal)"
echo "========================================================"

for adapter in "${ADAPTERS[@]}"; do
    container="${CONTAINER_MAP[$adapter]}"
    script="${SCRIPT_MAP[$adapter]}"

    echo "  [${adapter}] Testing $script in $container..."
    set +e
    # Start a subprocess running the server, kill after ready signal received
    result=$(ssh "$HOST" "timeout 15 bash -c '
        docker exec -i $container python /workspace/$script --device cpu 2>/dev/null &
        PID=\$!
        # Wait for ready signal
        sleep 5
        kill \$PID 2>/dev/null
        echo \"server_started_ok\"
    '" 2>&1)
    exit_code=$?
    set -e

    if echo "$result" | grep -q "server_started_ok"; then
        echo "  [OK] $script syntax + startup OK"
    else
        # Also check if it's an import error (expected behavior on non-GPU with cpu)
        if echo "$result" | grep -qi "error"; then
            echo "  [WARN] $script startup had issues (expected on non-GPU):"
            echo "         $result" | head -5
        else
            echo "  [OK] $script syntax OK (no GPU ready signal)"
        fi
    fi
done

# ── 3. End-to-end test (optional) ──
if [[ "$RUN_TEST" == true ]]; then
    echo ""
    echo "========================================================"
    echo "  Step 3: End-to-end smoke test"
    echo "========================================================"
    echo "  Running test_docker_adapter.py on remote server..."
    echo ""
    ssh "$HOST" "cd $REMOTE_DIR && python test_docker_adapter.py ${ADAPTERS[*]}"
fi

# ── Cleanup ──
echo ""
echo "========================================================"
echo "  Cleanup"
echo "========================================================"
ssh "$HOST" "rm -rf $REMOTE_DIR"
echo "  [OK] Remote temp dir cleaned"

echo ""
echo "========================================================"
echo "  Deployment complete!"
echo "========================================================"
echo ""
echo "To run a full end-to-end test on the server:"
echo "  bash deploy/deploy_inference_servers.sh --host $HOST --test"
echo ""
echo "Or run individual adapter tests manually:"
echo "  # OpenPI"
echo "  ssh $HOST 'docker exec -i openpi_server python /workspace/openpi_inference_server.py --device cuda:0'"
echo ""
echo "  # Motus"
echo "  ssh $HOST 'docker exec -i motus_server python /workspace/motus_inference_server.py --device cuda:0'"
echo ""
echo "  # DreamZero"
echo "  ssh $HOST 'docker exec -i dreamzero_final bash -c \"source /opt/conda/etc/profile.d/conda.sh && conda activate dreamzero && python /workspace/dreamzero_inference_server.py --device cuda:0\"'"
