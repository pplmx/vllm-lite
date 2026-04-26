#!/bin/bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-vllm-lite}"
RELEASE_NAME="${RELEASE_NAME:-vllm-lite}"
CHART_PATH="${CHART_PATH:-./k8s/charts/vllm-lite}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] COMMAND

Commands:
    install     Install vllm-lite to Kubernetes
    upgrade     Upgrade existing vllm-lite deployment
    uninstall   Remove vllm-lite from Kubernetes
    status      Show deployment status
    logs        Show pod logs
    port-forward  Forward local port to service

Options:
    -n, --namespace NAMESPACE   Kubernetes namespace (default: vllm-lite)
    -r, --release RELEASE       Helm release name (default: vllm-lite)
    -v, --values FILE           Values file to use

Examples:
    $(basename "$0") install
    $(basename "$0") -n production install
    $(basename "$0") -v values.prod.yaml upgrade
    $(basename "$0") port-forward
EOF
    exit 1
}

check_prereqs() {
    for cmd in kubectl helm; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: $cmd is required but not installed"
            exit 1
        fi
    done
}

create_namespace() {
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        echo "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
}

cmd_install() {
    check_prereqs
    create_namespace

    echo "Installing vllm-lite..."
    helm install "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --values "${VALUES_FILE:-/dev/null}" \
        --wait \
        --timeout 10m

    echo ""
    echo "Installation complete!"
    echo "  Service: $RELEASE_NAME.$NAMESPACE.svc.cluster.local:8000"
    echo "  gRPC: $RELEASE_NAME.$NAMESPACE.svc.cluster.local:50051"
    echo "  Peer discovery: $RELEASE_NAME-peer.$NAMESPACE.svc.cluster.local"
    echo ""
    echo "Run '$(basename "$0") status' to check the deployment"
}

cmd_upgrade() {
    check_prereqs

    echo "Upgrading vllm-lite..."
    helm upgrade "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "${VALUES_FILE:-/dev/null}" \
        --wait \
        --timeout 10m

    echo "Upgrade complete!"
}

cmd_uninstall() {
    check_prereqs

    echo "Uninstalling vllm-lite..."
    helm uninstall "$RELEASE_NAME" --namespace "$NAMESPACE" || true

    echo "Uninstall complete. PVCs may remain - check with:"
    echo "  kubectl get pvc -n $NAMESPACE"
}

cmd_status() {
    check_prereqs

    echo "=== Deployment Status ==="
    kubectl get deployment -n "$NAMESPACE" -l "app.kubernetes.io/name=vllm-lite"

    echo ""
    echo "=== Pod Status ==="
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=vllm-lite"

    echo ""
    echo "=== Service Status ==="
    kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/name=vllm-lite"
}

cmd_logs() {
    check_prereqs

    local pod
    pod=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=vllm-lite" \
        --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "$pod" ]]; then
        echo "No running pods found"
        exit 1
    fi

    echo "Showing logs for: $pod"
    kubectl logs -n "$NAMESPACE" -f "$pod"
}

cmd_port_forward() {
    check_prereqs

    local port="${PORT:-8000}"
    echo "Port-forwarding localhost:$port to vllm-lite service..."
    echo "Press Ctrl+C to stop"
    kubectl port-forward -n "$NAMESPACE" svc/"$RELEASE_NAME" "$port:8000"
}

CMD="${1:-}"
shift || true

case "$CMD" in
    install|upgrade|uninstall|status|logs|port-forward)
        "cmd_$CMD" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $CMD"
        usage
        ;;
esac
