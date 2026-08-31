#!/usr/bin/env bash
# =====================================================================
# Script to stop, kill, and disable all MLOps daemon services & background tasks
# =====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🛑 [MLOps Teardown] Stopping and cleaning up all running services..."

# 1. Kill Python processes spawned by the project
echo "🔪 Terminating background processes (MLflow, Uvicorn, Telemetry Collector, Drift Monitor)..."

pkill -9 -f "mlflow server" 2>/dev/null || true
pkill -9 -f "uvicorn api.main:app" 2>/dev/null || true
pkill -9 -f "telemetry_collector.py" 2>/dev/null || true
pkill -9 -f "drift_monitor.py" 2>/dev/null || true

# 2. Release network ports if still occupied
if command -v fuser >/dev/null 2>&1; then
    echo "🔌 Freeing ports 5000 and 8000..."
    fuser -k 5000/tcp 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
fi

# 3. Stop & Disable systemd daemon service (if registered & active/enabled)
SERVICE_NAME="systemmonitor.service"

if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null || systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "⚙️ Disabling & stopping systemd service ($SERVICE_NAME)..."
    sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    sudo systemctl reset-failed "$SERVICE_NAME" 2>/dev/null || true
    echo "✅ Systemd service stopped and disabled."
else
    echo "ℹ️ Systemd service ($SERVICE_NAME) is not active/enabled."
fi

echo ""
echo "========================================================"
echo "✅ All MLOps processes, background tasks, and systemd services"
echo "   have been successfully STOPPED, KILLED, and DISABLED."
echo "========================================================"
