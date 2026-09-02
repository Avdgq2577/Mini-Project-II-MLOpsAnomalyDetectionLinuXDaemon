#!/usr/bin/env bash
# =====================================================================
# Script to launch MLOps Anomaly Detection Services in separate terminals
# within the virtual environment (proenv), and open local browsers.
# =====================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VENV_PATH="$SCRIPT_DIR/proenv"
echo "🚀 [MLOps Launcher] Initializing MLOps Anomaly Detection System..."
# 1. Check Virtual Environment
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment 'proenv' not found at $VENV_PATH!"
    echo "   Please create it using: python3 -m venv proenv && source proenv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
# Activate virtual environment
source "$VENV_PATH/bin/activate"
# 2. Check if Model exists; if not, train baseline & model
MODEL_FILE="$SCRIPT_DIR/api/models/isolation_forest.joblib"
if [ ! -f "$MODEL_FILE" ]; then
    echo "⚠️  Model artifact missing at $MODEL_FILE. Generating baseline data and training model..."
    python "$SCRIPT_DIR/mlops/generate_baseline.py"
    python "$SCRIPT_DIR/mlops/train.py"
fi
# 3. Terminal Launcher Function
launch_in_terminal() {
    local title="$1"
    local cmd="$2"
    echo "▶️ Launching terminal: $title"
    local full_cmd="cd '$SCRIPT_DIR' && source proenv/bin/activate && echo '====================================================' && echo '  $title' && echo '====================================================' && $cmd; exec bash"
    if command -v konsole >/dev/null 2>&1; then
        konsole -p title="$title" --workdir "$SCRIPT_DIR" --noclose -e bash -c "$full_cmd" &
    elif command -v kitty >/dev/null 2>&1; then
        kitty --title "$title" --directory "$SCRIPT_DIR" bash -c "$full_cmd" &
    elif command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="$title" --working-directory="$SCRIPT_DIR" -- bash -c "$full_cmd" &
    elif command -v xfce4-terminal >/dev/null 2>&1; then
        xfce4-terminal --title="$title" --working-directory="$SCRIPT_DIR" -e "bash -c \"$full_cmd\"" &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -title "$title" -e "bash -c \"$full_cmd\"" &
    else
        echo "⚠️ No GUI terminal emulator detected. Running in background..."
        bash -c "$full_cmd" > "/tmp/${title// /_}.log" 2>&1 &
    fi
    sleep 1
}
# 4. Launch each component in its own terminal
echo "🖥️ Spawning terminals within virtual environment (proenv)..."
# Terminal 1: MLflow UI (Port 5000)
launch_in_terminal "MLflow Tracking Server" "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"
# Terminal 2: FastAPI Microservice (Port 8000)
launch_in_terminal "FastAPI Anomaly API" "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
# Terminal 3: Telemetry Collector Daemon
launch_in_terminal "Telemetry Collector Daemon" "python daemon/telemetry_collector.py"
# Terminal 4: Continuous Drift Monitor
launch_in_terminal "Continuous Drift Monitor" "python mlops/drift_monitor.py"
# 5. Wait for servers to spin up
echo "⏳ Waiting 4 seconds for HTTP services to start (ports 5000 & 8000)..."
sleep 4
# 6. Open Browser URLs
echo "🌐 Opening web interfaces in browser..."
URL_DASHBOARD="http://localhost:8000"
URL_SWAGGER="http://localhost:8000/docs"
URL_MLFLOW="http://localhost:5000"
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL_DASHBOARD" >/dev/null 2>&1 &
    sleep 1
    xdg-open "$URL_SWAGGER" >/dev/null 2>&1 &
    sleep 1
    xdg-open "$URL_MLFLOW" >/dev/null 2>&1 &
elif command -v python3 >/dev/null 2>&1; then
    python3 -m webbrowser "$URL_DASHBOARD" >/dev/null 2>&1 &
    python3 -m webbrowser "$URL_SWAGGER" >/dev/null 2>&1 &
    python3 -m webbrowser "$URL_MLFLOW" >/dev/null 2>&1 &
fi
echo ""
echo "========================================================"
echo "✅ All 4 services are running in separate terminals!"
echo "   1. Real-Time Dashboard: $URL_DASHBOARD"
echo "   2. FastAPI Swagger UI:  $URL_SWAGGER"
echo "   3. MLflow UI:           $URL_MLFLOW"
echo "========================================================"
