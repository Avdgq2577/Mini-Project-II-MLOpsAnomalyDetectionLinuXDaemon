from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os
import logging
import time
import uuid
import json
import subprocess
from collections import deque
from api.schemas import TelemetryPayload, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI")

app = FastAPI(title="Linux Telemetry Anomaly API", version="1.1.0")

# Paths and global variables
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
ANOMALY_JSONL_LOG = os.path.join(LOGS_DIR, "anomalies.jsonl")
ANOMALY_TEXT_LOG = os.path.join(LOGS_DIR, "anomalies.log")

model_pipeline = None
recent_history = deque(maxlen=300) # last 300 data ticks (5 mins)
total_anomalies_count = 0
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "isolation_forest.joblib")
RECENT_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recent_telemetry.csv")

def analyze_root_cause(data_dict: dict, score: float, prediction: int):
    if prediction == 1:
        return "NOMINAL", [], []
    
    suspected_causes = []
    recommended_actions = []
    
    cpu = data_dict.get("cpu_percent", 0.0)
    if cpu > 75.0:
        suspected_causes.append(f"High CPU utilization ({cpu:.1f}%). High compute workload or runaway process.")
        recommended_actions.append("Run 'top -b -n 1' or 'pidstat -u 1 5' to identify high-CPU processes.")
    
    mem = data_dict.get("mem_percent", 0.0)
    if mem > 80.0:
        suspected_causes.append(f"Elevated RAM usage ({mem:.1f}%). Possible memory leak or large buffer allocation.")
        recommended_actions.append("Execute 'ps aux --sort=-%mem | head -n 10' to inspect memory distribution.")
    
    ctx = data_dict.get("ctx_switches", 0)
    if ctx > 200000:
        suspected_causes.append(f"High context switch frequency ({ctx:,} switches/sec). Thread thrashing detected.")
        recommended_actions.append("Audit thread concurrency pool size and worker thread settings.")
    
    disk_write = data_dict.get("disk_write_bytes", 0)
    disk_read = data_dict.get("disk_read_bytes", 0)
    if disk_write > 50_000_000 or disk_read > 50_000_000:
        suspected_causes.append(f"Heavy Disk I/O activity (Write: {disk_write/1e6:.1f} MB, Read: {disk_read/1e6:.1f} MB).")
        recommended_actions.append("Execute 'iotop -o -b -n 1' to trace active disk read/write processes.")
    
    net_sent = data_dict.get("net_bytes_sent", 0)
    net_recv = data_dict.get("net_bytes_recv", 0)
    if net_sent > 20_000_000 or net_recv > 20_000_000:
        suspected_causes.append(f"Spike in network traffic (Sent: {net_sent/1e6:.1f} MB, Recv: {net_recv/1e6:.1f} MB).")
        recommended_actions.append("Execute 'ss -tulpn' or 'iftop' to inspect active network sockets.")

    if not suspected_causes:
        suspected_causes.append("Multi-metric statistical anomaly detected by Isolation Forest decision boundary.")
        recommended_actions.append("Check system logs ('journalctl -xe') for hardware or system daemon warnings.")
    
    if score < -0.25:
        severity = "CRITICAL"
    elif score < -0.15:
        severity = "HIGH"
    elif score < -0.05:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return severity, suspected_causes, recommended_actions

def log_anomaly_event(event_data: dict):
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(ANOMALY_JSONL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data) + "\n")
        
        timestamp = event_data.get("timestamp_utc", "")
        event_id = event_data.get("event_id", "")
        severity = event_data.get("severity", "")
        score = event_data.get("anomaly_score", 0.0)
        causes = "\n  - ".join(event_data.get("suspected_causes", []))
        actions = "\n  - ".join(event_data.get("recommended_actions", []))
        
        formatted_entry = (
            f"================================================================================\n"
            f"[{timestamp}] ANOMALY DETECTED | Event ID: {event_id} | Severity: {severity}\n"
            f"Isolation Forest Score: {score:.4f}\n"
            f"Suspected Causes:\n  - {causes}\n"
            f"Recommended Actions:\n  - {actions}\n"
            f"================================================================================\n"
        )
        with open(ANOMALY_TEXT_LOG, "a", encoding="utf-8") as f:
            f.write(formatted_entry)
    except Exception as e:
        logger.error(f"Failed to log anomaly event: {e}")

@app.on_event("startup")
async def load_model():
    global model_pipeline
    os.makedirs(LOGS_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Loaded Isolation Forest model from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Inference will fail until a model is trained.")

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(payload: TelemetryPayload):
    global model_pipeline
    
    if model_pipeline is None:
        if os.path.exists(MODEL_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info("Hot-reloaded model.")
        else:
            raise HTTPException(status_code=503, detail="Model is not yet loaded or trained.")
    
    data_dict = payload.model_dump()
    df = pd.DataFrame([data_dict])
    
    try:
        prediction = int(model_pipeline.predict(df)[0])  # 1 (normal) or -1 (anomaly)
        score = float(model_pipeline.decision_function(df)[0])
        
        severity, causes, actions = analyze_root_cause(data_dict, score, prediction)
        
        global total_anomalies_count
        if prediction == -1:
            total_anomalies_count += 1
            event_id = str(uuid.uuid4())[:8]
            event_record = {
                "event_id": event_id,
                "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "anomaly_score": score,
                "severity": severity,
                "metrics": data_dict,
                "suspected_causes": causes,
                "recommended_actions": actions
            }
            log_anomaly_event(event_record)
        
        # Log to history buffer for real-time dashboard
        history_item = data_dict.copy()
        history_item['timestamp'] = time.strftime("%H:%M:%S")
        history_item['anomaly_flag'] = prediction
        history_item['anomaly_score'] = score
        history_item['severity'] = severity
        history_item['suspected_causes'] = causes
        history_item['recommended_actions'] = actions
        recent_history.append(history_item)
        
        # Log to recent telemetry CSV for drift detection
        log_df = df.copy()
        log_df['anomaly_flag'] = prediction
        log_df['anomaly_score'] = score
        log_df.to_csv(RECENT_DATA_FILE, mode='a', header=not os.path.exists(RECENT_DATA_FILE), index=False)
        
        return PredictionResponse(
            status="success",
            anomaly_flag=prediction,
            anomaly_score=score,
            severity=severity,
            suspected_causes=causes,
            recommended_actions=actions,
            message="Anomalous behavior detected" if prediction == -1 else "Nominal operation"
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/live-metrics")
async def get_live_metrics():
    history_list = list(recent_history)
    total_samples = len(history_list)
    anomalies = [item for item in history_list if item.get('anomaly_flag') == -1]
    latest = history_list[-1] if history_list else None
    
    return {
        "status": "active",
        "total_ticks": total_samples,
        "recent_anomalies_count": len(anomalies),
        "total_anomalies_count": total_anomalies_count,
        "latest": latest,
        "history": history_list
    }

@app.get("/api/v1/anomaly-logs")
async def get_anomaly_logs(limit: int = 50):
    if not os.path.exists(ANOMALY_JSONL_LOG):
        return {"total": 0, "logs": []}
    
    logs = []
    try:
        with open(ANOMALY_JSONL_LOG, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line.strip()))
        return {"total": len(logs), "logs": logs[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read anomaly logs: {e}")

def run_stress_task():
    try:
        subprocess.run(["stress-ng", "--cpu", "4", "--timeout", "10s"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        end_time = time.time() + 10
        while time.time() < end_time:
            _ = [i**2 for i in range(100000)]

@app.post("/api/v1/trigger-stress")
async def trigger_stress(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_stress_task)
    return {"status": "started", "message": "Stress test workload initiated for 10 seconds!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Linux Daemon - Real-Time Anomaly Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-orange: #f97316;
            --accent-purple: #8b5cf6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: var(--bg-dark); color: var(--text-main); min-height: 100vh; padding: 20px; }
        
        .header {
            display: flex; justify-content: space-between; align-items: center;
            background: var(--card-bg); backdrop-filter: blur(12px);
            padding: 20px 30px; border-radius: 16px; border: 1px solid var(--border-color);
            margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .title-area h1 { font-size: 1.6rem; font-weight: 700; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .title-area p { color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }
        
        .status-badge {
            padding: 8px 18px; border-radius: 30px; font-weight: 600; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;
            transition: all 0.3s ease;
        }
        .badge-normal { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-anomaly { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.5); animation: pulse 1s infinite; }
        
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }

        .controls-area { display: flex; gap: 12px; }
        .btn-stress {
            background: linear-gradient(135deg, #ef4444, #dc2626); color: white; border: none; padding: 10px 20px;
            border-radius: 10px; font-weight: 600; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 14px rgba(239,68,68,0.4);
        }
        .btn-stress:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(239,68,68,0.6); }

        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .metric-card {
            background: var(--card-bg); backdrop-filter: blur(12px); padding: 20px; border-radius: 16px; border: 1px solid var(--border-color);
            position: relative; overflow: hidden;
        }
        .metric-label { color: var(--text-muted); font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-value { font-size: 2rem; font-weight: 700; margin-top: 8px; color: #ffffff; }
        .metric-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 6px; }

        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        @media (max-width: 900px) { .charts-grid { grid-template-columns: 1fr; } }
        
        .chart-card {
            background: var(--card-bg); backdrop-filter: blur(12px); padding: 20px; border-radius: 16px; border: 1px solid var(--border-color);
        }
        .chart-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; color: var(--text-main); }
        .chart-container { position: relative; height: 260px; width: 100%; }

        .logs-card {
            background: var(--card-bg); backdrop-filter: blur(12px); padding: 20px; border-radius: 16px; border: 1px solid var(--border-color);
        }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.88rem; }
        th { text-align: left; padding: 12px; color: var(--text-muted); border-bottom: 1px solid var(--border-color); }
        td { padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); vertical-align: top; }
        
        .sev-CRITICAL { color: #f87171; font-weight: 700; background: rgba(239, 68, 68, 0.25); padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(239,68,68,0.4); }
        .sev-HIGH { color: #fb923c; font-weight: 700; background: rgba(249, 115, 22, 0.2); padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(249,115,22,0.4); }
        .sev-MEDIUM { color: #facc15; font-weight: 600; background: rgba(234, 179, 8, 0.15); padding: 3px 8px; border-radius: 6px; }
        .sev-LOW { color: #38bdf8; font-weight: 600; background: rgba(56, 189, 248, 0.15); padding: 3px 8px; border-radius: 6px; }
        .sev-NOMINAL { color: var(--accent-green); font-weight: 600; background: rgba(16, 185, 129, 0.15); padding: 3px 8px; border-radius: 6px; }
        
        .cause-list { margin: 0; padding-left: 16px; color: #e2e8f0; font-size: 0.83rem; }
        .action-list { margin: 0; padding-left: 16px; color: #38bdf8; font-size: 0.83rem; }
    </style>
</head>
<body>

    <div class="header">
        <div class="title-area">
            <h1>Linux Daemon MLOps Anomaly Detection</h1>
            <p>Real-Time System Telemetry & Root Cause Diagnostic Engine</p>
        </div>
        <div class="controls-area">
            <div id="statusBadge" class="status-badge badge-normal">
                <span id="statusDot">🟢</span> <span id="statusText">NOMINAL OPERATION</span>
            </div>
            <button class="btn-stress" onclick="triggerStressTest()">🔥 Trigger Stress Test</button>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">CPU Utilization</div>
            <div class="metric-value" id="valCpu">0.0%</div>
            <div class="metric-sub" id="subCpu">Telemetry Stream Active</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RAM Utilization</div>
            <div class="metric-value" id="valMem">0.0%</div>
            <div class="metric-sub" id="subMem">Virtual Memory Usage</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Decision Score</div>
            <div class="metric-value" id="valScore">0.00</div>
            <div class="metric-sub">Isolation Forest Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Anomalies</div>
            <div class="metric-value" id="valAnomalies" style="color: var(--accent-red);">0</div>
            <div class="metric-sub">Logged to logs/anomalies.jsonl</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-card">
            <div class="chart-title">Real-Time Telemetry (CPU & RAM %)</div>
            <div class="chart-container">
                <canvas id="telemetryChart"></canvas>
            </div>
        </div>
        <div class="chart-card">
            <div class="chart-title">Isolation Forest Decision Score</div>
            <div class="chart-container">
                <canvas id="scoreChart"></canvas>
            </div>
        </div>
    </div>

    <div class="logs-card">
        <div class="chart-title">Live Telemetry & Diagnostic Analysis Feed</div>
        <table>
            <thead>
                <tr>
                    <th style="width:90px;">Time</th>
                    <th style="width:70px;">CPU %</th>
                    <th style="width:70px;">RAM %</th>
                    <th style="width:90px;">Score</th>
                    <th style="width:100px;">Severity</th>
                    <th>Suspected Cause(s)</th>
                    <th>Recommended Operational Actions</th>
                </tr>
            </thead>
            <tbody id="logsTable">
                <tr><td colspan="7" style="text-align:center; color:var(--text-muted);">Waiting for telemetry stream from daemon...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        let telemetryChart, scoreChart;

        function initCharts() {
            const ctx1 = document.getElementById('telemetryChart').getContext('2d');
            telemetryChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'CPU %', data: [], borderColor: '#06b6d4', backgroundColor: 'rgba(6, 182, 212, 0.1)', fill: true, tension: 0.3 },
                        { label: 'RAM %', data: [], borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)', fill: true, tension: 0.3 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { min: 0, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    },
                    plugins: { legend: { labels: { color: '#f8fafc' } } }
                }
            });

            const ctx2 = document.getElementById('scoreChart').getContext('2d');
            scoreChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Anomaly Decision Score', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true, tension: 0.3 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    },
                    plugins: { legend: { labels: { color: '#f8fafc' } } }
                }
            });
        }

        async function fetchMetrics() {
            try {
                const res = await fetch('/api/v1/live-metrics');
                const data = await res.json();
                
                if (!data.history || data.history.length === 0) return;

                const history = data.history;
                const latest = data.latest;

                // Update Metric Values
                document.getElementById('valCpu').innerText = latest.cpu_percent.toFixed(1) + '%';
                document.getElementById('valMem').innerText = latest.mem_percent.toFixed(1) + '%';
                document.getElementById('valScore').innerText = latest.anomaly_score.toFixed(3);
                document.getElementById('valAnomalies').innerText = data.total_anomalies_count !== undefined ? data.total_anomalies_count : data.recent_anomalies_count;

                // Update Status Badge
                const badge = document.getElementById('statusBadge');
                if (latest.anomaly_flag === -1) {
                    badge.className = 'status-badge badge-anomaly';
                    document.getElementById('statusDot').innerText = '🚨';
                    document.getElementById('statusText').innerText = `ANOMALY DETECTED (${latest.severity || 'HIGH'})`;
                } else {
                    badge.className = 'status-badge badge-normal';
                    document.getElementById('statusDot').innerText = '🟢';
                    document.getElementById('statusText').innerText = 'NOMINAL OPERATION';
                }

                // Update Charts
                const labels = history.map(h => h.timestamp);
                const cpuData = history.map(h => h.cpu_percent);
                const memData = history.map(h => h.mem_percent);
                const scoreData = history.map(h => h.anomaly_score);

                telemetryChart.data.labels = labels;
                telemetryChart.data.datasets[0].data = cpuData;
                telemetryChart.data.datasets[1].data = memData;
                telemetryChart.update('none');

                scoreChart.data.labels = labels;
                scoreChart.data.datasets[0].data = scoreData;
                scoreChart.data.datasets[0].borderColor = scoreData.map(s => s < 0 ? '#ef4444' : '#10b981');
                scoreChart.update('none');

                // Update Table (Last 10 items)
                const tbody = document.getElementById('logsTable');
                tbody.innerHTML = '';
                const recentSlice = history.slice(-10).reverse();
                recentSlice.forEach(item => {
                    const tr = document.createElement('tr');
                    const isAnomaly = item.anomaly_flag === -1;
                    const sev = item.severity || (isAnomaly ? 'HIGH' : 'NOMINAL');
                    
                    const causesHtml = (item.suspected_causes && item.suspected_causes.length > 0)
                        ? `<ul class="cause-list">${item.suspected_causes.map(c => `<li>${c}</li>`).join('')}</ul>`
                        : `<span style="color:var(--text-muted);">None (Nominal)</span>`;
                        
                    const actionsHtml = (item.recommended_actions && item.recommended_actions.length > 0)
                        ? `<ul class="action-list">${item.recommended_actions.map(a => `<li><code>${a}</code></li>`).join('')}</ul>`
                        : `<span style="color:var(--text-muted);">-</span>`;

                    tr.innerHTML = `
                        <td>${item.timestamp}</td>
                        <td>${item.cpu_percent.toFixed(1)}%</td>
                        <td>${item.mem_percent.toFixed(1)}%</td>
                        <td style="color:${isAnomaly ? '#ef4444' : '#10b981'}; font-weight:600;">${item.anomaly_score.toFixed(3)}</td>
                        <td><span class="sev-${sev}">${sev}</span></td>
                        <td>${causesHtml}</td>
                        <td>${actionsHtml}</td>
                    `;
                    tbody.appendChild(tr);
                });

            } catch (err) {
                console.error("Error fetching metrics:", err);
            }
        }

        async function triggerStressTest() {
            try {
                const res = await fetch('/api/v1/trigger-stress', { method: 'POST' });
                const data = await res.json();
                alert("🔥 Stress test workload initiated! Watch the CPU chart and anomaly score!");
            } catch (err) {
                alert("Failed to initiate stress test.");
            }
        }

        window.onload = () => {
            initCharts();
            fetchMetrics();
            setInterval(fetchMetrics, 1000);
        };
    </script>
</body>
</html>
    """
