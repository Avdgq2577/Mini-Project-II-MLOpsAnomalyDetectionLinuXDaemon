from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os
import logging
import time
import subprocess
from collections import deque
from api.schemas import TelemetryPayload, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI")

app = FastAPI(title="Linux Telemetry Anomaly API", version="1.0.0")

# Global model variable & history buffer
model_pipeline = None
recent_history = deque(maxlen=60) # last 60 data ticks
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "isolation_forest.joblib")
RECENT_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recent_telemetry.csv")

@app.on_event("startup")
async def load_model():
    global model_pipeline
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
        prediction = model_pipeline.predict(df)[0]  # returns 1 (normal) or -1 (anomaly)
        score = model_pipeline.decision_function(df)[0]
        
        # Log to history buffer for real-time dashboard
        history_item = data_dict.copy()
        history_item['timestamp'] = time.strftime("%H:%M:%S")
        history_item['anomaly_flag'] = int(prediction)
        history_item['anomaly_score'] = float(score)
        recent_history.append(history_item)
        
        # Log to recent telemetry CSV for drift detection
        log_df = df.copy()
        log_df['anomaly_flag'] = prediction
        log_df['anomaly_score'] = score
        log_df.to_csv(RECENT_DATA_FILE, mode='a', header=not os.path.exists(RECENT_DATA_FILE), index=False)
        
        return PredictionResponse(
            status="success",
            anomaly_flag=int(prediction),
            anomaly_score=float(score),
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
        "latest": latest,
        "history": history_list
    }

def run_stress_task():
    try:
        # Try running stress-ng if available
        subprocess.run(["stress-ng", "--cpu", "4", "--timeout", "10s"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # Fallback python intensive CPU load for 10s
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
        table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.9rem; }
        th { text-align: left; padding: 12px; color: var(--text-muted); border-bottom: 1px solid var(--border-color); }
        td { padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .tag-anomaly { color: var(--accent-red); font-weight: 700; background: rgba(239, 68, 68, 0.15); padding: 4px 10px; border-radius: 6px; }
        .tag-normal { color: var(--accent-green); font-weight: 600; background: rgba(16, 185, 129, 0.15); padding: 4px 10px; border-radius: 6px; }
    </style>
</head>
<body>

    <div class="header">
        <div class="title-area">
            <h1>Linux Daemon MLOps Anomaly Detection</h1>
            <p>Real-Time System Telemetry & Isolation Forest Inference Service</p>
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
            <div class="metric-sub">Detected in Current Session</div>
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
        <div class="chart-title">Live Telemetry Prediction Feed</div>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>CPU %</th>
                    <th>RAM %</th>
                    <th>Context Switches</th>
                    <th>Decision Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="logsTable">
                <tr><td colspan="6" style="text-align:center; color:var(--text-muted);">Waiting for telemetry stream from daemon...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        let telemetryChart, scoreChart;
        let anomalyCount = 0;

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
                document.getElementById('valAnomalies').innerText = data.recent_anomalies_count;

                // Update Status Badge
                const badge = document.getElementById('statusBadge');
                if (latest.anomaly_flag === -1) {
                    badge.className = 'status-badge badge-anomaly';
                    document.getElementById('statusDot').innerText = '🚨';
                    document.getElementById('statusText').innerText = 'ANOMALY DETECTED';
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
                    tr.innerHTML = `
                        <td>${item.timestamp}</td>
                        <td>${item.cpu_percent.toFixed(1)}%</td>
                        <td>${item.mem_percent.toFixed(1)}%</td>
                        <td>${item.ctx_switches.toLocaleString()}</td>
                        <td style="color:${isAnomaly ? '#ef4444' : '#10b981'}; font-weight:600;">${item.anomaly_score.toFixed(4)}</td>
                        <td><span class="${isAnomaly ? 'tag-anomaly' : 'tag-normal'}">${isAnomaly ? 'ANOMALY' : 'NORMAL'}</span></td>
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

