#  Automated MLOps Pipeline for Real-Time Anomaly Detection in Linux Daemons

An end-to-end MLOps system designed to monitor Linux system telemetry in real time, detect anomalies using machine learning (Isolation Forest), manage model tracking with MLflow, and trigger automated retraining upon statistical data drift (Kolmogorov-Smirnov test).

---

##  Architecture Overview

```
 ┌─────────────────────────┐      HTTP POST      ┌───────────────────────────────────┐
 │ Systemd Telemetry Daemon│ ──────────────────> │  FastAPI Microservice (Port 8000) │
 │ (psutil OS metrics)     │                     │  - Isolation Forest Inference     │
 └─────────────────────────┘                     └─────────────────┬─────────────────┘
                                                                   │
                                                                   ▼
 ┌─────────────────────────┐      Metrics        ┌───────────────────────────────────┐
 │   Drift Monitor (KS)    │ ──────────────────> │   MLflow Tracking Server          │
 │ (Auto Retraining Loop)  │                     │   (SQLite / Local Storage :5000)  │
 └─────────────────────────┘                     └───────────────────────────────────┘
```

---

##  Features

- ** Real-Time OS Telemetry Harvesting**: Lightweight daemon collecting CPU, Memory, Disk I/O, and Network traffic metrics using `psutil`.
- ** Machine Learning Anomaly Detection**: Unsupervised Isolation Forest model predicting system anomalies in `<50ms`.
- ** MLflow Experiment Tracking**: Automated logging of parameters, metrics, contaminated thresholds, and model artifacts.
- ** Continuous Data Drift Monitoring**: Kolmogorov-Smirnov (KS 2-sample) statistical testing to detect metric distribution drift and trigger automatic model retraining.
- ** Systemd Service Daemon**: Native Linux service setup for background persistence across reboots.
- **🐳 Docker & Local Deployment Options**: Support for Docker Compose orchestration as well as native virtualenv runtimes.

---

## Repository Structure

```text
.
├── api/                        # FastAPI microservice & Dockerfile
│   ├── main.py                 # REST API endpoints for inference & health checks
│   ├── models/                 # Model artifacts storage
│   └── Dockerfile              # Docker container specification
├── daemon/                     # System telemetry collection scripts & unit files
│   ├── telemetry_collector.py  # Continuous system metrics collector
│   └── systemmonitor.service   # Systemd service configuration
├── mlops/                      # MLOps training, baseline generation & drift monitoring
│   ├── generate_baseline.py    # Synthetic baseline telemetry generator
│   ├── train.py                # Model training & MLflow logging script
│   └── drift_monitor.py        # KS-test data drift detector
├── docker-compose.yml          # Docker orchestration for API & MLflow UI
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Prerequisites & Installation

### Requirements
- **OS**: Linux (Fedora, Ubuntu, Debian, etc.)
- **Python**: 3.10+
- **Tools**: Docker & Docker Compose (optional for containerized setup), `stress-ng` (for anomaly simulation)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

# Create and activate virtual environment
python3 -m venv proenv
source proenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Quick Start & Live Demonstration Guide

### Automated Startup (Recommended)

For a quick and automated start of all services, use the provided shell scripts:

#### Start All Services
```bash
chmod +x start_all.sh
./start_all.sh
```

This script automatically:
- Activates the virtual environment
- Generates baseline telemetry data
- Trains the Isolation Forest model
- Starts MLflow tracking server (Port 5000)
- Launches FastAPI microservice (Port 8000)
- Starts the telemetry collector daemon
- Initiates the continuous drift monitor

#### Stop All Services
```bash
chmod +x stop_all.sh
./stop_all.sh
```

This script gracefully shuts down all running processes.

---

### Option 1: Direct Python Local Execution

#### Step 1: Initialize Baseline Data & Train Model
```bash
source proenv/bin/activate
python mlops/generate_baseline.py
python mlops/train.py
```

#### Step 2: Start MLflow UI Dashboard (Terminal 1)
```bash
source proenv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```
> Access dashboard: [http://localhost:5000](http://localhost:5000)

#### Step 3: Launch FastAPI Microservice (Terminal 2)
```bash
source proenv/bin/activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
>  Access Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Step 4: Run Telemetry Collector Daemon (Terminal 3)
```bash
source proenv/bin/activate
python daemon/telemetry_collector.py
```

#### Step 5: Start Continuous Drift Monitor (Terminal 4)
```bash
source proenv/bin/activate
python mlops/drift_monitor.py
```

---

### Option 2: Docker Compose Orchestration

```bash
# Build and start services in detached mode
docker-compose up -d --build

# View container logs
docker-compose logs -f
```

---

##  Register Daemon as a Systemd Service

To register and run the collector as a system background service:

```bash
sudo cp daemon/systemmonitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now systemmonitor.service
sudo systemctl status systemmonitor.service
```

---

##  Stress Testing & Anomaly Verification

Simulate high system load to verify live anomaly detection:

```bash
# Install stress-ng (Fedora: sudo dnf install stress-ng / Ubuntu: sudo apt install stress-ng)
stress-ng --cpu 4 --timeout 15s
```

**Observed behavior:**
1. **Daemon Log**: Output transitions from `Normal (Score: 0.12)` to `Anomaly (Score: -0.28)`.
2. **FastAPI**: Logs payload anomaly with response time `<50ms`.
3. **MLflow UI**: Track metrics and contamination scores under active experiments.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
