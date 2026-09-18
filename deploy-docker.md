# Docker Deployment Guide

This guide details how to deploy the **Linux Telemetry Anomaly Detection System** using Docker and Docker Compose.

---

## Prerequisites

- **Docker Engine**: v20.10+
- **Docker Compose**: v2.0+
- Linux environment with appropriate kernel metrics support

---

## Architecture Overview

The Docker setup orchestrates two primary microservices using standard bridge networking:

1. **FastAPI Inference Microservice** (`anomaly_api`):
   - Exposes REST API endpoints on port `8000`.
   - Runs Isolation Forest model inference on incoming Linux OS telemetry data.
   - Provides an interactive web dashboard at `http://localhost:8000`.

2. **MLflow Tracking Server** (`mlflow_server`):
   - Exposes experiment tracking dashboard on port `5000`.
   - Logs hyperparameter runs, metric thresholds, and model artifacts to SQLite and `./mlruns`.

---

## Quick Start Deployment

### 1. Build and Launch Containers

Run the following command from the root of the repository to build images and launch services in detached mode:

```bash
docker-compose up -d --build
```

### 2. Verify Container Status

Check that both containers are running and healthy:

```bash
docker-compose ps
```

Expected output:
```text
NAME             COMMAND                  SERVICE   STATUS              PORTS
anomaly_api      "uvicorn api.main:app…"   api       running (healthy)   0.0.0.0:8000->8000/tcp
mlflow_server    "bash -c 'pip install…"   mlflow    running (healthy)   0.0.0.0:5000->5000/tcp
```

### 3. View Service Logs

To monitor container logs in real time:

```bash
# View logs from all services
docker-compose logs -f

# View logs for API microservice only
docker-compose logs -f api
```

---

## Environment & Port Mapping

| Service | Container Name | Host Port | Container Port | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **API & Dashboard** | `anomaly_api` | `8000` | `8000` | Telemetry inference & dashboard UI |
| **MLflow Server** | `mlflow_server` | `5000` | `5000` | Experiment & metric tracking UI |

---

## Running Telemetry Collector Daemon with Docker

To collect host metrics and send them to the containerized API:

### Native Host Daemon (Recommended)
Run the telemetry collector directly on the Linux host so it captures true host hardware telemetry:

```bash
source proenv/bin/activate
python daemon/telemetry_collector.py
```

---

## Stopping and Cleaning Up Services

To stop running containers without deleting data:

```bash
docker-compose down
```

To stop containers and remove named volumes/networks:

```bash
docker-compose down --volumes
```

---

## Container Security & Hardening Best Practices

1. **Non-Root User Execution**:
   - Production container images should run process workloads under non-privileged service accounts (e.g. `USER 10001:10001`).

2. **Docker Socket Management**:
   - Avoid mounting `/var/run/docker.sock` inside application containers unless container management capabilities are strictly required.
   - Restrict membership of the host `docker` Linux group to authorized administrative users only.

3. **Resource Boundaries**:
   - Set CPU and Memory limits in `docker-compose.yml` to prevent single-container resource starvation:
     ```yaml
     deploy:
       resources:
         limits:
           cpus: '1.50'
           memory: 1024M
     ```
