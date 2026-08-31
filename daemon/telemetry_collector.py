import time
import json
import logging
import psutil
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TelemetryDaemon")

API_URL = "http://localhost:8000/api/v1/predict"
POLL_INTERVAL = 1.0 # Poll every second

def collect_metrics():
    # CPU
    cpu_percent = psutil.cpu_percent(interval=None)
    
    # Memory
    vm = psutil.virtual_memory()
    mem_percent = vm.percent
    
    # Disk I/O
    disk_io = psutil.disk_io_counters()
    disk_read_bytes = disk_io.read_bytes if disk_io else 0
    disk_write_bytes = disk_io.write_bytes if disk_io else 0
    
    # Network
    net_io = psutil.net_io_counters()
    net_bytes_sent = net_io.bytes_sent if net_io else 0
    net_bytes_recv = net_io.bytes_recv if net_io else 0
    
    # Context Switches and Interrupts
    cpu_stats = psutil.cpu_stats()
    ctx_switches = cpu_stats.ctx_switches
    interrupts = cpu_stats.interrupts
    
    payload = {
        "cpu_percent": cpu_percent,
        "mem_percent": mem_percent,
        "disk_read_bytes": disk_read_bytes,
        "disk_write_bytes": disk_write_bytes,
        "net_bytes_sent": net_bytes_sent,
        "net_bytes_recv": net_bytes_recv,
        "ctx_switches": ctx_switches,
        "interrupts": interrupts
    }
    
    return payload

def main():
    logger.info(f"Starting Telemetry Daemon. Polling every {POLL_INTERVAL} seconds.")
    # Initialize CPU percent (first call gives 0.0 usually)
    psutil.cpu_percent(interval=None)
    time.sleep(0.5)
    
    while True:
        try:
            metrics = collect_metrics()
            logger.debug(f"Collected metrics: {metrics}")
            
            # Send to API
            try:
                response = requests.post(API_URL, json=metrics, timeout=2)
                if response.status_code == 200:
                    result = response.json()
                    status = "Anomaly" if result.get("anomaly_flag") == -1 else "Normal"
                    logger.info(f"Sent successfully. Prediction: {status} (Score: {result.get('anomaly_score', 0):.2f})")
                else:
                    logger.warning(f"API returned status {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to API: {e}")
                
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
