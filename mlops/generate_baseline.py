import pandas as pd
import numpy as np
import os
import psutil
import time

# Define output path
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUTPUT_DIR, "baseline_telemetry.csv")

def get_current_system_snapshot():
    # Warm up CPU percent
    psutil.cpu_percent(interval=None)
    time.sleep(0.2)
    
    cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    mem_percent = vm.percent
    
    disk_io = psutil.disk_io_counters()
    disk_read = disk_io.read_bytes if disk_io else 0
    disk_write = disk_io.write_bytes if disk_io else 0
    
    net_io = psutil.net_io_counters()
    net_sent = net_io.bytes_sent if net_io else 0
    net_recv = net_io.bytes_recv if net_io else 0
    
    cpu_stats = psutil.cpu_stats()
    ctx_switches = cpu_stats.ctx_switches
    interrupts = cpu_stats.interrupts
    
    return {
        "cpu_percent": cpu_percent,
        "mem_percent": mem_percent,
        "disk_read_bytes": disk_read,
        "disk_write_bytes": disk_write,
        "net_bytes_sent": net_sent,
        "net_bytes_recv": net_recv,
        "ctx_switches": ctx_switches,
        "interrupts": interrupts
    }

def generate_data(num_samples=2000):
    np.random.seed(42)
    base_snapshot = get_current_system_snapshot()
    print(f"Base system snapshot fetched: CPU={base_snapshot['cpu_percent']}%, RAM={base_snapshot['mem_percent']}%, CtxSwitches={base_snapshot['ctx_switches']}")

    # Generate realistic baseline around system's current live metrics
    cpu_percent = np.random.normal(loc=max(15, base_snapshot['cpu_percent']), scale=5, size=num_samples)
    cpu_percent = np.clip(cpu_percent, 2, 100)
    
    mem_percent = np.random.normal(loc=base_snapshot['mem_percent'], scale=3, size=num_samples)
    mem_percent = np.clip(mem_percent, 5, 100)
    
    # Cumulative metrics gradually increasing over time ticks
    time_steps = np.arange(num_samples)
    
    disk_read = base_snapshot['disk_read_bytes'] + time_steps * np.random.normal(loc=50000, scale=10000, size=num_samples)
    disk_write = base_snapshot['disk_write_bytes'] + time_steps * np.random.normal(loc=20000, scale=5000, size=num_samples)
    
    net_sent = base_snapshot['net_bytes_sent'] + time_steps * np.random.normal(loc=10000, scale=2000, size=num_samples)
    net_recv = base_snapshot['net_bytes_recv'] + time_steps * np.random.normal(loc=50000, scale=10000, size=num_samples)
    
    ctx_switches = base_snapshot['ctx_switches'] + time_steps * np.random.normal(loc=3000, scale=500, size=num_samples)
    interrupts = base_snapshot['interrupts'] + time_steps * np.random.normal(loc=1500, scale=300, size=num_samples)
    
    df = pd.DataFrame({
        "cpu_percent": cpu_percent,
        "mem_percent": mem_percent,
        "disk_read_bytes": disk_read,
        "disk_write_bytes": disk_write,
        "net_bytes_sent": net_sent,
        "net_bytes_recv": net_recv,
        "ctx_switches": ctx_switches,
        "interrupts": interrupts
    })
    
    # Add a few anomalies (about 5%)
    num_anomalies = int(num_samples * 0.05)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    # Anomaly conditions: High CPU (90%+), High Ctx Switches
    df.loc[anomaly_indices, 'cpu_percent'] = np.random.normal(loc=95, scale=3, size=num_anomalies)
    df.loc[anomaly_indices, 'ctx_switches'] += np.random.normal(loc=500000, scale=50000, size=num_anomalies)
    
    df['cpu_percent'] = df['cpu_percent'].clip(0, 100)
    df['mem_percent'] = df['mem_percent'].clip(0, 100)
    
    return df

if __name__ == "__main__":
    print("Generating system-calibrated baseline telemetry data...")
    df = generate_data(2000)
    df.to_csv(DATA_FILE, index=False)
    print(f"Generated {len(df)} calibrated samples and saved to {DATA_FILE}")
