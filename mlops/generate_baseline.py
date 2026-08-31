import pandas as pd
import numpy as np
import os

# Define output path
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUTPUT_DIR, "baseline_telemetry.csv")

def generate_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate baseline (normal) telemetry data
    # cpu_percent: mostly between 5-30%
    cpu_percent = np.random.normal(loc=15, scale=5, size=num_samples)
    cpu_percent = np.clip(cpu_percent, 0, 100)
    
    # mem_percent: mostly around 40-60%
    mem_percent = np.random.normal(loc=50, scale=10, size=num_samples)
    mem_percent = np.clip(mem_percent, 0, 100)
    
    # disk_read_bytes / disk_write_bytes: normal background activity
    disk_read = np.random.exponential(scale=500000, size=num_samples)
    disk_write = np.random.exponential(scale=200000, size=num_samples)
    
    # net_bytes_sent / net_bytes_recv: normal background network activity
    net_sent = np.random.exponential(scale=100000, size=num_samples)
    net_recv = np.random.exponential(scale=500000, size=num_samples)
    
    # ctx_switches / interrupts
    ctx_switches = np.random.normal(loc=5000, scale=1000, size=num_samples)
    interrupts = np.random.normal(loc=3000, scale=500, size=num_samples)
    
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
    
    # High CPU and memory for anomalies
    df.loc[anomaly_indices, 'cpu_percent'] = np.random.normal(loc=90, scale=5, size=num_anomalies)
    df.loc[anomaly_indices, 'mem_percent'] = np.random.normal(loc=95, scale=3, size=num_anomalies)
    df.loc[anomaly_indices, 'disk_read_bytes'] = np.random.exponential(scale=50000000, size=num_anomalies)
    
    # Ensure clipping
    df['cpu_percent'] = df['cpu_percent'].clip(0, 100)
    df['mem_percent'] = df['mem_percent'].clip(0, 100)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic baseline telemetry data...")
    df = generate_data(2000)
    df.to_csv(DATA_FILE, index=False)
    print(f"Generated {len(df)} samples and saved to {DATA_FILE}")
