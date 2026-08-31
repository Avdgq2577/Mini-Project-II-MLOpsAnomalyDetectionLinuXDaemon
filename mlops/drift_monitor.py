import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import logging
import time
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DriftMonitor")

# Configuration
MLOPS_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(MLOPS_DIR, "baseline_telemetry.csv")
# In a real scenario, this would be a database or a file where the API logs recent telemetry.
# For simplicity in this demo, we'll look for a 'recent_telemetry.csv' that the API will periodically flush.
RECENT_DATA_FILE = os.path.join(os.path.dirname(MLOPS_DIR), "api", "recent_telemetry.csv")
P_VALUE_THRESHOLD = 0.05
POLL_INTERVAL = 60 # Check every minute

def check_drift():
    if not os.path.exists(BASELINE_FILE):
        logger.warning(f"Baseline file {BASELINE_FILE} missing. Skipping drift check.")
        return False
        
    if not os.path.exists(RECENT_DATA_FILE):
        logger.debug(f"Recent data file {RECENT_DATA_FILE} not found yet. Skipping drift check.")
        return False

    baseline_df = pd.read_csv(BASELINE_FILE)
    try:
        recent_df = pd.read_csv(RECENT_DATA_FILE)
    except Exception as e:
        logger.error(f"Error reading recent data: {e}")
        return False
        
    if len(recent_df) < 50:
        logger.debug("Not enough recent data to perform a reliable KS test.")
        return False
        
    drift_detected = False
    drifted_features = []
    
    # Perform KS test for each feature
    features = [c for c in baseline_df.columns if c in recent_df.columns and c not in ['timestamp', 'anomaly_score', 'anomaly_flag']]
    
    for feature in features:
        stat, p_value = ks_2samp(baseline_df[feature], recent_df[feature])
        if p_value < P_VALUE_THRESHOLD:
            drift_detected = True
            drifted_features.append(feature)
            logger.warning(f"Drift detected in feature '{feature}' (p-value: {p_value:.4f})")
            
    if drift_detected:
        logger.info(f"Significant data drift detected in features: {drifted_features}. Triggering retraining...")
        return True
        
    logger.info("No significant data drift detected.")
    return False

def trigger_retraining():
    logger.info("Executing train.py for retraining model on new baseline...")
    # Optional: in a real scenario, you'd append recent data to baseline or replace baseline
    # before retraining. For demo, we just run the script.
    try:
        # We append recent data to baseline to simulate learning new patterns
        baseline_df = pd.read_csv(BASELINE_FILE)
        recent_df = pd.read_csv(RECENT_DATA_FILE)
        # Drop columns not in baseline
        cols = [c for c in recent_df.columns if c in baseline_df.columns]
        combined = pd.concat([baseline_df, recent_df[cols]], ignore_index=True)
        # Keep only the last 5000 rows to avoid infinite growth
        if len(combined) > 5000:
            combined = combined.tail(5000)
        combined.to_csv(BASELINE_FILE, index=False)
        logger.info("Updated baseline dataset with recent telemetry.")
        
        # Clear recent data file
        os.remove(RECENT_DATA_FILE)
        
        # Run training
        import sys
        train_script = os.path.join(MLOPS_DIR, "train.py")
        subprocess.run([sys.executable, train_script], check=True)
        logger.info("Retraining completed successfully.")
    except Exception as e:
        logger.error(f"Failed during retraining pipeline: {e}")

def main():
    logger.info(f"Starting Drift Monitor. Checking every {POLL_INTERVAL} seconds.")
    while True:
        if check_drift():
            trigger_retraining()
            # Wait longer after retraining before checking again
            time.sleep(POLL_INTERVAL * 5)
        else:
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
