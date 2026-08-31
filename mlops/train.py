import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainPipeline")

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "Linux_Telemetry_Anomaly_Detection"
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_telemetry.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "models")

def train_model():
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file {DATA_FILE} not found. Please run generate_baseline.py first.")
        return

    logger.info("Loading telemetry data...")
    df = pd.read_csv(DATA_FILE)
    
    # We use all features for unsupervised learning
    features = df.columns.tolist()
    X = df[features]
    
    logger.info(f"Checking MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    if MLFLOW_TRACKING_URI.startswith("http"):
        try:
            import urllib.request
            urllib.request.urlopen(MLFLOW_TRACKING_URI, timeout=2)
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        except Exception:
            logger.warning(f"MLflow server at {MLFLOW_TRACKING_URI} is offline. Using local SQLite store 'sqlite:///mlflow.db'.")
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment(EXPERIMENT_NAME)

    
    contamination = 0.05
    n_estimators = 100
    
    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("features", features)
        
        # Create pipeline: RobustScaler -> IsolationForest
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('iforest', IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42))
        ])
        
        logger.info("Training Isolation Forest Pipeline...")
        pipeline.fit(X)
        
        # Predict on training data to compute metrics
        preds = pipeline.predict(X)
        
        # Compute a pseudo-metric (Isolation forest doesn't have a standard accuracy without labels)
        # We can just log the number of anomalies found
        num_anomalies = (preds == -1).sum()
        mlflow.log_metric("train_anomalies_detected", num_anomalies)
        logger.info(f"Anomalies detected in training set: {num_anomalies}/{len(X)}")
        
        # Save locally for the API to load on startup
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_model_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
        joblib.dump(pipeline, local_model_path)
        logger.info(f"Model saved locally to {local_model_path}")

        # Log model to MLflow
        try:
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            logger.info("MLflow model logged successfully.")
        except Exception as e:
            logger.warning(f"MLflow model log skipped/warning: {e}")
        
if __name__ == "__main__":
    train_model()
