from pydantic import BaseModel
from typing import List, Optional

class TelemetryPayload(BaseModel):
    cpu_percent: float
    mem_percent: float
    disk_read_bytes: int
    disk_write_bytes: int
    net_bytes_sent: int
    net_bytes_recv: int
    ctx_switches: int
    interrupts: int

class PredictionResponse(BaseModel):
    status: str
    anomaly_flag: int
    anomaly_score: float
    severity: str = "NOMINAL"
    suspected_causes: List[str] = []
    recommended_actions: List[str] = []
    message: str

