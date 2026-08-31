from pydantic import BaseModel

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
    message: str
