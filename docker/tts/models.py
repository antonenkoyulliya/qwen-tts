from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TTSRequest(BaseModel):
    text: str
    language: str = "Auto"
    speaker: str = "Vivian"
    instruct: str = ""

class TTSResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float
    result: Optional[Dict] = None
    error: Optional[str] = None
    current_chunk: int = 0
    total_chunks: int = 0

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_jobs: Optional[int] = None