import uuid
from datetime import datetime
from typing import Dict, Optional, List
from collections import deque
import os

from .models import JobStatus
from .config import Config

class TTSJob:
    """Represents a single TTS generation job"""

    def __init__(self, job_id: str):
        self.id = job_id
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.result = None
        self.error = None
        self.progress = 0.0
        self.current_chunk = 0
        self.total_chunks = 0

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks
        }

class JobManager:
    """Manages all TTS jobs: creation, updates, retrieval, and cleanup"""

    def __init__(self):
        self._jobs: Dict[str, TTSJob] = {}
        self._history = deque(maxlen=Config.MAX_JOB_HISTORY)

    def create_job(self) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = TTSJob(job_id)
        return job_id

    def update_job(self, job_id: str, **kwargs):
        """Update job attributes"""
        job = self._jobs.get(job_id)
        if job:
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = datetime.now()

    def get_job(self, job_id: str) -> Optional[TTSJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50, status: Optional[JobStatus] = None) -> List[TTSJob]:
        """List recent jobs, optionally filtered by status"""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        if status:
            jobs = [j for j in jobs if j.status == status]

        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete job and its associated audio file"""
        job = self._jobs.get(job_id)
        if not job:
            return False

        # Only delete completed or failed jobs
        if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return False

        # Delete audio file if exists
        if job.result and 'filepath' in job.result:
            try:
                os.unlink(job.result['filepath'])
            except Exception as e:
                print(f"Error deleting file for job {job_id}: {e}")

        del self._jobs[job_id]
        return True

    def cleanup_old_jobs(self):
        """Remove jobs older than configured hours"""
        now = datetime.now()
        to_remove = []

        for job_id, job in self._jobs.items():
            age_hours = (now - job.created_at).total_seconds() / 3600

            if age_hours > Config.JOB_CLEANUP_HOURS:
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    to_remove.append(job_id)

        for job_id in to_remove:
            self.delete_job(job_id)

    def get_stats(self) -> dict:
        """Get job statistics"""
        jobs = self._jobs.values()
        return {
            "total": len(jobs),
            "pending": sum(1 for j in jobs if j.status == JobStatus.PENDING),
            "processing": sum(1 for j in jobs if j.status == JobStatus.PROCESSING),
            "completed": sum(1 for j in jobs if j.status == JobStatus.COMPLETED),
            "failed": sum(1 for j in jobs if j.status == JobStatus.FAILED)
        }