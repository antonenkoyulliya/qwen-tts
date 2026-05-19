from fastapi import APIRouter, HTTPException
from typing import Optional
from ..models import JobStatusResponse

router = APIRouter(tags=["jobs"])

# This will be set by tts/main.py
job_manager = None

@router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job.status}

@router.get("/list/{status}")
async def list_jobs(status: str):
    from ..models import JobStatus

    # Validate status if provided
    valid_statuses = [s.value for s in JobStatus]

    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Use: {valid_statuses}")

    status_enum = JobStatus(status)
    jobs = job_manager.list_jobs(limit=10, status=status_enum)

    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job.id,
                "status": job.status,
                "created_at": job.created_at,
                "progress": job.progress,
                "duration": job.result.get("duration") if job.result else None
            }
            for job in jobs
        ]
    }

@router.get("/stats")
async def get_job_stats():
    return job_manager.get_stats()

@router.delete("/job/{job_id}")
async def delete_job(job_id: str):
    success = job_manager.delete_job(job_id)

    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be deleted")

    return {"message": f"Job {job_id} deleted successfully"}