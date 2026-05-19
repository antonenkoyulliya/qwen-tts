from fastapi import APIRouter, Depends
from ..models import HealthResponse

router = APIRouter(tags=["health"])

# These will be set by main.py
tts_engine = None
job_manager = None

@router.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=tts_engine.is_loaded if tts_engine else False,
        active_jobs=job_manager.get_stats()["processing"] if job_manager else 0
    )

@router.get("/speakers")
async def get_speakers():
    if tts_engine:
        return {"speakers": tts_engine.get_speakers()}
    return {"speakers": []}

@router.get("/languages")
async def get_languages():
    if tts_engine:
        return {"languages": tts_engine.get_languages()}
    return {"languages": []}