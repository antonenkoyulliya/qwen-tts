from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import uuid
import asyncio
import os
import soundfile as sf
from typing import List

from ..models import TTSRequest, TTSResponse, JobStatus

router = APIRouter(tags=["tts"])

# These will be set by tts/main.py
tts_engine = None
job_manager = None

async def process_tts_job(job_id: str, request: TTSRequest, tts_engine, job_manager):
    """Background task to process TTS generation"""
    try:
        # Update status to processing
        job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=10.0)

        # Generate audio
        audio, sample_rate = await asyncio.to_thread(
            tts_engine.generate_audio,
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruct if request.instruct else None
        )

        job_manager.update_job(job_id, progress=90.0)

        # Save to file
        filename = f"{job_id}.wav"
        filepath = f"/app/output/{filename}"
        await asyncio.to_thread(sf.write, filepath, audio, sample_rate)

        # Store result
        result = {
            "audio_url": f"/output/{filename}",
            "filepath": filepath,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "speaker": request.speaker
        }

        job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            result=result,
            progress=100.0
        )

    except Exception as e:
        print(f"❌ TTS error for job {job_id}: {e}")
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
            progress=0.0
        )

async def stream_tts_job(job_id: str, request: TTSRequest, tts_engine, job_manager, livekit_session=None):
    """Process TTS job while streaming to LiveKit and saving to file"""
    try:
        job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=10.0)

        # Collect chunks for file saving
        audio_chunks = []
        sample_rate = 24000  # Qwen-TTS default

        async for audio_chunk, sr in tts_engine.stream_generate_audio(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                instruct=request.instruct if request.instruct else None
        ):
            # Store chunk for file
            audio_chunks.append(audio_chunk)
            sample_rate = sr

            # Stream to LiveKit if session provided
            if livekit_session:
                await livekit_session.send_audio(audio_chunk)

            # Update progress
            job_manager.update_job(job_id, progress=min(50.0 + len(audio_chunks) * 2, 90.0))

        job_manager.update_job(job_id, progress=90.0)

        # Save combined audio to file
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks)
            filename = f"{job_id}.wav"
            filepath = f"/app/output/{filename}"
            await asyncio.to_thread(sf.write, filepath, combined_audio, sample_rate)

            result = {
                "audio_url": f"/output/{filename}",
                "filepath": filepath,
                "sample_rate": sample_rate,
                "duration": len(combined_audio) / sample_rate,
                "speaker": request.speaker
            }

            job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=result,
                progress=100.0
            )
        else:
            raise Exception("No audio chunks generated")

    except Exception as e:
        print(f"❌ TTS error for job {job_id}: {e}")
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
            progress=0.0
        )

@router.post("/async", response_model=TTSResponse)
async def generate_speech_async(
        request: TTSRequest,
        background_tasks: BackgroundTasks
):
    """Submit TTS job asynchronously"""
    if not tts_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    if request.speaker not in tts_engine.get_speakers():
        raise HTTPException(status_code=400, detail=f"Speaker '{request.speaker}' not supported")

    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")

    # Create job
    job_id = job_manager.create_job()

    # Process in background
    background_tasks.add_task(process_tts_job, job_id, request, tts_engine, job_manager)

    return TTSResponse(
        job_id=job_id,
        status="pending",
        message="Job submitted successfully. Poll /tts/status/{job_id} for results."
    )

@router.post("/sync")
async def generate_speech_sync(
        request: TTSRequest
):
    """Synchronous TTS generation (waits for result)"""
    if not tts_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        audio, sample_rate = await asyncio.to_thread(
            tts_engine.generate_audio,
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruct
        )

        filename = f"{request.speaker}_{hash(request.text)}.wav"
        filepath = f"/app/output/{filename}"
        await asyncio.to_thread(sf.write, filepath, audio, sample_rate)

        return {
            "audio_url": f"/output/{filename}",
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "speaker": request.speaker
        }

    except Exception as e:
        print(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def generate_speech_batch(
        requests: List[TTSRequest]
):
    """Batch process multiple texts"""
    if not tts_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for req in requests:
        try:
            audio, sample_rate = await asyncio.to_thread(
                tts_engine.generate_audio,
                text=req.text,
                language=req.language,
                speaker=req.speaker,
                instruct=req.instruct
            )

            filename = f"{req.speaker}_{hash(req.text)}.wav"
            filepath = f"/app/output/{filename}"
            await asyncio.to_thread(sf.write, filepath, audio, sample_rate)

            results.append({
                "audio_url": f"/output/{filename}",
                "duration": len(audio) / sample_rate,
                "speaker": req.speaker
            })
        except Exception as e:
            results.append({"error": str(e), "text": req.text[:50]})

    return {"results": results, "total": len(results)}

@router.get("/output/{filename}")
async def get_audio(job_id: str):
    """Download generated audio file"""
    filename = f"{job_id}.wav"
    filepath = f"/app/output/{filename}"
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="File not found")