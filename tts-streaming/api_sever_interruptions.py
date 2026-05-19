import asyncio
from typing import Dict, Set
from contextlib import asynccontextmanager

# Store active generation tasks
active_generations: Dict[str, asyncio.Task] = {}
cancellation_events: Dict[str, asyncio.Event] = {}

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field("Vivian", description="Voice name")
    language: str = Field("en", description="Language code")
    stream: bool = Field(True, description="Enable streaming response")
    response_format: str = Field("pcm", description="pcm or wav")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed")
    emit_every_frames: int = Field(8, description="Stream chunk size")
    decode_window_frames: int = Field(80, description="Decoder context window")
    request_id: Optional[str] = Field(None, description="Unique request ID for cancellation")

class StopRequest(BaseModel):
    request_id: str = Field(..., description="ID of generation to stop")

async def stream_audio_generator(
        text: str,
        voice: str,
        language: str,
        emit_every_frames: int,
        decode_window_frames: int,
        response_format: str,
        request_id: str,
        cancel_event: asyncio.Event,
):
    """Generate streaming audio chunks with cancellation support"""

    try:
        if voice not in PREDEFINED_VOICES:
            voice = "Vivian"

        print(f"Generating streaming audio for voice: {voice} (id: {request_id})")

        generator = model.stream_generate_pcm(
            text=text,
            speaker=voice,
            language=language,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        )

        chunk_count = 0
        sample_rate = 24000

        for audio_chunk, sample_rate in generator:
            # Check for cancellation
            if cancel_event.is_set():
                print(f"❌ Generation {request_id} cancelled by user")
                break

            chunk_count += 1
            print(f"📦 Processing chunk {chunk_count}")

            if chunk_count == 1:
                print(f"⏱️ First chunk received!")

            # Convert to bytes
            if response_format == "pcm":
                audio_bytes = audio_to_pcm_bytes(audio_chunk, sample_rate)
            else:
                audio_bytes = audio_to_wav_bytes(audio_chunk, sample_rate)

            yield audio_bytes

            # Small yield to allow cancellation check
            await asyncio.sleep(0)

        print(f"✅ Generation {request_id} complete: {chunk_count} chunks")

    except asyncio.CancelledError:
        print(f"⚠️ Generation {request_id} was cancelled")
        raise
    except Exception as e:
        print(f"Streaming error for {request_id}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up
        active_generations.pop(request_id, None)
        cancellation_events.pop(request_id, None)

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    """Generate speech from text with streaming support"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.voice not in PREDEFINED_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.voice}' not found. Available: {list(PREDEFINED_VOICES)}"
        )

    # Generate unique ID if not provided
    request_id = request.request_id or f"gen_{int(time.time()*1000)}_{os.urandom(4).hex()}"
    cancel_event = asyncio.Event()
    cancellation_events[request_id] = cancel_event

    return StreamingResponse(
        stream_audio_generator(
            text=request.text,
            voice=request.voice,
            language=request.language,
            emit_every_frames=request.emit_every_frames,
            decode_window_frames=request.decode_window_frames,
            response_format=request.response_format,
            request_id=request_id,
            cancel_event=cancel_event,
        ),
        media_type="audio/L16" if request.response_format == "pcm" else "audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
            "X-Request-ID": request_id,  # Return ID for cancellation
        }
    )

@app.post("/v1/audio/stop")
async def stop_generation(request: StopRequest):
    """Stop an ongoing generation"""
    if request.request_id not in cancellation_events:
        raise HTTPException(
            status_code=404,
            detail=f"No active generation found with ID: {request.request_id}"
        )

    # Set cancellation event
    cancellation_events[request.request_id].set()

    return {
        "status": "stopped",
        "request_id": request.request_id,
        "message": "Generation stop requested"
    }

@app.get("/v1/audio/active")
async def list_active_generations():
    """List all active generations"""
    return {
        "active_generations": list(active_generations.keys()),
        "count": len(active_generations)
    }