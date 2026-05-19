#!/usr/bin/env python3
"""
Production API Server for Qwen3-TTS with Streaming Support
Adaptive GPU support - works on V100, A100, H100, RTX, etc.
"""

import os
import sys
import io
import wave
import numpy as np
import time
import warnings
from pathlib import Path
from typing import Optional, AsyncGenerator
import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add the qwen-tts package to path
sys.path.insert(0, '/app/Qwen3-TTS-streaming')

# Import the streaming-enabled Qwen3TTS
from qwen_tts import Qwen3TTSModel

# ============ Adaptive GPU Configuration ============
def get_optimal_dtype():
    """Automatically select the best dtype for the available GPU"""
    if not torch.cuda.is_available():
        return torch.float32

    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)

    print(f"Detected GPU: {gpu_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

    if compute_capability[0] >= 8:
        if torch.cuda.is_bf16_supported():
            print("✓ bfloat16 supported - using for best performance")
            return torch.bfloat16
        else:
            print("⚠️ bfloat16 not supported, falling back to float16")
            return torch.float16
    elif compute_capability[0] >= 7:
        print("Using float16 (optimal for V100/T4/RTX 20xx)")
        return torch.float16
    else:
        print("Using float32 (most compatible)")
        return torch.float32

def get_optimal_compile_mode():
    """Select the best torch.compile mode based on GPU"""
    if not torch.cuda.is_available():
        return None

    compute_capability = torch.cuda.get_device_capability(0)

    if compute_capability[0] >= 8:
        return "reduce-overhead"
    else:
        return "default"

def should_use_cuda_graphs():
    """Determine if CUDA graphs should be used"""
    if not torch.cuda.is_available():
        return False

    compute_capability = torch.cuda.get_device_capability(0)
    return compute_capability[0] >= 8

# ============ Configuration ============
MODEL_PATH = os.getenv("MODEL_PATH", "/app/qwen-tts-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = get_optimal_dtype()
USE_COMPILE = torch.cuda.is_available()
COMPILE_MODE = get_optimal_compile_mode()
USE_CUDA_GRAPHS = should_use_cuda_graphs()

# Suppress bfloat16 warnings on V100
if torch.cuda.is_available() and "V100" in torch.cuda.get_device_name(0):
    warnings.filterwarnings("ignore", message=".*bfloat16.*not support.*")

# ============ Global Model ============
model = None

# Predefined voices
PREDEFINED_VOICES = {
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee"
}

# ============ Request/Response Models ============
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field("Vivian", description="Voice name")
    language: str = Field("en", description="Language code")
    stream: bool = Field(True, description="Enable streaming response")
    response_format: str = Field("pcm", description="pcm or wav")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed")
    emit_every_frames: int = Field(8, description="Stream chunk size")
    decode_window_frames: int = Field(80, description="Decoder context window")

class VoiceCloneRequest(BaseModel):
    name: str = Field(..., description="Voice profile name")
    ref_audio: str = Field(..., description="Path to reference audio file")
    ref_text: str = Field(..., description="Transcript of reference audio")

# ============ Lifespan Management ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("=" * 60)
    print("Qwen3-TTS Streaming API Starting...")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Torch.compile: {USE_COMPILE} (mode: {COMPILE_MODE})")
    print(f"CUDA Graphs: {USE_CUDA_GRAPHS}")
    print("-" * 60)

    start_time = time.time()

    try:
        print("Loading model... (this may take 30-60 seconds on first run)")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE,
            device_map="cuda" if DEVICE == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")

        # Enable streaming optimizations
        if hasattr(model, 'enable_streaming_optimizations'):
            print("Configuring streaming optimizations...")
            model.enable_streaming_optimizations(
                decode_window_frames=80,
                use_compile=USE_COMPILE,
                use_cuda_graphs=USE_CUDA_GRAPHS,
                compile_mode=COMPILE_MODE if COMPILE_MODE else "reduce-overhead",
                compile_codebook_predictor=USE_COMPILE,
                compile_talker=USE_COMPILE,
            )
            print("✓ Streaming optimizations enabled")

        # ============ WARMUP (Fixed placement) ============
        print("-" * 60)
        print("🔥 Advanced warmup (pre-compiling common configurations)...")
        warmup_start = time.time()

        warmup_configs = [
            {"text": "A.", "speaker": "Vivian", "language": "english", "emit_every_frames": 8},
            {"text": "B.", "speaker": "Vivian", "language": "english", "emit_every_frames": 16},
            {"text": "C.", "speaker": "Serena", "language": "russian", "emit_every_frames": 16},
        ]

        for config in warmup_configs:
            print(f"   Warming: {config['speaker']} (frames={config['emit_every_frames']})")
            gen = model.stream_generate_pcm(
                text=config["text"],
                speaker=config["speaker"],
                language=config["language"],
                emit_every_frames=config["emit_every_frames"],
                decode_window_frames=80,
            )
            # Get first chunk only to trigger compilation
            for i, _ in enumerate(gen):
                if i >= 0:
                    break

        warmup_time = time.time() - warmup_start
        print(f"✓ Advanced warmup complete in {warmup_time:.2f} seconds")
        print("-" * 60)

        # Log available streaming methods
        streaming_methods = [m for m in dir(model) if 'stream' in m.lower()]
        print(f"Available streaming methods: {streaming_methods}")

        # Print GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        print("=" * 60)
        print("✓ API Server Ready! (Pre-warmed for fast responses)")
        print("=" * 60)

    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise

    yield

    print("Shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============ FastAPI App ============
app = FastAPI(
    title="Qwen3-TTS Streaming API",
    description="Streaming TTS API using dffdeeq/Qwen3-TTS-streaming fork",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Helper Functions ============
def audio_to_pcm_bytes(audio_chunk: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to PCM bytes"""
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)

    max_val = np.abs(audio_chunk).max()
    if max_val > 1.0:
        audio_chunk = audio_chunk / max_val

    audio_int16 = (audio_chunk * 32767).astype(np.int16)
    return audio_int16.tobytes()

def audio_to_wav_bytes(audio_chunk: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes"""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        wav.writeframes(audio_int16.tobytes())
    return buffer.getvalue()

# ============ Streaming Generator ============
async def stream_audio_generator(
        text: str,
        voice: str,
        language: str,
        emit_every_frames: int,
        decode_window_frames: int,
        response_format: str,
):
    """Generate streaming audio chunks"""

    try:
        # Use predefined voice with stream_generate_pcm
        if voice not in PREDEFINED_VOICES:
            voice = "Vivian"

        print(f"Generating streaming audio for voice: {voice}")

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
            chunk_count += 1
            print(f"📦 Processing chunk {chunk_count}")

            if chunk_count == 1:
                print(f"⏱️ First chunk received!")

            # Convert to bytes
            if response_format == "pcm":
                audio_bytes = audio_to_pcm_bytes(audio_chunk, sample_rate)
            else:
                audio_bytes = audio_to_wav_bytes(audio_chunk, sample_rate)

            print(f"   Yielding {len(audio_bytes)} bytes")
            yield audio_bytes
            print(f"   Chunk {chunk_count} yielded")

        print(f"✅ Streaming complete: {chunk_count} chunks")

    except Exception as e:
        print(f"Streaming error: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============ API Endpoints ============
@app.get("/")
async def root():
    return {
        "service": "Qwen3-TTS Streaming API",
        "status": "running",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "streaming": "enabled",
        "voices": list(PREDEFINED_VOICES)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "streaming_methods": [m for m in dir(model) if 'stream' in m.lower()] if model else [],
    }

@app.get("/voices")
async def list_voices():
    """List all available predefined voices"""
    return {
        "predefined_voices": list(PREDEFINED_VOICES),
        "total": len(PREDEFINED_VOICES)
    }

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text with streaming support
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.voice not in PREDEFINED_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.voice}' not found. Available: {list(PREDEFINED_VOICES)}"
        )

    # STREAMING MODE (always use streaming for best performance)
    return StreamingResponse(
        stream_audio_generator(
            text=request.text,
            voice=request.voice,
            language=request.language,
            emit_every_frames=request.emit_every_frames,
            decode_window_frames=request.decode_window_frames,
            response_format=request.response_format,
        ),
        media_type="audio/L16" if request.response_format == "pcm" else "audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )

# ============ Main Entry Point ============
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )