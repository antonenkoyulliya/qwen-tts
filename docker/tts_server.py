import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
from contextlib import asynccontextmanager
import os
import uuid
import asyncio
from typing import List
import numpy as np
from functools import lru_cache
import gc

# ============ PERFORMANCE OPTIMIZATIONS ============
# Enable TF32 on Ampere+ GPUs (20-30% speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN

# Enable faster memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

model = None
device = "cuda:0"

class TTSRequest(BaseModel):
    text: str
    language: str = "Auto"
    speaker: str = "Vivian"
    instruct: str = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device

    model_path = os.getenv("MODEL_PATH", "/app/models/qwen-tts")
    print(f"Loading model from: {model_path}")

    # Load model with optimizations
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16,  # bfloat16 is faster than float16 on Ampere+
        attn_implementation="sdpa",  # Use PyTorch's native SDPA (almost as fast as flash-attn!)
    )

    # Compile model for 20-40% speedup (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("🔥 Applying torch.compile optimization...")
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✓ torch.compile enabled")
        except Exception as e:
            print(f"⚠ torch.compile failed: {e}")

    # Warm up model (optional but reduces first-request latency)
    print("🔥 Warming up model...")
    with torch.no_grad():
        _ = model.generate_custom_voice(
            text="Warm up.",
            language="Auto",
            speaker=model.get_supported_speakers()[0],
            instruct=None,
        )
    print("✓ Model ready!")

    speakers = model.get_supported_speakers()
    print(f"✓ Supported speakers: {speakers}")
    print(f"✓ Supported languages: {model.get_supported_languages()}")
    print(f"✓ CUDA memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    yield

    # Cleanup
    model = None
    torch.cuda.empty_cache()
    gc.collect()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/speakers")
async def get_speakers():
    if model:
        return {"speakers": model.get_supported_speakers()}
    return {"speakers": []}

@app.get("/languages")
async def get_languages():
    if model:
        return {"languages": model.get_supported_languages()}
    return {"languages": []}

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Use larger chunks for fewer generations (200 → 350 chars)
        chunks = split_text_into_chunks(request.text, max_chars=350)

        with torch.no_grad():  # Disable gradient computation
            torch.cuda.synchronize()  # Ensure clean timing

            if len(chunks) == 1:
                wavs, sr = model.generate_custom_voice(
                    text=request.text,
                    language=request.language,
                    speaker=request.speaker,
                    instruct=request.instruct if request.instruct else None,
                )
                combined_audio = wavs[0]
            else:
                # Use list comprehension for parallel generation (if your model supports it)
                print(f"📝 Splitting text into {len(chunks)} chunks")
                combined_audio = None
                sr = None

                # Process chunks sequentially (most TTS models can't do parallel due to state)
                for i, chunk in enumerate(chunks):
                    print(f"  Chunk {i+1}/{len(chunks)}")

                    wavs, sr = model.generate_custom_voice(
                        text=chunk,
                        language=request.language,
                        speaker=request.speaker,
                        instruct=request.instruct if request.instruct else None,
                    )

                    if combined_audio is None:
                        combined_audio = wavs[0]
                    else:
                        # Reduce silence between chunks (300ms → 150ms)
                        combined_audio = add_silence(combined_audio, sr, 150)
                        combined_audio = np.concatenate([combined_audio, wavs[0]])

                print(f"✓ Combined {len(chunks)} chunks into {len(combined_audio)/sr:.2f}s")

        # Save async to not block
        filename = f"{uuid.uuid4()}.wav"
        filepath = f"/app/output/{filename}"

        # Write in non-blocking way
        await asyncio.to_thread(sf.write, filepath, combined_audio, sr)

        return {
            "audio_url": f"/output/{filename}",
            "sample_rate": sr,
            "duration": len(combined_audio) / sr,
            "speaker": request.speaker,
            "chunks_processed": len(chunks)
        }

    except Exception as e:
        print(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/output/{filename}")
async def get_audio(filename: str):
    filepath = f"/app/output/{filename}"
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/tts/batch")
async def generate_speech_batch(requests: List[TTSRequest]):
    """Batch processing for multiple texts - faster for bulk generation"""
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    with torch.no_grad():
        for req in requests:
            try:
                wavs, sr = model.generate_custom_voice(
                    text=req.text,
                    language=req.language,
                    speaker=req.speaker,
                    instruct=req.instruct if req.instruct else None,
                )

                filename = f"{uuid.uuid4()}.wav"
                filepath = f"/app/output/{filename}"
                await asyncio.to_thread(sf.write, filepath, wavs[0], sr)

                results.append({
                    "audio_url": f"/output/{filename}",
                    "duration": len(wavs[0]) / sr,
                    "speaker": req.speaker
                })
            except Exception as e:
                results.append({"error": str(e), "text": req.text[:50]})

    return {"results": results, "total": len(results)}

def split_text_into_chunks(text: str, max_chars: int = 350) -> List[str]:
    """Optimized text splitting with natural breaks"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by punctuation for better prosody
    import re
    sentences = re.split(r'(?<=[.!?;:])\s+', text)

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Ensure no empty chunks
    return [c for c in chunks if c]

def add_silence(audio: np.ndarray, sample_rate: int, silence_ms: int = 150) -> np.ndarray:
    """Add silence with optimized memory"""
    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    return np.concatenate([audio, silence])

if __name__ == "__main__":
    import uvicorn
    # Optimized uvicorn settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Keep 1 worker for GPU model
        loop="uvloop",  # Faster event loop
        limit_max_requests=1000,  # Prevent memory leaks
        timeout_keep_alive=30
    )