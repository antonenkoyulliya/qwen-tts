main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import health, jobs, tts
from .config import Config
from .tts_engine import TTSEngine
from .job_manager import JobManager

# Create FastAPI app
app = FastAPI(
    title="Qwen-TTS Service",
    description="Text-to-Speech service using Qwen models",
    version="1.0.0"
)

# Initialize global instances
tts_engine = TTSEngine()
job_manager = JobManager()

# Set the dependencies in route modules
health.tts_engine = tts_engine
health.job_manager = job_manager
tts.tts_engine = tts_engine
tts.job_manager = job_manager
jobs.job_manager = job_manager

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(tts.router, prefix="/tts", tags=["tts"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])

@app.get("/")
async def root():
    return {"message": "Qwen-TTS Service is running", "status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("Starting up TTS Service...")
    await tts_engine.initialize()
    print("TTS Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down TTS Service...")
    await tts_engine.cleanup()
    print("TTS Service shutdown complete")