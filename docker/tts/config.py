import os
import torch

class Config:
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/qwen-tts")
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    # CRITICAL: For V100, use float16 (not bfloat16!)
    # For H100, bfloat16 is better
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability(0)
        if compute_capability[0] >= 8:  # H100, A100, etc.
            DTYPE = torch.bfloat16
        else:  # V100 (SM70) and older
            DTYPE = torch.float16
    else:
        DTYPE = torch.float32

    # Mixed precision for inference
    USE_MIXED_PRECISION = os.getenv("USE_MIXED_PRECISION", "1") == "1"

    # Text processing
    MAX_TEXT_LENGTH = 5000
    CHUNK_SIZE = 350  # characters
    SILENCE_MS = 150  # between chunks

    # Job management
    MAX_JOB_HISTORY = 1000
    JOB_CLEANUP_HOURS = 24

    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    WORKERS = int(os.getenv("WORKERS", "1"))

    # Performance
    ENABLE_TF32 = True
    MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "500"))
    SAMPLE_RATE = 24000  # Default sample rate for Qwen-TTS
    ENABLE_COMPILE = os.getenv("ENABLE_COMPILE", "1") == "1"

    # Audio settings
    SILENCE_DURATION_MS = int(os.getenv("SILENCE_DURATION_MS", "500"))

    @classmethod
    def setup_performance(cls):
        """Setup PyTorch performance optimizations"""
        if cls.ENABLE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ TF32 enabled")

        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

        # Print GPU info
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"✓ Using dtype: {cls.DTYPE}")
            print(f"✓ Mixed precision: {cls.USE_MIXED_PRECISION}")

    @classmethod
    def __getitem__(cls, key):
        return getattr(cls, key)