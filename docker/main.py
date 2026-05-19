#!/usr/bin/env python3
"""
Main entry point for TTS Service
"""

import sys
import os
import torch
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_attention_backend():
    """Configure attention backend based on GPU capability"""
    if not torch.cuda.is_available():
        logging.info("No GPU detected, using CPU backend")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)

    logging.info(f"GPU detected: {gpu_name}")
    logging.info(f"Compute capability: SM{compute_capability[0]}{compute_capability[1]}")

    # Check if GPU supports Flash Attention (requires SM80+)
    if compute_capability[0] >= 8:
        try:
            import flash_attn
            logging.info("✓ Flash Attention available and enabled")
            return True
        except ImportError:
            logging.info("Flash Attention not installed, using PyTorch SDPA")
    else:
        logging.info(f"GPU {gpu_name} does not support Flash Attention, using PyTorch SDPA")

    return False

if __name__ == "__main__":
    # Import here to avoid circular imports
    from tts.config import Config
    import uvicorn

    # Run the application from tts.main
    uvicorn.run(
        "tts.main:app",
        host=Config.HOST,
        port=Config.PORT,
        workers=Config.WORKERS,
        loop="uvloop",
        limit_max_requests=1000,
        timeout_keep_alive=30
    )