FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MAX_JOBS=4
ENV PIP_ROOT_USER_ACTION=ignore

# Install Python 3.10 (keep as is since it works)
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libsndfile1 ffmpeg sox libsox-fmt-all \
    python3-sox build-essential ninja-build wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip numpy==1.24.3

# PyTorch 2.6.0 for CUDA 12.4
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchaudio==2.6.0

# For JIT compilation speed
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache

# Install onnxruntime-gpu
RUN pip3 install --no-cache-dir onnxruntime-gpu

# Try flash-attn (will install on H100, fail on V100)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || true

# Install uvloop for faster async (optional performance boost)
RUN pip3 install --no-cache-dir uvloop || echo "uvloop not available, using asyncio"

# Streaming fork version:
RUN pip3 install --no-cache-dir --ignore-installed --no-deps\
    "qwen-tts @ git+https://github.com/rekuenkdr/Qwen3-TTS-streaming.git"

# Install remaining requirements (excluding torch and flash-attn)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pin transformers to working version (overrides any other version)
RUN pip3 install --no-cache-dir transformers==4.57.3

# Verify streaming fork is correctly installed
RUN python3 -c "
from qwen_tts import Qwen3TTSModel
stream_methods = [m for m in dir(Qwen3TTSModel) if 'stream' in m.lower()]
print(f'✓ Streaming methods found: {stream_methods}')
assert hasattr(Qwen3TTSModel, 'stream_generate_custom_voice'), 'Streaming fork not installed!'
print('✓ Streaming fork verified successfully')
"

RUN mkdir -p /app/output /app/custom_voices
COPY tts/ ./tts/
COPY main.py ./
COPY ./models/qwen-tts /app/models/qwen-tts

EXPOSE 8000

# Set environment variables for model caching
ENV HF_HOME=/app/models
ENV MODEL_PATH=/app/models/qwen-tts
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

#CMD ["python3", "tts_server.py"]
CMD ["python3", "main.py"]