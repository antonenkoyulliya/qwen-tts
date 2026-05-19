FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/qwen-tts-model
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    sox \
    libsox-fmt-all \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3.10 to python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

RUN pip3 install --upgrade pip numpy==1.24.3

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention for CUDA 12.4 (prebuilt wheel)
RUN pip3 install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu124torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl || \
    echo "Warning: flash-attn installation failed, continuing without it"

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional API dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart

# Copy the streaming fork code
COPY Qwen3-TTS-streaming /app/Qwen3-TTS-streaming/

# Copy the pre-downloaded model
COPY qwen-tts-model /app/qwen-tts-model/

# Install the qwen-tts package in development mode
RUN cd /app/Qwen3-TTS-streaming && pip3 install -e .

# Add the Qwen3-TTS-streaming to Python path
ENV PYTHONPATH=/app/Qwen3-TTS-streaming:${PYTHONPATH}

# Create directory for voice profiles
RUN mkdir -p /app/voices /app/output

# Expose API port
EXPOSE 8000

# Copy the API server script
COPY api_server.py /app/api_server.py

# Set entrypoint
CMD ["python3", "/app/api_server.py"]