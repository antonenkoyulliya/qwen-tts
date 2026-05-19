# Qwen3-TTS Docker - Offline Text-to-Speech Service

Production-ready Docker container for Qwen3-TTS with model embedded in the image for offline deployment.

## Features

- **Fully Offline**: Model embedded in Docker image, no internet required at runtime
- **GPU Accelerated**: CUDA 12.4 support with automatic GPU memory management
- **Auto-Cleanup**: Automatic deletion of old audio files (configurable retention period)
- **Production Ready**: Health checks, input validation, error handling
- **Memory Safe**: Automatic GPU memory cleanup after each generation
- **REST API**: FastAPI-based API with automatic documentation

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA 12.x drivers
- nvidia-docker2 runtime
- Python 3.10+ (for model download only)

## Quick Start

### 1. Download the Model

First, download the Qwen3-TTS model on a machine **with internet access**:

```bash
cd docker
python3 download_model.py
```

This downloads ~1.7GB model to `./models/qwen-tts/`.

### 2. Build and Run

```bash
./setup.sh
```

Or manually:

```bash
docker compose build
docker compose up -d
```

### 3. Test the Service

```bash
python3 test_client.py "Hello world"
```

Or use curl:

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test", "voice": "default"}' | python3 -m json.tool
```

## API Endpoints

### POST /tts

Generate speech from text.

**Request:**
```json
{
  "text": "Text to convert to speech",
  "voice": "default"
}
```

**Response:**
```json
{
  "file_path": "/app/output/uuid.wav",
  "duration_seconds": 3.45
}
```

**Limits:**
- Max text length: 5000 characters (configurable via `MAX_TEXT_LENGTH`)
- Supported voices: `default` or custom voice files in `./custom_voices/`

### GET /health

Check service health and status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "offline_mode": true,
  "model_path": "/app/models/qwen-tts",
  "model_loaded": true,
  "sampling_rate": 12000
}
```

### GET /docs

Interactive API documentation (Swagger UI).

## Configuration

Environment variables can be set in `docker-compose.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID to use |
| `MODEL_PATH` | `/app/models/qwen-tts` | Path to model files |
| `OUTPUT_DIR` | `/app/output` | Directory for generated audio |
| `CUSTOM_VOICES_DIR` | `/app/custom_voices` | Directory for custom voice samples |
| `SAMPLING_RATE` | `12000` | Audio sampling rate (Hz) |
| `MAX_TEXT_LENGTH` | `5000` | Maximum characters per request |
| `FILE_RETENTION_HOURS` | `24` | Hours to keep generated files before cleanup |

## Custom Voices

Add custom voice samples to `./custom_voices/`:

```bash
cp my_voice.wav ./custom_voices/
```

Use in API:

```json
{
  "text": "Hello with custom voice",
  "voice": "my_voice"
}
```

## File Management

Generated audio files are saved to `./output/` and automatically cleaned up after the retention period (default 24 hours).

To retrieve files, they're available at the path returned in the API response.

## Troubleshooting

### Check logs

```bash
docker-compose logs -f
```

### Verify GPU access

```bash
docker exec -it qwen-tts-gpu nvidia-smi
```

### Check health

```bash
curl http://localhost:8000/health
```

### Out of memory errors

- Reduce `shm_size` in docker-compose.yaml if system RAM is limited
- Set `PYTORCH_CUDA_ALLOC_CONF` to manage GPU memory fragmentation
- Lower `MAX_TEXT_LENGTH` to limit generation size

### Service won't start

1. Verify model files exist: `ls -lh ./models/qwen-tts/`
2. Check GPU drivers: `nvidia-smi`
3. Ensure ports are available: `lsof -i :8000`

## Architecture

- **Base Image**: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- **Python**: 3.10
- **Framework**: FastAPI + Uvicorn
- **TTS Model**: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- **Inference**: PyTorch with CUDA acceleration

## Performance

- **Model Size**: ~1.7GB
- **Docker Image**: ~8-10GB (with model embedded)
- **GPU Memory**: ~3-4GB during inference
- **Generation Speed**: ~2-5x real-time (GPU dependent)

## Production Deployment

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Multiple GPUs

Set `CUDA_VISIBLE_DEVICES` to select GPU:

```bash
CUDA_VISIBLE_DEVICES=1 docker-compose up -d
```

### Scaling

For multiple concurrent requests, consider:
- Running multiple containers on different GPUs
- Load balancer in front (nginx, traefik)
- Async queue system (Celery, RQ) for request handling

## Security

- No internet access required after build
- Input validation on all API endpoints
- File cleanup prevents disk exhaustion
- Runs as non-root user (configurable)

## License

Follows Qwen3-TTS model license terms. Check model repository for details.

## Support

For issues specific to this Docker setup, check logs and configuration.
For model-related issues, refer to the Qwen3-TTS documentation.
