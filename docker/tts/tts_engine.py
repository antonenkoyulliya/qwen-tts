import torch
import soundfile as sf
import numpy as np
from typing import Tuple, List, Dict, Optional, AsyncGenerator
from qwen_tts import Qwen3TTSModel
from torch.cuda.amp import autocast
import asyncio

from .config import Config
from .text_processing import split_text_into_chunks, add_silence

class TTSEngine:
    """Handles TTS model loading and audio generation"""

    def __init__(self):
        self.model = None
        self.device = Config.DEVICE
        self.is_loaded = False
        self._speaker_cache = {}
        self.flash_attn_available = False

    async def initialize(self):
        """Async wrapper for load_model"""
        self.load_model()
        return True

    def load_model(self):
        """Load the TTS model with optimizations"""
        print(f"Loading model from: {Config.MODEL_PATH}")
        print(f"Device: {self.device}, Dtype: {Config.DTYPE}")
        self._check_flash_attention()
        # For V100: "sdpa" (PyTorch native) works best
        # For H100: "flash_attention_2" if available
        attn_impl = "flash_attention_2" if self.flash_attn_available else "sdpa"
        print(f"Using attention implementation: {attn_impl}")

        # Load base model
        self.model = Qwen3TTSModel.from_pretrained(
            Config.MODEL_PATH,
            device_map=self.device,
            dtype=Config.DTYPE,
            attn_implementation="sdpa",
        )

        # Enable optimizations (recommended for streaming)
        self.model.enable_streaming_optimizations(
            decode_window_frames=80,
            use_compile=True,
            compile_mode="reduce-overhead",
        )

        # Apply torch.compile for speed if enabled
        if Config.ENABLE_COMPILE and hasattr(torch, 'compile'):
            print("🔥 Applying torch.compile optimization...")
            try:
                if hasattr(self.model, 'model'):
                    self.model.model = torch.compile(
                        self.model.model,
                        mode="reduce-overhead",  # Best for inference
                        fullgraph=False  # Less aggressive, more stable
                    )
                print("✓ torch.compile enabled")
            except Exception as e:
                print(f"⚠ torch.compile failed: {e}")

        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.use_cache = True
            print("✓ KV caching enabled")

        # Warm up the model
        self._warmup()

        self.is_loaded = True
        self._print_info()

    def _check_flash_attention(self):
        """Check if Flash Attention is available for this GPU"""
        if not torch.cuda.is_available():
            return

        compute_capability = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)

        # Flash Attention requires SM80+ (A100, H100, etc.)
        if compute_capability[0] >= 8:
            try:
                import flash_attn
                self.flash_attn_available = True
                print(f"✓ Flash Attention available for {gpu_name}")
            except ImportError:
                print(f"⚠ Flash Attention not installed for {gpu_name}, using SDPA")
        else:
            print(f"ℹ {gpu_name} (SM{compute_capability[0]}{compute_capability[1]}) does not support Flash Attention, using PyTorch SDPA")

    def _warmup(self):
        """Run a small warmup inference to initialize CUDA"""
        print("🔥 Warming up model...")
        with torch.no_grad():
            _ = self.model.generate_custom_voice(
                text="Warm up.",
                language="Auto",
                speaker=self.get_speakers()[0],
                instruct=None,
            )
        print("✓ Warmup complete")

    def _print_info(self):
        """Print model information"""
        speakers = self.get_speakers()
        print(f"✓ Supported speakers: {speakers}")
        print(f"✓ Supported languages: {self.get_languages()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA memory used: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
            print(f"✓ CUDA memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

    def get_speakers(self) -> List[str]:
        """Get list of available speakers"""
        if self.model:
            return self.model.get_supported_speakers()
        return []

    def get_languages(self) -> List[str]:
        """Get list of supported languages"""
        if self.model:
            return self.model.get_supported_languages()
        return []

    def generate_audio(self, text: str, language: str, speaker: str, instruct: str = None) -> Tuple[np.ndarray, int]:
        """Generate audio with speaker caching."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        # Cache speaker (just store the speaker name, not embedding)
        # Qwen3TTS handles embedding internally, so we just pass speaker

        # Split long text into chunks
        chunks = split_text_into_chunks(text)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            with autocast(enabled=Config.USE_MIXED_PRECISION):
                if len(chunks) == 1:
                    wavs, sr = self.model.generate_custom_voice(
                        text=text,
                        language=language,
                        speaker=speaker,  # Pass speaker directly (Qwen handles caching)
                        instruct=instruct if instruct else None,
                    )
                    combined_audio = wavs[0]
                else:
                    print(f"📝 Splitting text into {len(chunks)} chunks")
                    combined_audio = None
                    sr = None

                    # OPTIMIZATION 5: Process multiple chunks if batch processing is possible
                    # For now, sequential with caching benefits
                    for i, chunk in enumerate(chunks):
                        print(f"  Processing chunk {i+1}/{len(chunks)}")

                        wavs, sr = self.model.generate_custom_voice(
                            text=chunk,
                            language=language,
                            speaker=speaker,  # Same speaker, Qwen handles optimization
                            instruct=instruct if instruct else None,
                        )

                        if combined_audio is None:
                            combined_audio = wavs[0]
                        else:
                            combined_audio = add_silence(combined_audio, sr)
                            combined_audio = np.concatenate([combined_audio, wavs[0]])

                    print(f"✓ Combined {len(chunks)} chunks into {len(combined_audio)/sr:.2f}s")

        return combined_audio, sr

    def generate_audio_stream(self,
                              text: str,
                              language: str,
                              speaker: str,
                              instruct: str =None,
                              cancel_event: Optional[asyncio.Event] = None) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio"""
        if not self.model:
            raise RuntimeError("Model not loaded")

        sample_rate = 24000  # Qwen-TTS default

        if hasattr(self.model, 'stream_generate_custom_voice'):
            # Enable streaming optimizations
            if hasattr(self.model, 'enable_streaming_optimizations'):
                self.model.enable_streaming_optimizations(
                    decode_window_frames=80,
                    use_compile=Config.ENABLE_COMPILE,
                )

            with torch.no_grad():
                with autocast(enabled=Config.USE_MIXED_PRECISION):
                    # Stream audio chunks
                    for chunk, sr in self.model.stream_generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=speaker,
                            instruct=instruct if instruct else None,
                    ):
                        if first_chunk:
                            sample_rate = sr
                            first_chunk = False
                        yield chunk
        else:
            # Fallback: generate all at once (no streaming)
            print("⚠ Streaming not available, generating full audio")
            audio, sr = self.generate_audio(text, language, speaker, instruct)
            yield audio, sr

        async def stream_generate_audio(
                self,
                text: str,
                language: str,
                speaker: str,
                instruct: str = None
        ):
            """Stream audio chunks as they're generated."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        # Enable streaming optimizations
        if hasattr(self.model, 'enable_streaming_optimizations'):
            self.model.enable_streaming_optimizations(
                decode_window_frames=80,
                use_compile=Config.ENABLE_COMPILE,
            )

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=Config.USE_MIXED_PRECISION):
                # Use stream_generate_voice_clone (the available method)
                for chunk, sr in self.model.stream_generate_voice_clone(
                        text=text,
                        language=language,
                        speaker=speaker,
                        instruct=instruct if instruct else None,
                ):
                    yield chunk

        def unload(self):
            """Unload model and free GPU memory"""
            if self.model:
                self.model = None
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                self.is_loaded = False
                print("✓ Model unloaded and GPU memory freed")