import re
import numpy as np
from typing import List
from .config import Config

def split_text_into_chunks(text: str, max_chars: int = None) -> List[str]:
    """
    Split text into chunks at natural boundaries (punctuation).

    Args:
        text: Input text to split
        max_chars: Maximum characters per chunk (default from Config)

    Returns:
        List of text chunks
    """
    if max_chars is None:
        max_chars = Config.CHUNK_SIZE

    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # Split at sentence boundaries (punctuation followed by space)
    sentences = re.split(r'(?<=[.!?;:])\s+', text)

    for sentence in sentences:
        # If adding this sentence would exceed limit, save current chunk
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Add sentence to current chunk
            separator = " " if current_chunk else ""
            current_chunk = current_chunk + separator + sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out any empty chunks
    return [chunk for chunk in chunks if chunk]

def add_silence(audio: np.ndarray, sample_rate: int, silence_ms: int = None) -> np.ndarray:
    """
    Add silence between audio chunks for better flow.

    Args:
        audio: Audio numpy array
        sample_rate: Audio sample rate in Hz
        silence_ms: Silence duration in milliseconds

    Returns:
        Audio with silence appended
    """
    if silence_ms is None:
        silence_ms = Config.SILENCE_MS

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=audio.dtype)

    return np.concatenate([audio, silence])