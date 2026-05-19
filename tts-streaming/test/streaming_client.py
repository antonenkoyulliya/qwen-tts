#!/usr/bin/env python3
"""
streaming_client_fixed.py - Fixed streaming client with better timeout handling
"""

import requests
import numpy as np
import time
import sys

API_URL = "http://195.209.214.219:8000/v1/audio/speech"
SAMPLE_RATE = 24000

def stream_and_save(output_file="output.pcm"):
    """Stream audio and save to file (most reliable)"""

    print("Connecting to streaming API...")

    # Use session with longer timeout
    session = requests.Session()
    session.timeout = (30, 300)  # (connect, read) timeout

    try:
        response = session.post(
            API_URL,
            json={
                "text": "Я — ваш AI-ассистент, и я готов помочь вам с самыми разными задачами! Вот чем я могу быть полезен:\n\n*   Помощь в учебе: Я могу объяснять сложные темы, помогать с домашними заданиями, разбирать упражнения (например, по немецкому языку), проверять грамматику и орфографию.",
                "voice": "Vivian",
                "language": "russian",
                "response_format": "pcm",
                "emit_every_frames": 8,
            },
            stream=True,
            timeout=(30, 300)
        )

        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code != 200:
            print(f"Error: {response.text}")
            return

        chunk_count = 0
        total_bytes = 0

        print("Receiving audio chunks...")

        # Open file to save all chunks
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    f.write(chunk)
                    print(f"Chunk {chunk_count}: {len(chunk)} bytes (total: {total_bytes})")

        print(f"\n✅ Success! Received {chunk_count} chunks")
        print(f"💾 Saved to {output_file} ({total_bytes} bytes)")

        # Try to play if on Linux/Mac
        if sys.platform != 'win32':
            import subprocess
            subprocess.run(['aplay', '-f', 'S16_LE', '-r', '24000', '-c', '1', output_file])

    except requests.exceptions.Timeout:
        print("❌ Timeout - connection took too long")
    except requests.exceptions.ChunkedEncodingError as e:
        print(f"❌ Chunked encoding error: {e}")
        print("This might be because the server is taking too long between chunks")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        session.close()

def stream_and_play_with_buffering():
    """Stream audio with buffering and playback"""

    print("Connecting to streaming API...")

    response = requests.post(
        API_URL,
        json={
            "text": "This is a test of streaming audio with buffering.",
            "voice": "Vivian",
            "language": "english",
            "response_format": "pcm",
        },
        stream=True,
        timeout=(30, 60)
    )

    print(f"Status: {response.status_code}")

    chunk_count = 0
    buffer = bytearray()

    for chunk in response.iter_content(chunk_size=8192):  # Smaller chunks
        if chunk:
            chunk_count += 1
            buffer.extend(chunk)
            print(f"Chunk {chunk_count}: {len(chunk)} bytes (buffer: {len(buffer)})")

            # Play when buffer reaches certain size (e.g., 0.5 seconds)
            if len(buffer) >= SAMPLE_RATE:  # 1 second of audio
                audio = np.frombuffer(buffer[:SAMPLE_RATE*2], dtype=np.int16).astype(np.float32) / 32767.0

                # Play using pyaudio (Windows compatible)
                try:
                    import pyaudio
                    p = pyaudio.PyAudio()
                    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
                    stream.write(buffer[:SAMPLE_RATE*2])
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                except ImportError:
                    print("pyaudio not installed, saving to file instead")
                    with open("temp_audio.pcm", "ab") as f:
                        f.write(buffer[:SAMPLE_RATE*2])

                buffer = buffer[SAMPLE_RATE*2:]  # Remove played portion

    print(f"\n✅ Received {chunk_count} chunks")

if __name__ == "__main__":
    print("Streaming Client (Windows compatible)")
    print("=" * 40)

    # Option 1: Save to file (works always)
    stream_and_save("test_audio.pcm")

    # Option 2: Try to play with pygame (Windows alternative)
    print("\nConverting to WAV for playback...")
    import subprocess
    try:
        # Convert PCM to WAV using ffmpeg if available
        subprocess.run(['ffmpeg', '-f', 's16le', '-ar', '24000', '-ac', '1',
                        '-i', 'test_audio.pcm', 'test_audio.wav'],
                       capture_output=True)
        print("Created test_audio.wav - double-click to play")
    except:
        print("Install ffmpeg or use a player that supports PCM files")