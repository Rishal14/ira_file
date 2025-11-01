"""Generate 3 short synthetic WAV files for testing the indexer.
Run from project root after activating the virtualenv:
    python generate_sample_wavs.py
This writes files to data/wavs/sample_*.wav
"""
import os
import numpy as np
import soundfile as sf

OUT_DIR = "data/wavs"
os.makedirs(OUT_DIR, exist_ok=True)

sr = 16000
dur = 1.5  # seconds

t = np.linspace(0, dur, int(sr*dur), endpoint=False)

# three different sine tones (different frequencies)
freqs = [440.0, 660.0, 880.0]

for i, f in enumerate(freqs, 1):
    wave = 0.3 * np.sin(2 * np.pi * f * t)
    path = os.path.join(OUT_DIR, f"sample_{i}_{int(f)}Hz.wav")
    sf.write(path, wave.astype(np.float32), sr)
    print("Wrote:", path)

print("Done. You can now run: python extract_and_index.py")
