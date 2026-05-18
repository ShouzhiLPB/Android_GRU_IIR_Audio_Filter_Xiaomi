import math
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 48_000
AMPLITUDE = 0.9
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def to_int16(signal: np.ndarray) -> np.ndarray:
    signal = np.clip(signal, -1.0, 1.0)
    return (signal * 32767.0).astype(np.int16)


def write_wav(path: Path, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    pcm = to_int16(signal)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def generate_log_sweep(seconds: int = 60) -> np.ndarray:
    n = seconds * SAMPLE_RATE
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f0 = 20.0
    f1 = 20_000.0
    k = seconds / math.log(f1 / f0)
    phase = 2.0 * math.pi * f0 * k * (np.exp(t / k) - 1.0)
    sig = np.sin(phase)
    return (AMPLITUDE * sig).astype(np.float32)


def generate_band_limited_noise(seconds: int = 30) -> np.ndarray:
    n = seconds * SAMPLE_RATE
    white = np.random.randn(n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    band_mask = (freqs >= 20.0) & (freqs <= 20_000.0)
    spectrum[~band_mask] = 0.0
    noise = np.fft.irfft(spectrum, n=n)
    noise = noise / (np.max(np.abs(noise)) + 1e-12)
    return (AMPLITUDE * noise).astype(np.float32)


def generate_multi_sine() -> np.ndarray:
    segment_seconds = 6
    frequencies = [100.0, 500.0, 1000.0, 3000.0, 8000.0]
    chunks = []
    n_seg = segment_seconds * SAMPLE_RATE
    t = np.arange(n_seg, dtype=np.float64) / SAMPLE_RATE
    for f in frequencies:
        chunks.append(np.sin(2.0 * math.pi * f * t))
    sig = np.concatenate(chunks, axis=0)
    return (AMPLITUDE * sig).astype(np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep = generate_log_sweep(60)
    noise = generate_band_limited_noise(30)
    sine = generate_multi_sine()

    write_wav(OUTPUT_DIR / "sweep_48k_mono.wav", sweep)
    write_wav(OUTPUT_DIR / "noise_48k_mono.wav", noise)
    write_wav(OUTPUT_DIR / "sine_48k_mono.wav", sine)

    n10 = 10 * SAMPLE_RATE
    t10 = np.arange(n10, dtype=np.float64) / SAMPLE_RATE

    # 4) impulse_48k_mono.wav
    impulse = np.zeros(n10, dtype=np.float32)
    impulse[::SAMPLE_RATE] = 1.0  # 32767 after int16 conversion
    write_wav(OUTPUT_DIR / "impulse_48k_mono.wav", impulse)

    # 5) dc_offset_48k_mono.wav
    dc_sig = 0.5 * np.sin(2.0 * math.pi * 1000.0 * t10) + 0.3
    dc_sig = dc_sig / max(1.0, float(np.max(np.abs(dc_sig))))
    write_wav(OUTPUT_DIR / "dc_offset_48k_mono.wav", dc_sig.astype(np.float32))

    # 6) loud_48k_mono.wav
    loud = 0.99 * np.sin(2.0 * math.pi * 1000.0 * t10)
    write_wav(OUTPUT_DIR / "loud_48k_mono.wav", loud.astype(np.float32))

    # 7) quiet_48k_mono.wav
    quiet = 0.001 * np.sin(2.0 * math.pi * 1000.0 * t10)
    write_wav(OUTPUT_DIR / "quiet_48k_mono.wav", quiet.astype(np.float32))

    print("[OK] Tests signals generated in tests/data/")
    print("  - sweep_48k_mono.wav : 60s")
    print("  - noise_48k_mono.wav : 30s")
    print("  - sine_48k_mono.wav : 30s")
    print("  - impulse_48k_mono.wav : 10s")
    print("  - dc_offset_48k_mono.wav : 10s")
    print("  - loud_48k_mono.wav : 10s")
    print("  - quiet_48k_mono.wav : 10s")


if __name__ == "__main__":
    main()
