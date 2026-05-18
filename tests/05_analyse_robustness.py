from pathlib import Path

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"
RAW_DIR = REPO_ROOT / "tests" / "results" / "raw"
NPY_DIR = REPO_ROOT / "tests" / "results" / "npy"
OUT_PATH = NPY_DIR / "robustness_results.npy"

SAMPLE_RATE = 48_000
ZERO_RUN_LIMIT = SAMPLE_RATE  # 1 second


TEST_CASES = [
    ("robustness_impulse", "impulse", "impulse_48k_mono.wav"),
    ("robustness_dc", "dc_offset", "dc_offset_48k_mono.wav"),
    ("robustness_loud", "loud", "loud_48k_mono.wav"),
    ("robustness_quiet", "quiet", "quiet_48k_mono.wav"),
]


def read_wav_mono(path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate in {path}: {sr}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32767.0
    if data.dtype == np.float32:
        return data.astype(np.float32)
    return data.astype(np.float32)


def resolve_output_path(base_name: str) -> Path | None:
    # Accept both naming conventions:
    # robustness_xxx.wav (old) and robustness_xxx_cpu.wav (new matrix)
    direct = RAW_DIR / f"{base_name}.wav"
    if direct.exists():
        return direct
    cpu = RAW_DIR / f"{base_name}_cpu.wav"
    if cpu.exists():
        return cpu
    return None


def longest_zero_run(signal: np.ndarray, eps: float = 1e-9) -> int:
    mask = np.abs(signal) <= eps
    if not np.any(mask):
        return 0
    max_run = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


def high_freq_attenuation_db(x_in: np.ndarray, x_out: np.ndarray) -> float:
    n = min(len(x_in), len(x_out))
    x_in = x_in[:n]
    x_out = x_out[:n]
    fin = np.fft.rfft(x_in)
    fout = np.fft.rfft(x_out)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    high = freqs >= 4000.0
    p_in = np.mean(np.abs(fin[high]) ** 2) + 1e-12
    p_out = np.mean(np.abs(fout[high]) ** 2) + 1e-12
    return float(10.0 * np.log10(p_out / p_in))


def main() -> None:
    NPY_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    print("=== Résultats des tests de robustesse (exécutés sur téléphone) ===")
    print("Test | Statut | Amplitude max | Rapport puissance | Résumé")

    for test_name, input_type, input_file in TEST_CASES:
        in_path = DATA_DIR / input_file
        out_path = resolve_output_path(test_name)

        if not in_path.exists():
            row = {
                "test_name": test_name,
                "input_type": input_type,
                "status": "FAIL",
                "max_amplitude": float("nan"),
                "has_zeros": True,
                "power_ratio_db": float("nan"),
                "high_freq_attenuation_db": float("nan"),
            }
            results.append(row)
            print(f"{test_name} | FAIL | N/A | N/A | entrée manquante")
            continue

        if out_path is None:
            row = {
                "test_name": test_name,
                "input_type": input_type,
                "status": "FAIL",
                "max_amplitude": float("nan"),
                "has_zeros": True,
                "power_ratio_db": float("nan"),
                "high_freq_attenuation_db": float("nan"),
            }
            results.append(row)
            print(f"{test_name} | FAIL | N/A | N/A | sortie téléphone manquante")
            continue

        x_in = read_wav_mono(in_path)
        x_out = read_wav_mono(out_path)
        n = min(len(x_in), len(x_out))
        x_in = x_in[:n]
        x_out = x_out[:n]

        max_amp = float(np.max(np.abs(x_out)))
        has_nan_inf = bool(np.any(~np.isfinite(x_out)))
        zero_run = longest_zero_run(x_out)
        has_zeros = bool(zero_run > ZERO_RUN_LIMIT)

        pin = float(np.mean(x_in**2) + 1e-12)
        pout = float(np.mean(x_out**2) + 1e-12)
        power_ratio_db = float(10.0 * np.log10(pout / pin))

        hf_att_db = high_freq_attenuation_db(x_in, x_out)

        has_signal = bool(np.max(np.abs(x_out)) > 1e-6)
        passed = (max_amp < 1.5) and (not has_zeros) and has_signal and (not has_nan_inf)
        status = "PASS" if passed else "FAIL"

        row = {
            "test_name": test_name,
            "input_type": input_type,
            "status": status,
            "max_amplitude": max_amp,
            "has_zeros": has_zeros,
            "power_ratio_db": power_ratio_db,
            "high_freq_attenuation_db": hf_att_db,
        }
        results.append(row)

        summary = (
            f"zero_run={zero_run} échantillons"
            + ("; NaN/Inf détecté" if has_nan_inf else "")
        )
        print(
            f"{test_name} | {status} | {max_amp:.4f} | {power_ratio_db:+.2f} dB | {summary}"
        )

    np.save(OUT_PATH, np.array(results, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
