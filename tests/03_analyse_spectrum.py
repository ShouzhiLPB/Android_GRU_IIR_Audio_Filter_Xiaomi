from pathlib import Path

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "tests" / "data"
RESULTS_RAW = REPO_ROOT / "tests" / "results" / "raw"
RESULTS_NPY = REPO_ROOT / "tests" / "results" / "npy"

INPUT_WAV = TEST_DATA / "sweep_48k_mono.wav"
FC_LIST = [500, 1000, 3000]
PHONE_WAV = {
    500: RESULTS_RAW / "sweep_fc500_cpu.wav",
    1000: RESULTS_RAW / "sweep_fc1000_cpu.wav",
    3000: RESULTS_RAW / "sweep_fc3000_cpu.wav",
}
SAMPLE_RATE = 48_000


def read_wav_mono(path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate mismatch in {path}: {sr} != {SAMPLE_RATE}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.float32:
        data = data.copy()
    else:
        data = data.astype(np.float32)
    return data


def spectrum_db(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag_db = 20.0 * np.log10(np.abs(np.fft.rfft(signal)) + 1e-10)
    return freqs, mag_db


def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.std(x))


def find_fc_actual(
    freqs: np.ndarray, transfer: np.ndarray, threshold_db: float, fc_theory: float
) -> float:
    # Search -3 dB crossing around theoretical cutoff only.
    search_mask = (freqs >= 0.1 * fc_theory) & (freqs <= 5.0 * fc_theory)
    freqs_search = freqs[search_mask]
    transfer_search = transfer[search_mask]
    idx = np.where(transfer_search <= threshold_db)[0]
    if idx.size == 0:
        return float("nan")
    return float(freqs_search[idx[0]])


def nearest_value(freqs: np.ndarray, values: np.ndarray, target_freq: float) -> float:
    idx = int(np.argmin(np.abs(freqs - target_freq)))
    return float(values[idx])


def main() -> None:
    RESULTS_NPY.mkdir(parents=True, exist_ok=True)

    if not INPUT_WAV.exists():
        raise FileNotFoundError(f"Missing input wav: {INPUT_WAV}")
    missing = [str(PHONE_WAV[fc]) for fc in FC_LIST if not PHONE_WAV[fc].exists()]
    if missing:
        raise FileNotFoundError("Missing phone output wav files:\n" + "\n".join(missing))

    x_in = read_wav_mono(INPUT_WAV)
    results_dtype = np.dtype([
        ("fc_theory_hz", np.float64),
        ("fc_actual_hz", np.float64),
        ("fc_error_hz", np.float64),
        ("passband_gain_db", np.float64),
        ("attenuation_2fc_db", np.float64),
        ("stopband_attenuation_db", np.float64),
        ("passband_flatness_db", np.float64),
    ])
    table = np.zeros(len(FC_LIST), dtype=results_dtype)
    raw = {}

    print("=== Caractéristiques fréquentielles mesurées sur téléphone ===")
    print("fc_théorique | fc_réelle | écart | atténuation@2fc | planéité_passebande")

    for i, fc in enumerate(FC_LIST):
        x_out = read_wav_mono(PHONE_WAV[fc])
        n = min(len(x_in), len(x_out))
        xin = x_in[:n]
        xout = x_out[:n]

        freqs, mag_in = spectrum_db(xin, SAMPLE_RATE)
        _, mag_out = spectrum_db(xout, SAMPLE_RATE)
        transfer = mag_out - mag_in

        passband_mask = (freqs >= 20.0) & (freqs <= 0.3 * fc)
        passband_gain = safe_mean(transfer[passband_mask])
        threshold = passband_gain - 3.0
        fc_actual = find_fc_actual(freqs, transfer, threshold, fc)
        fc_error = float(abs(fc_actual - fc)) if np.isfinite(fc_actual) else float("nan")

        att_2fc = nearest_value(freqs, transfer, 2.0 * fc) - passband_gain
        stopband_mask = freqs >= 5.0 * fc
        stopband_att = safe_mean(transfer[stopband_mask]) - passband_gain
        flatness_mask = (freqs >= 20.0) & (freqs <= 0.5 * fc)
        flatness = safe_std(transfer[flatness_mask])

        table[i] = (
            float(fc),
            fc_actual,
            fc_error,
            passband_gain,
            att_2fc,
            stopband_att,
            flatness,
        )

        raw[f"fc_{fc}"] = {
            "freqs": freqs,
            "mag_input": mag_in,
            "mag_output": mag_out,
            "transfer": transfer,
        }

        def fmt(v: float) -> str:
            return "nan" if not np.isfinite(v) else f"{v:.2f}"

        print(f"{fc:>11} | {fmt(fc_actual):>9} | {fmt(fc_error):>5} | {fmt(att_2fc):>15} | {fmt(flatness):>19}")

    np.save(RESULTS_NPY / "spectrum_results.npy", table)
    np.save(RESULTS_NPY / "spectrum_raw.npy", raw, allow_pickle=True)


if __name__ == "__main__":
    main()
