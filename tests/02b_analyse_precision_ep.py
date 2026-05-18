from pathlib import Path

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_NPY = REPO_ROOT / "tests" / "results" / "npy"
RESULTS_RAW = REPO_ROOT / "tests" / "results" / "raw"

WAVEFORMS_PATH = RESULTS_NPY / "precision_waveforms.npy"
OUT_PATH = RESULTS_NPY / "precision_results_ep.npy"

FC_LIST = [500, 1000, 3000]
EP_LIST = ["cpu", "xnnpack", "nnapi"]  # axis1 order
GROUP_NAMES = ["pytorch_vs_phone", "ort_pc_vs_phone"]  # axis2 order


def read_wav_mono(path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.float32:
        data = data.copy()
    else:
        data = data.astype(np.float32)
    return data


def align_signals(a: np.ndarray, b: np.ndarray, max_shift: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-n + 1, n)
    lag = int(lags[np.argmax(corr)])
    lag = int(np.clip(lag, -max_shift, max_shift))
    if lag > 0:
        return a[lag:], b[: len(b) - lag]
    if lag < 0:
        lag = -lag
        return a[: len(a) - lag], b[lag:]
    return a, b


def metrics(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mae = float(np.mean(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))
    pearson = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    max_err = float(np.max(np.abs(a - b)))
    psnr = float(20.0 * np.log10(np.max(np.abs(a)) / (np.sqrt(mse) + 1e-10)))
    return np.array([mae, mse, pearson, max_err, psnr], dtype=np.float64)


def load_waveforms(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        data = obj.item()
        if isinstance(data, dict):
            return data
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported precision_waveforms.npy format: {type(obj)}")


def main() -> None:
    if not WAVEFORMS_PATH.exists():
        raise FileNotFoundError(f"Missing: {WAVEFORMS_PATH}")

    waveforms = load_waveforms(WAVEFORMS_PATH)
    res = np.full((3, 3, 2, 5), np.nan, dtype=np.float64)  # fc x ep x group x metric

    print("=== Analyse de précision multi-EP : PyTorch/ORT-PC vs Téléphone ===")

    for fi, fc in enumerate(FC_LIST):
        key = f"fc_{fc}"
        if key not in waveforms:
            raise KeyError(f"Missing key in precision_waveforms.npy: {key}")

        y_pt = np.asarray(waveforms[key]["y_pytorch"], dtype=np.float32).reshape(-1)
        y_ort = np.asarray(waveforms[key]["y_ort_pc"], dtype=np.float32).reshape(-1)

        print(f"\nfc = {fc} Hz")
        print("ep | groupe | MAE | MSE | Pearson | Max_Error | PSNR")

        for ei, ep in enumerate(EP_LIST):
            phone_wav = RESULTS_RAW / f"sweep_fc{fc}_{ep}.wav"
            if not phone_wav.exists():
                print(f"{ep} | {GROUP_NAMES[0]} | N/A (fichier manquant: {phone_wav.name})")
                print(f"{ep} | {GROUP_NAMES[1]} | N/A (fichier manquant: {phone_wav.name})")
                continue

            y_phone = read_wav_mono(phone_wav)

            a1, b1 = align_signals(y_pt, y_phone)
            m1 = metrics(a1, b1)
            res[fi, ei, 0, :] = m1

            a2, b2 = align_signals(y_ort, y_phone)
            m2 = metrics(a2, b2)
            res[fi, ei, 1, :] = m2

            print(
                f"{ep} | {GROUP_NAMES[0]} | {m1[0]:.6e} | {m1[1]:.6e} | "
                f"{m1[2]:.6f} | {m1[3]:.6e} | {m1[4]:.3f}"
            )
            print(
                f"{ep} | {GROUP_NAMES[1]} | {m2[0]:.6e} | {m2[1]:.6e} | "
                f"{m2[2]:.6f} | {m2[3]:.6e} | {m2[4]:.3f}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, res)
    print(f"\n[OK] Résultats sauvegardés: {OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
