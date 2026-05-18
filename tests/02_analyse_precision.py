import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "tests" / "data"
RESULTS_RAW = REPO_ROOT / "tests" / "results" / "raw"
RESULTS_NPY = REPO_ROOT / "tests" / "results" / "npy"

INPUT_WAV = TEST_DATA / "sweep_48k_mono.wav"
PHONE_WAV = {
    500: RESULTS_RAW / "sweep_fc500_cpu.wav",
    1000: RESULTS_RAW / "sweep_fc1000_cpu.wav",
    3000: RESULTS_RAW / "sweep_fc3000_cpu.wav",
}
ONNX_PATH = REPO_ROOT / "workspaces" / "gru_model" / "exported_onnx_models" / "lowpass_rnn.onnx"
PT_PATH = REPO_ROOT / "workspaces" / "gru_model" / "order1" / "butter_lowpass.pt"

SAMPLE_RATE = 48_000
BUFFER_SIZE = 96
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FC_LIST = [500, 1000, 3000]


def read_wav_mono(path: Path) -> np.ndarray:
    from scipy.io import wavfile

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


def fc_norm(fc_hz: float) -> float:
    return np.log2(2.0 * fc_hz / SAMPLE_RATE) / np.log2(SAMPLE_RATE)


def trim_to_buffer(x: np.ndarray, buffer_size: int) -> np.ndarray:
    n = (len(x) // buffer_size) * buffer_size
    return x[:n]


def align_signals(a: np.ndarray, b: np.ndarray, max_shift: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-n + 1, n)
    idx = np.argmax(corr)
    lag = int(lags[idx])
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


def run_pytorch_inference(x: np.ndarray, fc_hz: int) -> np.ndarray:
    import torch

    sys.path.insert(0, str(REPO_ROOT / "workspaces" / "gru_model"))
    from model import LowpassRNN

    model = LowpassRNN(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, conditioned=True)
    state = torch.load(str(PT_PATH), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    hidden = None
    out_blocks = []
    fcn = fc_norm(fc_hz)
    with torch.no_grad():
        for i in range(0, len(x), BUFFER_SIZE):
            xb = x[i : i + BUFFER_SIZE]
            x_audio = torch.from_numpy(xb).float()
            x_fc = torch.full((BUFFER_SIZE,), float(fcn), dtype=torch.float32)
            xin = torch.stack([x_audio, x_fc], dim=-1).unsqueeze(0)  # (1, B, 2)
            y, hidden = model(xin, hidden)
            out_blocks.append(y.squeeze(-1).squeeze(0).cpu().numpy())
    return np.concatenate(out_blocks, axis=0).astype(np.float32)


def run_ort_pc_inference(x: np.ndarray, fc_hz: int) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    hidden = np.zeros((NUM_LAYERS, 1, HIDDEN_SIZE), dtype=np.float32)
    out_blocks = []
    fcn = np.float32(fc_norm(fc_hz))
    for i in range(0, len(x), BUFFER_SIZE):
        xb = x[i : i + BUFFER_SIZE].astype(np.float32)
        fc_channel = np.full((BUFFER_SIZE,), fcn, dtype=np.float32)
        x_in = np.stack([xb, fc_channel], axis=-1)[None, :, :]  # (1, B, 2)
        outputs = sess.run(
            ["output", "hidden_out"],
            {"x": x_in, "hidden_in": hidden},
        )
        yb, hidden = outputs[0], outputs[1]
        out_blocks.append(yb.reshape(-1))
    return np.concatenate(out_blocks, axis=0).astype(np.float32)


def main() -> None:
    RESULTS_NPY.mkdir(parents=True, exist_ok=True)

    # Check required files first.
    required = [INPUT_WAV, ONNX_PATH, PT_PATH] + [PHONE_WAV[fc] for fc in FC_LIST]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    x = trim_to_buffer(read_wav_mono(INPUT_WAV), BUFFER_SIZE)
    metrics_array = np.zeros((3, 3, 5), dtype=np.float64)  # fc x group x metric
    waveforms = {}
    diffs = {}

    print("=== Analyse de précision : PyTorch vs ORT-PC vs Téléphone ===")
    group_names = ["pytorch_vs_ort_pc", "pytorch_vs_phone", "ort_pc_vs_phone"]

    for fi, fc in enumerate(FC_LIST):
        y_pt = run_pytorch_inference(x, fc)
        y_ort = run_ort_pc_inference(x, fc)
        y_phone = trim_to_buffer(read_wav_mono(PHONE_WAV[fc]), BUFFER_SIZE)

        a1, b1 = align_signals(y_pt, y_ort)
        a2, b2 = align_signals(y_pt, y_phone)
        a3, b3 = align_signals(y_ort, y_phone)

        m1 = metrics(a1, b1)
        m2 = metrics(a2, b2)
        m3 = metrics(a3, b3)
        metrics_array[fi, 0, :] = m1
        metrics_array[fi, 1, :] = m2
        metrics_array[fi, 2, :] = m3

        diffs[f"diff_pytorch_phone_fc{fc}"] = (a2 - b2).astype(np.float32)
        waveforms[f"fc_{fc}"] = {
            "y_pytorch": y_pt[: 2 * SAMPLE_RATE],
            "y_ort_pc": y_ort[: 2 * SAMPLE_RATE],
            "y_phone": y_phone[: 2 * SAMPLE_RATE],
        }

        print(f"\nfc = {fc} Hz")
        print("groupe | MAE | MSE | Pearson | Max_Error | PSNR")
        for gi, gname in enumerate(group_names):
            vals = metrics_array[fi, gi]
            print(
                f"{gname} | {vals[0]:.6e} | {vals[1]:.6e} | {vals[2]:.6f} | "
                f"{vals[3]:.6e} | {vals[4]:.3f}"
            )

    np.save(RESULTS_NPY / "precision_results.npy", metrics_array)
    waveforms["diffs"] = diffs
    np.save(RESULTS_NPY / "precision_waveforms.npy", waveforms, allow_pickle=True)


if __name__ == "__main__":
    main()
