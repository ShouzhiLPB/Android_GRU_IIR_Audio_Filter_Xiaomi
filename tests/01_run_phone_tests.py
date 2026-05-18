import re
import subprocess
from pathlib import Path

import numpy as np

# Configurable paths
DEVICE_DIR = "/data/local/tmp/filtered_app"
MODEL_PATH = "./lowpass_rnn.onnx"
BINARY = "./filtered"
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
RESULTS_RAW_DIR = REPO_ROOT / "tests" / "results" / "raw"
LATENCY_LOG_DIR = REPO_ROOT / "tests" / "results" / "npy"
LATENCY_OUT = LATENCY_LOG_DIR / "all_latency_raw.npy"


TEST_MATRIX = [
    # CPU
    ("sweep_48k_mono", 500, "cpu", "sweep_fc500_cpu.wav"),
    ("sweep_48k_mono", 1000, "cpu", "sweep_fc1000_cpu.wav"),
    ("sweep_48k_mono", 3000, "cpu", "sweep_fc3000_cpu.wav"),
    ("noise_48k_mono", 1000, "cpu", "noise_fc1000_cpu.wav"),
    ("sine_48k_mono", 500, "cpu", "sine_fc500_cpu.wav"),
    ("sine_48k_mono", 1000, "cpu", "sine_fc1000_cpu.wav"),
    ("sine_48k_mono", 3000, "cpu", "sine_fc3000_cpu.wav"),
    ("impulse_48k_mono", 1000, "cpu", "robustness_impulse_cpu.wav"),
    ("dc_offset_48k_mono", 1000, "cpu", "robustness_dc_cpu.wav"),
    ("loud_48k_mono", 1000, "cpu", "robustness_loud_cpu.wav"),
    ("quiet_48k_mono", 1000, "cpu", "robustness_quiet_cpu.wav"),
    # XNNPACK
    ("sweep_48k_mono", 500, "xnnpack", "sweep_fc500_xnnpack.wav"),
    ("sweep_48k_mono", 1000, "xnnpack", "sweep_fc1000_xnnpack.wav"),
    ("sweep_48k_mono", 3000, "xnnpack", "sweep_fc3000_xnnpack.wav"),
    ("noise_48k_mono", 1000, "xnnpack", "noise_fc1000_xnnpack.wav"),
    ("sine_48k_mono", 500, "xnnpack", "sine_fc500_xnnpack.wav"),
    ("sine_48k_mono", 1000, "xnnpack", "sine_fc1000_xnnpack.wav"),
    ("sine_48k_mono", 3000, "xnnpack", "sine_fc3000_xnnpack.wav"),
    ("impulse_48k_mono", 1000, "xnnpack", "robustness_impulse_xnnpack.wav"),
    ("dc_offset_48k_mono", 1000, "xnnpack", "robustness_dc_xnnpack.wav"),
    ("loud_48k_mono", 1000, "xnnpack", "robustness_loud_xnnpack.wav"),
    ("quiet_48k_mono", 1000, "xnnpack", "robustness_quiet_xnnpack.wav"),
    # NNAPI
    ("sweep_48k_mono", 500, "nnapi", "sweep_fc500_nnapi.wav"),
    ("sweep_48k_mono", 1000, "nnapi", "sweep_fc1000_nnapi.wav"),
    ("sweep_48k_mono", 3000, "nnapi", "sweep_fc3000_nnapi.wav"),
    ("noise_48k_mono", 1000, "nnapi", "noise_fc1000_nnapi.wav"),
    ("sine_48k_mono", 500, "nnapi", "sine_fc500_nnapi.wav"),
    ("sine_48k_mono", 1000, "nnapi", "sine_fc1000_nnapi.wav"),
    ("sine_48k_mono", 3000, "nnapi", "sine_fc3000_nnapi.wav"),
    ("impulse_48k_mono", 1000, "nnapi", "robustness_impulse_nnapi.wav"),
    ("dc_offset_48k_mono", 1000, "nnapi", "robustness_dc_nnapi.wav"),
    ("loud_48k_mono", 1000, "nnapi", "robustness_loud_nnapi.wav"),
    ("quiet_48k_mono", 1000, "nnapi", "robustness_quiet_nnapi.wav"),
]


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def parse_summary(text: str) -> dict | None:
    # Expected pattern from filtered output:
    # [SUMMARY] blocks=... min_ms=... mean_ms=... max_ms=... p95_ms=...
    match = re.search(
        r"\[SUMMARY\].*?blocks=(\d+).*?min_ms=([0-9.]+).*?mean_ms=([0-9.]+).*?max_ms=([0-9.]+).*?p95_ms=([0-9.]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return {
        "blocks": int(match.group(1)),
        "min_ms": float(match.group(2)),
        "mean_ms": float(match.group(3)),
        "max_ms": float(match.group(4)),
        "p95_ms": float(match.group(5)),
    }


def main() -> None:
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    LATENCY_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Quick device availability check.
    run_cmd(["adb", "get-state"])

    all_latency_rows: list[dict] = []
    pulled_ok = 0
    total = len(TEST_MATRIX)

    for idx, (input_file, fc_hz, ep, output_name) in enumerate(TEST_MATRIX, start=1):
        test_name = f"{input_file}_fc{fc_hz}_{ep}"
        print(f"[{idx}/{total}] Running {test_name}")

        input_path = TEST_DATA_DIR / f"{input_file}.wav"
        if not input_path.exists():
            print(f"  [WARN] Missing input file: {input_path}")
            all_latency_rows.append(
                {
                    "test_name": test_name,
                    "input_file": input_file,
                    "fc_hz": fc_hz,
                    "ep": ep,
                    "blocks": None,
                    "min_ms": None,
                    "mean_ms": None,
                    "max_ms": None,
                    "p95_ms": None,
                }
            )
            continue

        try:
            run_cmd(["adb", "push", str(input_path), f"{DEVICE_DIR}/input_test.wav"])

            shell_cmd = (
                f"cd {DEVICE_DIR} && "
                f"LD_LIBRARY_PATH=. {BINARY} --model {MODEL_PATH} --ep {ep} --fc {fc_hz} "
                f"--input ./input_test.wav --output ./output_test.wav 2>&1"
            )
            proc = run_cmd(["adb", "shell", shell_cmd], check=False)
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

            summary = parse_summary(combined)
            if summary is None:
                print("  [WARN] SUMMARY not found for this run.")
                row = {
                    "test_name": test_name,
                    "input_file": input_file,
                    "fc_hz": fc_hz,
                    "ep": ep,
                    "blocks": None,
                    "min_ms": None,
                    "mean_ms": None,
                    "max_ms": None,
                    "p95_ms": None,
                }
            else:
                print(
                    "  [SUMMARY] "
                    f"blocks={summary['blocks']} min_ms={summary['min_ms']:.4f} "
                    f"mean_ms={summary['mean_ms']:.4f} max_ms={summary['max_ms']:.4f} "
                    f"p95_ms={summary['p95_ms']:.4f}"
                )
                row = {
                    "test_name": test_name,
                    "input_file": input_file,
                    "fc_hz": fc_hz,
                    "ep": ep,
                    **summary,
                }

            all_latency_rows.append(row)

            out_local = RESULTS_RAW_DIR / output_name
            pull_proc = run_cmd(
                ["adb", "pull", f"{DEVICE_DIR}/output_test.wav", str(out_local)],
                check=False,
            )
            if pull_proc.returncode == 0:
                pulled_ok += 1
            else:
                print("  [WARN] adb pull failed for output_test.wav")

        except Exception as e:
            print(f"  [ERROR] {e}")
            all_latency_rows.append(
                {
                    "test_name": test_name,
                    "input_file": input_file,
                    "fc_hz": fc_hz,
                    "ep": ep,
                    "blocks": None,
                    "min_ms": None,
                    "mean_ms": None,
                    "max_ms": None,
                    "p95_ms": None,
                }
            )

    np.save(LATENCY_OUT, np.array(all_latency_rows, dtype=object), allow_pickle=True)

    print("=== Tous les tests téléphone terminés ===")
    print(f"Fichiers audio récupérés : {pulled_ok} / {total}")
    print(f"Données de latence sauvegardées : {LATENCY_OUT.as_posix()}")


if __name__ == "__main__":
    main()
