"""
Step 6: long-run stability on device — 30× file-mode runs on the same WAV,
recording latency drift (mean / p95 per run from [SUMMARY]).
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import numpy as np

# Same layout as tests/01_run_phone_tests.py
DEVICE_DIR = "/data/local/tmp/filtered_app"
MODEL_PATH = "./lowpass_rnn.onnx"
BINARY = "./filtered"

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
NPY_OUT_DIR = REPO_ROOT / "tests" / "results" / "npy"
STABILITY_OUT = NPY_OUT_DIR / "stability_results.npy"

TOTAL_RUNS = 30
WAV_BASENAME = "sweep_48k_mono.wav"
FC_HZ = 1000
EP = "cpu"
PROGRESS_EVERY = 5


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def parse_summary(text: str) -> dict | None:
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


def single_run() -> tuple[dict | None, str]:
    """Run filtered once on device (expects stability_input.wav already pushed)."""
    shell_cmd = (
        f"cd {DEVICE_DIR} && "
        f"LD_LIBRARY_PATH=. {BINARY} --model {MODEL_PATH} --ep {EP} --fc {FC_HZ} "
        f"--input ./stability_input.wav --output ./stability_output.wav 2>&1"
    )
    proc = run_cmd(["adb", "shell", shell_cmd], check=False)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    summary = parse_summary(combined)
    return summary, combined


def main() -> None:
    NPY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = TEST_DATA_DIR / WAV_BASENAME
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}; run tests/00_generate_signals.py first."
        )

    run_cmd(["adb", "get-state"])

    t0 = time.monotonic()

    timestamps_min: list[float] = []
    mean_latency_per_run: list[float] = []
    p95_latency_per_run: list[float] = []
    failed_runs = 0

    for i in range(TOTAL_RUNS):
        print(f"[{i + 1}/{TOTAL_RUNS}] stability run…")
        try:
            run_cmd(["adb", "push", str(input_path), f"{DEVICE_DIR}/stability_input.wav"])
            summary, _ = single_run()
            elapsed_min = (time.monotonic() - t0) / 60.0
            timestamps_min.append(float(elapsed_min))
            if summary is None:
                failed_runs += 1
                mean_latency_per_run.append(float("nan"))
                p95_latency_per_run.append(float("nan"))
                print("  [WARN] SUMMARY introuvable pour ce run.")
            else:
                mean_latency_per_run.append(float(summary["mean_ms"]))
                p95_latency_per_run.append(float(summary["p95_ms"]))
                print(
                    f"  [SUMMARY] blocks={summary['blocks']} mean_ms={summary['mean_ms']:.4f} "
                    f"p95_ms={summary['p95_ms']:.4f} (t={elapsed_min:.2f} min)"
                )
        except Exception as e:
            failed_runs += 1
            elapsed_min = (time.monotonic() - t0) / 60.0
            timestamps_min.append(float(elapsed_min))
            mean_latency_per_run.append(float("nan"))
            p95_latency_per_run.append(float("nan"))
            print(f"  [ERROR] {e}")

        if (i + 1) % PROGRESS_EVERY == 0:
            print(
                f"--- Progrès : {i + 1}/{TOTAL_RUNS} runs terminés "
                f"({(i + 1) * 1} min d’audio traité, horloge ~{(time.monotonic() - t0) / 60.0:.1f} min) ---"
            )

    mean_arr = np.asarray(mean_latency_per_run, dtype=np.float64)
    early_mean = float(np.nanmean(mean_arr[:5])) if mean_arr.size >= 5 else float("nan")
    late_mean = float(np.nanmean(mean_arr[-5:])) if mean_arr.size >= 5 else float("nan")
    if np.isnan(early_mean) or early_mean == 0.0:
        drift_percent = float("nan")
    else:
        drift_percent = (late_mean - early_mean) / early_mean * 100.0

    std_latency = float(np.nanstd(mean_arr))

    result = {
        "timestamps": np.asarray(timestamps_min, dtype=np.float64),
        "mean_latency_per_run": mean_arr,
        "p95_latency_per_run": np.asarray(p95_latency_per_run, dtype=np.float64),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "drift_percent": drift_percent,
        "std_latency": std_latency,
        "total_runs": TOTAL_RUNS,
        "failed_runs": int(failed_runs),
    }
    np.save(STABILITY_OUT, np.array(result, dtype=object), allow_pickle=True)

    print()
    print("=== Test de stabilité 30 minutes (Xiaomi 13 Ultra) ===")
    print(f"Nombre de runs : {TOTAL_RUNS}")
    print(f"Latence initiale (5 premiers runs) : {early_mean:.4f} ms")
    print(f"Latence finale (5 derniers runs) : {late_mean:.4f} ms")
    print(f"Dérive : {drift_percent:.2f}%")
    print(f"Écart-type : {std_latency:.4f} ms")
    print(f"Runs échoués : {failed_runs}")
    print(f"Résultats sauvegardés : {STABILITY_OUT.as_posix()}")


if __name__ == "__main__":
    main()
