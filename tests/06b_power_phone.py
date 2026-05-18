"""
Mesure de consommation sur téléphone : baseline veille puis sweep 60 s
par backend (cpu / xnnpack / nnapi), échantillonnage courant/tension via adb.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import numpy as np

DEVICE_DIR = "/data/local/tmp/filtered_app"
MODEL_PATH = "./lowpass_rnn.onnx"
BINARY = "./filtered"
FC_HZ = 1000
INPUT_ON_DEVICE = "./power_input.wav"

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
NPY_OUT_DIR = REPO_ROOT / "tests" / "results" / "npy"
POWER_OUT = NPY_OUT_DIR / "power_results.npy"

WAV_BASENAME = "sweep_48k_mono.wav"

CURRENT_PATHS = [
    "/sys/class/power_supply/battery/current_now",
    "/sys/class/power_supply/battery/BatteryAverageCurrent",
]
VOLTAGE_PATH = "/sys/class/power_supply/battery/voltage_now"

BASELINE_SAMPLES = 10
BASELINE_INTERVAL_S = 1.0
RUN_SAMPLE_INTERVAL_S = 5.0

_CURRENT_READ_WARNED = False


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def read_current_ua() -> float | None:
    global _CURRENT_READ_WARNED
    for path in CURRENT_PATHS:
        proc = run_cmd(["adb", "shell", f"cat {path}"], check=False)
        if proc.returncode != 0:
            continue
        raw = (proc.stdout or "").strip()
        if not raw or raw in ("", "null"):
            continue
        try:
            return abs(float(raw))
        except ValueError:
            continue
    if not _CURRENT_READ_WARNED:
        print(
            "[WARN] Lecture du courant impossible (current_now et BatteryAverageCurrent). "
            "Échantillons concernés ignorés."
        )
        _CURRENT_READ_WARNED = True
    return None


def read_voltage_uv() -> float | None:
    proc = run_cmd(["adb", "shell", f"cat {VOLTAGE_PATH}"], check=False)
    if proc.returncode != 0:
        return None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def power_mw(current_ua: float, voltage_uv: float) -> float:
    return (current_ua / 1000.0) * (voltage_uv / 1_000_000.0)


def sample_power_mw() -> float | None:
    c = read_current_ua()
    v = read_voltage_uv()
    if c is None or v is None:
        return None
    return power_mw(c, v)


def parse_dumpsys_battery() -> tuple[int | None, float | None]:
    """Niveau (%) et tension depuis dumpsys (tension souvent en mV)."""
    proc = run_cmd(["adb", "shell", "dumpsys battery"], check=False)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    level_m = re.search(r"level:\s*(\d+)", text, re.IGNORECASE)
    volt_m = re.search(r"voltage:\s*(\d+)", text, re.IGNORECASE)
    level = int(level_m.group(1)) if level_m else None
    voltage_mv = float(volt_m.group(1)) if volt_m else None
    return level, voltage_mv


def filtered_shell_cmd(ep: str) -> str:
    return (
        f"cd {DEVICE_DIR} && "
        f"LD_LIBRARY_PATH=. {BINARY} --model {MODEL_PATH} --ep {ep} --fc {FC_HZ} "
        f"--input {INPUT_ON_DEVICE} --output ./power_output.wav 2>&1"
    )


def measure_baseline_mw() -> tuple[float, list[float]]:
    powers: list[float] = []
    for _ in range(BASELINE_SAMPLES):
        p = sample_power_mw()
        if p is not None:
            powers.append(p)
        time.sleep(BASELINE_INTERVAL_S)
    if not powers:
        return float("nan"), []
    return float(np.mean(np.asarray(powers, dtype=np.float64))), powers


def measure_run_mw_while_filtered(ep: str) -> tuple[list[float], str]:
    """
    Lance filtered en arrière-plan ; échantillonne toutes les RUN_SAMPLE_INTERVAL_S
    jusqu'à la fin du processus, puis une lecture après la fin.
    """
    shell_cmd = filtered_shell_cmd(ep)
    proc = subprocess.Popen(
        ["adb", "shell", shell_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    samples: list[float] = []
    while proc.poll() is None:
        p = sample_power_mw()
        if p is not None:
            samples.append(p)
        elapsed = 0.0
        while elapsed < RUN_SAMPLE_INTERVAL_S and proc.poll() is None:
            time.sleep(0.1)
            elapsed += 0.1

    out, _ = proc.communicate(timeout=30)
    log = out or ""

    p_end = sample_power_mw()
    if p_end is not None:
        samples.append(p_end)

    return samples, log


def main() -> None:
    NPY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = TEST_DATA_DIR / WAV_BASENAME
    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier manquant : {input_path} (exécuter tests/00_generate_signals.py)."
        )

    run_cmd(["adb", "get-state"])

    lvl0, v0 = parse_dumpsys_battery()
    print(f"[dumpsys battery] niveau={lvl0} %, voltage(dumpsys)={v0} mV (indicatif)")

    print("Mesure baseline (veille), 10 échantillons / 1 s…")
    baseline_mw, _baseline_raw = measure_baseline_mw()
    if np.isnan(baseline_mw):
        print("[WARN] Baseline sans échantillon valide (courant ou tension illisible).")

    run_cmd(["adb", "push", str(input_path), f"{DEVICE_DIR}/power_input.wav"])

    cpu_samples: list[float] = []
    xnnpack_samples: list[float] = []
    nnapi_samples: list[float] = []

    for ep, store in (
        ("cpu", "cpu"),
        ("xnnpack", "xnnpack"),
        ("nnapi", "nnapi"),
    ):
        print(f"\n=== Run filtered EP={ep} (sweep 60 s) + échantillons puissance ===")
        lvl, vmv = parse_dumpsys_battery()
        print(f"  [dumpsys] niveau={lvl} %, voltage={vmv} mV")
        samples, log = measure_run_mw_while_filtered(ep)
        if "[SUMMARY]" not in log and "SUMMARY" not in log.upper():
            print("  [WARN] Sortie filtered : résumé latence absent ou tronqué.")
        if store == "cpu":
            cpu_samples = samples
        elif store == "xnnpack":
            xnnpack_samples = samples
        else:
            nnapi_samples = samples
        print(f"  Échantillons valides : {len(samples)}")

    def mean_samples(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        return float(np.mean(np.asarray(xs, dtype=np.float64)))

    cpu_mw = mean_samples(cpu_samples)
    xnnpack_mw = mean_samples(xnnpack_samples)
    nnapi_mw = mean_samples(nnapi_samples)

    cpu_extra_mw = cpu_mw - baseline_mw if not np.isnan(cpu_mw) and not np.isnan(baseline_mw) else float("nan")
    xnnpack_extra_mw = (
        xnnpack_mw - baseline_mw if not np.isnan(xnnpack_mw) and not np.isnan(baseline_mw) else float("nan")
    )
    nnapi_extra_mw = (
        nnapi_mw - baseline_mw if not np.isnan(nnapi_mw) and not np.isnan(baseline_mw) else float("nan")
    )

    result = {
        "baseline_mw": float(baseline_mw),
        "cpu_mw": float(cpu_mw),
        "xnnpack_mw": float(xnnpack_mw),
        "nnapi_mw": float(nnapi_mw),
        "cpu_extra_mw": float(cpu_extra_mw),
        "xnnpack_extra_mw": float(xnnpack_extra_mw),
        "nnapi_extra_mw": float(nnapi_extra_mw),
        "cpu_samples": np.asarray(cpu_samples, dtype=np.float64),
        "xnnpack_samples": np.asarray(xnnpack_samples, dtype=np.float64),
        "nnapi_samples": np.asarray(nnapi_samples, dtype=np.float64),
    }
    np.save(POWER_OUT, np.array(result, dtype=object), allow_pickle=True)

    def fmt_mw(x: float) -> str:
        if np.isnan(x):
            return "nan"
        return f"{x:.1f}"

    def fmt_extra(x: float) -> str:
        if np.isnan(x):
            return "nan"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.1f} mW"

    print()
    print("=== Mesure de consommation énergétique (Xiaomi 13 Ultra) ===")
    print("Mode | Puissance moy (mW) | Surconsommation vs veille")
    print(f"Veille      | {fmt_mw(baseline_mw)} mW | -")
    print(f"CPU         | {fmt_mw(cpu_mw)} mW | {fmt_extra(cpu_extra_mw)}")
    print(f"XNNPACK     | {fmt_mw(xnnpack_mw)} mW | {fmt_extra(xnnpack_extra_mw)}")
    print(f"NNAPI/GPU   | {fmt_mw(nnapi_mw)} mW | {fmt_extra(nnapi_extra_mw)}")
    print(f"Données sauvegardées : {POWER_OUT.as_posix()}")


if __name__ == "__main__":
    main()
