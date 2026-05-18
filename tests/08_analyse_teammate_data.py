"""
Analyse des mesures puissance / latence (Pixel 9a, données camarade) et
génération des figures fig8–fig10.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TEAMMATE_DIR = REPO_ROOT / "tests" / "data" / "teammate"
FIG_DIR = REPO_ROOT / "tests" / "results" / "figures"
NPY_OUT_DIR = REPO_ROOT / "tests" / "results" / "npy"
STATS_OUT = NPY_OUT_DIR / "teammate_stats.npy"

# Sources puissance (W), une ligne = un échantillon (~500 ms)
PATH_POWER_CPU = TEAMMATE_DIR / "power_samplesCPU_Butter_Pixel9a.txt"
PATH_POWER_XNN = TEAMMATE_DIR / "power_samplesXnnpack_Butter_Pixel9a.txt"
PATH_POWER_WGPU = TEAMMATE_DIR / "power_samplesWebGpu_Butter_Pixel9a.txt"
PATH_POWER_NO_AIR = TEAMMATE_DIR / "power_samples_Pixel9a_butter.txt"
PATH_POWER_AIR = TEAMMATE_DIR / "power_samples_mode_avion_Pixel9a_butter.txt"

PATH_LAT_CPU = TEAMMATE_DIR / "latency_CPUExec_ProviderButter.npy"
PATH_LAT_XNN = TEAMMATE_DIR / "latency_xnnpackButter.npy"
PATH_LAT_WGPU = TEAMMATE_DIR / "latency_WebGpuButter.npy"

# Latences WebGPU > seuil considérées comme cold start / aberrations pour stats
WEBGPU_COLDSTART_THRESHOLD_MS = 100.0

FIG_A = FIG_DIR / "fig8_pixel9a_power_ep.png"
FIG_B = FIG_DIR / "fig9_pixel9a_power_airplane.png"
FIG_C = FIG_DIR / "fig10_pixel9a_latency_ep.png"


def use_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass


def load_power_txt(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    # Exports Excel/Android parfois en UTF-16 LE (BOM ff fe)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")
    values: list[float] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        values.append(float(s.replace(",", ".")))
    return np.asarray(values, dtype=np.float64)


def load_latency_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=True)
    return np.asarray(raw, dtype=np.float64).ravel()


def mean_p95(arr: np.ndarray) -> tuple[float, float]:
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.percentile(arr, 95))


def require_inputs() -> None:
    paths = [
        PATH_POWER_CPU,
        PATH_POWER_XNN,
        PATH_POWER_WGPU,
        PATH_POWER_NO_AIR,
        PATH_POWER_AIR,
        PATH_LAT_CPU,
        PATH_LAT_XNN,
        PATH_LAT_WGPU,
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers camarade manquants (copier dans tests/data/teammate/) :\n"
            + "\n".join(missing)
        )


def plot_fig_a(cpu: np.ndarray, xnn: np.ndarray, wgpu: np.ndarray) -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"CPU": "#1f77b4", "XNNPACK": "#ff7f0e", "WebGPU": "#2ca02c"}
    for name, arr in [("CPU", cpu), ("XNNPACK", xnn), ("WebGPU", wgpu)]:
        x = np.arange(len(arr))
        ax.plot(x, arr, label=f"{name}", color=colors[name], linewidth=1.2, alpha=0.85)
        mu = float(np.mean(arr))
        ax.axhline(mu, color=colors[name], linestyle="--", linewidth=1.0, alpha=0.85, label=f"{name} (moy. {mu:.3f} W)")
    ax.set_xlabel("Index d'échantillon (~500 ms par point)")
    ax.set_ylabel("Puissance (W)")
    ax.set_title("Consommation énergétique par backend — Pixel 9a (Google Tensor G4)")
    ax.legend(loc="upper right", fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_A, dpi=150)
    plt.close(fig)


def plot_fig_b(no_air: np.ndarray, air: np.ndarray) -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(no_air)), no_air, label="Sans mode avion (CPU Butter)", color="#1f77b4", linewidth=1.2)
    ax.plot(np.arange(len(air)), air, label="Mode avion (CPU Butter)", color="#d62728", linewidth=1.2)
    m0 = float(np.mean(no_air))
    m1 = float(np.mean(air))
    ax.axhline(m0, color="#1f77b4", linestyle="--", alpha=0.85, label=f"Sans mode avion (moy. {m0:.3f} W)")
    ax.axhline(m1, color="#d62728", linestyle="--", alpha=0.85, label=f"Mode avion (moy. {m1:.3f} W)")
    ax.set_xlabel("Index d'échantillon (~500 ms par point)")
    ax.set_ylabel("Puissance (W)")
    ax.set_title("Impact du mode avion sur la consommation — Pixel 9a")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_B, dpi=150)
    plt.close(fig)


def plot_fig_c(lat_cpu: np.ndarray, lat_xnn: np.ndarray, lat_wgpu: np.ndarray, mean_wgpu_clean: float) -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"CPU": "#1f77b4", "XNNPACK": "#ff7f0e", "WebGPU": "#2ca02c"}
    mu_cpu = float(np.mean(lat_cpu))
    mu_xnn = float(np.mean(lat_xnn))
    ax.plot(np.arange(len(lat_cpu)), lat_cpu, label="CPU", color=colors["CPU"], linewidth=0.8, alpha=0.9)
    ax.plot(np.arange(len(lat_xnn)), lat_xnn, label="XNNPACK", color=colors["XNNPACK"], linewidth=0.8, alpha=0.9)
    ax.plot(np.arange(len(lat_wgpu)), lat_wgpu, label="WebGPU", color=colors["WebGPU"], linewidth=0.8, alpha=0.9)
    ax.axhline(mu_cpu, color=colors["CPU"], linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(mu_xnn, color=colors["XNNPACK"], linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(mean_wgpu_clean, color=colors["WebGPU"], linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Index de buffer")
    ax.set_ylabel("Latence (ms)")
    ax.set_title("Latence d'inférence par buffer — Pixel 9a")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_C, dpi=150)
    plt.close(fig)


def main() -> None:
    require_inputs()

    p_cpu = load_power_txt(PATH_POWER_CPU)
    p_xnn = load_power_txt(PATH_POWER_XNN)
    p_wgpu = load_power_txt(PATH_POWER_WGPU)
    p_no_air = load_power_txt(PATH_POWER_NO_AIR)
    p_air = load_power_txt(PATH_POWER_AIR)

    lat_cpu = load_latency_npy(PATH_LAT_CPU)
    lat_xnn = load_latency_npy(PATH_LAT_XNN)
    lat_wgpu = load_latency_npy(PATH_LAT_WGPU)

    lat_wgpu_clean = lat_wgpu[lat_wgpu <= WEBGPU_COLDSTART_THRESHOLD_MS]
    m_wgpu_clean, p95_wgpu_clean = mean_p95(lat_wgpu_clean)

    m_cpu, p95_cpu = mean_p95(lat_cpu)
    m_xnn, p95_xnn = mean_p95(lat_xnn)

    mu_p_cpu = float(np.mean(p_cpu))
    mu_p_xnn = float(np.mean(p_xnn))
    mu_p_wgpu = float(np.mean(p_wgpu))
    mu_air = float(np.mean(p_air))
    mu_no_air = float(np.mean(p_no_air))

    plot_fig_a(p_cpu, p_xnn, p_wgpu)
    plot_fig_b(p_no_air, p_air)
    plot_fig_c(lat_cpu, lat_xnn, lat_wgpu, m_wgpu_clean)

    stats: dict = {
        "device": "Pixel 9a (Google Tensor G4)",
        "power_mean_w": {
            "cpu": mu_p_cpu,
            "xnnpack": mu_p_xnn,
            "webgpu": mu_p_wgpu,
        },
        "power_airplane_mode_mean_w": mu_air,
        "power_no_airplane_mode_mean_w": mu_no_air,
        "latency_ms": {
            "cpu": {"mean": m_cpu, "p95": p95_cpu},
            "xnnpack": {"mean": m_xnn, "p95": p95_xnn},
            "webgpu_all": {"mean": float(np.mean(lat_wgpu)), "p95": float(np.percentile(lat_wgpu, 95))},
            "webgpu_no_cold_start": {
                "mean": m_wgpu_clean,
                "p95": p95_wgpu_clean,
                "threshold_ms": WEBGPU_COLDSTART_THRESHOLD_MS,
                "n_excluded": int(lat_wgpu.size - lat_wgpu_clean.size),
            },
        },
    }

    NPY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(STATS_OUT, stats, allow_pickle=True)

    print("=== Statistiques Pixel 9a (Google Tensor G4) ===")
    print(f"Puissance moyenne CPU : {mu_p_cpu:.3f} W")
    print(f"Puissance moyenne XNNPACK : {mu_p_xnn:.3f} W")
    print(f"Puissance moyenne WebGPU : {mu_p_wgpu:.3f} W")
    print(f"Mode avion : {mu_air:.3f} W | Sans mode avion : {mu_no_air:.3f} W")
    print(f"Latence moyenne CPU : {m_cpu:.2f} ms | P95 : {p95_cpu:.2f} ms")
    print(f"Latence moyenne XNNPACK : {m_xnn:.2f} ms | P95 : {p95_xnn:.2f} ms")
    print(
        f"Latence moyenne WebGPU (hors cold start) : {m_wgpu_clean:.2f} ms | P95 : {p95_wgpu_clean:.2f} ms"
    )
    print(f"Figures : {FIG_A.name}, {FIG_B.name}, {FIG_C.name}")
    print(f"Stats : {STATS_OUT}")


if __name__ == "__main__":
    main()
