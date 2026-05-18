"""
Analyse des latences par buffer (logs Xiaomi 13 Ultra).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "tests" / "results" / "figures"
NPY_OUT_DIR = REPO_ROOT / "tests" / "results" / "npy"
OUT_NPY = NPY_OUT_DIR / "block_latency_xiaomi.npy"

PATH_CPU = REPO_ROOT / "latency_block_cpu_xiaomi.txt"
PATH_XNN = REPO_ROOT / "latency_block_xnnpack_xiaomi.txt"
PATH_NNAPI = REPO_ROOT / "latency_block_nnapi_xiaomi.txt"

FIG_TIMELINE = FIG_DIR / "fig11_xiaomi_latency_timeline.png"
FIG13_BACKEND = FIG_DIR / "fig13_xiaomi_latency_backend.png"

PATTERN = re.compile(r"infer_ms=([0-9.]+)")

BUDGET_MS = 5.33


def use_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass


def read_text_flexible(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def parse_infer_ms_lines(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    vals: list[float] = []
    for line in read_text_flexible(path).splitlines():
        m = PATTERN.search(line)
        if m:
            vals.append(float(m.group(1)))
    if not vals:
        raise ValueError(f"Aucune latence trouvée dans {path}")
    return np.asarray(vals, dtype=np.float64)


def summarize(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "frames": int(arr.size),
    }


def plot_timeline(
    cpu: np.ndarray,
    xnn: np.ndarray,
    nnapi: np.ndarray,
) -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    n = len(cpu)
    x = np.arange(n)
    mu_cpu = float(np.mean(cpu))
    mu_xnn = float(np.mean(xnn))
    mu_nnapi = float(np.mean(nnapi))

    ax.plot(x, cpu, color="#1f77b4", linewidth=0.7, alpha=0.9, label="CPU")
    ax.plot(x, xnn, color="#ff7f0e", linewidth=0.7, alpha=0.9, label="XNNPACK")
    ax.plot(x, nnapi, color="#2ca02c", linewidth=0.7, alpha=0.9, label="NNAPI-GPU")

    ax.axhline(mu_cpu, color="#1f77b4", linestyle="--", linewidth=1.1, alpha=0.9, label=f"CPU μ={mu_cpu:.2f} ms")
    ax.axhline(mu_xnn, color="#ff7f0e", linestyle="--", linewidth=1.1, alpha=0.9, label=f"XNNPACK μ={mu_xnn:.2f} ms")
    ax.axhline(
        mu_nnapi, color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.9, label=f"NNAPI μ={mu_nnapi:.2f} ms"
    )

    ax.axhline(BUDGET_MS, color="red", linestyle="--", linewidth=1.2, zorder=5)
    ax.text(
        0.99,
        0.08,
        "Budget temps réel : 5.33 ms",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="red",
    )

    ax.set_ylim(0, 10)
    ax.set_xlabel("Numéro de buffer")
    ax.set_ylabel("Latence (ms)")
    ax.set_title("Latence d'inférence par buffer — Xiaomi 13 Ultra (Snapdragon 8 Gen 2)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_TIMELINE, dpi=150)
    plt.close(fig)


def plot_fig13(data: dict, out_path: Path) -> None:
    """Moyenne et P95 par backend (Xiaomi uniquement), fichier fig13."""
    eps = ["CPU", "XNNPACK", "NNAPI-GPU"]
    means = [data["cpu"]["mean"], data["xnnpack"]["mean"], data["nnapi"]["mean"]]
    p95s = [data["cpu"]["p95"], data["xnnpack"]["p95"], data["nnapi"]["p95"]]

    x = np.arange(len(eps))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(
        x - w / 2,
        means,
        width=w,
        label="Latence moyenne",
        color=["#1f77b4", "#1f9b4e", "#1f4eb4"],
    )
    b2 = ax.bar(
        x + w / 2,
        p95s,
        width=w,
        label="Latence P95",
        color=["#6baed6", "#74c476", "#6b9fd6"],
    )
    ax.axhline(
        5.33,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Budget temps réel : 5.33 ms",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(eps)
    ax.set_ylabel("Latence (ms)")
    ax.set_title("Comparaison de latence par backend — Xiaomi 13 Ultra (Snapdragon 8 Gen 2)")
    ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cpu = parse_infer_ms_lines(PATH_CPU)
    xnn = parse_infer_ms_lines(PATH_XNN)
    nnapi = parse_infer_ms_lines(PATH_NNAPI)

    if not (len(cpu) == len(xnn) == len(nnapi)):
        print(
            f"[WARN] Longueurs différentes : CPU {len(cpu)}, XNNPACK {len(xnn)}, NNAPI {len(nnapi)} — timeline tronquée au min."
        )
        m = min(len(cpu), len(xnn), len(nnapi))
        cpu, xnn, nnapi = cpu[:m], xnn[:m], nnapi[:m]

    s_cpu = summarize(cpu)
    s_xnn = summarize(xnn)
    s_nnapi = summarize(nnapi)

    plot_timeline(cpu, xnn, nnapi)
    plot_fig13(
        {"cpu": s_cpu, "xnnpack": s_xnn, "nnapi": s_nnapi},
        FIG13_BACKEND,
    )

    payload: dict = {
        "device": "Xiaomi 13 Ultra (Snapdragon 8 Gen 2)",
        "infer_ms_arrays": {
            "cpu": cpu,
            "xnnpack": xnn,
            "nnapi": nnapi,
        },
        "summary": {
            "cpu": s_cpu,
            "xnnpack": s_xnn,
            "nnapi": s_nnapi,
        },
        "paths": {
            "cpu": str(PATH_CPU),
            "xnnpack": str(PATH_XNN),
            "nnapi": str(PATH_NNAPI),
        },
    }

    NPY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, payload, allow_pickle=True)

    print("=== Latence par buffer — Xiaomi 13 Ultra ===")
    print(
        f"CPU     : mean={s_cpu['mean']:.2f} ms | P95={s_cpu['p95']:.2f} ms | "
        f"max={s_cpu['max']:.2f} ms | frames={s_cpu['frames']}"
    )
    print(
        f"XNNPACK : mean={s_xnn['mean']:.2f} ms | P95={s_xnn['p95']:.2f} ms | "
        f"max={s_xnn['max']:.2f} ms | frames={s_xnn['frames']}"
    )
    print(
        f"NNAPI   : mean={s_nnapi['mean']:.2f} ms | P95={s_nnapi['p95']:.2f} ms | "
        f"max={s_nnapi['max']:.2f} ms | frames={s_nnapi['frames']}"
    )
    print(f"Figures : {FIG_TIMELINE.name}, {FIG13_BACKEND.name}")
    print(f"NPY     : {OUT_NPY}")


if __name__ == "__main__":
    main()
