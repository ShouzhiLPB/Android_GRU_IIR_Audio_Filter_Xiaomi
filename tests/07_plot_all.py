"""
Génère les 7 figures de synthèse à partir des fichiers .npy du dossier results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NPY_DIR = REPO_ROOT / "tests" / "results" / "npy"
FIG_DIR = REPO_ROOT / "tests" / "results" / "figures"

SAMPLE_RATE = 48_000
WAVE_PREVIEW_S = 0.05
FRAME_BUDGET_MS = 5.33

PATHS = {
    "precision_results": NPY_DIR / "precision_results.npy",
    "precision_waveforms": NPY_DIR / "precision_waveforms.npy",
    "precision_results_ep": NPY_DIR / "precision_results_ep.npy",
    "spectrum_results": NPY_DIR / "spectrum_results.npy",
    "spectrum_raw": NPY_DIR / "spectrum_raw.npy",
    "latency_analysis": NPY_DIR / "latency_analysis.npy",
    "robustness_results": NPY_DIR / "robustness_results.npy",
    "stability_results": NPY_DIR / "stability_results.npy",
}

FC_LIST = [500, 1000, 3000]
EP_LIST = ["cpu", "xnnpack", "nnapi"]
EP_LABEL = {"cpu": "CPU", "xnnpack": "XNNPACK", "nnapi": "NNAPI"}


def use_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass


def require_files() -> None:
    missing = [str(p) for k, p in PATHS.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("Fichiers manquants :\n" + "\n".join(missing))


def validate_aux_npy() -> None:
    pr = np.load(PATHS["precision_results"])
    if pr.shape != (3, 3, 5):
        raise ValueError(f"precision_results.npy : forme attendue (3,3,5), obtenue {pr.shape}")
    np.load(PATHS["spectrum_results"], allow_pickle=True)


def load_waveforms_dict(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        data = obj.item()
    else:
        data = obj
    if not isinstance(data, dict):
        raise TypeError(f"precision_waveforms.npy inattendu : {type(data)}")
    return data


def load_stability_dict(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    data = arr.item() if getattr(arr, "ndim", 0) == 0 else arr.flat[0]
    if not isinstance(data, dict):
        raise TypeError(f"stability_results.npy inattendu : {type(data)}")
    return data


def load_latency_dict(path: Path) -> dict:
    arr = np.load(path, allow_pickle=True)
    data = arr.item() if getattr(arr, "ndim", 0) == 0 else arr.flat[0]
    if not isinstance(data, dict):
        raise TypeError(f"latency_analysis.npy inattendu : {type(data)}")
    return data


def load_spectrum_raw(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        data = obj.item()
    else:
        data = obj
    if not isinstance(data, dict):
        raise TypeError(f"spectrum_raw.npy inattendu : {type(data)}")
    return data


def load_robustness_list(path: Path) -> list:
    raw = np.load(path, allow_pickle=True)
    return list(raw.tolist())


def fig1_waveforms(out_path: Path) -> None:
    wave = load_waveforms_dict(PATHS["precision_waveforms"])
    n = int(WAVE_PREVIEW_S * SAMPLE_RATE)
    fc_cols = FC_LIST
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, fc in enumerate(fc_cols):
        key = f"fc_{fc}"
        block = wave[key]
        y_pt = np.asarray(block["y_pytorch"], dtype=np.float64).reshape(-1)[:n]
        y_ph = np.asarray(block["y_phone"], dtype=np.float64).reshape(-1)[:n]
        m = max(np.max(np.abs(y_pt)), np.max(np.abs(y_ph)), 1e-12)
        y_pt_n = y_pt / m
        y_ph_n = y_ph / m
        t = np.arange(len(y_pt_n)) / float(SAMPLE_RATE)
        err = np.abs(y_pt_n - y_ph_n)

        ax0 = axes[0, j]
        ax0.plot(t, y_pt_n, label="PyTorch (référence)", linewidth=1.0)
        ax0.plot(t, y_ph_n, label="Téléphone CPU", linewidth=1.0, alpha=0.85)
        ax0.set_title(f"Comparaison forme d'onde — fc = {fc} Hz")
        ax0.set_xlabel("Temps (s)")
        ax0.set_ylabel("Amplitude normalisée")
        ax0.legend(loc="upper right", fontsize=8)

        ax1 = axes[1, j]
        ax1.plot(t, err, color="C2", linewidth=1.0)
        ax1.set_title(f"Erreur absolue — fc = {fc} Hz")
        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Erreur absolue")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.text(
        0.5,
        0.01,
        "Remarque : les courbes « Téléphone » proviennent de precision_waveforms.npy "
        "(inférence CPU uniquement). Les sorties XNNPACK et NNAPI ne sont pas "
        "superposées ici ; elles figurent dans le tableau de précision (fig. 2).",
        ha="center",
        fontsize=8,
        wrap=True,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_precision_table(out_path: Path) -> None:
    ep_arr = np.load(PATHS["precision_results_ep"])
    if ep_arr.ndim != 4 or ep_arr.shape[:2] != (3, 3):
        raise ValueError(f"precision_results_ep forme inattendue : {ep_arr.shape}")

    rows_data: list[list[str]] = []
    for fi, fc in enumerate(FC_LIST):
        for ei, ep in enumerate(EP_LIST):
            # Axe groupe : 0 = pytorch_vs_phone, 1 = ort_pc_vs_phone (tableau : groupe 0 uniquement).
            row_metrics = ep_arr[fi, ei, 0, :]
            mae, mse, pearson, mx, psnr = row_metrics
            row = [
                str(fc),
                EP_LABEL[ep],
                f"{mae:.2e}" if np.isfinite(mae) else "N/A",
                f"{mse:.2e}" if np.isfinite(mse) else "N/A",
                f"{pearson:.6f}" if np.isfinite(pearson) else "N/A",
                f"{mx:.2e}" if np.isfinite(mx) else "N/A",
                f"{psnr:.1f}" if np.isfinite(psnr) else "N/A",
            ]
            rows_data.append(row)

    col_labels = ["fc (Hz)", "Backend", "MAE", "MSE", "Pearson", "Max Error", "PSNR (dB)"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_title("Précision de l'inférence : PyTorch vs Téléphone", pad=12)

    tbl = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_bode(out_path: Path) -> None:
    raw = load_spectrum_raw(PATHS["spectrum_raw"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_flat = axes.ravel()

    key_in = "fc_500"
    freqs_in = np.asarray(raw[key_in]["freqs"], dtype=np.float64)
    mag_in = np.asarray(raw[key_in]["mag_input"], dtype=np.float64)
    mask = (freqs_in >= 20.0) & (freqs_in <= 20_000.0)
    ax_in = axes_flat[0]
    ax_in.plot(freqs_in[mask], mag_in[mask], color="C0")
    ax_in.set_xscale("log")
    ax_in.set_xlim(20.0, 20_000.0)
    ax_in.set_xlabel("Fréquence (Hz)")
    ax_in.set_ylabel("Amplitude (dB)")
    ax_in.set_title("Signal d'entrée (sweep 20Hz–20kHz)")

    for k, fc in enumerate(FC_LIST):
        ax = axes_flat[k + 1]
        key = f"fc_{fc}"
        freqs = np.asarray(raw[key]["freqs"], dtype=np.float64)
        mag_out = np.asarray(raw[key]["mag_output"], dtype=np.float64)
        m = (freqs >= 20.0) & (freqs <= 20_000.0)
        ax.plot(freqs[m], mag_out[m], color="C1")
        ax.axvline(fc, color="red", linestyle="--", linewidth=1.0)
        ymax = float(np.nanmax(mag_out[m]))
        ymin = float(np.nanmin(mag_out[m]))
        y_ann = ymax - 0.08 * (ymax - ymin + 1e-6)
        ax.annotate(
            f"fc = {fc} Hz",
            xy=(fc, y_ann),
            color="red",
            fontsize=9,
            ha="center",
            va="bottom",
        )
        # Pas de ligne « -3 dB » ici : mag_output est une amplitude absolue (FFT en dB).
        # Le seuil -3 dB relatif au passe-bande figure sur la fonction de transfert (fig. 7).
        ax.set_xscale("log")
        ax.set_xlim(20.0, 20_000.0)
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Amplitude (dB)")
        ax.set_title(f"Filtre passe-bas — fc = {fc} Hz")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_latency(out_path: Path) -> None:
    lat = load_latency_dict(PATHS["latency_analysis"])
    by_ep = lat["by_ep"]
    labels = ["CPU", "XNNPACK", "NNAPI-GPU"]
    keys = ["cpu", "xnnpack", "nnapi"]
    means = [float(by_ep[k]["mean"]) for k in keys]
    p95s = [float(by_ep[k]["p95"]) for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)
    fig.subplots_adjust(bottom=0.22)

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax1.bar(x - w / 2, means, width=w, label="Latence moyenne", color="#1f77b4")
    b2 = ax1.bar(x + w / 2, p95s, width=w, label="Latence P95", color="#ff7f0e")
    ax1.axhline(FRAME_BUDGET_MS, color="red", linestyle="--", linewidth=1.2)
    ax1.annotate(
        f"Budget temps réel : {FRAME_BUDGET_MS:.2f} ms",
        xy=(0.99, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color="red",
        fontsize=9,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Latence (ms)")
    ax1.set_title("Latence d'inférence par backend")
    ax1.legend(loc="upper left")

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        if isinstance(h, (float, int)) and np.isfinite(h):
            ax1.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    cpu = by_ep["cpu"]
    util = float(cpu["budget_utilization_percent"])
    if np.isnan(util):
        sizes = [1.0, 0.0]
        labels_p = ["N/A", ""]
    else:
        inf = min(util, 100.0)
        mrg = max(0.0, 100.0 - util)
        if util > 100.0:
            sizes = [100.0, 0.0]
        else:
            sizes = [inf, mrg]
        labels_p = [
            f"Inférence ({inf:.1f}%)",
            f"Marge disponible ({mrg:.1f}%)",
        ]
    ax2.pie(
        sizes,
        labels=labels_p,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff7f0e", "#2ca02c"],
    )
    ax2.set_title("Utilisation du budget temps réel (CPU)")

    fig.text(
        0.5,
        0.06,
        "Note : NNAPI utilise le GPU Adreno 740. Le NPU est incompatible avec l'architecture GRU récurrente.",
        ha="center",
        fontsize=9,
        wrap=True,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_stability(out_path: Path) -> None:
    st = load_stability_dict(PATHS["stability_results"])
    mean_lr = np.asarray(st["mean_latency_per_run"], dtype=np.float64).reshape(-1)
    p95_lr = np.asarray(st["p95_latency_per_run"], dtype=np.float64).reshape(-1)
    n = len(mean_lr)
    runs = np.arange(1, n + 1)
    mu = float(np.nanmean(mean_lr))
    sigma = float(np.nanstd(mean_lr))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    ax1.plot(runs, mean_lr, marker="o", linewidth=1.2, markersize=4)
    if np.isfinite(mu):
        ax1.axhline(mu, color="blue", linestyle="--", linewidth=1.0, label=f"Moyenne : {mu:.4f} ms")
    if np.isfinite(sigma) and sigma > 0:
        ax1.fill_between(
            runs,
            mu - sigma,
            mu + sigma,
            color="blue",
            alpha=0.15,
            label="±1 écart-type",
        )
    ax1.set_xlabel("Numéro du run")
    ax1.set_ylabel("Latence moyenne (ms)")
    ax1.set_title("Évolution de la latence sur 30 runs (sweep 60s × 30, CPU)")
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(runs, p95_lr, marker="s", color="C1", linewidth=1.2, markersize=4)
    ax2.set_xlabel("Numéro du run")
    ax2.set_ylabel("Latence P95 (ms)")
    ax2.set_title("Latence P95 par run")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig6_robustness_table(out_path: Path) -> None:
    rows = load_robustness_list(PATHS["robustness_results"])
    col_labels = [
        "Cas de test",
        "Type d'entrée",
        "Statut",
        "Amplitude max",
        "Rapport puissance (dB)",
        "Zéros continus",
    ]
    cell_text: list[list[str]] = []
    cell_colours: list[list[str]] = []
    pass_bg = "#d4edda"
    fail_bg = "#f8d7da"

    for r in rows:
        ztxt = "Aucun" if not r.get("has_zeros") else "Oui (> 1 s)"
        max_a = r.get("max_amplitude")
        pr = r.get("power_ratio_db")
        cell_text.append(
            [
                str(r.get("test_name", "")),
                str(r.get("input_type", "")),
                str(r.get("status", "")),
                f"{max_a:.4f}" if isinstance(max_a, (int, float)) and np.isfinite(max_a) else "N/A",
                f"{pr:+.2f}" if isinstance(pr, (int, float)) and np.isfinite(pr) else "N/A",
                ztxt,
            ]
        )
        bg = pass_bg if str(r.get("status")) == "PASS" else fail_bg
        cell_colours.append([bg] * len(col_labels))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.set_title("Résultats des tests de robustesse — Xiaomi 13 Ultra", pad=10)
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        cellColours=cell_colours,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#e9ecef")
    tbl.scale(1.0, 1.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_transfer(out_path: Path) -> None:
    raw = load_spectrum_raw(PATHS["spectrum_raw"])
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = ["C0", "C1", "C2"]
    for idx, fc in enumerate(FC_LIST):
        key = f"fc_{fc}"
        freqs = np.asarray(raw[key]["freqs"], dtype=np.float64)
        transfer = np.asarray(raw[key]["transfer"], dtype=np.float64)
        m = (freqs >= 20.0) & (freqs <= 20_000.0)
        ax.plot(freqs[m], transfer[m], label=f"fc = {fc} Hz", color=colors[idx])
        ax.axvline(fc, color=colors[idx], linestyle="--", linewidth=0.9, alpha=0.7)

    ax.axhline(-3.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(20.0, 20_000.0)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Atténuation (dB)")
    ax.set_title(
        "Fonction de transfert mesurée (sweep 60s, backend CPU, Xiaomi 13 Ultra)"
    )
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    use_style()
    require_files()
    validate_aux_npy()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    p1 = FIG_DIR / "fig1_waveform_comparison.png"
    fig1_waveforms(p1)
    outputs.append(p1)

    p2 = FIG_DIR / "fig2_precision_table.png"
    fig2_precision_table(p2)
    outputs.append(p2)

    p3 = FIG_DIR / "fig3_bode_plot.png"
    fig3_bode(p3)
    outputs.append(p3)

    p4 = FIG_DIR / "fig4_latency_comparison.png"
    fig4_latency(p4)
    outputs.append(p4)

    p5 = FIG_DIR / "fig5_stability_curve.png"
    fig5_stability(p5)
    outputs.append(p5)

    p6 = FIG_DIR / "fig6_robustness_table.png"
    fig6_robustness_table(p6)
    outputs.append(p6)

    p7 = FIG_DIR / "fig7_spectrum_transfer.png"
    fig7_transfer(p7)
    outputs.append(p7)

    print("Figures générées :")
    for p in outputs:
        print(p.as_posix())


if __name__ == "__main__":
    main()
