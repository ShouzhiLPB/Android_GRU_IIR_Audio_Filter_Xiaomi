from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_LATENCY_PATH = REPO_ROOT / "tests" / "results" / "npy" / "all_latency_raw.npy"
OUT_PATH = REPO_ROOT / "tests" / "results" / "npy" / "latency_analysis.npy"

FRAME_BUDGET_MS = 5.33


def percentile_or_nan(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def min_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.min(np.asarray(values, dtype=np.float64)))


def max_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.max(np.asarray(values, dtype=np.float64)))


def summarize_latency(values_ms: list[float]) -> dict:
    mean_v = mean_or_nan(values_ms)
    p50_v = percentile_or_nan(values_ms, 50)
    p95_v = percentile_or_nan(values_ms, 95)
    p99_v = percentile_or_nan(values_ms, 99)
    min_v = min_or_nan(values_ms)
    max_v = max_or_nan(values_ms)

    if np.isnan(mean_v):
        rtf = float("nan")
        util = float("nan")
    else:
        rtf = mean_v / FRAME_BUDGET_MS
        util = rtf * 100.0

    if np.isnan(p95_v):
        margin = float("nan")
    else:
        margin = (FRAME_BUDGET_MS - p95_v) / FRAME_BUDGET_MS * 100.0

    return {
        "mean": mean_v,
        "p50": p50_v,
        "p95": p95_v,
        "p99": p99_v,
        "min": min_v,
        "max": max_v,
        "real_time_factor": rtf,
        "budget_utilization_percent": util,
        "safety_margin_percent": margin,
    }


def main() -> None:
    if not RAW_LATENCY_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_LATENCY_PATH}")

    raw = np.load(RAW_LATENCY_PATH, allow_pickle=True)
    rows = list(raw.tolist())

    valid_rows = [
        r
        for r in rows
        if isinstance(r, dict)
        and r.get("mean_ms") is not None
        and r.get("ep") in {"cpu", "xnnpack", "nnapi"}
    ]

    by_ep = {}
    for ep in ["cpu", "xnnpack", "nnapi"]:
        vals = [float(r["mean_ms"]) for r in valid_rows if r.get("ep") == ep]
        by_ep[ep] = summarize_latency(vals)

    cpu_mean = by_ep["cpu"]["mean"]
    xnn_mean = by_ep["xnnpack"]["mean"]
    nnapi_mean = by_ep["nnapi"]["mean"]
    by_ep["cpu"]["speedup_vs_cpu"] = 1.0 if not np.isnan(cpu_mean) else float("nan")
    by_ep["xnnpack"]["speedup_vs_cpu"] = (
        cpu_mean / xnn_mean if (not np.isnan(cpu_mean) and not np.isnan(xnn_mean) and xnn_mean != 0) else float("nan")
    )
    by_ep["nnapi"]["speedup_vs_cpu"] = (
        cpu_mean / nnapi_mean if (not np.isnan(cpu_mean) and not np.isnan(nnapi_mean) and nnapi_mean != 0) else float("nan")
    )

    # By input-file group: sweep / noise / sine / robustness...
    input_groups = sorted({str(r.get("input_file")) for r in valid_rows})
    by_input = {}
    for inp in input_groups:
        vals = [float(r["mean_ms"]) for r in valid_rows if str(r.get("input_file")) == inp]
        by_input[inp] = summarize_latency(vals)

    # By cutoff group (500/1000/3000)
    by_fc = {}
    for fc in [500, 1000, 3000]:
        vals = [float(r["mean_ms"]) for r in valid_rows if int(r.get("fc_hz", -1)) == fc]
        by_fc[str(fc)] = summarize_latency(vals)

    result = {
        "by_ep": by_ep,
        "by_input": by_input,
        "by_fc": by_fc,
        "frame_budget_ms": FRAME_BUDGET_MS,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, np.array(result, dtype=object), allow_pickle=True)

    print("=== Analyse de latence d'inférence (Xiaomi 13 Ultra, Snapdragon 8 Gen 2) ===")
    print(f"Budget temps réel par frame : {FRAME_BUDGET_MS:.2f} ms (256 samples @ 48kHz)")
    print("CPU = inférence CPU pure | XNNPACK = CPU optimisé | NNAPI = GPU Adreno 740")
    print()
    print("Backend | mean_ms | p50_ms | p95_ms | p99_ms | min_ms | max_ms | Utilisation_budget | Marge_securite | Speedup_vs_CPU")
    for ep in ["cpu", "xnnpack", "nnapi"]:
        s = by_ep[ep]
        print(
            f"{ep:7} | {s['mean']:.4f} | {s['p50']:.4f} | {s['p95']:.4f} | {s['p99']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['budget_utilization_percent']:.2f}% | "
            f"{s['safety_margin_percent']:.2f}% | {s['speedup_vs_cpu']:.4f}"
        )
    print("Note : NNAPI utilise le GPU Adreno (pas le NPU), car la compilation NPU est incompatible avec l'architecture GRU récurrente.")


if __name__ == "__main__":
    main()
