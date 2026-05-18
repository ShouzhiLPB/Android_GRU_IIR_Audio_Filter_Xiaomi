# Android GRU–IIR audio filter

Real-time low-pass filtering on Android: a small GRU approximates a Butterworth IIR, runs through ONNX Runtime, and processes live audio with Oboe (mic in → filter → speaker out). There is also a command-line `filtered` binary for file-based benchmarks on the device.

## What the supervisor’s code already provided

This repo builds on Nicolas Gry’s two upstream projects (kept read-only; copies live under `workspaces/`):

| Upstream | In this repo | Role |
|----------|--------------|------|
| `GRU-IRR-Filter-main` | `workspaces/gru_model/` | Frequency-conditioned GRU training, dataset generation, PyTorch → ONNX export |
| `RT-GRU-IRR-filter-on-Android-main` | `workspaces/android_runtime/` | C++ Oboe + ONNX Runtime, file/CLI path via the `filtered` binary |

The core idea is unchanged: a GRU with hidden state carried across 96-sample blocks approximates an IIR low-pass; cutoff is a second input (`fc_norm`).

## What I added on top

- **Single delivery tree** — `Android_GRU_IIR_Audio_Filter/` with `automation/`, `third_party/`, `artifacts/`, and pinned Oboe 1.9.0 + ORT Android (fetch script + SHA256 check) so the project rebuilds without hand-patched paths.
- **Android build fixes** — CMake points at `third_party/`; per-ABI ONNX Runtime linking; `cpu_only` / stream API fixes so `arm64-v8a` and `armeabi-v7a` both build.
- **Realtime engine hardening** — inference moved off the Oboe callback onto a worker thread; ring buffer between capture/playback; overflow/underrun counters; CPU / XNNPACK / NNAPI execution providers.
- **APK (`workspaces/android_app/`)** — Kotlin UI (mic permission, start/stop, cutoff slider) + JNI (`jni_bridge.cpp`) wrapping the same Oboe + GRU stack as the CLI, not in the original Android repo.
- **Phone test pipeline (`tests/`)** — adb scripts to push WAVs, run `filtered` on device, pull outputs; Python analysis for accuracy (PyTorch vs phone), spectrum, latency, robustness, long-run stability, and plots.
- **Deploy helpers** — `build_android.py`, `deploy_android.py`, `.bat`/`.sh` wrappers; documented gates in `env_check_all.py`.

Training scripts and model architecture in `gru_model/` are essentially the supervisor’s; my work is integration, realtime reliability, the app shell, and reproducible on-device evaluation.

## What is in the repo

| Path | Role |
|------|------|
| `workspaces/gru_model/` | PyTorch training, ONNX export, PC-side checks |
| `workspaces/android_runtime/` | C++ Oboe + ORT engine (`filtered` CLI) |
| `workspaces/android_app/` | Kotlin UI + JNI wrapper around the same engine |
| `workspaces/oboe_boilerplates/` | Shared Oboe stream helpers |
| `automation/` | Copy sources, fetch deps, env check, NDK build, deploy |
| `tests/` | Phone test scripts (adb) and analysis plots |
| `third_party/` | Oboe + ORT Android (not committed; run fetch script) |

## Architecture

```
Mic (Oboe in) → Recorder → ring buffer → Player worker
                              ↓
                    GRUBinding (ORT, block=96)
                    inputs: audio + normalized cutoff
                    state: 2-layer GRU hidden (64)
                              ↓
                    Oboe out → speaker
```

- **Model:** 2-layer GRU, hidden 64, block size 96 @ 48 kHz. Cutoff frequency is a second input channel (normalized Hz).
- **Backends:** CPU, XNNPACK; NNAPI optional where the device supports it.
- **Two entry points:** `android_runtime` for WAV/CLI tests pushed via adb; `android_app` APK for interactive use.

## Quick start

Requirements: Python 3.10+, git, cmake, Android NDK (r27d), optional adb for device tests.

```bash
python automation/fetch_third_party.py
python automation/env_check_all.py
python automation/build_android.py          # NDK → artifacts/build/<abi>/filtered
# APK: open workspaces/android_app in Android Studio, or ./gradlew assembleDebug
```

Copy `workspaces/gru_model/exported_onnx_models/lowpass_rnn.onnx` into the app assets before building the APK (see `workspaces/android_app/app/src/main/assets/README.txt`).

Phone benchmarks: `python tests/00_generate_signals.py` then `python tests/01_run_phone_tests.py` (USB debugging on).

## Why the folder looks big on disk

A full checkout can exceed 1 GB locally because of **generated** trees that are **not** meant for Git:

- `artifacts/build/` — CMake + fetched Oboe/ORT during NDK builds (~350 MB)
- `tests/results/` — WAV/npy/plots from phone runs (~300 MB)
- `workspaces/android_app/app/build/` — Gradle/APK intermediates (~150 MB)
- `third_party/` — Oboe clone + ORT AAR (~110 MB, re-downloaded via script)

With `.gitignore` as configured, a GitHub clone is on the order of **~5–15 MB** (source, scripts, ONNX, small test WAVs). Students clone, run `fetch_third_party.py`, then build.

## Legacy sources

Original upstream trees (`GRU-IRR-Filter-main`, `RT-GRU-IRR-filter-on-Android-main`) stay read-only elsewhere. Editable copies live under `workspaces/`. Use `automation/copy_legacy_sources.py` only if you need to refresh from those paths.

More iteration detail: `docs/ARCHITECTURE_AND_ITERATIONS.md`.
