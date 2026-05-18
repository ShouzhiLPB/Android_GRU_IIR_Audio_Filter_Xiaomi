"""
Run host-side environment checks for the Android GRU-IIR pipeline.

Checks Python version, optional ML imports, CMake/ADB on PATH, ANDROID_NDK,
local third_party layout, and workspace copies. Writes JSON under
artifacts/test_results/env/ and appends a short Chinese summary to env_check.md.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _repo_paths import artifacts_dir, repo_root


def _guess_cmake_on_windows() -> str | None:
    """
    If cmake is not on PATH, try well-known install locations (Windows only).

    Returns absolute path to cmake.exe if found, else None.
    """
    if sys.platform != "win32":
        return None
    candidates = [
        Path(r"C:\Program Files\CMake\bin\cmake.exe"),
        Path(r"C:\Program Files (x86)\CMake\bin\cmake.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return None


def _which(name: str) -> str | None:
    """Return executable path if found on PATH."""
    return shutil.which(name)


def _run_version(cmd: list[str]) -> tuple[int, str]:
    """Run a --version command and return (returncode, combined stdout/stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main() -> int:
    """Collect checks, write JSON + markdown summary, print Chinese overview."""
    root = repo_root()
    env_out = artifacts_dir() / "test_results" / "env"
    env_out.mkdir(parents=True, exist_ok=True)

    py_ok = sys.version_info >= (3, 10)
    checks: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "ok_ge_3_10": py_ok,
        },
        "tools": {},
        "third_party": {},
        "workspaces": {},
        "optional_imports": {},
    }

    for tool, args in (("cmake", ["cmake", "--version"]), ("adb", ["adb", "version"])):
        code, out = _run_version(args)
        checks["tools"][tool] = {"on_path": _which(tool) is not None, "returncode": code, "output_head": out[:300]}

    cmake_guess = None if checks["tools"]["cmake"]["on_path"] else _guess_cmake_on_windows()
    checks["tools"]["cmake"]["guessed_off_path"] = cmake_guess

    ndk = os.environ.get("ANDROID_NDK_HOME") or os.environ.get("ANDROID_NDK")
    checks["android_ndk"] = {"env_set": bool(ndk), "path": ndk}

    oboe = root / "third_party" / "oboe" / "include" / "oboe" / "Oboe.h"
    ort_v7 = root / "third_party" / "onnxruntime-android" / "jni" / "armeabi-v7a" / "libonnxruntime.so"
    ort_v8 = root / "third_party" / "onnxruntime-android" / "jni" / "arm64-v8a" / "libonnxruntime.so"
    checks["third_party"] = {
        "oboe_header": oboe.is_file(),
        "ort_armeabi_v7a_so": ort_v7.is_file(),
        "ort_arm64_v8a_so": ort_v8.is_file(),
    }

    gru_model = root / "workspaces" / "gru_model" / "model.py"
    android_cmake = root / "workspaces" / "android_runtime" / "CMakeLists.txt"
    checks["workspaces"] = {
        "gru_model_py": gru_model.is_file(),
        "android_cmake": android_cmake.is_file(),
    }

    for mod in ("torch", "onnxruntime", "numpy"):
        try:
            __import__(mod)
            checks["optional_imports"][mod] = "ok"
        except Exception as e:  # noqa: BLE001 - user-facing diagnostics
            checks["optional_imports"][mod] = f"missing: {e.__class__.__name__}"

    # Tier A: enough to work on Python / models (copy + third_party).
    gate_copy_and_deps = (
        py_ok
        and checks["workspaces"]["gru_model_py"]
        and checks["workspaces"]["android_cmake"]
        and checks["third_party"]["oboe_header"]
        and checks["third_party"]["ort_armeabi_v7a_so"]
        and checks["third_party"]["ort_arm64_v8a_so"]
    )
    # Tier B: ready to configure Android CMake (host toolchain).
    gate_android_build = gate_copy_and_deps and checks["tools"]["cmake"]["on_path"]

    checks["gate_copy_and_deps_ok"] = gate_copy_and_deps
    checks["gate_android_build_ready"] = gate_android_build
    checks["gate_all_required"] = gate_android_build

    json_path = env_out / "host_check.json"
    json_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    md_block = (
        f"\n\n## Auto check {checks['timestamp_utc']}\n\n"
        f"- Python OK (>=3.10): **{py_ok}**\n"
        f"- cmake on PATH: **{checks['tools']['cmake']['on_path']}**\n"
        f"- adb on PATH: **{checks['tools']['adb']['on_path']}**\n"
        f"- ANDROID_NDK set: **{checks['android_ndk']['env_set']}**\n"
        f"- third_party Oboe header: **{checks['third_party']['oboe_header']}**\n"
        f"- ORT v7a/v8a so: **{checks['third_party']['ort_armeabi_v7a_so']}** / **{checks['third_party']['ort_arm64_v8a_so']}**\n"
        f"- workspaces copy OK: **{checks['workspaces']['gru_model_py']}** / **{checks['workspaces']['android_cmake']}**\n"
        f"- **GATE A copy+deps OK: {gate_copy_and_deps}**\n"
        f"- **GATE B android build ready (needs cmake): {gate_android_build}**\n"
        f"- JSON: `{json_path.as_posix()}`\n"
    )
    env_md = root / "env_check.md"
    env_md.write_text(env_md.read_text(encoding="utf-8") + md_block, encoding="utf-8")

    print("========== Environment check (I-3) ==========")
    print("Python:", sys.version.split()[0], "(need >=3.10) ->", "OK" if py_ok else "FAIL")
    print("cmake on PATH:", "yes" if checks["tools"]["cmake"]["on_path"] else "no")
    print("adb on PATH:", "yes" if checks["tools"]["adb"]["on_path"] else "no (optional without a device)")
    print("ANDROID_NDK_HOME:", "set" if checks["android_ndk"]["env_set"] else "not set (required before Android build)")
    print("Oboe headers:", "ok" if checks["third_party"]["oboe_header"] else "missing — run fetch_third_party.py")
    print("ORT jni libs (v7a / v8a):", checks["third_party"]["ort_armeabi_v7a_so"], checks["third_party"]["ort_arm64_v8a_so"])
    print("workspaces copied:", checks["workspaces"]["gru_model_py"], checks["workspaces"]["android_cmake"])
    print("optional torch/onnxruntime:", checks["optional_imports"])
    print("GATE A (copy + deps):", "PASS" if gate_copy_and_deps else "FAIL")
    print("GATE B (cmake ready for NDK build):", "PASS" if gate_android_build else "FAIL")
    if not gate_android_build and gate_copy_and_deps:
        print("")
        print("Gate B failed: install CMake or add it to PATH.")
        if cmake_guess:
            print("Found CMake at:", cmake_guess)
            print("Add its bin directory to PATH, restart the terminal, rerun this script.")
        else:
            print("Install from https://cmake.org/download/ (tick 'Add to PATH'),")
            print("or enable 'CMake tools' in Visual Studio Installer, then run: cmake --version")
        print("")
    print("JSON report:", json_path)
    print("=============================================")
    return 0 if gate_copy_and_deps else 2


if __name__ == "__main__":
    raise SystemExit(main())
