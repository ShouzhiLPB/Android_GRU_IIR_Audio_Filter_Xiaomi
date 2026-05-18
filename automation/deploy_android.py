#!/usr/bin/env python3
"""
Deploy Android runtime artifacts to a connected device and verify startup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DEVICE_DIR = "/data/local/tmp/filtered_app"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def detect_device_abi() -> str:
    result = run(["adb", "shell", "getprop", "ro.product.cpu.abi"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query device ABI: {result.stderr.strip()}")
    abi = result.stdout.strip()
    if "arm64-v8a" in abi or "aarch64" in abi:
        return "arm64-v8a"
    if "armeabi-v7a" in abi or "armv7" in abi:
        return "armeabi-v7a"
    raise RuntimeError(f"Unsupported device ABI: {abi}")


def push_file(local_path: Path, remote_path: str) -> None:
    result = run(["adb", "push", str(local_path), remote_path])
    if result.returncode != 0:
        raise RuntimeError(f"adb push failed for {local_path}: {result.stderr.strip()}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "artifacts" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / "deploy_help.txt"

    try:
        abi = detect_device_abi()
        build_dir = repo_root / "artifacts" / "build" / abi

        filtered_bin = build_dir / "filtered"
        ort_so = build_dir / "_deps" / "onnxruntime-src" / "jni" / abi / "libonnxruntime.so"

        missing = [str(p) for p in (filtered_bin, ort_so) if not p.exists()]
        if missing:
            raise RuntimeError("Missing build artifacts:\n" + "\n".join(missing))

        run(["adb", "shell", "mkdir", "-p", DEVICE_DIR])
        push_file(filtered_bin, f"{DEVICE_DIR}/filtered")
        push_file(ort_so, f"{DEVICE_DIR}/libonnxruntime.so")

        chmod_result = run(["adb", "shell", "chmod", "+x", f"{DEVICE_DIR}/filtered"])
        if chmod_result.returncode != 0:
            raise RuntimeError(f"chmod failed: {chmod_result.stderr.strip()}")

        help_cmd = (
            f"cd {DEVICE_DIR} && "
            "LD_LIBRARY_PATH=. ./filtered --help"
        )
        help_result = run(["adb", "shell", help_cmd])
        combined_output = (
            f"[device_abi] {abi}\n"
            f"[command] {help_cmd}\n\n"
            f"[stdout]\n{help_result.stdout}\n"
            f"[stderr]\n{help_result.stderr}\n"
            f"[exit_code] {help_result.returncode}\n"
        )
        output_path.write_text(combined_output, encoding="utf-8")

        if help_result.returncode != 0:
            print(f"[FAIL] Deploy finished but --help failed. See: {output_path}")
            return 1

        print(f"[PASS] Deploy and --help check succeeded. Log: {output_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        output_path.write_text(f"[FAIL] {exc}\n", encoding="utf-8")
        print(f"[FAIL] {exc}")
        print(f"Details written to: {output_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
