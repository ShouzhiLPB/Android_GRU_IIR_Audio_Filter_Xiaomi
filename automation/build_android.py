"""
Configure and build the Android runtime with the Android NDK CMake toolchain.

Requires ANDROID_NDK_HOME or ANDROID_NDK pointing to NDK r27d (or compatible).
Produces binaries under artifacts/build/<abi>/ and writes logs for CI-style review.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from _repo_paths import artifacts_dir, repo_root


def find_ndk() -> Path | None:
    """Return NDK root path from common environment variables."""
    for key in ("ANDROID_NDK_HOME", "ANDROID_NDK", "NDK_ROOT"):
        v = os.environ.get(key)
        if v and Path(v).is_dir():
            return Path(v)
    return None


def configure_and_build(abi: str, ndk: Path, dry_run: bool) -> int:
    """Run cmake configure + build for one ANDROID_ABI."""
    root = repo_root()
    src = root / "workspaces" / "android_runtime"
    build_dir = artifacts_dir() / "build" / abi
    build_dir.mkdir(parents=True, exist_ok=True)
    toolchain = ndk / "build" / "cmake" / "android.toolchain.cmake"
    if not toolchain.is_file():
        print("ERROR: android.toolchain.cmake not found under NDK:", toolchain)
        return 3

    cmake_args = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        f"-DANDROID_ABI={abi}",
        "-DANDROID_PLATFORM=android-24",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]

    log_dir = artifacts_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cmake_{abi}.log"

    if dry_run:
        print("DRY-RUN would run:", " ".join(cmake_args))
        print("then:", " ".join(build_cmd))
        return 0

    print("Configuring", abi, "...")
    r1 = subprocess.run(cmake_args, capture_output=True, text=True)
    log_path.write_text((r1.stdout or "") + "\n" + (r1.stderr or ""), encoding="utf-8")
    if r1.returncode != 0:
        print("CMake configure failed; log:", log_path)
        print(r1.stderr or r1.stdout)
        return r1.returncode

    print("Building", abi, "...")
    r2 = subprocess.run(build_cmd, capture_output=True, text=True)
    log_path.write_text(log_path.read_text(encoding="utf-8") + "\n--- BUILD ---\n" + (r2.stdout or "") + "\n" + (r2.stderr or ""), encoding="utf-8")
    if r2.returncode != 0:
        print("Build failed; log:", log_path)
        print(r2.stderr or r2.stdout)
        return r2.returncode

    exe = build_dir / "filtered"
    if not exe.is_file():
        exe = build_dir / "Release" / "filtered"
    if exe.is_file():
        marker = build_dir / "build_ok.marker"
        marker.write_text("ok\n", encoding="utf-8")
        print("OK: built", exe)
    else:
        print("WARNING: filtered binary not found; see log:", log_path)
        return 4
    return 0


def main() -> int:
    """Parse CLI and build one or two Android ABIs."""
    parser = argparse.ArgumentParser(description="Configure/build android_runtime with NDK")
    parser.add_argument("--abi", action="append", help="ANDROID_ABI (repeat for multiple)", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ndk = find_ndk()
    if not ndk:
        print("========== Android build (I-5) ==========")
        print("FAIL: ANDROID_NDK_HOME (or ANDROID_NDK) not set. Point it at NDK r27d root.")
        print("==========================================")
        return 5
    if not shutil.which("cmake"):
        print("========== Android build (I-5) ==========")
        print("FAIL: cmake not on PATH. Install CMake and restart the terminal.")
        print("==========================================")
        return 6

    abis = args.abi or ["arm64-v8a", "armeabi-v7a"]
    code = 0
    for abi in abis:
        rc = configure_and_build(abi, ndk, args.dry_run)
        if rc != 0:
            code = rc
            break
    print("========== Build summary ==========")
    print("NDK:", ndk)
    print("Exit code:", code, "(0 = all ABIs OK)")
    print("Logs:", artifacts_dir() / "logs")
    print("===================================")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
