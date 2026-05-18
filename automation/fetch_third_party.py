"""
Download and unpack pinned third-party dependencies (Oboe, ONNX Runtime Android).

Clones Oboe tag 1.9.0 into third_party/oboe and unpacks the official
onnxruntime-android 1.24.2 AAR into third_party/onnxruntime-android with
SHA-256 verification. Writes a JSON report under artifacts/logs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from _repo_paths import artifacts_dir, repo_root

OBOE_TAG = "1.9.0"
OBOE_URL = f"https://github.com/google/oboe.git"

ORT_URL = (
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/"
    "onnxruntime-android/1.24.2/onnxruntime-android-1.24.2.aar"
)
ORT_SHA256 = "bc461499a735653dff285a6a3477d28b9cfd119a09c7753eaf003426b577f223"


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_oboe(dest: Path, dry_run: bool) -> dict:
    """Clone Oboe at the pinned tag if missing."""
    if dest.exists() and (dest / ".git").is_dir():
        return {"ok": True, "message": f"Oboe already present at {dest}", "path": str(dest)}
    if dry_run:
        return {"ok": True, "message": f"DRY-RUN would git clone {OBOE_URL} tag {OBOE_TAG} -> {dest}", "path": str(dest)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "--branch", OBOE_TAG, OBOE_URL, str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {
            "ok": False,
            "message": f"git clone failed: {r.stderr or r.stdout}",
            "path": str(dest),
        }
    return {"ok": True, "message": f"Cloned Oboe {OBOE_TAG} to {dest}", "path": str(dest)}


def ensure_onnxruntime(dest: Path, cache_dir: Path, dry_run: bool) -> dict:
    """Download and extract ONNX Runtime Android AAR after SHA-256 check."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    aar_path = cache_dir / "onnxruntime-android-1.24.2.aar"
    if dry_run:
        return {
            "ok": True,
            "message": f"DRY-RUN would download ORT AAR to {aar_path} and extract to {dest}",
            "path": str(dest),
        }

    if not aar_path.is_file():
        print("Downloading ONNX Runtime Android AAR (~tens of MB)...")
        urllib.request.urlretrieve(ORT_URL, aar_path)

    digest = sha256_file(aar_path)
    if digest.lower() != ORT_SHA256.lower():
        return {
            "ok": False,
            "message": f"SHA256 mismatch: got {digest}, expected {ORT_SHA256}",
            "path": str(dest),
        }

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(aar_path, "r") as zf:
        zf.extractall(dest)

    return {"ok": True, "message": f"Extracted ORT AAR to {dest}", "path": str(dest)}


def main() -> int:
    """CLI entry: fetch Oboe + ORT and write a JSON report."""
    root = repo_root()
    parser = argparse.ArgumentParser(description="Fetch Oboe + ONNX Runtime Android dependencies")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    third = root / "third_party"
    oboe_dir = third / "oboe"
    ort_dir = third / "onnxruntime-android"
    cache = artifacts_dir() / "logs" / "download_cache"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "oboe": ensure_oboe(oboe_dir, args.dry_run),
        "onnxruntime": ensure_onnxruntime(ort_dir, cache, args.dry_run),
    }
    ok = report["oboe"]["ok"] and report["onnxruntime"]["ok"]

    out = artifacts_dir() / "logs" / "fetch_third_party.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("========== Fetch third-party deps (I-2) ==========")
    print("Oboe:", report["oboe"]["message"])
    print("ONNX Runtime:", report["onnxruntime"]["message"])
    print("Report:", out)
    if ok:
        print("PASS (or dry-run OK).")
        print("Check: third_party/oboe/include/oboe and ORT jni libs under armeabi-v7a + arm64-v8a.")
    else:
        print("FAIL: see messages above (git missing or SHA256 mismatch).")
    print("==================================================")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
