"""
Copy read-only legacy GRU and Android projects into workspaces/.

Creates fresh editable copies under workspaces/gru_model and
workspaces/android_runtime. Skips .venv in the GRU tree to save time and space.
Writes a JSON manifest under artifacts/logs/ for traceability.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from _repo_paths import artifacts_dir, repo_root


def _ignore_venv(dir_path: str, names: list[str]) -> set[str]:
    """Return names to skip when copying (drop Python virtual env folders)."""
    skip = {n for n in names if n == ".venv" or n == "__pycache__"}
    return skip


def copy_tree(src: Path, dst: Path, dry_run: bool) -> dict:
    """
    Copy directory src to dst.

    Returns a small dict with keys ok, message, file_count (approx).
    """
    if not src.is_dir():
        return {"ok": False, "message": f"Source missing or not a directory: {src}", "file_count": 0}
    if dry_run:
        count = sum(1 for _ in src.rglob("*") if _.is_file())
        return {"ok": True, "message": f"DRY-RUN would copy {src} -> {dst} ({count} files)", "file_count": count}
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore_venv)
    count = sum(1 for _ in dst.rglob("*") if _.is_file())
    return {"ok": True, "message": f"Copied {src} -> {dst}", "file_count": count}


def main() -> int:
    """Parse CLI, perform copies, write manifest, print Chinese summary."""
    root = repo_root()
    default_gru = root.parent / "GRU-IRR-Filter-main" / "GRU-IRR-Filter-main"
    default_android = (
        root.parent / "RT-GRU-IRR-filter-on-Android-main" / "RT-GRU-IRR-filter-on-Android-main"
    )

    parser = argparse.ArgumentParser(description="Copy legacy sources into workspaces/")
    parser.add_argument("--gru-source", type=Path, default=default_gru, help="Path to GRU project root (contains model.py)")
    parser.add_argument(
        "--android-source",
        type=Path,
        default=default_android,
        help="Path to Android C++ project root (contains CMakeLists.txt)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report what would happen")
    args = parser.parse_args()

    dst_gru = root / "workspaces" / "gru_model"
    dst_android = root / "workspaces" / "android_runtime"
    dst_gru.mkdir(parents=True, exist_ok=True)
    dst_android.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gru_source": str(args.gru_source.resolve()),
        "android_source": str(args.android_source.resolve()),
        "dry_run": args.dry_run,
        "gru": copy_tree(args.gru_source, dst_gru, args.dry_run),
        "android": copy_tree(args.android_source, dst_android, args.dry_run),
    }

    ok = results["gru"]["ok"] and results["android"]["ok"]
    if ok and not args.dry_run:
        (dst_gru / "exported_onnx_models").mkdir(parents=True, exist_ok=True)

    manifest_path = artifacts_dir() / "logs" / "copy_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("========== Copy legacy sources (I-1) ==========")
    print(f"GRU source: {args.gru_source}")
    print(f"Android source: {args.android_source}")
    print(f"GRU dest: {dst_gru}")
    print(f"Android dest: {dst_android}")
    print("---------------------------------------------")
    print("GRU:", results["gru"]["message"], f"({results['gru']['file_count']} files)")
    print("Android:", results["android"]["message"], f"({results['android']['file_count']} files)")
    print("Manifest:", manifest_path)
    if ok:
        print("PASS: both trees copied (or dry-run OK).")
        print("Check: workspaces/gru_model/model.py and workspaces/android_runtime/CMakeLists.txt exist.")
    else:
        print("FAIL: at least one source path missing. Fix --gru-source / --android-source.")
    print("=============================================")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
