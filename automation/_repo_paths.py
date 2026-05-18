"""
Central path helpers for all automation scripts.

Resolves the workspace root (parent of the automation/ directory) so every
script can build paths in a relocatable way without hard-coded drive letters.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the Android_GRU_IIR_Audio_Filter directory (repository root)."""
    return Path(__file__).resolve().parent.parent


def artifacts_dir() -> Path:
    """Return the artifacts/ root; creates artifacts/logs if missing."""
    root = repo_root() / "artifacts"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "test_results" / "env").mkdir(parents=True, exist_ok=True)
    return root
