"""
Smoke tests for repository layout after bootstrap scripts.

Ensures key directories exist relative to the automation package root.
Run with: python automation/tests/test_paths.py   (no pytest required)
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestRepoLayout(unittest.TestCase):
    """Layout checks for workspaces and third_party boilerplate."""

    def test_workspaces_have_copies(self) -> None:
        """After copy_legacy_sources, GRU model.py and Android CMakeLists must exist."""
        self.assertTrue((ROOT / "workspaces" / "gru_model" / "model.py").is_file())
        self.assertTrue((ROOT / "workspaces" / "android_runtime" / "CMakeLists.txt").is_file())

    def test_oboe_boilerplate_present(self) -> None:
        """Boilerplate acquire_audio_stream sources must exist in third_party."""
        base = ROOT / "third_party" / "oboe_boilerplates"
        self.assertTrue((base / "acquire_audio_stream.cpp").is_file())
        self.assertTrue((base / "acquire_audio_stream.hpp").is_file())


if __name__ == "__main__":
    unittest.main()
