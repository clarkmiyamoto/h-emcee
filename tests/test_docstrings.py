"""Tests for ensuring docstrings follow the Google style guide."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_docstrings_follow_google_style() -> None:
    """Ensure package docstrings conform to the Google style guide."""
    cmd = [
        sys.executable,
        "-m",
        "pydocstyle",
        "--convention=google",
        "--add-ignore=D100,D101,D102,D103,D104,D105,D106,D107",
        str(PROJECT_ROOT / "src" / "hemcee"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            "pydocstyle reported Google style violations:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
