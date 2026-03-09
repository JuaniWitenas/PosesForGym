"""Basic tests for reusable geometry helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow tests to import from src without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pose.angles import compute_angle


def test_compute_angle_right_angle() -> None:
    a = [1.0, 0.0]
    b = [0.0, 0.0]
    c = [0.0, 1.0]
    angle = compute_angle(a, b, c)
    assert math.isclose(angle, 90.0, rel_tol=1e-6, abs_tol=1e-6)


def test_compute_angle_straight_line() -> None:
    a = [-1.0, 0.0]
    b = [0.0, 0.0]
    c = [1.0, 0.0]
    angle = compute_angle(a, b, c)
    assert math.isclose(angle, 180.0, rel_tol=1e-6, abs_tol=1e-6)
