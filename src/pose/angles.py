"""Reusable angle and geometry helpers for pose analysis."""

from __future__ import annotations

import argparse
import math


def compute_angle(a: list[float], b: list[float], c: list[float]) -> float:
    """Return angle ABC in degrees, where B is the vertex."""
    ba_x, ba_y = a[0] - b[0], a[1] - b[1]
    bc_x, bc_y = c[0] - b[0], c[1] - b[1]

    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)
    if mag_ba == 0 or mag_bc == 0:
        return float("nan")

    cosine = (ba_x * bc_x + ba_y * bc_y) / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def parse_window_size(value: str) -> tuple[int, int]:
    """Parse window size from WIDTHxHEIGHT string."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window size must be WIDTHxHEIGHT, e.g. 1600x900")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window size values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("window size values must be > 0")
    return width, height
