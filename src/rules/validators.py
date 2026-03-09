"""Biomechanical validation rules for exercise classification."""

from __future__ import annotations


def is_shoulder_angle_neutral(shoulder_angle: float, max_angle: float) -> bool:
    """Shoulder should not be raised into overhead-press posture."""
    return shoulder_angle <= max_angle


def shoulder_displacement_ok(
    shoulder_start_y: float | None,
    shoulder_y: float,
    torso_len: float,
    displacement_ratio_max: float,
) -> tuple[float, bool]:
    """Return (current_displacement_pixels, is_within_limit)."""
    if shoulder_start_y is None or torso_len <= 0:
        return 0.0, False
    current_disp = abs(shoulder_y - shoulder_start_y)
    current_disp_ok = current_disp <= displacement_ratio_max * torso_len
    return current_disp, current_disp_ok


def is_rep_valid(
    rep_delta: float,
    min_rep_delta: float,
    shoulder_angle_ok: bool,
    shoulder_disp_ok: bool,
) -> bool:
    """Final decision rule for counting a repetition."""
    return rep_delta >= min_rep_delta and shoulder_angle_ok and shoulder_disp_ok
