"""Bicep curl detection logic based on right-arm pose landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pose.angles import compute_angle
from pose.landmarks import ArmLandmarks
from rules.validators import is_rep_valid, is_shoulder_angle_neutral, shoulder_displacement_ok


@dataclass
class CurlMetrics:
    """Per-frame exercise metrics used for display and logging."""

    elbow_angle: float
    shoulder_angle: float
    curl_valid: bool
    elbow_point: tuple[int, int] | None
    reps: int
    state: str | None
    rep_delta: float


class BicepsCurlDetector:
    """State-machine curl detector using elbow and shoulder validation."""

    def __init__(
        self,
        down_threshold: float = 150.0,
        up_threshold: float = 70.0,
        min_rep_delta: float = 70.0,
        shoulder_angle_max: float = 110.0,
        shoulder_disp_ratio_max: float = 0.18,
    ) -> None:
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.min_rep_delta = min_rep_delta
        self.shoulder_angle_max = shoulder_angle_max
        self.shoulder_disp_ratio_max = shoulder_disp_ratio_max
        self.state: str | None = None
        self.reps = 0
        self.down_peak_angle: float | None = None
        self.up_min_angle: float | None = None
        self.last_rep_delta = 0.0
        self.max_shoulder_angle_in_cycle = float("-inf")
        self.shoulder_start_y: float | None = None
        self.max_shoulder_disp_in_cycle = 0.0
        self.cycle_torso_len_ref = 1.0

    def _start_cycle(self, shoulder_angle: float, shoulder_y: float, torso_len: float) -> None:
        self.max_shoulder_angle_in_cycle = shoulder_angle
        self.shoulder_start_y = shoulder_y
        self.max_shoulder_disp_in_cycle = 0.0
        self.cycle_torso_len_ref = max(torso_len, 1.0)

    def _update_shoulder_tracking(self, shoulder_angle: float, shoulder_y: float) -> None:
        self.max_shoulder_angle_in_cycle = max(self.max_shoulder_angle_in_cycle, shoulder_angle)
        if self.shoulder_start_y is not None:
            self.max_shoulder_disp_in_cycle = max(
                self.max_shoulder_disp_in_cycle, abs(shoulder_y - self.shoulder_start_y)
            )

    def update(
        self, elbow_angle: float, shoulder_angle: float, shoulder_y: float, torso_len: float
    ) -> tuple[str | None, int, float, bool]:
        """Return (state, reps, rep_delta, curl_valid)."""
        if (
            math.isnan(elbow_angle)
            or math.isnan(shoulder_angle)
            or math.isnan(shoulder_y)
            or torso_len <= 0
        ):
            return self.state, self.reps, self.last_rep_delta, False

        if self.state is None:
            if elbow_angle > self.down_threshold:
                self.state = "DOWN"
                self.down_peak_angle = elbow_angle
                self._start_cycle(shoulder_angle, shoulder_y, torso_len)
            elif elbow_angle < self.up_threshold:
                self.state = "UP"
                self.up_min_angle = elbow_angle
                self._start_cycle(shoulder_angle, shoulder_y, torso_len)
            return (
                self.state,
                self.reps,
                self.last_rep_delta,
                is_shoulder_angle_neutral(shoulder_angle, self.shoulder_angle_max),
            )

        if self.state == "DOWN":
            if self.down_peak_angle is None:
                self.down_peak_angle = elbow_angle
            else:
                self.down_peak_angle = max(self.down_peak_angle, elbow_angle)
            self._update_shoulder_tracking(shoulder_angle, shoulder_y)

            if elbow_angle < self.up_threshold:
                self.state = "UP"
                self.up_min_angle = elbow_angle

        elif self.state == "UP":
            if self.up_min_angle is None:
                self.up_min_angle = elbow_angle
            else:
                self.up_min_angle = min(self.up_min_angle, elbow_angle)
            self._update_shoulder_tracking(shoulder_angle, shoulder_y)

            if elbow_angle > self.down_threshold:
                rep_delta = 0.0
                if self.down_peak_angle is not None and self.up_min_angle is not None:
                    rep_delta = self.down_peak_angle - self.up_min_angle
                self.last_rep_delta = rep_delta

                shoulder_angle_ok = is_shoulder_angle_neutral(
                    self.max_shoulder_angle_in_cycle, self.shoulder_angle_max
                )
                max_disp_allowed = self.shoulder_disp_ratio_max * self.cycle_torso_len_ref
                shoulder_disp_ok = self.max_shoulder_disp_in_cycle <= max_disp_allowed
                curl_valid = is_rep_valid(
                    rep_delta,
                    self.min_rep_delta,
                    shoulder_angle_ok,
                    shoulder_disp_ok,
                )
                if curl_valid:
                    self.reps += 1

                self.state = "DOWN"
                self.down_peak_angle = elbow_angle
                self.up_min_angle = None
                self._start_cycle(shoulder_angle, shoulder_y, torso_len)
                return self.state, self.reps, self.last_rep_delta, curl_valid

        _, current_disp_ok = shoulder_displacement_ok(
            self.shoulder_start_y,
            shoulder_y,
            self.cycle_torso_len_ref,
            self.shoulder_disp_ratio_max,
        )
        current_curl_valid = (
            is_shoulder_angle_neutral(shoulder_angle, self.shoulder_angle_max) and current_disp_ok
        )
        return self.state, self.reps, self.last_rep_delta, current_curl_valid


def detect_bicep_curl(landmarks: ArmLandmarks | None, detector: BicepsCurlDetector) -> CurlMetrics:
    """Detect bicep curl state from current frame landmarks."""
    if landmarks is None:
        return CurlMetrics(
            elbow_angle=float("nan"),
            shoulder_angle=float("nan"),
            curl_valid=False,
            elbow_point=None,
            reps=detector.reps,
            state=detector.state,
            rep_delta=detector.last_rep_delta,
        )

    torso_len = math.hypot(
        landmarks.right_shoulder[0] - landmarks.right_hip[0],
        landmarks.right_shoulder[1] - landmarks.right_hip[1],
    )
    elbow_angle = compute_angle(landmarks.right_shoulder, landmarks.right_elbow, landmarks.right_wrist)
    shoulder_angle = compute_angle(landmarks.right_hip, landmarks.right_shoulder, landmarks.right_elbow)

    state, reps, rep_delta, curl_valid = detector.update(
        elbow_angle,
        shoulder_angle,
        landmarks.right_shoulder[1],
        torso_len,
    )

    return CurlMetrics(
        elbow_angle=elbow_angle,
        shoulder_angle=shoulder_angle,
        curl_valid=curl_valid,
        elbow_point=(int(landmarks.right_elbow[0]), int(landmarks.right_elbow[1])),
        reps=reps,
        state=state,
        rep_delta=rep_delta,
    )
