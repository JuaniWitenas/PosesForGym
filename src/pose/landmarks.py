"""Landmark extraction and temporal smoothing for YOLO pose keypoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArmLandmarks:
    """Structured right-side body landmarks used by curl detection."""

    right_hip: list[float]
    right_shoulder: list[float]
    right_elbow: list[float]
    right_wrist: list[float]


class KeypointEMASmoother:
    """Exponential moving average smoothing for 2D keypoints."""

    def __init__(self, alpha: float = 0.3) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("EMA alpha must be in the range (0, 1].")
        self.alpha = alpha
        self._state: dict[str, list[float]] = {}

    def smooth_point(self, name: str, point: list[float]) -> list[float]:
        if name not in self._state:
            self._state[name] = [point[0], point[1]]
            return [point[0], point[1]]

        prev_x, prev_y = self._state[name]
        x = self.alpha * point[0] + (1.0 - self.alpha) * prev_x
        y = self.alpha * point[1] + (1.0 - self.alpha) * prev_y
        self._state[name] = [x, y]
        return [x, y]


def extract_primary_landmarks(result, smoother: KeypointEMASmoother) -> ArmLandmarks | None:
    """Extract and smooth right-side landmarks from the first detected person."""
    if result.keypoints is None:
        return None

    xy = result.keypoints.xy
    if len(xy) == 0:
        return None

    person_kpts = xy[0]
    right_hip = smoother.smooth_point("right_hip", person_kpts[12].tolist())
    right_shoulder = smoother.smooth_point("right_shoulder", person_kpts[6].tolist())
    right_elbow = smoother.smooth_point("right_elbow", person_kpts[8].tolist())
    right_wrist = smoother.smooth_point("right_wrist", person_kpts[10].tolist())

    return ArmLandmarks(
        right_hip=right_hip,
        right_shoulder=right_shoulder,
        right_elbow=right_elbow,
        right_wrist=right_wrist,
    )
