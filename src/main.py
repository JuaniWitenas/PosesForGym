"""Entry point for pose inference and exercise detection."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

from exercises.bicep_curl import BicepsCurlDetector, detect_bicep_curl
from pose.angles import parse_window_size
from pose.inference import YoloPoseInference
from pose.landmarks import KeypointEMASmoother, extract_primary_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics YOLO26n pose inference on an image, video, URL, or webcam."
    )
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--source", default="0", help="Path/URL/webcam index (e.g. 0, 1).")
    parser.add_argument("--project", default="YOLO Pose")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", dest="save", action="store_true", default=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--shoulder-angle-max", type=float, default=110.0)
    parser.add_argument("--shoulder-disp-ratio-max", type=float, default=0.18)
    parser.add_argument("--window-size", type=parse_window_size, default=(1280, 720))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    return parser.parse_args()


def configure_logging(args: argparse.Namespace) -> logging.Logger:
    logger = logging.getLogger("pose_app")
    logger.setLevel(getattr(logging, args.log_level))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


class PoseRepPipeline:
    """Per-frame keypoint extraction + bicep curl detection pipeline."""

    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.logger = logger
        self.smoother = KeypointEMASmoother(alpha=args.ema_alpha)
        self.detector = BicepsCurlDetector(
            min_rep_delta=70.0,
            shoulder_angle_max=args.shoulder_angle_max,
            shoulder_disp_ratio_max=args.shoulder_disp_ratio_max,
        )

    def process(self, result, frame_idx: int):
        landmarks = extract_primary_landmarks(result, self.smoother)
        metrics = detect_bicep_curl(landmarks, self.detector)
        self.logger.debug(
            "Frame %s | elbow=%.2f shoulder=%.2f state=%s reps=%s delta=%.2f valid=%s",
            frame_idx,
            metrics.elbow_angle,
            metrics.shoulder_angle,
            metrics.state,
            metrics.reps,
            metrics.rep_delta,
            metrics.curl_valid,
        )
        return metrics

    @staticmethod
    def draw_overlay(frame, metrics, cv2) -> None:
        elbow_text = "NaN" if math.isnan(metrics.elbow_angle) else f"{metrics.elbow_angle:.1f} deg"
        shoulder_text = "NaN" if math.isnan(metrics.shoulder_angle) else f"{metrics.shoulder_angle:.1f} deg"

        cv2.putText(
            frame,
            f"Reps: {metrics.reps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Elbow angle: {elbow_text}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Shoulder angle: {shoulder_text}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Curl valid: {'yes' if metrics.curl_valid else 'no'}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if metrics.curl_valid else (0, 0, 255),
            2,
        )

        if metrics.elbow_point is not None:
            cv2.circle(frame, metrics.elbow_point, 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                elbow_text,
                (metrics.elbow_point[0] + 10, metrics.elbow_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )


def run(args: argparse.Namespace, logger: logging.Logger) -> int:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        logger.error("Missing dependencies. Install with: python -m pip install ultralytics")
        return 1

    inference = YoloPoseInference(args, logger)
    source = inference.normalize_source()
    if not inference.validate_camera_source(cv2, source):
        return 1

    model = inference.load_model(YOLO)
    results = inference.predict_stream(model, source)

    pipeline = PoseRepPipeline(args, logger)
    window_name = "Biceps Curl Detector"

    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.window_size[0], args.window_size[1])

    frames_processed = 0
    total_boxes = 0
    save_dir = None

    try:
        loop_t0 = time.perf_counter()
        for idx, result in enumerate(results, start=1):
            frames_processed = idx
            total_boxes += len(result.boxes) if result.boxes is not None else 0
            if save_dir is None:
                save_dir = getattr(result, "save_dir", None)

            metrics = pipeline.process(result, idx)

            if args.show:
                frame = result.plot()
                pipeline.draw_overlay(frame, metrics, cv2)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        elapsed = time.perf_counter() - loop_t0
        if frames_processed > 0:
            logger.info(
                "Processed %s frames in %.2fs (%.2f FPS)",
                frames_processed,
                elapsed,
                frames_processed / max(elapsed, 1e-6),
            )
    except ConnectionError as exc:
        logger.error("Source error: %s", exc)
        return 1
    finally:
        if args.show:
            cv2.destroyAllWindows()

    if frames_processed == 0:
        logger.warning("No frames were processed.")
        return 0

    if args.save and save_dir:
        logger.info("Saved outputs to: %s", Path(save_dir))
    elif not args.save:
        logger.info("Saving disabled (--no-save).")

    logger.info("Total detections: %s", total_boxes)
    logger.info("Total reps counted: %s", pipeline.detector.reps)
    return 0


def main() -> int:
    args = parse_args()
    logger = configure_logging(args)
    return run(args, logger)


if __name__ == "__main__":
    raise SystemExit(main())
