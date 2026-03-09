"""YOLO pose model loading and inference streaming."""

from __future__ import annotations

import time
from pathlib import Path


class YoloPoseInference:
    """Loads YOLO pose model and returns streaming inference results."""

    def __init__(self, args, logger) -> None:
        self.args = args
        self.logger = logger

    def resolve_model(self) -> str:
        model_arg = self.args.model
        model_path = Path(model_arg)
        if model_path.is_absolute():
            return str(model_path)

        # Prefer model file next to repository root / script execution path.
        local_candidate = Path.cwd() / model_arg
        if local_candidate.exists():
            self.logger.info("Using local model: %s", local_candidate)
            return str(local_candidate)

        self.logger.warning(
            "Model '%s' not found in cwd. Ultralytics may download it (slow on first run).",
            model_arg,
        )
        return model_arg

    def normalize_source(self):
        source = self.args.source
        return int(source) if isinstance(source, str) and source.isdigit() else source

    def validate_camera_source(self, cv2, source) -> bool:
        if not isinstance(source, int):
            return True
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            self.logger.error("Camera index %s is not available. Try --source 0 (or 1/2).", source)
        return ok

    def load_model(self, YOLO):
        startup_t0 = time.perf_counter()
        model_ref = self.resolve_model()
        self.logger.info("Loading model: %s", model_ref)
        model = YOLO(model_ref)
        self.logger.info("Model loaded in %.2fs", time.perf_counter() - startup_t0)
        return model

    def predict_stream(self, model, source):
        infer_t0 = time.perf_counter()
        results = model.predict(
            source=source,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            device=self.args.device,
            save=self.args.save,
            show=False,
            stream=True,
            save_txt=self.args.save_txt,
            save_conf=self.args.save_conf,
            project=self.args.project,
            name=self.args.name,
            exist_ok=self.args.exist_ok,
            verbose=False,
        )
        self.logger.info("Predictor initialized in %.2fs", time.perf_counter() - infer_t0)
        return results
