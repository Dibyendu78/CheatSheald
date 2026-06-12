#!/usr/bin/env python3
"""Headless ML Model Engine for Redis-based worker.

Adapted from unified_system.py and worker/model_engine.py.
- Removes all GUI/display code.
- Accepts base64-encoded JPEG frames (no S3/file I/O).
- Returns structured result dicts.
- NO per-user state here — managed by the worker's UserStateRegistry.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# PyTorch 2.6+ defaults to weights_only=True in torch.load().
# Our YOLO .pt files use pickle serialization — allowlist the
# model classes so they can be loaded safely.
torch.serialization.add_safe_globals([DetectionModel])

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
GAZE_DIRECTIONS = {"LEFT", "RIGHT", "UP", "DOWN"}

SUSPICIOUS_OBJECT_KEYWORDS = {
    "book",
    "cell phone",
    "phone",
    "mobile",
    "laptop",
    "tablet",
    "computer",
    "monitor",
    "keyboard",
    "mouse",
    "remote",
    "tv",
}

PERSON_CONF_THRESHOLD = 0.25
OBJECT_CONF_THRESHOLD = 0.20
INFERENCE_WIDTH = 416      # resize input to this width for speed


def is_suspicious_object(name: str) -> bool:
    normalized = name.lower().replace("_", " ").replace("-", " ")
    return any(kw in normalized for kw in SUSPICIOUS_OBJECT_KEYWORDS)


def resize_for_inference(frame: np.ndarray, target_width: int) -> tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    if target_width <= 0 or w <= target_width:
        return frame, 1.0, 1.0
    target_height = int(h * (target_width / w))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, w / target_width, h / target_height


# ──────────────────────────────────────────────
# Object Detector
# ──────────────────────────────────────────────
class ObjectDetector:
    """Runs custom YOLO model (best.pt) for suspicious object detection."""

    def __init__(self, model_path: str, conf: float = OBJECT_CONF_THRESHOLD):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Object model not found: {path}")
        self.model = YOLO(str(path))
        self.conf = conf
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect objects in frame, return list of detection dicts."""
        resized, sx, sy = resize_for_inference(frame, INFERENCE_WIDTH)
        results = self.model(resized, conf=self.conf, iou=0.45, verbose=False)

        detections = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = str(self.names.get(class_id, class_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "name": name,
                "confidence": round(confidence, 3),
                "box": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                "suspicious": is_suspicious_object(name),
            })

        return detections


# ──────────────────────────────────────────────
# Face & Gaze Tracker
# ──────────────────────────────────────────────
class FaceGazeTracker:
    """Counts persons (YOLOv8n) and determines gaze direction (MediaPipe)."""

    def __init__(self, person_model_path: str, face_landmarker_path: str):
        person_path = Path(person_model_path)
        face_path = Path(face_landmarker_path)

        if not person_path.exists():
            raise FileNotFoundError(f"Person model not found: {person_path}")
        if not face_path.exists():
            raise FileNotFoundError(f"Face landmarker not found: {face_path}")

        self.person_model = YOLO(str(person_path))

        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(face_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _h_ratio(lm, li: int, ri: int, ii: int, w: int, h: int) -> float:
        left = np.array([lm[li].x * w, lm[li].y * h])
        right = np.array([lm[ri].x * w, lm[ri].y * h])
        iris = np.array([lm[ii].x * w, lm[ii].y * h])
        eye_w = np.linalg.norm(right - left)
        return float(np.linalg.norm(iris - left) / eye_w) if eye_w > 0 else 0.5

    @staticmethod
    def _v_ratio(lm, ti: int, bi: int, ii: int, w: int, h: int) -> float:
        top = np.array([lm[ti].x * w, lm[ti].y * h])
        bot = np.array([lm[bi].x * w, lm[bi].y * h])
        iris = np.array([lm[ii].x * w, lm[ii].y * h])
        eye_h = np.linalg.norm(bot - top)
        return float(np.linalg.norm(iris - top) / eye_h) if eye_h > 0 else 0.5

    def _gaze(self, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_img)

        if not result.face_landmarks:
            return "NO_FACE"

        lm = result.face_landmarks[0]
        lh = self._h_ratio(lm, 33, 133, 468, w, h)
        rh = self._h_ratio(lm, 362, 263, 473, w, h)
        lv = self._v_ratio(lm, 159, 145, 468, w, h)
        rv = self._v_ratio(lm, 386, 374, 473, w, h)

        avg_h = (lh + rh) / 2
        avg_v = (lv + rv) / 2

        if avg_v < 0.40:
            return "UP"
        if avg_v > 0.60:
            return "DOWN"
        if avg_h < 0.40:
            return "RIGHT"
        if avg_h > 0.60:
            return "LEFT"
        return "CENTER"

    def analyze(self, frame: np.ndarray) -> dict:
        """Count persons and determine gaze direction."""
        resized, _, _ = resize_for_inference(frame, INFERENCE_WIDTH)

        # Person detection
        pr = self.person_model(resized, classes=[0], conf=PERSON_CONF_THRESHOLD, verbose=False)
        boxes = pr[0].boxes
        face_count = 0 if boxes is None else len(boxes)

        # Gaze direction
        gaze_direction = self._gaze(resized)

        return {
            "face_count": face_count,
            "gaze_direction": gaze_direction,
        }


# ──────────────────────────────────────────────
# Model Engine — Entry Point for Worker
# ──────────────────────────────────────────────
class ModelEngine:
    """
    Fully STATELESS ML engine. Each frame is analyzed independently.

    Rules:
      - No counters, no thresholds, no memory between frames.
      - CENTER gaze  → no alert (green on frontend).
      - Any other gaze (LEFT/RIGHT/UP/DOWN/NO_FACE) → alert.
      - Gaze alert message is simple — never exposes direction.
      - Multiple workers can safely process any user's frames in any order.
    """

    def __init__(
        self,
        object_model_path: str,
        person_model_path: str,
        face_landmarker_path: str,
    ):
        import logging
        self.logger = logging.getLogger("model_engine")
        self.logger.info("Loading ML models...")
        t0 = time.time()

        self.object_detector = ObjectDetector(object_model_path)
        self.face_gaze = FaceGazeTracker(person_model_path, face_landmarker_path)

        self.logger.info(f"Models loaded in {time.time() - t0:.1f}s")

    def analyze_frame_b64(
        self,
        frame_b64: str,
        user_id: str,
    ) -> dict:
        """
        Analyze a single base64-encoded JPEG frame — fully stateless.

        Each call is completely independent of previous frames.
        Gaze logic:
            CENTER  → gaze_ok = True  (no alert)
            anything else → gaze_ok = False (alert: "Student is not looking at screen")
        """
        # Decode base64 → JPEG bytes → OpenCV frame
        try:
            img_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            return self._error_result(user_id, f"Frame decode error: {e}")

        if frame is None:
            return self._error_result(user_id, "Could not decode image")

        # ── Object detection ───────────────────────────────────────────
        object_detections = self.object_detector.detect(frame)
        suspicious_objects = [d for d in object_detections if d["suspicious"]]

        # ── Face & gaze detection ──────────────────────────────────────
        face_gaze_result = self.face_gaze.analyze(frame)
        face_count: int = face_gaze_result["face_count"]
        gaze_direction: str = face_gaze_result["gaze_direction"]

        # ── Build alerts (stateless — each frame decides independently) ───
        alerts = []

        # Check 1: Multiple persons
        if face_count > 1:
            alerts.append({
                "type": "multiple_persons",
                "message": "Multiple persons detected in frame",
                "severity": "critical",
            })

        # Check 2: No face visible
        if face_count == 0:
            alerts.append({
                "type": "no_face",
                "message": "Student not visible in frame",
                "severity": "warning",
            })

        # Check 3: Suspicious objects
        for obj in suspicious_objects:
            alerts.append({
                "type": "suspicious_object",
                "message": f"Suspicious object detected: {obj['name']}",
                "severity": "critical",
            })

        # Check 4: Gaze away — CENTER = OK (green), anything else = alert
        # Simple message only — direction is NOT exposed in the alert text
        gaze_ok = (gaze_direction == "CENTER")
        if not gaze_ok and face_count > 0:  # only alert if face is present
            alerts.append({
                "type": "gaze_away",
                "message": "Student is not looking at the screen",
                "severity": "warning",
            })

        cheating = len(alerts) > 0
        primary = alerts[0] if alerts else None

        return {
            "cheating": cheating,
            "type": primary["type"] if primary else "none",
            "message": primary["message"] if primary else "Normal",
            "user_id": user_id,
            "timestamp": int(time.time()),
            "details": {
                "face_count": face_count,
                # gaze_direction is kept for debugging only — never shown as alert text
                "gaze_direction": gaze_direction,
                "gaze_ok": gaze_ok,
                "objects_detected": [
                    {"name": d["name"], "confidence": d["confidence"]}
                    for d in object_detections
                ],
                "suspicious_objects": [
                    {"name": d["name"], "confidence": d["confidence"]}
                    for d in suspicious_objects
                ],
            },
            "alerts": alerts,
        }

    @staticmethod
    def _error_result(user_id: str, message: str) -> dict:
        return {
            "cheating": False,
            "type": "error",
            "message": message,
            "user_id": user_id,
            "timestamp": int(time.time()),
            "details": {},
            "alerts": [],
        }
