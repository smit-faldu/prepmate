"""
vision_processor.py — Advanced Multimodal Behavioral Analysis Pipeline
Principal Architect: Refactored for production-grade LLM-ready output.

Architecture Overview:
  - TemporalBuffer: Accumulates per-frame signals over a sliding window (default 5s).
  - HeadPoseEstimator: Derives yaw/pitch/roll from YOLO facial keypoints via solvePnP.
  - PostureAnalyzer: Derives confidence score from shoulder/spine/hip geometry.
  - FidgetTracker: Detects high-frequency low-amplitude wrist movement via rolling variance.
  - EmotionTracker: Tracks transitions and micro-expression onset via delta scoring.
  - BehavioralSummarizer: Quantizes all continuous signals into semantic LLM-ready JSON.
"""

import asyncio
import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from ultralytics import YOLO
from hsemotion.facial_emotions import HSEmotionRecognizer

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame, UserImageRawFrame, LLMMessagesAppendFrame, OutputTransportMessageFrame
)
import torch
import json

# ---------------------------------------------------------------------------
# Constants — Model Geometry
# ---------------------------------------------------------------------------

# YOLO 17-keypoint indices (COCO format)
KP_NOSE        = 0
KP_L_EYE      = 1
KP_R_EYE      = 2
KP_L_EAR      = 3
KP_R_EAR      = 4
KP_L_SHOULDER  = 5
KP_R_SHOULDER  = 6
KP_L_ELBOW     = 7
KP_R_ELBOW     = 8
KP_L_WRIST     = 9
KP_R_WRIST     = 10
KP_L_HIP       = 11
KP_R_HIP       = 12

# Canonical 3D face model points (nose, chin, eye corners, mouth corners)
# Used for solvePnP head-pose estimation
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],  # Nose tip
    [0.0,   -63.6, -12.5  ],  # Chin (estimated)
    [-43.3,  32.7,  -26.0 ],  # Left eye outer corner
    [43.3,   32.7,  -26.0 ],  # Right eye outer corner
    [-28.9, -28.9,  -24.1 ],  # Left mouth corner
    [28.9,  -28.9,  -24.1 ],  # Right mouth corner
], dtype=np.float32)

# Corresponding YOLO keypoint indices for the 6 face landmarks above
FACE_KP_INDICES = [KP_NOSE, KP_L_EAR, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR]

EMOTION_LABELS = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FrameSignal:
    """All extracted signals for a single video frame."""
    timestamp: float
    emotion_scores: np.ndarray          # shape (8,) — raw HSEmotion probabilities
    dominant_emotion: str
    confidence_score: float             # 0–100 posture/composure
    head_yaw: float                     # degrees; negative = looking left
    head_pitch: float                   # degrees; negative = looking up
    head_roll: float                    # degrees
    l_wrist_pos: Optional[np.ndarray]   # (x, y) normalized [0,1]
    r_wrist_pos: Optional[np.ndarray]
    gesture_state: str                  # semantic label
    keypoints_visible: bool


@dataclass
class WindowSummary:
    """Aggregated behavioral summary over a time window, ready for LLM injection."""
    timestamp_window: str
    primary_emotion: str
    emotion_confidence: str             # "High / Medium / Low"
    posture_state: str
    confidence_level: str               # "High / Moderate / Low"
    gesture_summary: str
    attention: str
    notable_event: Optional[str]
    raw_metrics: dict                   # for frontend / debug


# ---------------------------------------------------------------------------
# Head Pose Estimator
# ---------------------------------------------------------------------------

class HeadPoseEstimator:
    """
    Estimates yaw, pitch, roll from YOLO facial keypoints using OpenCV solvePnP.
    Falls back to simple angle heuristics if the geometry is degenerate.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.update_camera(frame_width, frame_height)

    def update_camera(self, w: int, h: int):
        focal = w  # Approximation: focal ≈ image width
        self.camera_matrix = np.array([
            [focal,     0,  w / 2],
            [0,     focal,  h / 2],
            [0,         0,      1],
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        self._frame_size = (w, h)

    def estimate(self, keypoints: np.ndarray) -> tuple[float, float, float]:
        """
        Returns (yaw_deg, pitch_deg, roll_deg).
        Yaw:   + = face turned right,  − = face turned left
        Pitch: + = face tilted down,   − = face tilted up
        Roll:  + = head tilted right,  − = head tilted left
        """
        try:
            nose  = keypoints[KP_NOSE]
            l_eye = keypoints[KP_L_EYE]
            r_eye = keypoints[KP_R_EYE]
            l_ear = keypoints[KP_L_EAR]
            r_ear = keypoints[KP_R_EAR]

            # Require all 5 points to be detected (non-zero)
            pts = np.array([nose, l_eye, r_eye, l_ear, r_ear])
            if np.any(pts == 0):
                return self._heuristic_pose(keypoints)

            # Build 2D image points matching FACE_3D_MODEL layout
            # We approximate chin as below nose using inter-ocular distance
            iod = np.linalg.norm(r_eye - l_eye)
            chin_2d = nose + np.array([0, iod * 1.2])
            mouth_l = nose + np.array([-iod * 0.45, iod * 0.5])
            mouth_r = nose + np.array([ iod * 0.45, iod * 0.5])

            image_points = np.array([
                nose, chin_2d, l_eye, r_eye, mouth_l, mouth_r
            ], dtype=np.float32)

            # Update camera matrix if frame size has changed
            success, rot_vec, _ = cv2.solvePnP(
                FACE_3D_MODEL, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return self._heuristic_pose(keypoints)

            rot_mat, _ = cv2.Rodrigues(rot_vec)
            # Decompose rotation matrix to Euler angles (ZYX convention)
            sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                pitch = np.degrees(np.arctan2( rot_mat[2, 1], rot_mat[2, 2]))
                yaw   = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
                roll  = np.degrees(np.arctan2( rot_mat[1, 0], rot_mat[0, 0]))
            else:
                pitch = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
                yaw   = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
                roll  = 0.0
            return float(yaw), float(pitch), float(roll)

        except Exception:
            return self._heuristic_pose(keypoints)

    def _heuristic_pose(self, keypoints: np.ndarray) -> tuple[float, float, float]:
        """Lightweight fallback using eye/ear symmetry."""
        try:
            l_eye, r_eye = keypoints[KP_L_EYE], keypoints[KP_R_EYE]
            l_ear, r_ear = keypoints[KP_L_EAR], keypoints[KP_R_EAR]
            if np.all(l_eye == 0) or np.all(r_eye == 0):
                return 0.0, 0.0, 0.0
            eye_mid_x = (l_eye[0] + r_eye[0]) / 2.0
            frame_center_x = self._frame_size[0] / 2.0
            yaw = (eye_mid_x - frame_center_x) / frame_center_x * 30.0  # crude scaling
            # Roll from eye line angle
            d = r_eye - l_eye
            roll = float(np.degrees(np.arctan2(d[1], d[0])))
            return float(yaw), 0.0, roll
        except Exception:
            return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Posture Analyzer
# ---------------------------------------------------------------------------

class PostureAnalyzer:
    """
    Derives a 0–100 Confidence Score and a semantic posture label from
    YOLO pose keypoints.

    Metrics:
      - Shoulder symmetry: level shoulders ↑ score
      - Forward lean: nose X vs shoulder midpoint X as proxy for leaning
      - Slouch: shoulder Y vs hip Y compressed distance → slouch penalty
      - Shoulder width: broad shoulders (relative to body) → confidence ↑
    """

    def analyze(
        self,
        keypoints: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> tuple[float, str]:
        """Returns (confidence_score 0-100, posture_label)."""
        try:
            ls = keypoints[KP_L_SHOULDER]
            rs = keypoints[KP_R_SHOULDER]
            lh = keypoints[KP_L_HIP]
            rh = keypoints[KP_R_HIP]
            nose = keypoints[KP_NOSE]

            # Require shoulders to be visible
            if np.any(ls == 0) or np.any(rs == 0):
                return 50.0, "posture unclear"

            score = 50.0  # Baseline

            # --- 1. Shoulder Level (symmetry) ---
            shoulder_dy = abs(ls[1] - rs[1])
            shoulder_width = abs(rs[0] - ls[0])
            if shoulder_width > 0:
                tilt_ratio = shoulder_dy / shoulder_width
                # Tilt ratio > 0.15 = notably tilted
                score -= min(tilt_ratio * 60, 15)  # up to -15pts

            # --- 2. Shoulder Width (breadth = confidence) ---
            # Normalize by frame width; broader → more confident
            normalized_width = shoulder_width / frame_width
            # Typical seated person: 0.25–0.45 of frame
            width_score = np.clip((normalized_width - 0.20) / 0.25, 0, 1) * 20
            score += width_score  # up to +20pts

            # --- 3. Slouch Detection (shoulder–hip vertical compression) ---
            if np.any(lh == 0) or np.any(rh == 0):
                pass  # Skip hip check if not visible
            else:
                hip_mid_y = (lh[1] + rh[1]) / 2.0
                shoulder_mid_y = (ls[1] + rs[1]) / 2.0
                torso_height = abs(hip_mid_y - shoulder_mid_y)
                # Normalize by frame height
                torso_ratio = torso_height / frame_height
                # Upright person has longer visible torso
                if torso_ratio < 0.15:
                    score -= 20  # Strong slouch
                elif torso_ratio < 0.22:
                    score -= 10  # Mild slouch
                else:
                    score += 10  # Upright posture bonus

            # --- 4. Forward Lean (engagement signal) ---
            if nose[0] > 0 and nose[1] > 0:
                shoulder_mid_x = (ls[0] + rs[0]) / 2.0
                shoulder_mid_y = (ls[1] + rs[1]) / 2.0
                # If nose is significantly above shoulder mid-Y, person leans in
                nose_lead = shoulder_mid_y - nose[1]  # positive = nose above shoulders
                lean_ratio = nose_lead / max(shoulder_width, 1)
                if lean_ratio > 0.6:
                    score += 10   # Leaning forward (engaged)
                elif lean_ratio < 0.1:
                    score -= 10  # Leaning back (withdrawn)

            score = float(np.clip(score, 0, 100))

            # --- Semantic Label ---
            shoulder_tilt = abs(ls[1] - rs[1]) / max(shoulder_width, 1)
            lean_label = ""
            if nose[0] > 0 and nose[1] > 0:
                shoulder_mid_y = (ls[1] + rs[1]) / 2.0
                lean_ratio = (shoulder_mid_y - nose[1]) / max(shoulder_width, 1)
                if lean_ratio > 0.6:
                    lean_label = ", leaning slightly forward"
                elif lean_ratio < 0.1:
                    lean_label = ", leaning back"

            if score >= 70:
                posture = f"Upright and open{lean_label}"
            elif score >= 45:
                posture = f"Neutral posture{lean_label}"
            else:
                posture = f"Slouched or closed{lean_label}"

            return score, posture

        except Exception as e:
            logger.warning(f"PostureAnalyzer error: {e}")
            return 50.0, "posture unclear"


# ---------------------------------------------------------------------------
# Fidget Tracker
# ---------------------------------------------------------------------------

class FidgetTracker:
    """
    Calculates a "fidget index" (0–100) by measuring rolling variance
    in wrist positions over a short history window.

    High-frequency, low-amplitude movement = fidgeting.
    Low variance = still / composed.
    """

    def __init__(self, history_len: int = 15):
        # Store normalized (x, y) per wrist, deque of fixed size
        self.l_history: deque = deque(maxlen=history_len)
        self.r_history: deque = deque(maxlen=history_len)

    def update(
        self,
        l_wrist: Optional[np.ndarray],
        r_wrist: Optional[np.ndarray],
        frame_w: int,
        frame_h: int
    ) -> tuple[float, str]:
        """Returns (fidget_index 0-100, gesture_state_label)."""
        if l_wrist is not None and l_wrist[0] > 0:
            self.l_history.append(l_wrist / np.array([frame_w, frame_h], dtype=float))
        if r_wrist is not None and r_wrist[0] > 0:
            self.r_history.append(r_wrist / np.array([frame_w, frame_h], dtype=float))

        fidget_index = self._compute_variance_score()
        return fidget_index

    def _compute_variance_score(self) -> float:
        variances = []
        for hist in (self.l_history, self.r_history):
            if len(hist) >= 5:
                arr = np.array(hist)
                variances.append(np.mean(np.var(arr, axis=0)))
        if not variances:
            return 0.0
        # Typical variance for stationary wrist ≈ 0.0001–0.001
        # Typical variance for fidgeting ≈ 0.002–0.01+
        raw = float(np.mean(variances))
        # Map to 0–100: threshold at 0.008 = max fidget
        score = np.clip(raw / 0.008, 0, 1) * 100
        return float(score)


# ---------------------------------------------------------------------------
# Gesture Classifier
# ---------------------------------------------------------------------------

class GestureClassifier:
    """
    Classifies arm/hand state into semantic labels using YOLO keypoints.
    Priority order (most salient first):
      crossed_arms > face_touching > hands_raised > open_gesturing > hands_resting
    """

    def classify(self, keypoints: np.ndarray, frame_h: int) -> str:
        try:
            ls, rs   = keypoints[KP_L_SHOULDER], keypoints[KP_R_SHOULDER]
            le, re   = keypoints[KP_L_ELBOW],    keypoints[KP_R_ELBOW]
            lw, rw   = keypoints[KP_L_WRIST],    keypoints[KP_R_WRIST]
            nose     = keypoints[KP_NOSE]

            def visible(p): return p[0] > 0 and p[1] > 0

            # --- Face touching: wrist near nose ---
            if visible(lw) and visible(nose):
                dist_l = np.linalg.norm(lw - nose)
                if dist_l < frame_h * 0.12:
                    return "touching face / chin"
            if visible(rw) and visible(nose):
                dist_r = np.linalg.norm(rw - nose)
                if dist_r < frame_h * 0.12:
                    return "touching face / chin"

            # --- Crossed arms: wrists cross body centerline ---
            if visible(ls) and visible(rs) and visible(lw) and visible(rw):
                body_center_x = (ls[0] + rs[0]) / 2.0
                l_crossed = lw[0] > body_center_x + abs(rs[0] - ls[0]) * 0.1
                r_crossed = rw[0] < body_center_x - abs(rs[0] - ls[0]) * 0.1
                if l_crossed and r_crossed:
                    return "arms crossed"

            # --- Hands raised high (above shoulders) ---
            l_high = visible(lw) and visible(ls) and lw[1] < ls[1]
            r_high = visible(rw) and visible(rs) and rw[1] < rs[1]
            if l_high or r_high:
                return "hands raised"

            # --- Active gesturing (wrists above elbows = forearms up) ---
            l_gesture = visible(lw) and visible(le) and lw[1] < le[1]
            r_gesture = visible(rw) and visible(re) and rw[1] < re[1]
            if l_gesture and r_gesture:
                return "gesturing with both hands"
            if l_gesture or r_gesture:
                return "gesturing with one hand"

            return "hands at rest"
        except Exception:
            return "hands at rest"


# ---------------------------------------------------------------------------
# Emotion Tracker (Transition & Micro-Expression Detection)
# ---------------------------------------------------------------------------

class EmotionTracker:
    """
    Maintains a short emotion history to detect:
      - State transitions (e.g., neutral → anxious)
      - Micro-expressions: brief high-confidence spikes of a non-baseline emotion
      - Smoothed dominant emotion via exponential moving average
    """

    def __init__(self, ema_alpha: float = 0.3, micro_threshold: float = 0.55):
        self.ema_alpha = ema_alpha
        self.micro_threshold = micro_threshold
        self.ema_scores: Optional[np.ndarray] = None
        self.prev_dominant: Optional[str] = None
        self.score_history: deque = deque(maxlen=30)  # ~10s at 3fps
        self.micro_event: Optional[str] = None

    def update(self, scores: np.ndarray) -> tuple[str, Optional[str]]:
        """
        Args:
            scores: raw HSEmotion probability array (8,)
        Returns:
            (smoothed_dominant_emotion, notable_event_or_None)
        """
        self.score_history.append(scores.copy())
        self.micro_event = None

        # --- EMA smoothing ---
        if self.ema_scores is None:
            self.ema_scores = scores.copy()
        else:
            self.ema_scores = self.ema_alpha * scores + (1 - self.ema_alpha) * self.ema_scores

        smoothed_dominant_idx = int(np.argmax(self.ema_scores))
        smoothed_dominant = EMOTION_LABELS[smoothed_dominant_idx]

        # --- Micro-expression detection ---
        # Look for a raw score spike (single frame) for a non-dominant emotion
        raw_dominant_idx = int(np.argmax(scores))
        raw_dominant = EMOTION_LABELS[raw_dominant_idx]
        if (raw_dominant != smoothed_dominant and
                scores[raw_dominant_idx] > self.micro_threshold):
            self.micro_event = f"micro-expression of {raw_dominant.lower()} detected"

        # --- State transition detection ---
        notable = None
        if self.prev_dominant and self.prev_dominant != smoothed_dominant:
            notable = f"emotion shifted from {self.prev_dominant.lower()} to {smoothed_dominant.lower()}"
        elif self.micro_event:
            notable = self.micro_event

        # Broad smile detection (Happy score spikes above 0.75)
        if EMOTION_LABELS.index('Happy') < len(scores):
            if scores[EMOTION_LABELS.index('Happy')] > 0.75:
                notable = "user smiled broadly"

        self.prev_dominant = smoothed_dominant
        return smoothed_dominant, notable

    def get_window_emotion(self) -> tuple[str, str]:
        """
        Aggregates emotion history over the current window.
        Returns (primary_emotion_label, confidence_str).
        """
        if len(self.score_history) < 2:
            return "Neutral", "Low"
        arr = np.array(self.score_history)
        mean_scores = arr.mean(axis=0)
        dominant_idx = int(np.argmax(mean_scores))
        dominant = EMOTION_LABELS[dominant_idx]
        confidence = float(mean_scores[dominant_idx])

        # Map to secondary emotion if close runner-up
        sorted_idxs = np.argsort(mean_scores)[::-1]
        runner_up = EMOTION_LABELS[sorted_idxs[1]]
        runner_up_score = float(mean_scores[sorted_idxs[1]])

        label = dominant
        if runner_up_score > 0.25 and runner_up != dominant:
            label = f"{dominant} / {runner_up}"

        if confidence > 0.60:
            conf_str = "High"
        elif confidence > 0.40:
            conf_str = "Medium"
        else:
            conf_str = "Low"

        # Clear history after window read
        self.score_history.clear()
        return label, conf_str


# ---------------------------------------------------------------------------
# Temporal Buffer & Summarizer
# ---------------------------------------------------------------------------

class TemporalBuffer:
    """Accumulates FrameSignals over a fixed time window and produces WindowSummary."""

    def __init__(self, window_secs: float = 5.0):
        self.window_secs = window_secs
        self.signals: list[FrameSignal] = []
        self.window_start: float = time.time()
        self.window_index: int = 0

    def add(self, signal: FrameSignal):
        self.signals.append(signal)

    def is_window_ready(self) -> bool:
        return (time.time() - self.window_start) >= self.window_secs

    def flush(self, emotion_tracker: "EmotionTracker") -> Optional[WindowSummary]:
        if not self.signals:
            return None

        signals = self.signals
        self.signals = []
        t_start = self.window_start
        t_end = time.time()
        self.window_start = t_end
        idx = self.window_index
        self.window_index += 1

        def fmt_time(t: float) -> str:
            total_secs = int(t - (t_start - idx * self.window_secs))
            return f"{total_secs // 60:02d}:{total_secs % 60:02d}"

        window_label = f"{fmt_time(t_start)} – {fmt_time(t_end)}"

        # --- Emotion ---
        primary_emotion, emotion_confidence = emotion_tracker.get_window_emotion()

        # --- Posture / Confidence ---
        conf_scores = [s.confidence_score for s in signals]
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 50.0
        if avg_conf >= 65:
            confidence_level = "High"
        elif avg_conf >= 40:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        # Most common posture label
        posture_labels = [s.gesture_state for s in signals]  # reuse field? No:
        # Use the posture computed per-frame (stored in a dedicated field below)
        posture_states = [s.__dict__.get('posture_label', 'neutral') for s in signals]
        posture_state = _most_common(posture_states) if posture_states else "unclear"

        # --- Gestures / Fidget ---
        gesture_states = [s.gesture_state for s in signals]
        dominant_gesture = _most_common(gesture_states)
        fidget_vals = [s.__dict__.get('fidget_index', 0.0) for s in signals]
        avg_fidget = float(np.mean(fidget_vals)) if fidget_vals else 0.0

        if avg_fidget > 60:
            fidget_str = "notable fidgeting / restlessness"
        elif avg_fidget > 30:
            fidget_str = "some hand movement"
        else:
            fidget_str = "minimal fidgeting"

        gesture_summary = f"{dominant_gesture.capitalize()}, {fidget_str}"

        # --- Attention / Eye Contact ---
        yaws   = [s.head_yaw for s in signals]
        pitches = [s.head_pitch for s in signals]
        avg_yaw   = float(np.mean(np.abs(yaws)))
        avg_pitch = float(np.mean(np.abs(pitches)))

        # Thresholds: ±15° yaw / ±10° pitch considered "on camera"
        on_camera_frames = sum(
            1 for s in signals
            if abs(s.head_yaw) < 15 and abs(s.head_pitch) < 12
        )
        attention_ratio = on_camera_frames / max(len(signals), 1)

        if attention_ratio > 0.80:
            attention = "Sustained eye contact with camera"
        elif attention_ratio > 0.50:
            attention = "Intermittent eye contact, some distraction"
        else:
            attention = "Mostly looking away from camera"

        # --- Notable Events ---
        notable_events = [s.__dict__.get('notable_event') for s in signals
                          if s.__dict__.get('notable_event')]
        notable = notable_events[-1] if notable_events else None
        if notable:
            notable = notable.capitalize()

        raw_metrics = {
            "avg_confidence_score": round(avg_conf, 1),
            "avg_fidget_index": round(avg_fidget, 1),
            "avg_head_yaw_deg": round(float(np.mean(yaws)), 1),
            "avg_head_pitch_deg": round(float(np.mean(pitches)), 1),
            "attention_ratio": round(attention_ratio, 2),
            "frame_count": len(signals),
        }

        return WindowSummary(
            timestamp_window=window_label,
            primary_emotion=primary_emotion,
            emotion_confidence=emotion_confidence,
            posture_state=posture_state,
            confidence_level=confidence_level,
            gesture_summary=gesture_summary,
            attention=attention,
            notable_event=notable,
            raw_metrics=raw_metrics,
        )


def _most_common(lst: list) -> str:
    if not lst:
        return "unknown"
    return max(set(lst), key=lst.count)


# ---------------------------------------------------------------------------
# Extended FrameSignal (with extra fields the buffer needs)
# ---------------------------------------------------------------------------

@dataclass
class RichFrameSignal(FrameSignal):
    posture_label: str = "unclear"
    fidget_index: float = 0.0
    notable_event: Optional[str] = None


# ---------------------------------------------------------------------------
# Main Processor
# ---------------------------------------------------------------------------

class MultimodalVisionProcessor(FrameProcessor):
    """
    Production-grade vision processor for real-time behavioral analysis.

    Pipeline per frame:
      decode → YOLOv11-pose → head pose (solvePnP) → posture score →
      gesture classification → face crop → HSEmotion → fidget tracker →
      emotion tracker → temporal buffer → [every 5s] semantic LLM summary
    """

    def __init__(
        self,
        target_fps: float = 3.0,
        yolo_model: str = "yolo11n-pose.pt",
        window_secs: float = 5.0,
    ):
        super().__init__()
        self.target_fps = target_fps
        self.min_frame_interval = 1.0 / target_fps
        self.last_process_time: float = 0.0
        self.window_secs = window_secs

        # Track frame dimensions for lazy camera calibration
        self._last_frame_size: tuple[int, int] = (640, 480)

        logger.info("Loading Vision Models...")
        self.yolo = YOLO(yolo_model, task='pose')

        # Patch torch.load to handle PyTorch 2.6+ weights_only default change
        original_load = torch.load
        def legacy_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        try:
            torch.load = legacy_load
            self.emotion_recognizer = HSEmotionRecognizer(
                model_name='enet_b0_8_best_vgaf',
                device='cpu'
            )
        finally:
            torch.load = original_load

        logger.success("Vision Models loaded.")

        # Sub-components
        self.head_pose_estimator = HeadPoseEstimator(640, 480)
        self.posture_analyzer    = PostureAnalyzer()
        self.fidget_tracker      = FidgetTracker(history_len=15)
        self.gesture_classifier  = GestureClassifier()
        self.emotion_tracker     = EmotionTracker(ema_alpha=0.3, micro_threshold=0.55)
        self.temporal_buffer     = TemporalBuffer(window_secs=window_secs)
        self._is_processing = False

    # ------------------------------------------------------------------
    # Frame entry point
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # 1. Always push the frame down for Audio/VAD to work!
        await self.push_frame(frame, direction)

        if not isinstance(frame, UserImageRawFrame):
            return

        current_time = time.time()
        if current_time - self.last_process_time < self.min_frame_interval:
            return

        # 2. ADD THIS: If vision is already analyzing a frame, skip this one
        if self._is_processing:
            return 

        self.last_process_time = current_time
        
        # 3. ADD THIS: Lock the state while processing
        self._is_processing = True
        try:
            # Offload heavy inference to a thread pool
            signal = await asyncio.to_thread(self._analyze_frame, frame)
        finally:
            # Unlock the state when done
            self._is_processing = False

        if signal is None:
            return
        # Push real-time per-frame update to the frontend
        realtime_payload = {
            "type": "vision_frame_update",
            "emotion": signal.dominant_emotion,
            "confidence_score": round(signal.confidence_score),
            "gesture": signal.gesture_state,
            "head_yaw": round(signal.head_yaw, 1),
            "head_pitch": round(signal.head_pitch, 1),
            "fidget_index": round(signal.fidget_index, 1),
        }
        await self.push_frame(
            OutputTransportMessageFrame(message=realtime_payload), direction
        )

        # Accumulate into temporal buffer
        self.temporal_buffer.add(signal)

        # Every N seconds: flush buffer → produce LLM-ready semantic summary
        if self.temporal_buffer.is_window_ready():
            summary = self.temporal_buffer.flush(self.emotion_tracker)
            if summary:
                await self._dispatch_llm_summary(summary, direction)

    # ------------------------------------------------------------------
    # Frame Analysis (runs in thread)
    # ------------------------------------------------------------------

    def _analyze_frame(self, frame: UserImageRawFrame) -> Optional[RichFrameSignal]:
        try:
            img = self._decode_frame(frame)
            if img is None:
                return None

            h, w = img.shape[:2]

            # Lazy camera matrix update
            if (w, h) != self._last_frame_size:
                self.head_pose_estimator.update_camera(w, h)
                self._last_frame_size = (w, h)

            # --- YOLO Pose Inference ---
            results = self.yolo(img, verbose=False, conf=0.45, device="cpu")

            if not results or len(results[0].boxes) == 0:
                return RichFrameSignal(
                    timestamp=time.time(),
                    emotion_scores=np.ones(8) / 8,
                    dominant_emotion="Neutral",
                    confidence_score=50.0,
                    head_yaw=0.0, head_pitch=0.0, head_roll=0.0,
                    l_wrist_pos=None, r_wrist_pos=None,
                    gesture_state="not visible",
                    keypoints_visible=False,
                    posture_label="not visible",
                    fidget_index=0.0,
                    notable_event="User is not visible on camera",
                )

            # Take the highest-confidence detection
            boxes = results[0].boxes
            best_idx = int(torch.argmax(boxes.conf).item())
            keypoints = results[0].keypoints.xy[best_idx].cpu().numpy()

            # --- Head Pose ---
            yaw, pitch, roll = self.head_pose_estimator.estimate(keypoints)

            # --- Posture ---
            conf_score, posture_label = self.posture_analyzer.analyze(keypoints, w, h)

            # --- Gesture ---
            gesture_state = self.gesture_classifier.classify(keypoints, h)

            # --- Fidget ---
            lw = keypoints[KP_L_WRIST] if len(keypoints) > KP_L_WRIST else None
            rw = keypoints[KP_R_WRIST] if len(keypoints) > KP_R_WRIST else None
            fidget_index = self.fidget_tracker.update(lw, rw, w, h)

            # --- Emotion (HSEmotion on face crop) ---
            emotion_scores, dominant_emotion = self._run_emotion(img, keypoints, w, h)

            # --- Emotion Tracker (EMA + transitions) ---
            smoothed_emotion, notable_event = self.emotion_tracker.update(emotion_scores)

            return RichFrameSignal(
                timestamp=time.time(),
                emotion_scores=emotion_scores,
                dominant_emotion=smoothed_emotion,
                confidence_score=conf_score,
                head_yaw=yaw,
                head_pitch=pitch,
                head_roll=roll,
                l_wrist_pos=lw,
                r_wrist_pos=rw,
                gesture_state=gesture_state,
                keypoints_visible=True,
                posture_label=posture_label,
                fidget_index=fidget_index,
                notable_event=notable_event,
            )

        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_frame(self, frame: UserImageRawFrame) -> Optional[np.ndarray]:
        """Decode raw bytes to BGR OpenCV image."""
        try:
            img_array = np.frombuffer(frame.image, dtype=np.uint8)
            width, height = frame.size
            n_pixels = width * height

            if len(img_array) == n_pixels * 4:
                img = img_array.reshape((height, width, 4))
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif len(img_array) == n_pixels * 3:
                img = img_array.reshape((height, width, 3))
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                # Try JPEG decode as fallback (some WebRTC stacks send encoded frames)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
                logger.warning(f"Unsupported frame format: {len(img_array)} bytes for {width}x{height}")
                return None
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None

    def _run_emotion(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        w: int, h: int
    ) -> tuple[np.ndarray, str]:
        """Crop face region and run HSEmotion. Returns (scores_array, dominant_label)."""
        default_scores = np.zeros(8)
        default_scores[EMOTION_LABELS.index('Neutral')] = 1.0

        try:
            nose = keypoints[KP_NOSE]
            l_eye = keypoints[KP_L_EYE]
            r_eye = keypoints[KP_R_EYE]

            # Prefer eye-based crop (more accurate than nose heuristic)
            if l_eye[0] > 0 and r_eye[0] > 0:
                iod = np.linalg.norm(r_eye - l_eye)
                cx = float((l_eye[0] + r_eye[0]) / 2)
                cy = float((l_eye[1] + r_eye[1]) / 2)
                # Face bounding box: ~3× inter-ocular distance
                half_size = int(iod * 1.8)
                fx1 = max(0, int(cx - half_size))
                fy1 = max(0, int(cy - half_size))
                fx2 = min(w, int(cx + half_size))
                fy2 = min(h, int(cy + int(iod * 2.5)))
            elif nose[0] > 0 and nose[1] > 0:
                # Fallback: nose-centered crop
                face_size = 80
                fx1 = max(0, int(nose[0] - face_size))
                fy1 = max(0, int(nose[1] - face_size))
                fx2 = min(w, int(nose[0] + face_size))
                fy2 = min(h, int(nose[1] + face_size))
            else:
                return default_scores, "Neutral"

            face_crop = img[fy1:fy2, fx1:fx2]
            if face_crop.size < 400:  # Too small to be useful
                return default_scores, "Neutral"

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            emotion_label, scores_dict = self.emotion_recognizer.predict_emotions(
                face_rgb, logits=True
            )

            # Convert scores dict to ordered numpy array
            if isinstance(scores_dict, dict):
                scores_arr = np.array([scores_dict.get(e, 0.0) for e in EMOTION_LABELS], dtype=np.float32)
                # Softmax normalize
                scores_arr = np.exp(scores_arr - np.max(scores_arr))
                scores_arr /= scores_arr.sum()
            else:
                scores_arr = np.array(scores_dict, dtype=np.float32)
                if scores_arr.sum() > 0:
                    scores_arr /= scores_arr.sum()

            return scores_arr, emotion_label

        except Exception as e:
            logger.warning(f"Emotion inference error: {e}")
            return default_scores, "Neutral"

    async def _dispatch_llm_summary(self, summary: WindowSummary, direction: FrameDirection):
        """Formats the window summary and injects it into the LLM pipeline."""

        llm_json = {
            "timestamp_window": summary.timestamp_window,
            "primary_emotion": summary.primary_emotion,
            "posture_state": summary.posture_state,
            "confidence_level": summary.confidence_level,
            "gesture_summary": summary.gesture_summary,
            "attention": summary.attention,
        }
        if summary.notable_event:
            llm_json["notable_event"] = summary.notable_event

        json_str = json.dumps(llm_json, indent=2)
        system_msg = (
            f"[VISUAL CONTEXT UPDATE — DO NOT READ ALOUD OR REFERENCE DIRECTLY. "
            f"Use this as background context to adjust your tone and responses subtly.]\n"
            f"{json_str}"
        )

        logger.info(f"📊 Vision Summary:\n{json_str}")

        # Inject into LLM context
        await self.push_frame(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": system_msg}]),
            direction
        )

        # Also push full payload (including raw metrics) to frontend for debugging/UI
        await self.push_frame(
            OutputTransportMessageFrame(message={
                "type": "vision_window_summary",
                "data": {**llm_json, "raw_metrics": summary.raw_metrics},
            }),
            direction
        )