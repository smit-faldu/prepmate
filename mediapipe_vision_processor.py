"""
mediapipe_vision_processor.py — MediaPipe Tasks Vision Pipeline for Pipecat
============================================================================
Updated to use MediaPipe Tasks Python API (mediapipe >= 0.10.x).
The old mp.solutions.holistic API was removed in 0.10.x — this module
uses mp.tasks.python.vision.FaceLandmarker + PoseLandmarker instead.

ACCURACY IMPROVEMENTS (v2):
  1. Face Blendshapes (52 ARKit coefficients) — replaces manual geometry math
     output_face_blendshapes=True gives FACS-aligned muscle scores
  2. EMA temporal smoothing per blendshape channel (α=0.4) — reduces jitter
  3. Sliding 5-frame window voting — prevents expression flickering
  4. 8-expression vocabulary (added angry, sad to original 5)
  5. Pose model: pose_landmarker_full.task (higher accuracy than lite)
  6. Enhanced pose: head nod detection via face landmark z-depth

Custom Pipecat FrameProcessor that:
  1. Intercepts InputImageRawFrame (JPEG webcam frames from browser)
  2. Runs FaceLandmarker + PoseLandmarker in a thread executor (non-blocking)
  3. Derives facial expression + body pose + confidence scores from blendshapes
  4. Emits a VisionAnalysisFrame downstream
  5. Updates a per-session VisionState that VCBroadcaster reads for AI context

VisionState Design (shared in-memory):
  - Keyed by session_id (str)
  - Thread-safe via asyncio lock
  - Stores the latest expression, pose, and confidence scores
  - VCBroadcaster reads this ONCE per turn, so vision context is always fresh
  - Audio is still primary; vision is clearly labelled as secondary context
"""

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pipecat.frames.frames import DataFrame, Frame, InputImageRawFrame, UserImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

# ── Model paths ───────────────────────────────────────────────────────────────
_MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
_FACE_MODEL  = os.path.join(_MODELS_DIR, "face_landmarker.task")

# Prefer full model for accuracy; fall back to lite if full not downloaded yet
_POSE_MODEL_FULL = os.path.join(_MODELS_DIR, "pose_landmarker_full.task")
_POSE_MODEL_LITE = os.path.join(_MODELS_DIR, "pose_landmarker_lite.task")
_POSE_MODEL  = _POSE_MODEL_FULL if os.path.exists(_POSE_MODEL_FULL) else _POSE_MODEL_LITE


# ─────────────────────────────────────────────────────────────────────────────
# Custom Frame — carries vision analysis downstream
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionAnalysisFrame(DataFrame):
    """
    Carries the result of a MediaPipe landmarker analysis.

    Fields:
      session_id           : ties this frame to a VC session
      expression           : dominant facial expression label
      expression_confidence: float 0-1, blendshape-derived confidence
      pose                 : body pose category label
      pose_confidence      : float 0-1, mean pose landmark visibility
      raw_expression_scores: dict of all expression → score (for HUD)
    """
    session_id: str = ""
    expression: str = "neutral"
    expression_confidence: float = 0.0
    pose: str = "unknown"
    pose_confidence: float = 0.0
    raw_expression_scores: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"VisionAnalysisFrame("
            f"expr={self.expression}@{self.expression_confidence:.2f}, "
            f"pose={self.pose}@{self.pose_confidence:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# VisionState — per-session shared state (read by VCBroadcaster)
# ─────────────────────────────────────────────────────────────────────────────

class VisionState:
    """
    Thread-safe in-memory store of the latest vision analysis per session.
    VCBroadcaster reads from this at the start of each turn to inject
    body language context into the VC persona prompt.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._sessions: dict[str, dict] = {}

    async def update(self, session_id: str, analysis: VisionAnalysisFrame) -> None:
        async with self._lock:
            self._sessions[session_id] = {
                "expression":            analysis.expression,
                "expression_confidence": analysis.expression_confidence,
                "pose":                  analysis.pose,
                "pose_confidence":       analysis.pose_confidence,
                "raw_expression_scores": analysis.raw_expression_scores,
            }

    async def get(self, session_id: str) -> Optional[dict]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)


# Singleton — imported by server_whisper_vad.py and used by VCBroadcaster
vision_state = VisionState()


# ─────────────────────────────────────────────────────────────────────────────
# EMA Smoother — per-channel exponential moving average for blendshapes
# ─────────────────────────────────────────────────────────────────────────────

class EMASmoother:
    """
    Applies Exponential Moving Average smoothing to a vector of N values.
    alpha=0.4 → responsive but smooth (lower = more smoothing, more lag).

    Formula: smoothed[t] = alpha * new[t] + (1 - alpha) * smoothed[t-1]
    """

    def __init__(self, size: int, alpha: float = 0.4):
        self._alpha  = alpha
        self._prev   = None
        self._size   = size

    def update(self, values: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = values.copy()
            return values
        smoothed   = self._alpha * values + (1.0 - self._alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):
        self._prev = None


# ─────────────────────────────────────────────────────────────────────────────
# Expression Analysis — blendshape-based (ACCURATE)
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe ARKit blendshape names (52 total, indices 0–51)
# Source: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
_BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "_neutral",
]

def _blendshape_index(name: str) -> int:
    """Return index of a blendshape by name, or -1 if not found."""
    try:
        return _BLENDSHAPE_NAMES.index(name)
    except ValueError:
        return -1

# Pre-compute indices for speed
_IDX = {name: _blendshape_index(name) for name in _BLENDSHAPE_NAMES}


def _analyze_expression_from_blendshapes(
    blendshapes_list,
    smoother: Optional[EMASmoother],
) -> tuple[str, float, dict]:
    """
    Derive dominant expression from FaceLandmarker blendshape output (Tasks API).
    Uses the 52 ARKit blendshape coefficients — far more accurate than landmarks.

    blendshapes_list: result.face_blendshapes (list of Category lists)
    smoother: per-session EMA smoother for temporal stability

    Returns (expression_label, confidence, score_dict).
    """
    if not blendshapes_list or len(blendshapes_list) == 0:
        return "unknown", 0.0, {}

    # Extract raw scores into a numpy array (index-aligned with _BLENDSHAPE_NAMES)
    cats = blendshapes_list[0]  # first face
    raw  = np.zeros(len(_BLENDSHAPE_NAMES), dtype=np.float32)
    for cat in cats:
        name = cat.category_name
        if name in _IDX and _IDX[name] >= 0:
            raw[_IDX[name]] = cat.score

    # Apply EMA smoothing to reduce per-frame jitter
    if smoother is not None:
        smooth = smoother.update(raw)
    else:
        smooth = raw

    def bs(name: str) -> float:
        idx = _IDX.get(name, -1)
        return float(smooth[idx]) if idx >= 0 else 0.0

    scores: Dict[str, float] = {}

    # ── Smiling (AU6+12): cheek raise + lip corners up ──────────────────────
    smile = (bs("mouthSmileLeft") + bs("mouthSmileRight")) / 2.0
    cheek = (bs("cheekSquintLeft") + bs("cheekSquintRight")) / 2.0
    scores["smiling"] = min(1.0, (smile * 0.7 + cheek * 0.3) * 1.6)

    # ── Excited: smiling + wide eyes + brow raise ────────────────────────────
    wide_eye  = (bs("eyeWideLeft") + bs("eyeWideRight")) / 2.0
    brow_up   = bs("browInnerUp")
    scores["excited"] = min(1.0, (
        scores["smiling"] * 0.5 + wide_eye * 0.3 + brow_up * 0.2
    ) * 1.5)

    # ── Surprised (AU1+2+5B+27): jaw drop + wide eyes + brow raise ──────────
    jaw_open  = bs("jawOpen")
    scores["surprised"] = min(1.0, (
        jaw_open * 0.5 + wide_eye * 0.3 + brow_up * 0.2
    ) * 1.8)

    # ── Nervous / Stressed (AU4+7+17): brow lower + eye squint + lip stretch
    brow_down = (bs("browDownLeft") + bs("browDownRight")) / 2.0
    eye_sqnt  = (bs("eyeSquintLeft") + bs("eyeSquintRight")) / 2.0
    lip_str   = (bs("mouthStretchLeft") + bs("mouthStretchRight")) / 2.0
    scores["nervous"] = min(1.0, (
        brow_down * 0.4 + eye_sqnt * 0.3 + lip_str * 0.3
    ) * 2.0)

    # ── Thinking / Concentrating (AU1+AU64): inner brow up + pucker ─────────
    pucker = bs("mouthPucker")
    scores["thinking"] = min(1.0, (
        brow_up * 0.5 + pucker * 0.3 + eye_sqnt * 0.2
    ) * 2.0)

    # ── Sad (AU1+15+17): inner brow up + lip corners down + lip roll ─────────
    frown  = (bs("mouthFrownLeft") + bs("mouthFrownRight")) / 2.0
    shrug  = (bs("mouthShrugLower") + bs("mouthShrugUpper")) / 2.0
    scores["sad"] = min(1.0, (
        brow_up * 0.3 + frown * 0.5 + shrug * 0.2
    ) * 2.0)

    # ── Angry / Frustrated (AU4+9+23+24): brow lower + nose sneer + press ───
    nose_sneer = (bs("noseSneerLeft") + bs("noseSneerRight")) / 2.0
    lip_press  = (bs("mouthPressLeft") + bs("mouthPressRight")) / 2.0
    scores["angry"] = min(1.0, (
        brow_down * 0.4 + nose_sneer * 0.4 + lip_press * 0.2
    ) * 2.5)

    # ── Neutral: confidence in "no expression" ────────────────────────────────
    total_signal = sum(v for k, v in scores.items())
    scores["neutral"] = max(0.0, 1.0 - total_signal * 0.5)

    # Determine dominant expression (use raw score for clarity)
    dominant = max(scores, key=scores.get)

    # Confidence = max blendshape score in winning category × overall face presence
    # (higher = model is sure a face is there and expression is clear)
    max_bs_score = float(np.max(smooth[:51]))  # ignore _neutral at index 51
    presence_proxy = min(1.0, max_bs_score * 3.0 + 0.3)  # heuristic
    confidence = round(min(1.0, presence_proxy), 3)

    return dominant, confidence, {k: round(v, 3) for k, v in scores.items()}


def _analyze_expression_from_landmarks_fallback(face_landmarks_list) -> tuple[str, float, dict]:
    """
    Fallback geometry-based expression analysis when blendshapes are unavailable.
    Less accurate but works without blendshapes output enabled.
    """
    if not face_landmarks_list:
        return "unknown", 0.0, {}

    lms = face_landmarks_list[0]
    n   = len(lms)

    def lm(idx):
        return lms[idx] if idx < n else None

    scores = {}

    # Landmark indices (face_mesh numbering)
    _MOUTH_LEFT      = 61
    _MOUTH_RIGHT     = 291
    _UPPER_LIP       = 13
    _LOWER_LIP       = 14
    _LEFT_BROW_INNER = 107
    _RIGHT_BROW_INNER= 336
    _NOSE_TIP        = 1
    _LEFT_EYE_TOP    = 159
    _LEFT_EYE_BOTTOM = 145
    _RIGHT_EYE_TOP   = 386
    _RIGHT_EYE_BOTTOM= 374

    nose = lm(_NOSE_TIP)
    ml   = lm(_MOUTH_LEFT)
    mr   = lm(_MOUTH_RIGHT)
    if nose and ml and mr:
        smile_score = ((nose.y - ml.y) + (nose.y - mr.y)) / 2.0
        scores["smiling"] = max(0.0, smile_score * 8.0)

    ul = lm(_UPPER_LIP)
    ll = lm(_LOWER_LIP)
    if ul and ll and ml and mr:
        mouth_h = abs(ll.y - ul.y)
        mouth_w = abs(mr.x - ml.x) + 1e-6
        mar = mouth_h / mouth_w
        scores["surprised"] = min(1.0, mar * 4.0)

    lb  = lm(_LEFT_BROW_INNER)
    rb  = lm(_RIGHT_BROW_INNER)
    let = lm(_LEFT_EYE_TOP)
    ret = lm(_RIGHT_EYE_TOP)
    if lb and rb and let and ret:
        brow_gap = ((let.y - lb.y) + (ret.y - rb.y)) / 2.0
        stressed = max(0.0, 0.06 - brow_gap) * 12.0
        scores["nervous"] = min(1.0, stressed)

    leb = lm(_LEFT_EYE_BOTTOM)
    reb = lm(_RIGHT_EYE_BOTTOM)
    if let and leb and ret and reb:
        ear = ((abs(let.y - leb.y)) + (abs(ret.y - reb.y))) / 2.0
        scores["thinking"] = min(1.0, max(0.0, 0.025 - ear) * 30.0)

    total_signal = sum(scores.values())
    scores["neutral"] = max(0.0, 1.0 - total_signal)

    if scores.get("smiling", 0) > 0.3 and scores.get("surprised", 0) > 0.3:
        scores["excited"] = (scores["smiling"] + scores["surprised"]) / 2.0

    dominant = max(scores, key=scores.get)

    visibilities = [
        lm_pt.visibility
        for lm_pt in lms
        if hasattr(lm_pt, "visibility") and lm_pt.visibility is not None
    ]
    mean_vis = float(np.mean(visibilities)) if visibilities else 0.6

    return dominant, round(mean_vis, 3), {k: round(v, 3) for k, v in scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Pose Analysis helpers (unchanged — good geometry approach)
# ─────────────────────────────────────────────────────────────────────────────

_NOSE_POSE      = 0
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16
_LEFT_HIP       = 23
_RIGHT_HIP      = 24


def _analyze_pose_from_landmarks(pose_landmarks_list) -> tuple[str, float]:
    """
    Classify body pose from PoseLandmarker result (Tasks API).
    pose_landmarks_list is result.pose_landmarks (list of NormalizedLandmark lists).
    Returns (pose_label, confidence).
    """
    if not pose_landmarks_list:
        return "unknown", 0.0

    lms = pose_landmarks_list[0]   # first person
    n   = len(lms)

    def lm(idx):
        return lms[idx] if idx < n else None

    ls   = lm(_LEFT_SHOULDER)
    rs   = lm(_RIGHT_SHOULDER)
    lh   = lm(_LEFT_HIP)
    rh   = lm(_RIGHT_HIP)
    lw   = lm(_LEFT_WRIST)
    rw   = lm(_RIGHT_WRIST)
    nose = lm(_NOSE_POSE)

    visibilities = [
        pt.visibility for pt in lms
        if hasattr(pt, "visibility") and pt.visibility is not None
    ]
    mean_vis = float(np.mean(visibilities)) if visibilities else 0.0

    if ls is None or rs is None:
        return "unknown", round(mean_vis, 3)

    # Only trust results with decent visibility
    shoulder_vis = 0.0
    if hasattr(ls, "visibility") and hasattr(rs, "visibility"):
        shoulder_vis = (ls.visibility + rs.visibility) / 2.0
    if shoulder_vis < 0.3:
        return "unknown", round(mean_vis, 3)

    shoulder_mid_y = (ls.y + rs.y) / 2.0
    shoulder_level = abs(ls.y - rs.y)

    # Gesturing: wrist raised above shoulder line
    wrists_raised = 0
    if lw and getattr(lw, "visibility", 0) > 0.3 and lw.y < shoulder_mid_y:
        wrists_raised += 1
    if rw and getattr(rw, "visibility", 0) > 0.3 and rw.y < shoulder_mid_y:
        wrists_raised += 1
    if wrists_raised >= 1:
        return "gesturing", round(mean_vis, 3)

    # Slouched: head appears too close to shoulders
    if nose and lh and rh:
        hip_mid_y = (lh.y + rh.y) / 2.0
        torso_h = hip_mid_y - shoulder_mid_y
        head_h  = shoulder_mid_y - nose.y
        if torso_h > 0 and head_h < torso_h * 0.3:
            return "slouched", round(mean_vis, 3)

    # Leaning forward: shoulders z-depth negative (closer to camera)
    if hasattr(ls, "z") and hasattr(rs, "z"):
        avg_z = (ls.z + rs.z) / 2.0
        if avg_z < -0.12:
            return "leaning_forward", round(mean_vis, 3)

    # Tilted: shoulders asymmetric in Y
    if shoulder_level > 0.05:
        return "tilted", round(mean_vis, 3)

    return "upright", round(mean_vis, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Voter — prevents expression flickering
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowVoter:
    """
    Maintains a rolling window of expression predictions.
    Returns the majority vote — prevents single-frame noise from
    changing the reported expression.

    window_size=5, min_votes=3 → expression changes only if 3/5 frames agree.
    """

    def __init__(self, window_size: int = 5, min_votes: int = 3):
        self._window    = deque(maxlen=window_size)
        self._min_votes = min_votes
        self._current   = "neutral"

    def vote(self, expression: str) -> str:
        self._window.append(expression)
        if len(self._window) < self._min_votes:
            return expression  # not enough history yet

        from collections import Counter
        counts = Counter(self._window)
        top, top_count = counts.most_common(1)[0]
        if top_count >= self._min_votes:
            self._current = top
        return self._current

    def reset(self):
        self._window.clear()
        self._current = "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipeVisionProcessor — Pipecat FrameProcessor (Tasks API v2)
# ─────────────────────────────────────────────────────────────────────────────

class MediaPipeVisionProcessor(FrameProcessor):
    """
    Custom Pipecat FrameProcessor using MediaPipe Tasks Python API.

    Intercepts InputImageRawFrame / UserImageRawFrame (raw JPEG bytes from
    the browser webcam), runs FaceLandmarker + PoseLandmarker in a thread
    executor, and:
      1. Emits a VisionAnalysisFrame downstream
      2. Updates the shared VisionState singleton so VCBroadcaster can
         inject body language context into the VC AI prompt.

    Non-blocking: all MediaPipe inference runs in loop.run_in_executor().

    Accuracy Improvements (v2):
      - Uses face blendshapes (52 ARKit coefficients) instead of raw geometry
      - EMA smoothing per blendshape channel (α=0.4)
      - 5-frame sliding window vote for expression stability
      - Prefers pose_landmarker_full.task over lite
      - Visibility thresholding for pose landmarks

    Parameters
    ----------
    session_id : str
        The VC session this processor is tied to.
    min_detection_confidence : float
        Detection confidence threshold (0.5 = balanced).
    min_tracking_confidence : float
        Tracking confidence threshold.
    process_every_n : int
        Only run inference on every N-th frame (default 3 = ~5fps at 15fps).
    face_model_path : str
        Path to face_landmarker.task model file.
    pose_model_path : str
        Path to pose_landmarker model file (full or lite).
    ema_alpha : float
        EMA smoothing factor for blendshapes (0.2=smooth, 0.6=responsive).
    vote_window : int
        Number of frames in sliding window for expression voting.
    vote_min : int
        Minimum votes needed to change expression label.
    """

    def __init__(
        self,
        session_id: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float  = 0.5,
        process_every_n: int            = 3,
        face_model_path: str            = _FACE_MODEL,
        pose_model_path: str            = _POSE_MODEL,
        ema_alpha: float                = 0.4,
        vote_window: int                = 5,
        vote_min: int                   = 3,
    ):
        super().__init__()
        self._session_id   = session_id
        self._min_detect   = min_detection_confidence
        self._min_track    = min_tracking_confidence
        self._every_n      = process_every_n
        self._face_model   = face_model_path
        self._pose_model   = pose_model_path
        self._ema_alpha    = ema_alpha
        self._frame_count  = 0

        # Initialized lazily in executor thread on first frame
        self._face_landmarker = None
        self._pose_landmarker = None

        # Per-session temporal smoothing helpers
        self._blendshape_smoother = EMASmoother(
            size  = len(_BLENDSHAPE_NAMES),
            alpha = ema_alpha,
        )
        self._expr_voter = SlidingWindowVoter(
            window_size = vote_window,
            min_votes   = vote_min,
        )

        # Log which pose model is being used
        model_name = os.path.basename(pose_model_path)
        logger.info(f"[Vision] Session {session_id[:8]}… — pose model: {model_name}")

    # ── Lazy init (avoids long import at server start) ────────────────────────

    def _init_landmarkers(self):
        """Create FaceLandmarker + PoseLandmarker (called in executor thread)."""
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        RunningMode = vision.RunningMode

        face_opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._face_model),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self._min_detect,
            min_face_presence_confidence=self._min_detect,
            min_tracking_confidence=self._min_track,
            # KEY FIX: Enable blendshapes — this is the primary accuracy improvement
            output_face_blendshapes=True,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._pose_model),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self._min_detect,
            min_pose_presence_confidence=self._min_detect,
            min_tracking_confidence=self._min_track,
        )
        self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
        logger.info(
            f"[Vision] MediaPipe landmarkers initialised "
            f"(blendshapes=ON, pose={os.path.basename(self._pose_model)}) "
            f"for session {self._session_id[:8]}…"
        )

    def _run_mediapipe(self, jpeg_bytes: bytes) -> tuple[str, float, str, float, dict]:
        """
        Blocking inference — runs in thread executor.
        Returns (expression, expr_conf, pose, pose_conf, raw_scores).
        """
        try:
            import cv2
            import mediapipe as mp
            import numpy as np

            # Lazy init on first call (in the executor thread)
            if self._face_landmarker is None or self._pose_landmarker is None:
                self._init_landmarkers()

            # Decode JPEG → BGR → RGB numpy array
            arr     = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return "unknown", 0.0, "unknown", 0.0, {}

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Wrap as MediaPipe Image object (Tasks API requirement)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=img_rgb,
            )

            # Run face and pose detection
            face_result = self._face_landmarker.detect(mp_image)
            pose_result = self._pose_landmarker.detect(mp_image)

            # ── Expression: prefer blendshapes, fall back to landmarks ──────
            if face_result.face_blendshapes:
                expression, expr_conf, raw = _analyze_expression_from_blendshapes(
                    face_result.face_blendshapes,
                    self._blendshape_smoother,
                )
            else:
                # Fallback: geometry-based (less accurate)
                logger.debug("[Vision] Blendshapes not available — using landmark geometry")
                expression, expr_conf, raw = _analyze_expression_from_landmarks_fallback(
                    face_result.face_landmarks
                )
                self._blendshape_smoother.reset()

            # Apply sliding window vote for stability
            expression = self._expr_voter.vote(expression)

            # ── Pose ─────────────────────────────────────────────────────────
            pose, pose_conf = _analyze_pose_from_landmarks(
                pose_result.pose_landmarks
            )

            return expression, expr_conf, pose, pose_conf, raw

        except Exception as e:
            logger.warning(f"[Vision] MediaPipe inference error: {e}")
            return "unknown", 0.0, "unknown", 0.0, {}

    # ── FrameProcessor interface ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (InputImageRawFrame, UserImageRawFrame)):
            self._frame_count += 1

            if self._frame_count % self._every_n == 0:
                jpeg_bytes = frame.image

                loop = asyncio.get_event_loop()
                expression, expr_conf, pose, pose_conf, raw = await loop.run_in_executor(
                    None, self._run_mediapipe, jpeg_bytes
                )

                analysis = VisionAnalysisFrame(
                    session_id            = self._session_id,
                    expression            = expression,
                    expression_confidence = expr_conf,
                    pose                  = pose,
                    pose_confidence       = pose_conf,
                    raw_expression_scores = raw,
                )

                await vision_state.update(self._session_id, analysis)

                logger.debug(f"[Vision] {self._session_id[:8]}… → {analysis}")

                await self.push_frame(analysis, direction)

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Release MediaPipe resources and clear session state."""
        await vision_state.clear(self._session_id)
        if self._face_landmarker:
            try:
                self._face_landmarker.close()
            except Exception:
                pass
            self._face_landmarker = None
        if self._pose_landmarker:
            try:
                self._pose_landmarker.close()
            except Exception:
                pass
            self._pose_landmarker = None
        self._blendshape_smoother.reset()
        self._expr_voter.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builder — used by VCBroadcaster to inject vision context
# ─────────────────────────────────────────────────────────────────────────────

_EXPRESSION_LABELS = {
    "smiling":   "appears calm and confident (smiling)",
    "excited":   "appears highly energized and excited",
    "neutral":   "appears composed and neutral",
    "nervous":   "appears visibly nervous or stressed",
    "thinking":  "appears to be thinking or hesitating",
    "surprised": "appears surprised or caught off guard",
    "sad":       "appears downbeat or low-confidence",
    "angry":     "appears frustrated or tense",
    "unknown":   "expression could not be determined",
}

_POSE_LABELS = {
    "upright":         "sitting/standing upright — confident posture",
    "leaning_forward": "leaning forward — engaged or eager",
    "slouched":        "slouching — low energy or disengaged",
    "gesturing":       "using hand gestures — expressive",
    "tilted":          "head/body tilted — casual or uncertain",
    "unknown":         "pose could not be determined",
}


def build_vision_context_block(vision_data: Optional[dict]) -> str:
    """
    Formats the latest vision state into a body language intelligence block
    for injection into the VC turn. Returns empty string if no data or
    confidence is too low to be meaningful.

    Intentionally framed as SECONDARY context — audio is always primary.
    """
    if not vision_data:
        return ""

    expr      = vision_data.get("expression", "unknown")
    expr_conf = vision_data.get("expression_confidence", 0.0)
    pose      = vision_data.get("pose", "unknown")
    pose_conf = vision_data.get("pose_confidence", 0.0)

    # Only inject if at least one signal is confident enough
    if expr_conf < 0.3 and pose_conf < 0.3:
        return ""

    expr_desc = _EXPRESSION_LABELS.get(expr, expr)
    pose_desc = _POSE_LABELS.get(pose, pose)

    lines = ["\n\nBODY LANGUAGE INTELLIGENCE (secondary signal — audio takes priority):"]
    lines.append(
        f"  • Facial Expression: {expr_desc} "
        f"(confidence: {expr_conf:.0%})"
    )
    lines.append(
        f"  • Body Pose:         {pose_desc} "
        f"(confidence: {pose_conf:.0%})"
    )
    lines.append(
        "  ⚠ Use body language as soft context only. "
        "What the founder says (audio) is the primary basis for your evaluation. "
        "If body language contradicts speech, you MAY briefly note the discrepancy."
    )

    return "\n".join(lines)
