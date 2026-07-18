"""
app/vision/processor.py — MediaPipeVisionProcessor

Custom Pipecat FrameProcessor using MediaPipe Tasks Python API.

Intercepts InputImageRawFrame / UserImageRawFrame (raw JPEG bytes from the
browser webcam), runs FaceLandmarker + PoseLandmarker in a thread executor,
and:
  1. Emits a VisionAnalysisFrame downstream
  2. Updates the shared VisionState singleton so VCBroadcaster can inject
     body language context into the VC AI prompt.

Non-blocking: all MediaPipe inference runs in loop.run_in_executor().
"""

import asyncio
import logging
import os

from pipecat.frames.frames import Frame, InputImageRawFrame, UserImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.vision.expression import (
    BLENDSHAPE_NAMES,
    _analyze_expression_from_blendshapes,
    _analyze_expression_from_landmarks_fallback,
)
from app.vision.pose import analyze_pose_from_landmarks
from app.vision.smoothers import EMASmoother, SlidingWindowVoter
from app.vision.state import VisionAnalysisFrame, vision_state

logger = logging.getLogger(__name__)

# ── Model paths ───────────────────────────────────────────────────────────────
_MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_FACE_MODEL      = os.path.join(_MODELS_DIR, "face_landmarker.task")
_POSE_MODEL_FULL = os.path.join(_MODELS_DIR, "pose_landmarker_full.task")
_POSE_MODEL_LITE = os.path.join(_MODELS_DIR, "pose_landmarker_lite.task")
_POSE_MODEL      = _POSE_MODEL_FULL if os.path.exists(_POSE_MODEL_FULL) else _POSE_MODEL_LITE


class MediaPipeVisionProcessor(FrameProcessor):
    """
    Custom Pipecat FrameProcessor using MediaPipe Tasks Python API.

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
        self._session_id  = session_id
        self._min_detect  = min_detection_confidence
        self._min_track   = min_tracking_confidence
        self._every_n     = process_every_n
        self._face_model  = face_model_path
        self._pose_model  = pose_model_path
        self._frame_count = 0

        # Initialized lazily in executor thread on first frame
        self._face_landmarker = None
        self._pose_landmarker = None

        # Per-session temporal smoothing helpers
        self._blendshape_smoother = EMASmoother(size=len(BLENDSHAPE_NAMES), alpha=ema_alpha)
        self._expr_voter          = SlidingWindowVoter(window_size=vote_window, min_votes=vote_min)

        model_name = os.path.basename(pose_model_path)
        logger.info(f"[Vision] Session {session_id[:8]}… — pose model: {model_name}")

    # ── Lazy init ─────────────────────────────────────────────────────────────

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
            output_face_blendshapes=True,  # KEY: primary accuracy improvement
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

            if self._face_landmarker is None or self._pose_landmarker is None:
                self._init_landmarkers()

            arr     = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return "unknown", 0.0, "unknown", 0.0, {}

            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            face_result = self._face_landmarker.detect(mp_image)
            pose_result = self._pose_landmarker.detect(mp_image)

            # Expression: prefer blendshapes, fall back to landmarks
            if face_result.face_blendshapes:
                expression, expr_conf, raw = _analyze_expression_from_blendshapes(
                    face_result.face_blendshapes,
                    self._blendshape_smoother,
                )
            else:
                logger.debug("[Vision] Blendshapes not available — using landmark geometry")
                expression, expr_conf, raw = _analyze_expression_from_landmarks_fallback(
                    face_result.face_landmarks
                )
                self._blendshape_smoother.reset()

            expression = self._expr_voter.vote(expression)

            pose, pose_conf = analyze_pose_from_landmarks(pose_result.pose_landmarks)

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
        for attr in ("_face_landmarker", "_pose_landmarker"):
            obj = getattr(self, attr, None)
            if obj:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._blendshape_smoother.reset()
        self._expr_voter.reset()
