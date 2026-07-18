"""
app/vision/state.py — VisionAnalysisFrame dataclass and VisionState singleton.

VisionState is a thread-safe in-memory store of the latest vision analysis
per session. VCBroadcaster reads from this at the start of each turn to inject
body language context into the VC persona prompt.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from pipecat.frames.frames import DataFrame


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
        self._lock: asyncio.Lock = asyncio.Lock()
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


# Singleton — imported by VCBroadcaster and VisionBroadcaster
vision_state = VisionState()
