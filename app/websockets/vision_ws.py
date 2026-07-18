"""
app/websockets/vision_ws.py — Vision WebSocket endpoint (/ws/vision),
VisionBroadcaster, and RawJPEGSerializer.

The browser opens this as a SECOND WebSocket alongside /ws/vc.
It receives JPEG frames, runs MediaPipe Holistic, and:
  1. Sends VisionAnalysisFrame events back as JSON to the browser (for HUD)
  2. Updates the shared VisionState so VCBroadcaster can inject body
     language context into the VC AI on the next turn.
The session_id MUST match the /ws/vc session_id so both pipelines share state.
"""

import json

from fastapi import WebSocket
from loguru import logger

from pipecat.frames.frames import Frame, InputImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from app.config import VISION_MIN_DETECT, VISION_MIN_TRACK, VISION_PROCESS_EVERY_N
from app.vc import new_session
from app.vision.processor import MediaPipeVisionProcessor
from app.vision.state import VisionAnalysisFrame


class RawJPEGSerializer(FrameSerializer):
    """
    Simple serializer that wraps raw binary (JPEG) WebSocket messages
    directly as InputImageRawFrame. The browser sends webcam frames as
    raw JPEG binary blobs with no envelope.
    """

    async def serialize(self, frame: Frame) -> bytes | None:
        return None  # server never sends binary on this vision socket

    async def deserialize(self, data: bytes | str) -> Frame | None:
        if not data or not isinstance(data, bytes):
            return None
        return InputImageRawFrame(image=data, size=(0, 0), format="JPEG")


class VisionBroadcaster(FrameProcessor):
    """
    Downstream of MediaPipeVisionProcessor — catches VisionAnalysisFrame
    and sends real-time expression/pose data back to the browser as JSON
    for the live HUD overlay.
    All other frames pass through unchanged.
    """

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket = websocket
        # Once the WebSocket closes any send attempt raises an ASGI error.
        # Track that state so we log once and skip silently thereafter.
        self._ws_closed: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VisionAnalysisFrame) and not self._ws_closed:
            try:
                await self._websocket.send_text(json.dumps({
                    "type":                  "vision",
                    "expression":            frame.expression,
                    "expression_confidence": frame.expression_confidence,
                    "pose":                  frame.pose,
                    "pose_confidence":       frame.pose_confidence,
                    "raw_scores":            frame.raw_expression_scores,
                }))
            except Exception as e:
                self._ws_closed = True
                logger.debug(f"[VisionBroadcaster] WS closed, stopping sends: {e}")

        await self.push_frame(frame, direction)


async def vision_websocket_endpoint(websocket: WebSocket):
    """
    Handler for the computer vision WebSocket (/ws/vision).
    Browser sends JPEG frames; server responds with expression/pose JSON.
    """
    await websocket.accept()

    session_id = websocket.query_params.get("session_id", new_session())
    logger.info(f"[Vision] WebSocket connected. session_id={session_id}")

    serializer = RawJPEGSerializer()
    transport  = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=False,
            audio_out_enabled=False,
            video_in_enabled=True,
            serializer=serializer,
        ),
    )

    vision_processor   = MediaPipeVisionProcessor(
        session_id=session_id,
        min_detection_confidence=VISION_MIN_DETECT,
        min_tracking_confidence=VISION_MIN_TRACK,
        process_every_n=VISION_PROCESS_EVERY_N,
    )
    vision_broadcaster = VisionBroadcaster(websocket)

    pipeline = Pipeline([transport.input(), vision_processor, vision_broadcaster, transport.output()])
    task     = PipelineWorker(pipeline, params=PipelineParams())
    runner   = WorkerRunner()

    try:
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"[Vision] Pipeline exception: {e}")
    finally:
        await vision_processor.cleanup()
        logger.info(f"[Vision] WebSocket closed. session_id={session_id}")
