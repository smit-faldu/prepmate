"""
app/websockets/stt_ws.py — Raw STT WebSocket endpoint (/ws) and TranscriptionBroadcaster.

Pipeline: WebSocket audio → VAD → StreamingWhisper → TranscriptionBroadcaster → WebSocket output
"""

import json
import time

from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from app.config import VAD_STOP_SECS, resolve_whisper_config
from app.stt.serializer import WhisperLiveSerializer
from app.stt.streaming_processor import StreamingWhisperProcessor
from app.stt.whisper_model import get_whisper_model


class TranscriptionBroadcaster(FrameProcessor):
    """
    Sends transcription and VAD events directly to the browser via WebSocket JSON.

    The Pipecat output transport only sends audio and OutputTransportMessageFrames.
    TranscriptionFrame and VAD frames are routed to write_transport_frame() which
    is a no-op in FastAPIWebsocketOutputTransport — they are silently dropped.
    This processor intercepts them and sends JSON directly over the WebSocket.
    """

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket         = websocket
        self._speech_stopped_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        payload = None
        if isinstance(frame, TranscriptionFrame):
            if self._speech_stopped_at:
                latency_ms = (time.monotonic() - self._speech_stopped_at) * 1000
                logger.info(f"[Broadcaster] ✅ Final transcript ({latency_ms:.0f}ms after silence): {frame.text!r}")
                self._speech_stopped_at = None
            else:
                logger.info(f"[Broadcaster] ✅ Final transcript: {frame.text!r}")
            payload = json.dumps({"type": "final", "text": frame.text})
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.debug(f"[Broadcaster] 🔄 Interim: {frame.text!r}")
            payload = json.dumps({"type": "interim", "text": frame.text})
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            logger.info("[Broadcaster] 🎤 VAD: User started speaking")
            self._speech_stopped_at = None
            payload = json.dumps({"type": "status", "status": "speaking"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.info("[Broadcaster] 🔇 VAD: User stopped speaking")
            self._speech_stopped_at = time.monotonic()
            payload = json.dumps({"type": "status", "status": "silence"})

        if payload:
            try:
                await self._websocket.send_text(payload)
            except Exception as e:
                logger.warning(f"[Broadcaster] Failed to send WebSocket message: {e}")

        await self.push_frame(frame, direction)


async def stt_websocket_endpoint(websocket: WebSocket):
    """Handler for the raw STT WebSocket (/ws)."""
    await websocket.accept()
    logger.info(f"WebSocket client connected. Whisper model: {__import__('os').getenv('WHISPER_MODEL', 'tiny')}")

    wcfg          = resolve_whisper_config()
    whisper_model = await get_whisper_model(wcfg)

    serializer = WhisperLiveSerializer()
    transport  = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=False,
            serializer=serializer,
        ),
    )

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=VAD_STOP_SECS,
            start_secs=0.2,
            confidence=0.7,
            min_volume=0.6,
        )
    )
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)
    streaming_stt = StreamingWhisperProcessor(whisper_model=whisper_model, language=wcfg["language"])
    broadcaster   = TranscriptionBroadcaster(websocket)

    pipeline = Pipeline([transport.input(), vad_processor, streaming_stt, broadcaster, transport.output()])
    task     = PipelineWorker(pipeline, params=PipelineParams())
    runner   = WorkerRunner()

    try:
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"Exception during pipeline task: {e}")
    finally:
        logger.info("WebSocket connection closed, pipeline shut down.")
