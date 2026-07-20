"""
app/websockets/stt_ws.py — STT WebSocket endpoint (/ws).

Pipeline:
═════════
  FastAPIWebsocketTransport.input()
    → VADProcessor( SileroVADAnalyzer )
    → HybridWhisperSTTProcessor           ← real-time interim + accurate final
        ├─ InterimTranscriptionFrame every ~0.8 s while speaking (beam_size=1)
        └─ TranscriptionFrame immediately on VAD stop  (beam_size=5, full context)
    → TranscriptionBroadcaster            ← sends JSON to browser
    → FastAPIWebsocketTransport.output()

Key design:
  • SileroVADAnalyzer detects speech boundaries and emits VADUserStarted/
    StoppedSpeakingFrame which HybridWhisperSTTProcessor listens to.
  • During speech: greedy (beam=1) Whisper pass every INTERIM_INTERVAL_SECS
    → low latency, good-enough accuracy for live preview.
  • After VAD stop: full beam-search (beam=5) pass over the complete utterance
    → highest accuracy for the final committed transcript.
  • Both passes share the same loaded model (loaded once, reused).
  • TranscriptionLogObserver attached to PipelineWorker logs all frames
    automatically without coupling logging to broadcaster logic.

Browser JSON messages:
  { "type": "final",   "text": "…" }         TranscriptionFrame
  { "type": "interim", "text": "…" }         InterimTranscriptionFrame
  { "type": "status",  "status": "speaking" | "silence" }
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
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from app.config import VAD_STOP_SECS, resolve_whisper_config
from app.stt.serializer import WhisperLiveSerializer
from app.stt.streaming_processor import create_whisper_stt_service


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptionBroadcaster
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionBroadcaster(FrameProcessor):
    """
    Intercepts transcription and VAD frames in the Pipecat pipeline and
    forwards them to the connected browser as JSON WebSocket messages.

    Pipecat's FastAPIWebsocketOutputTransport only serialises audio and
    OutputTransportMessageFrame objects — all other frame types are silently
    dropped at the output transport.  This processor sits *before* the output
    transport and bridges those frames to the WebSocket directly.

    Message schema (all messages sent as text / UTF-8 JSON):
      { "type": "final",   "text": "…" }    — TranscriptionFrame
      { "type": "interim", "text": "…" }    — InterimTranscriptionFrame
      { "type": "status",  "status": "speaking" | "silence" }  — VAD events

    All frames are still passed downstream unchanged so the rest of the
    pipeline can observe them (e.g. the TranscriptionLogObserver attached
    to the PipelineWorker).
    """

    def __init__(self, websocket: WebSocket) -> None:
        super().__init__()
        self._websocket = websocket
        self._speech_stopped_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        payload: str | None = None

        if isinstance(frame, TranscriptionFrame):
            # Final transcript produced by WhisperSTTService after utterance end.
            if self._speech_stopped_at is not None:
                latency_ms = (time.monotonic() - self._speech_stopped_at) * 1000
                logger.info(
                    f"[STT] ✅ Final ({latency_ms:.0f} ms after silence): {frame.text!r}"
                )
                self._speech_stopped_at = None
            else:
                logger.info(f"[STT] ✅ Final: {frame.text!r}")
            payload = json.dumps({"type": "final", "text": frame.text})

        elif isinstance(frame, InterimTranscriptionFrame):
            # In-progress partial transcript while user is still speaking.
            logger.debug(f"[STT] 🔄 Interim: {frame.text!r}")
            payload = json.dumps({"type": "interim", "text": frame.text})

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            logger.info("[STT] 🎤 VAD: user started speaking")
            self._speech_stopped_at = None
            payload = json.dumps({"type": "status", "status": "speaking"})

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.info("[STT] 🔇 VAD: user stopped speaking")
            self._speech_stopped_at = time.monotonic()
            payload = json.dumps({"type": "status", "status": "silence"})

        if payload is not None:
            try:
                await self._websocket.send_text(payload)
            except Exception as exc:
                logger.warning(f"[STT] WebSocket send failed: {exc}")

        # Always forward frame downstream (TranscriptionLogObserver sees it too).
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

async def stt_websocket_endpoint(websocket: WebSocket) -> None:
    """
    FastAPI WebSocket handler for /ws.

    Builds and runs a fully Pipecat-native STT pipeline:

        FastAPIWebsocketTransport (audio in)
          → VADProcessor + SileroVADAnalyzer
          → WhisperSTTService (local faster-whisper)
          → TranscriptionBroadcaster (JSON → browser)
          → FastAPIWebsocketTransport (output)

    A TranscriptionLogObserver is attached to the PipelineWorker so every
    TranscriptionFrame and InterimTranscriptionFrame is automatically logged
    without any extra code in this function.
    """
    await websocket.accept()

    wcfg = resolve_whisper_config()
    logger.info(
        f"[STT] WebSocket client connected. "
        f"Model={wcfg['model']}  device={wcfg['device']}  "
        f"compute_type={wcfg['compute_type']}"
    )

    # ── 1. Transport ──────────────────────────────────────────────────────────
    serializer = WhisperLiveSerializer()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16_000,
            audio_out_enabled=False,
            serializer=serializer,
        ),
    )

    # ── 2. VAD ────────────────────────────────────────────────────────────────
    # SileroVADAnalyzer detects speech / silence boundaries.
    # VADProcessor wraps it as a FrameProcessor and emits:
    #   • VADUserStartedSpeakingFrame when speech begins
    #   • VADUserStoppedSpeakingFrame when silence is detected
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=VAD_STOP_SECS,     # seconds of silence before stop event
            start_secs=0.2,              # seconds of speech before start event
            confidence=0.7,              # Silero confidence threshold
            min_volume=0.6,              # ignore whisper-quiet audio
        )
    )
    vad_processor = VADProcessor(
        vad_analyzer=vad_analyzer,
        audio_idle_timeout=1.0,          # force stop if no audio while "speaking"
    )

    # ── 3. STT ────────────────────────────────────────────────────────────────
    # WhisperSTTService (Pipecat-native, backed by faster-whisper).
    # Listens for VAD frames emitted by the upstream VADProcessor and uses
    # them to delineate speech segments.  Emits:
    #   • InterimTranscriptionFrame — while user is still speaking
    #   • TranscriptionFrame        — final result after utterance ends
    whisper_stt = create_whisper_stt_service(
        model=wcfg["model"],
        language=wcfg.get("language", "en"),
        device=wcfg["device"],
        compute_type=wcfg["compute_type"],
        no_speech_prob=wcfg.get("no_speech_prob", 0.4),
    )

    # ── 4. Broadcaster ────────────────────────────────────────────────────────
    # Intercepts transcription + VAD frames and sends JSON to the browser.
    broadcaster = TranscriptionBroadcaster(websocket)

    # ── 5. Pipeline ───────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),   # reads audio from WebSocket
            vad_processor,       # VAD → speech-boundary frames
            whisper_stt,         # STT → transcription frames
            broadcaster,         # JSON → browser WebSocket
            transport.output(),  # (no audio output; required for pipeline close)
        ]
    )

    # ── 6. Worker + Observer ─────────────────────────────────────────────────
    # PipelineWorker manages pipeline execution.
    # TranscriptionLogObserver automatically logs all TranscriptionFrames
    # and InterimTranscriptionFrames — zero extra wiring needed.
    worker = PipelineWorker(pipeline, params=PipelineParams())
    worker.add_observer(TranscriptionLogObserver())

    # ── 7. Run ────────────────────────────────────────────────────────────────
    runner = WorkerRunner()
    try:
        await runner.add_workers(worker)
        await runner.run()
    except Exception as exc:
        logger.error(f"[STT] Pipeline error: {exc}", exc_info=True)
    finally:
        logger.info("[STT] WebSocket closed — pipeline shut down.")
