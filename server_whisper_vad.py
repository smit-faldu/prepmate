import asyncio
import json
import os
import sys
import time
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# VC Agent — streaming entry point (concurrent analyst + persona, see
# vc_agent.py module docstring for the architecture).
from vc_agent import new_session, run_turn_streaming

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

# Load configuration from .env
load_dotenv(override=True)

# Configure logger
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "DEBUG"))

app = FastAPI(title="PrepMate — STT + VC Pitch Evaluator")


def _resolve_whisper_config() -> dict:
    """
    Resolves Whisper device, compute_type, model, and language from environment.

    Auto-selects the optimal compute type:
      - CUDA  → float16  (native GPU half-precision, fastest + most accurate)
      - CPU   → int8     (~3× faster than float32, no GPU needed)

    Override compute type at any time via WHISPER_COMPUTE_TYPE in .env.

    Language:
      - "auto" or empty → multilingual auto-detect (set language=None)
      - "en", "hi", etc. → force a specific language (faster, more accurate)
    """
    device   = os.getenv("WHISPER_DEVICE", "auto")          # auto | cpu | cuda
    model    = os.getenv("WHISPER_MODEL", "tiny")            # tiny | base | large-v3 | …
    lang_env = os.getenv("WHISPER_LANGUAGE", "en").strip()   # en | hi | auto | ""

    # ── Compute type: auto-select unless explicitly set ─────────────────────
    compute_override = os.getenv("WHISPER_COMPUTE_TYPE", "").strip()
    if compute_override:
        compute_type = compute_override
    else:
        # Detect effective device when device="auto" so we can pick wisely
        effective_device = device
        if device == "auto":
            try:
                import torch
                effective_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                effective_device = "cpu"  # torch not installed → assume CPU

        compute_type = "float16" if effective_device == "cuda" else "int8"

    # ── Language: None triggers Whisper multilingual auto-detect ────────────
    language: str | None = None if lang_env in ("", "auto") else lang_env

    logger.info(
        f"Whisper config → device={device!r} | model={model!r} "
        f"| compute_type={compute_type!r} "
        f"| language={'auto-detect' if language is None else language!r}"
    )
    return {"device": device, "model": model, "compute_type": compute_type, "language": language}

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# TranscriptionBroadcaster
# ---------------------------------------------------------------------------
# The Pipecat output transport only sends audio and OutputTransportMessageFrames.
# TranscriptionFrame and VAD frames are routed to write_transport_frame() which
# is a no-op in FastAPIWebsocketOutputTransport — they are silently dropped.
# This processor intercepts them and sends JSON directly over the WebSocket.
# ---------------------------------------------------------------------------
class TranscriptionBroadcaster(FrameProcessor):
    """Sends transcription and VAD events directly to the browser via WebSocket JSON."""

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket = websocket
        self._speech_stopped_at: float | None = None  # For latency measurement

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        payload = None
        if isinstance(frame, TranscriptionFrame):
            # Log latency from when speech stopped to when transcript arrives
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
            self._speech_stopped_at = time.monotonic()  # Start latency timer
            payload = json.dumps({"type": "status", "status": "silence"})

        if payload:
            try:
                await self._websocket.send_text(payload)
            except Exception as e:
                logger.warning(f"[Broadcaster] Failed to send WebSocket message: {e}")

        # Always pass the frame downstream so the rest of the pipeline still works
        await self.push_frame(frame, direction)

# Custom Serializer for bridging custom browser client with Pipecat pipeline
class WhisperLiveSerializer(FrameSerializer):
    def __init__(self):
        super().__init__()

    async def serialize(self, frame: Frame) -> bytes | str | None:
        """
        Converts pipeline frames to WebSocket text messages sent back to the browser client.
        """
        if isinstance(frame, InterimTranscriptionFrame):
            # Interim (live) text as user is speaking
            logger.debug(f"[Serializer] Interim: {frame.text!r}")
            return json.dumps({"type": "interim", "text": frame.text})
        elif isinstance(frame, TranscriptionFrame):
            # Finalized sentence after VAD detects turn stopped
            logger.info(f"[Serializer] Final transcript: {frame.text!r}")
            return json.dumps({"type": "final", "text": frame.text})
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            # User speaking status
            logger.debug("[Serializer] VAD: User started speaking")
            return json.dumps({"type": "status", "status": "speaking"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # User silent status
            logger.debug("[Serializer] VAD: User stopped speaking")
            return json.dumps({"type": "status", "status": "silence"})
        return None

    async def deserialize(self, data: bytes | str) -> Frame | None:
        """
        Converts incoming raw binary frames from browser websocket into Pipecat input frames.
        """
        if isinstance(data, bytes):
            # Capture mono 16kHz PCM audio bytes and wrap in input frame
            return InputAudioRawFrame(
                audio=data,
                sample_rate=16000,
                num_channels=1
            )
        return None

# Serve a minimal favicon to suppress browser 404 noise
@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    # Minimal 1×1 transparent ICO (base64-encoded)
    import base64
    ico_b64 = (
        "AAABAAEAAQEAAAEAGAAoAAAAFgAAACgAAAABAAAAAgAAAAEAGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
    )
    ico_bytes = base64.b64decode(ico_b64)
    return Response(content=ico_bytes, media_type="image/x-icon")


# Serve index.html — raw STT demo
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join("templates", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>templates/index.html not found!</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# Serve vc.html — VC Pitch Arena
@app.get("/vc", response_class=HTMLResponse)
async def get_vc():
    vc_path = os.path.join("templates", "vc.html")
    if not os.path.exists(vc_path):
        return HTMLResponse(content="<h1>templates/vc.html not found!</h1>", status_code=404)
    with open(vc_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# REST endpoint: create a new VC pitch session
@app.post("/api/vc/session")
async def create_vc_session():
    """Creates and returns a new unique session ID for a pitch session."""
    session_id = new_session()
    logger.info(f"[VC] New pitch session created: {session_id}")
    return JSONResponse({"session_id": session_id})

# WebSocket connection endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Model is always driven by .env (WHISPER_MODEL), browser param is ignored
    # This prevents accidentally loading a heavier model via URL
    model = os.getenv("WHISPER_MODEL", "tiny")
    
    logger.info(f"WebSocket client connected. Whisper model: {model}")

    # Set up custom transport and serializer
    serializer = WhisperLiveSerializer()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,   # Must match what the browser sends (16kHz PCM)
            audio_out_enabled=False,       # We only send JSON text, not audio back
            serializer=serializer,
        )
    )

    # Resolve device / compute_type / language from env (auto-selects best defaults)
    wcfg = _resolve_whisper_config()

    # Initialize Whisper STT Service
    stt_service = WhisperSTTService(
        device=wcfg["device"],
        compute_type=wcfg["compute_type"],
        settings=WhisperSTTService.Settings(
            model=wcfg["model"],
            language=wcfg["language"],  # None = multilingual auto-detect
            no_speech_prob=0.6,          # Filter silence/noise hallucinations
        )
    )

    # Initialize VAD Processor with Silero analyzer
    # stop_secs: how long to wait after silence before declaring end-of-turn
    # Default is 0.2s — we keep it low to minimize pause before transcription
    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "0.3"))
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=vad_stop_secs,  # Tunable via env: lower = faster response
            start_secs=0.2,           # Min speech duration before confirming start
            confidence=0.7,           # Silero speech confidence threshold
            min_volume=0.6,           # Min audio volume to count as speech
        )
    )
    logger.info(f"VAD stop_secs={vad_stop_secs}s | model={wcfg['model']} | device={wcfg['device']} | compute_type={wcfg['compute_type']}")
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

    # Broadcaster: sends TranscriptionFrame & VAD events directly over WebSocket
    # (the output transport's write_transport_frame() is a no-op for these types)
    broadcaster = TranscriptionBroadcaster(websocket)

    # Assemble the Pipecat pipeline:
    # WebSocket input -> VAD -> Whisper STT -> Broadcaster -> WebSocket output
    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        stt_service,
        broadcaster,
        transport.output()
    ])

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            # Standard configuration
        )
    )

    runner = WorkerRunner()

    try:
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"Exception during pipeline task: {e}")
    finally:
        logger.info("WebSocket connection closed, pipeline shut down.")

# ---------------------------------------------------------------------------
# VCBroadcaster — intercepts final transcripts → feeds LangGraph VC agent
# ---------------------------------------------------------------------------
class VCBroadcaster(FrameProcessor):
    """
    Replaces TranscriptionBroadcaster for the /ws/vc endpoint.
    On each final transcript:
      1. Sends 'vc_thinking' status to the browser
      2. Streams the VC's reply via run_turn_streaming() — tokens are
         forwarded to the browser as 'vc_token' messages AS THEY ARRIVE,
         so the user sees/hears the response start within ~1s instead of
         waiting for the full turn (analyst + persona) to complete.
      3. Sends a final 'vc_response' message with full metadata once the
         turn (including the concurrently-running analyst) completes.
    VAD events and interim transcripts are still forwarded for UI feedback.
    """

    def __init__(self, websocket: WebSocket, session_id: str):
        super().__init__()
        self._websocket = websocket
        self._session_id = session_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        payload = None

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"[VC] Founder said: {text!r}")
                turn_start = time.monotonic()
                try:
                    await self._websocket.send_text(json.dumps({
                        "type": "vc_thinking",
                        "founder_text": text,
                    }))
                except Exception:
                    pass

                first_token_at = None
                try:
                    async for event in run_turn_streaming(self._session_id, text):
                        if event["type"] == "token":
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                                logger.info(
                                    f"[VC] First token in {first_token_at - turn_start:.2f}s"
                                )
                            try:
                                await self._websocket.send_text(json.dumps({
                                    "type": "vc_token",
                                    "text": event["text"],
                                }))
                            except Exception as e:
                                logger.warning(f"[VCBroadcaster] Token send error: {e}")
                        elif event["type"] == "final":
                            total_s = time.monotonic() - turn_start
                            logger.info(
                                f"[VC] Turn complete in {total_s:.2f}s | "
                                f"Stage={event['stage']} | "
                                f"Exchange={event['exchange_count']} | "
                                f"is_out={event['is_out']}"
                            )
                            payload = json.dumps({
                                "type": "vc_response",
                                "founder_text": text,
                                "vc_text": event["vc_response"],
                                "stage": event["stage"],
                                "exchange_count": event["exchange_count"],
                                "pitch_metrics": event["pitch_metrics"],
                                "is_out": event["is_out"],
                                "pitch_ended": event["pitch_ended"],
                                "latency_ms": round(total_s * 1000),
                            })
                except Exception as e:
                    logger.error(f"[VC] Agent error: {e}")
                    payload = json.dumps({"type": "error", "message": str(e)})

        elif isinstance(frame, InterimTranscriptionFrame):
            payload = json.dumps({"type": "interim", "text": frame.text})
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            payload = json.dumps({"type": "status", "status": "speaking"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            payload = json.dumps({"type": "status", "status": "silence"})

        if payload:
            try:
                await self._websocket.send_text(payload)
            except Exception as e:
                logger.warning(f"[VCBroadcaster] WS send error: {e}")

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# /ws/vc — VC Pitch WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/vc")
async def vc_websocket_endpoint(websocket: WebSocket):
    """WebSocket for the VC pitch mode: VAD+STT → LangGraph VC agent → JSON response."""
    await websocket.accept()

    # Client sends session_id as query param
    session_id = websocket.query_params.get("session_id", new_session())
    logger.info(f"[VC] WebSocket connected. session_id={session_id}")

    serializer = WhisperLiveSerializer()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=False,
            serializer=serializer,
        )
    )

    wcfg = _resolve_whisper_config()

    stt_service = WhisperSTTService(
        device=wcfg["device"],
        compute_type=wcfg["compute_type"],
        settings=WhisperSTTService.Settings(
            model=wcfg["model"],
            language=wcfg["language"],
            no_speech_prob=0.6,
        )
    )

    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "0.3"))
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=vad_stop_secs,
            start_secs=0.2,
            confidence=0.7,
            min_volume=0.6,
        )
    )
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

    # VCBroadcaster routes transcripts through the LangGraph multi-agent system
    vc_broadcaster = VCBroadcaster(websocket, session_id)

    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        stt_service,
        vc_broadcaster,
        transport.output()
    ])

    task = PipelineWorker(
        pipeline,
        params=PipelineParams()
    )
    runner = WorkerRunner()

    try:
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"[VC] Pipeline exception: {e}")
    finally:
        logger.info(f"[VC] WebSocket closed. session_id={session_id}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PrepMate server on http://localhost:8000")
    logger.info("  Raw STT demo: http://localhost:8000/")
    logger.info("  VC Pitch Arena: http://localhost:8000/vc")
    uvicorn.run("server_whisper_vad:app", host="127.0.0.1", port=8000, reload=True)