import asyncio
import io
import json
import os
import re
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

# ── TTS feature flag ──────────────────────────────────────────────────────────
_TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"

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


# =============================================================================
# TTSEngine — ElevenLabs WebSocket streaming TTS
# =============================================================================
# Feeds LLM token chunks (accumulated into natural sentences) to ElevenLabs and
# yields raw 16-bit PCM audio bytes as they stream back. Turn-based: the caller
# awaits the full audio stream before re-enabling the mic.
# =============================================================================

# Regex to detect sentence boundaries (., !, ?) for natural chunking
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


class TTSEngine:
    """
    Async wrapper around ElevenLabs HTTP streaming TTS.

    Usage:
        engine = TTSEngine()
        async for pcm_chunk in engine.synthesize(text):
            await websocket.send_bytes(pcm_chunk)
    """

    def __init__(self):
        self._api_key   = os.getenv("ELEVENLABS_API_KEY", "")
        self._voice_id  = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
        self._model     = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
        self._enabled   = _TTS_ENABLED and bool(self._api_key) and self._api_key != "your_elevenlabs_api_key_here"

        if not self._enabled:
            logger.warning("[TTS] Disabled — set ELEVENLABS_API_KEY and TTS_ENABLED=true in .env")
        else:
            logger.info(f"[TTS] ElevenLabs ready. voice={self._voice_id!r} model={self._model!r}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def synthesize(self, text: str):
        """
        Synthesize *text* via ElevenLabs streaming API.
        Yields raw PCM audio bytes (mp3 chunks from ElevenLabs).
        Call once per complete VC response sentence/paragraph.
        """
        if not self._enabled or not text.strip():
            return

        import aiohttp

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.85,
                "style": 0.30,
                "use_speaker_boost": True,
            },
            "output_format": "mp3_44100_128",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"[TTS] ElevenLabs error {resp.status}: {body[:200]}")
                        return
                    async for chunk in resp.content.iter_chunked(4096):
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"[TTS] Synthesis error: {e}")

    async def synthesize_full_response(self, full_text: str, websocket: WebSocket):
        """
        Synthesize the complete VC response and stream audio binary frames to
        the browser. Sends a JSON 'tts_done' event when finished so the client
        knows to re-enable the mic.
        """
        if not self._enabled:
            # No TTS — tell client to re-enable mic immediately
            try:
                await websocket.send_text(json.dumps({"type": "tts_done"}))
            except Exception:
                pass
            return

        # Strip any internal tags before synthesizing
        clean = full_text.replace("<END_PITCH>", "").strip()
        if not clean:
            await websocket.send_text(json.dumps({"type": "tts_done"}))
            return

        logger.info(f"[TTS] Synthesizing {len(clean)} chars for Marcus Reid...")
        t0 = time.monotonic()
        chunk_count = 0

        try:
            async for audio_chunk in self.synthesize(clean):
                await websocket.send_bytes(audio_chunk)
                chunk_count += 1
        except Exception as e:
            logger.error(f"[TTS] Stream send error: {e}")

        elapsed = time.monotonic() - t0
        logger.info(f"[TTS] Done — {chunk_count} chunks in {elapsed:.2f}s")

        # Signal client: TTS finished, mic can open
        try:
            await websocket.send_text(json.dumps({"type": "tts_done"}))
        except Exception:
            pass


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

    Turn-based flow (investor does NOT get interrupted):
      1. Sends 'vc_thinking' to browser → client gates mic
      2. Streams LLM tokens → forwards as 'vc_token' for live text display
      3. On 'final' event: sends 'vc_response' JSON, THEN synthesizes the full
         VC reply via ElevenLabs and streams audio as binary WebSocket frames.
      4. Sends 'tts_done' → browser re-enables mic for next founder turn.

    VAD / interim events are still forwarded for UI feedback.
    """

    def __init__(self, websocket: WebSocket, session_id: str, tts_engine: "TTSEngine"):
        super().__init__()
        self._websocket  = websocket
        self._session_id = session_id
        self._tts        = tts_engine

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        payload = None

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"[VC] Founder said: {text!r}")
                turn_start = time.monotonic()

                # ── 1. Tell browser Marcus is thinking (gate the mic) ────────
                try:
                    await self._websocket.send_text(json.dumps({
                        "type": "vc_thinking",
                        "founder_text": text,
                    }))
                except Exception:
                    pass

                # ── 2. Stream LLM tokens (text appears word-by-word) ─────────
                first_token_at = None
                final_vc_text  = ""
                final_event    = None

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
                            final_event   = event
                            final_vc_text = event.get("vc_response", "")
                except Exception as e:
                    logger.error(f"[VC] Agent error: {e}")
                    try:
                        await self._websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                    except Exception:
                        pass
                    # Still send tts_done so client can recover
                    try:
                        await self._websocket.send_text(json.dumps({"type": "tts_done"}))
                    except Exception:
                        pass
                    await self.push_frame(frame, direction)
                    return

                # ── 3. Send full metadata response ───────────────────────────
                if final_event:
                    total_s = time.monotonic() - turn_start
                    logger.info(
                        f"[VC] Turn complete in {total_s:.2f}s | "
                        f"Stage={final_event['stage']} | "
                        f"Exchange={final_event['exchange_count']} | "
                        f"is_out={final_event['is_out']}"
                    )
                    try:
                        await self._websocket.send_text(json.dumps({
                            "type": "vc_response",
                            "founder_text": text,
                            "vc_text": final_vc_text,
                            "stage": final_event["stage"],
                            "exchange_count": final_event["exchange_count"],
                            "pitch_metrics": final_event["pitch_metrics"],
                            "is_out": final_event["is_out"],
                            "pitch_ended": final_event["pitch_ended"],
                            "latency_ms": round(total_s * 1000),
                        }))
                    except Exception as e:
                        logger.warning(f"[VCBroadcaster] vc_response send error: {e}")

                # ── 4. Stream TTS audio, then signal mic re-enable ───────────
                # synthesize_full_response() sends binary MP3 chunks then
                # sends {type: 'tts_done'} — turn-based, no interrupts.
                await self._tts.synthesize_full_response(final_vc_text, self._websocket)

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
    """WebSocket for the VC pitch mode: VAD+STT → LangGraph VC agent → TTS audio."""
    await websocket.accept()

    # Client sends session_id as query param
    session_id = websocket.query_params.get("session_id", new_session())
    logger.info(f"[VC] WebSocket connected. session_id={session_id}")

    # One TTSEngine per connection (stateless, so sharing is fine too)
    tts_engine = TTSEngine()

    serializer = WhisperLiveSerializer()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=False,   # We send audio manually as raw binary frames
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

    # VCBroadcaster: transcripts → LangGraph → token stream → TTS audio
    vc_broadcaster = VCBroadcaster(websocket, session_id, tts_engine)

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