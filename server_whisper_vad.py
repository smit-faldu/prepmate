import asyncio
import io
import json
import os
import re
import sys
import time
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# VC Agent — streaming entry point (concurrent analyst + persona, see
# vc_agent.py module docstring for the architecture).
from vc_agent import new_session, run_turn_streaming

# Vision pipeline — MediaPipe Holistic expression/pose analysis
from mediapipe_vision_processor import (
    MediaPipeVisionProcessor,
    VisionAnalysisFrame,
    build_vision_context_block,
    vision_state,
)

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


# =============================================================================
# StreamingWhisperProcessor — Real-time chunked Whisper inference
# =============================================================================
# Runs faster-whisper on a rolling audio buffer while the user is speaking,
# emitting InterimTranscriptionFrames every CHUNK_INTERVAL_SECS seconds.
# When VAD signals silence, runs a final accurate inference on the full buffer
# and emits a TranscriptionFrame. Replaces WhisperSTTService in the pipeline.
# =============================================================================

# Singleton Whisper model — shared across all WebSocket connections so the
# model is only loaded from disk once at server start-up.
_WHISPER_MODEL = None
_WHISPER_MODEL_LOCK = asyncio.Lock()

async def _get_whisper_model(wcfg: dict):
    """Lazily load faster-whisper WhisperModel as a singleton."""
    global _WHISPER_MODEL
    async with _WHISPER_MODEL_LOCK:
        if _WHISPER_MODEL is None:
            from faster_whisper import WhisperModel
            logger.info(
                f"[Whisper] Loading model '{wcfg['model']}' "
                f"device={wcfg['device']} compute_type={wcfg['compute_type']} ..."
            )
            _WHISPER_MODEL = WhisperModel(
                wcfg["model"],
                device=wcfg["device"],
                compute_type=wcfg["compute_type"],
            )
            logger.info("[Whisper] Model ready.")
    return _WHISPER_MODEL


class StreamingWhisperProcessor(FrameProcessor):
    """
    Custom Pipecat FrameProcessor that provides streaming-like transcription
    from local faster-whisper by running chunked inference on a rolling buffer.

    Behaviour:
      • On VADUserStartedSpeakingFrame  → reset buffer, start interim loop
      • On InputAudioRawFrame           → append PCM bytes to buffer
      • Every CHUNK_INTERVAL_SECS       → transcribe current buffer, emit
                                          InterimTranscriptionFrame if text changed
      • On VADUserStoppedSpeakingFrame  → cancel interim loop, final transcribe,
                                          emit TranscriptionFrame
      All frames are still passed downstream unchanged.
    """

    # How often (seconds) to run Whisper on the accumulated buffer mid-speech.
    # 1.5s balances latency vs. token accuracy. Lower = more frequent but less
    # accurate interim results (Whisper needs context to decode properly).
    CHUNK_INTERVAL_SECS: float = 1.5
    SAMPLE_RATE: int = 16_000

    def __init__(self, whisper_model, language: str | None = None):
        super().__init__()
        self._model    = whisper_model
        self._language = language

        # Audio accumulation buffer (raw Int16 PCM bytes from the browser)
        self._buffer: bytearray = bytearray()
        self._speaking: bool    = False
        self._last_interim: str = ""

        # Background task that fires interim transcriptions
        self._interim_task: asyncio.Task | None = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _transcribe_buffer(self, audio_bytes: bytes) -> str:
        """
        Run faster-whisper inference on raw Int16 PCM bytes.
        Returns the concatenated transcript text (empty string on no speech).
        """
        if len(audio_bytes) < self.SAMPLE_RATE * 2 * 0.3:  # < 0.3 s of audio
            return ""

        # Convert Int16 PCM → float32 normalised [-1, 1] (Whisper format)
        pcm_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_f32 = pcm_i16.astype(np.float32) / 32768.0

        try:
            segments, _ = self._model.transcribe(
                audio_f32,
                language=self._language,
                beam_size=5,
                vad_filter=True,           # built-in Silero filter inside Whisper
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    threshold=0.5,
                ),
                no_speech_threshold=0.6,   # drop silence hallucinations
                condition_on_previous_text=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        except Exception as e:
            logger.warning(f"[Whisper] Transcription error: {e}")
            return ""

    async def _run_transcribe(self, audio_bytes: bytes) -> str:
        """Run blocking Whisper inference in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_buffer, audio_bytes)

    async def _interim_loop(self):
        """
        Background coroutine: every CHUNK_INTERVAL_SECS, transcribe the current
        buffer and emit an InterimTranscriptionFrame if the text changed.
        Cancelled cleanly by VADUserStoppedSpeakingFrame.
        """
        try:
            while True:
                await asyncio.sleep(self.CHUNK_INTERVAL_SECS)

                snapshot = bytes(self._buffer)  # copy to avoid race
                if not snapshot:
                    continue

                text = await self._run_transcribe(snapshot)

                if text and text != self._last_interim:
                    self._last_interim = text
                    logger.debug(f"[Whisper] ⟳ Interim: {text!r}")
                    frame = InterimTranscriptionFrame(
                        text=text, user_id="", timestamp=""
                    )
                    await self.push_frame(frame)
        except asyncio.CancelledError:
            pass  # normal shutdown

    # ── FrameProcessor interface ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # ── Speech start: reset state and kick off interim loop ────────
            self._buffer.clear()
            self._last_interim = ""
            self._speaking = True

            if self._interim_task and not self._interim_task.done():
                self._interim_task.cancel()
            self._interim_task = asyncio.create_task(self._interim_loop())
            logger.debug("[Whisper] 🎤 Speech start — interim loop running")

        elif isinstance(frame, InputAudioRawFrame) and self._speaking:
            # ── Accumulate audio while user is speaking ───────────────────
            self._buffer.extend(frame.audio)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # ── Speech end: cancel interim loop, run final inference ──────
            self._speaking = False

            if self._interim_task and not self._interim_task.done():
                self._interim_task.cancel()
                self._interim_task = None

            final_audio = bytes(self._buffer)
            self._buffer.clear()
            self._last_interim = ""

            if final_audio:
                logger.debug("[Whisper] 🔇 Speech end — running final transcription")
                t0   = time.monotonic()
                text = await self._run_transcribe(final_audio)
                elapsed_ms = (time.monotonic() - t0) * 1000

                if text:
                    logger.info(
                        f"[Whisper] ✅ Final ({elapsed_ms:.0f}ms): {text!r}"
                    )
                    final_frame = TranscriptionFrame(
                        text=text, user_id="", timestamp=""
                    )
                    # Push final BEFORE passing the VAD stop frame downstream
                    # so broadcasters receive transcript first
                    await self.push_frame(final_frame)
                else:
                    logger.debug(f"[Whisper] ⚠ No speech detected ({elapsed_ms:.0f}ms)")

        # Always pass every frame downstream unchanged
        await self.push_frame(frame, direction)


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


# REST endpoint: return current STT engine info (for frontend display)
@app.get("/api/stt-info")
async def stt_info():
    """Returns the Whisper model config so the frontend can display it."""
    wcfg = _resolve_whisper_config()
    return JSONResponse({
        "engine": "local-whisper",
        "model": wcfg["model"],
        "device": wcfg["device"],
        "compute_type": wcfg["compute_type"],
        "language": wcfg["language"] or "auto",
        "streaming": True,
        "chunk_interval_secs": StreamingWhisperProcessor.CHUNK_INTERVAL_SECS,
    })

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

    # Load shared Whisper model (singleton — loaded once, reused across connections)
    whisper_model = await _get_whisper_model(wcfg)

    # Initialize VAD Processor with Silero analyzer
    # stop_secs: how long to wait after silence before declaring end-of-turn
    # Default is 0.3s — we keep it low to minimize pause before final transcription
    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "0.3"))
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=vad_stop_secs,  # Tunable via env: lower = faster response
            start_secs=0.2,           # Min speech duration before confirming start
            confidence=0.7,           # Silero speech confidence threshold
            min_volume=0.6,           # Min audio volume to count as speech
        )
    )
    logger.info(
        f"VAD stop_secs={vad_stop_secs}s | Streaming Whisper model={wcfg['model']} "
        f"| device={wcfg['device']} | compute_type={wcfg['compute_type']} "
        f"| interim_interval={StreamingWhisperProcessor.CHUNK_INTERVAL_SECS}s"
    )
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

    # StreamingWhisperProcessor: runs Whisper on rolling buffer mid-speech
    # (emits InterimTranscriptionFrames) and final accurate pass on silence.
    streaming_stt = StreamingWhisperProcessor(
        whisper_model=whisper_model,
        language=wcfg["language"],
    )

    # Broadcaster: sends TranscriptionFrame & VAD events directly over WebSocket
    # (the output transport's write_transport_frame() is a no-op for these types)
    broadcaster = TranscriptionBroadcaster(websocket)

    # Assemble the Pipecat pipeline:
    # WebSocket input -> VAD -> StreamingWhisper -> Broadcaster -> WebSocket output
    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        streaming_stt,
        broadcaster,
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

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        tts_engine: "TTSEngine",
        vision_session_id: str = "",
    ):
        super().__init__()
        self._websocket         = websocket
        self._session_id        = session_id
        self._tts               = tts_engine
        # Vision session_id: links this broadcaster to a /ws/vision session.
        # When set, body language context is injected into every VC turn prompt.
        # Audio is always primary; vision is a soft secondary signal.
        self._vision_session_id = vision_session_id or session_id

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

                # ── 1.5 Inject vision body language context (secondary) ─────
                # Read the latest expression/pose BEFORE calling the agent so
                # Marcus Reid has multimodal context for this turn.
                # Audio remains the primary basis; vision is a soft signal.
                vision_data = await vision_state.get(self._vision_session_id)
                vision_block = build_vision_context_block(vision_data)
                if vision_block:
                    logger.debug(
                        f"[VC] Vision context injected → "
                        f"expr={vision_data.get('expression')} "
                        f"pose={vision_data.get('pose')}"
                    )

                # Enrich the founder text with body language as a bracketed
                # annotation so the VC LLM receives it as part of the input
                # rather than as a system prompt modification (keeps turn
                # history clean and avoids prompt injection surface).
                enriched_text = text
                if vision_block:
                    expr = vision_data.get("expression", "")
                    pose = vision_data.get("pose", "")
                    expr_conf = vision_data.get("expression_confidence", 0)
                    pose_conf = vision_data.get("pose_confidence", 0)
                    if expr_conf > 0.35 or pose_conf > 0.35:
                        enriched_text = (
                            f"{text}\n"
                            f"[Body Language Context — secondary signal, audio is primary: "
                            f"facial expression = {expr} ({expr_conf:.0%} confidence), "
                            f"body pose = {pose} ({pose_conf:.0%} confidence)]"
                        )

                try:
                    async for event in run_turn_streaming(self._session_id, enriched_text):
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

    # Reuse the already-loaded singleton model (no second disk load)
    whisper_model = await _get_whisper_model(wcfg)

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

    # StreamingWhisperProcessor: interim results while speaking + final on silence
    streaming_stt = StreamingWhisperProcessor(
        whisper_model=whisper_model,
        language=wcfg["language"],
    )

    # VCBroadcaster: transcripts → LangGraph → token stream → TTS audio
    vc_broadcaster = VCBroadcaster(websocket, session_id, tts_engine)

    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        streaming_stt,
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


# ---------------------------------------------------------------------------
# /ws/vision — Dedicated webcam vision WebSocket endpoint
# ---------------------------------------------------------------------------
# The browser opens this as a SECOND WebSocket alongside /ws/vc.
# It receives JPEG frames, runs MediaPipe Holistic, and:
#   1. Sends VisionAnalysisFrame events back as JSON to the browser (for HUD)
#   2. Updates the shared VisionState so VCBroadcaster can inject body
#      language context into the VC AI on the next turn.
# The session_id MUST match the /ws/vc session_id so both pipelines share state.
# ---------------------------------------------------------------------------

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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VisionAnalysisFrame):
            try:
                await self._websocket.send_text(json.dumps({
                    "type":                   "vision",
                    "expression":             frame.expression,
                    "expression_confidence":  frame.expression_confidence,
                    "pose":                   frame.pose,
                    "pose_confidence":        frame.pose_confidence,
                    "raw_scores":             frame.raw_expression_scores,
                }))
            except Exception as e:
                logger.warning(f"[VisionBroadcaster] WS send error: {e}")

        await self.push_frame(frame, direction)


class RawJPEGSerializer(FrameSerializer):
    """
    Simple serializer that wraps raw binary (JPEG) WebSocket messages
    directly as InputImageRawFrame. The browser sends webcam frames as
    raw JPEG binary blobs with no envelope.
    """

    async def serialize(self, frame: Frame) -> bytes | None:
        return None  # server never sends binary on this vision socket

    async def deserialize(self, data: bytes | str) -> Frame | None:
        from pipecat.frames.frames import InputImageRawFrame
        if not data or not isinstance(data, bytes):
            return None
        return InputImageRawFrame(
            image=data,
            size=(0, 0),    # dimensions decoded inside MediaPipeVisionProcessor
            format="JPEG",
        )


@app.websocket("/ws/vision")
async def vision_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for the computer vision pipeline.
    Browser sends JPEG frames; server responds with expression/pose JSON.

    Query params:
      session_id — MUST match the /ws/vc session_id so VisionState is shared.
    """
    await websocket.accept()

    session_id = websocket.query_params.get("session_id", new_session())
    logger.info(f"[Vision] WebSocket connected. session_id={session_id}")

    # Throttle: process every 3rd frame (15fps webcam → ~5fps inference)
    process_every_n = int(os.getenv("VISION_PROCESS_EVERY_N", "3"))
    min_detect = float(os.getenv("VISION_MIN_DETECT", "0.5"))
    min_track  = float(os.getenv("VISION_MIN_TRACK", "0.5"))

    serializer = RawJPEGSerializer()
    transport  = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=False,
            audio_out_enabled=False,
            video_in_enabled=True,
            serializer=serializer,
        )
    )

    vision_processor = MediaPipeVisionProcessor(
        session_id=session_id,
        min_detection_confidence=min_detect,
        min_tracking_confidence=min_track,
        process_every_n=process_every_n,
    )

    vision_broadcaster = VisionBroadcaster(websocket)

    pipeline = Pipeline([
        transport.input(),
        vision_processor,
        vision_broadcaster,
        transport.output(),
    ])

    task   = PipelineWorker(pipeline, params=PipelineParams())
    runner = WorkerRunner()

    try:
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"[Vision] Pipeline exception: {e}")
    finally:
        await vision_processor.cleanup()
        logger.info(f"[Vision] WebSocket closed. session_id={session_id}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PrepMate server on http://localhost:8000")
    logger.info("  Raw STT demo: http://localhost:8000/")
    logger.info("  VC Pitch Arena: http://localhost:8000/vc")
    uvicorn.run("server_whisper_vad:app", host="127.0.0.1", port=8000, reload=True)