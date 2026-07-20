"""
app/stt/streaming_processor.py — Hybrid real-time Pipecat STT processor.

Architecture
────────────
Pipecat's WhisperSTTService (SegmentedSTTService) is accurate but NOT
real-time: it only runs Whisper ONCE after VADUserStoppedSpeakingFrame,
so no InterimTranscriptionFrames appear while the user is speaking.

This module implements a HybridWhisperSTTProcessor that provides BOTH:

  ① Real-time interim transcription
     A background asyncio task runs Whisper on the growing audio buffer
     every INTERIM_INTERVAL_SECS while the user speaks.  Each pass
     emits an InterimTranscriptionFrame so the UI updates instantly.

  ② Accurate final transcription
     When VADUserStoppedSpeakingFrame arrives, the interim task is
     cancelled and a clean beam-search pass runs over the complete
     utterance buffer, emitting a TranscriptionFrame.

Frame flow:
  VADProcessor
    → HybridWhisperSTTProcessor     ← this file
        ├─ [during speech] InterimTranscriptionFrame (every INTERIM_INTERVAL_SECS)
        └─ [after VAD stop] TranscriptionFrame (final beam-search pass)
    → TranscriptionBroadcaster / VCBroadcaster
    → transport.output()

── Why words at the start of an utterance used to go missing ──────────────
Silero VAD requires `start_secs` (commonly ~0.2s) of *confirmed* speech
before it fires VADUserStartedSpeakingFrame. Pipecat's VADProcessor forwards
every audio frame downstream immediately — it does NOT wait for that
confirmation — so the first ~200-400ms of real speech (e.g. the "Hello" in
"Hello, nice to meet you") reaches this processor *before* the started-
speaking event does.

The previous version of this file cleared `_audio_buffer` on
VADUserStartedSpeakingFrame and only appended audio once `_speaking` was
already True, which silently discarded exactly that leading fragment on
every single utterance.

The fix mirrors what Pipecat's own SegmentedSTTService.process_audio_frame
does internally (see pipecat/services/stt_service.py): keep a small rolling
"pre-roll" buffer at all times, trimmed to `preroll_secs` while idle, and
simply stop trimming (let it keep growing) once speech is confirmed. The
buffer is never cleared on speech start — it already contains the leading
edge of the utterance by the time VAD confirms it.

── Why the model is now loaded once, not per-connection ───────────────────
The previous version built a brand new WhisperSTTService (and therefore a
brand new faster-whisper WhisperModel) inside every websocket handler, so
every browser tab reloaded the whole model from disk/GPU memory. Whisper
inference calls are stateless, so a single WhisperModel per (model, device,
compute_type) combination can safely be shared across every connection via
asyncio.to_thread(). See get_shared_whisper_model() / preload_whisper_model()
below — call preload_whisper_model() once at FastAPI startup so the first
user never pays the load cost.

Key design decisions:
  • Interim: beam_size=1 (fast greedy decode) — speed > accuracy for interim
  • Final:   beam_size=5 — accuracy for committed text
  • Pre-roll: rolling buffer, trimmed only while idle, carried into the
    utterance buffer unmodified when speech starts
  • Model: loaded once per (model, device, compute_type), shared by every
    connection; loading itself runs in a thread so it never blocks the
    event loop
  • Thread safety: asyncio.to_thread() for all blocking Whisper calls
"""

import asyncio

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz  (must match transport audio_in_sample_rate)
MIN_AUDIO_SECS = 0.3          # ignore buffers shorter than this (avoid hallucinations)
INTERIM_INTERVAL_SECS = 0.8   # how often to produce an interim frame while speaking
PREROLL_SECS = 1.0            # rolling pre-speech buffer retained while idle


# ── Shared model cache ──────────────────────────────────────────────────────────
# Keyed on (model, device, compute_type). Language/no_speech_prob are NOT part
# of the key because they're passed per-call to model.transcribe(), never
# baked into the loaded model — so the same model instance can safely be
# reused by connections configured with different languages.
_model_cache: dict[tuple[str, str, str], WhisperModel] = {}
_model_cache_lock = asyncio.Lock()


async def get_shared_whisper_model(model: str, device: str, compute_type: str) -> WhisperModel:
    """Return a process-wide shared WhisperModel, loading it at most once.

    Safe to call concurrently from multiple connections/tasks — the lock
    ensures only the first caller actually loads the model; everyone else
    reuses the cached instance. Loading runs in a worker thread so it never
    blocks the event loop (important the first time a large model is
    downloaded from the Hugging Face hub).
    """
    key = (model, device, compute_type)
    async with _model_cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached
        logger.info(
            f"[HybridSTT] Loading Whisper model {model!r} "
            f"(device={device!r}, compute_type={compute_type!r}) …"
        )
        loaded = await asyncio.to_thread(WhisperModel, model, device=device, compute_type=compute_type)
        _model_cache[key] = loaded
        logger.info(f"[HybridSTT] Whisper model {model!r} ready — shared across all connections")
        return loaded


async def preload_whisper_model(model: str, device: str, compute_type: str) -> None:
    """Warm the shared model cache. Call this once at FastAPI startup so the
    first websocket connection doesn't have to wait for a multi-second (or
    multi-minute, on first-ever download) model load.
    """
    await get_shared_whisper_model(model, device, compute_type)


# ── HybridWhisperSTTProcessor ──────────────────────────────────────────────────

class HybridWhisperSTTProcessor(FrameProcessor):
    """
    Drop-in Pipecat FrameProcessor providing low-latency interim transcription
    AND high-accuracy final transcription from a single local Whisper model.

    Interim strategy
    ────────────────
    While speech is in progress, a background task wakes up every
    INTERIM_INTERVAL_SECS, snapshots the growing PCM buffer, runs a fast
    greedy Whisper pass (beam_size=1) in a thread pool, and emits an
    InterimTranscriptionFrame if the text changed.

    Final strategy
    ──────────────
    On VADUserStoppedSpeakingFrame the interim task is cancelled and the
    accumulated PCM (pre-roll + full utterance) is run through one clean,
    high-quality pass (beam_size=5), emitting a TranscriptionFrame.

    Pre-roll strategy
    ──────────────────
    Audio is accumulated continuously, regardless of speaking state. While
    idle, the buffer is trimmed to the trailing `preroll_secs` seconds. Once
    speech is confirmed, trimming simply stops — the buffer already holds
    the leading edge of the utterance that arrived during VAD's confirmation
    window, so nothing spoken is ever thrown away.
    """

    def __init__(
        self,
        model: WhisperModel,
        language: str | None = "en",
        no_speech_prob: float = 0.4,
        interim_interval: float = INTERIM_INTERVAL_SECS,
        preroll_secs: float = PREROLL_SECS,
    ) -> None:
        super().__init__()

        # ── language normalisation ──────────────────────────────────────────
        self._lang_enum: Language | None = None
        self._lang_str: str | None = None
        if language:
            try:
                self._lang_enum = Language(language.lower())
                self._lang_str = language.lower()
            except ValueError:
                logger.warning(f"[HybridSTT] Unknown language {language!r}, using auto-detect")

        # ── shared, already-loaded model (see get_shared_whisper_model) ─────
        self._model = model

        self._no_speech_prob = no_speech_prob
        self._interim_interval = interim_interval
        self._preroll_bytes = int(SAMPLE_RATE * 2 * preroll_secs)  # 16-bit mono PCM

        # ── state ───────────────────────────────────────────────────────────
        self._audio_buffer: bytearray = bytearray()
        self._speaking: bool = False
        self._last_interim: str = ""
        self._interim_task: asyncio.Task | None = None

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _pcm_to_float(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw Int16 PCM bytes → float32 normalised [-1, 1]."""
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _is_long_enough(self, audio_bytes: bytes) -> bool:
        return len(audio_bytes) >= SAMPLE_RATE * 2 * MIN_AUDIO_SECS

    async def _transcribe_fast(self, audio_bytes: bytes) -> str:
        """
        Fast greedy decode for interim frames.
        beam_size=1 cuts latency in half vs beam_size=5; accuracy is
        good enough for a 'live preview' that the user never commits to.
        """
        if not self._is_long_enough(audio_bytes):
            return ""
        audio_f32 = self._pcm_to_float(audio_bytes)
        try:
            segments, _ = await asyncio.to_thread(
                self._model.transcribe,
                audio_f32,
                language=self._lang_str,
                beam_size=1,                    # greedy = fast
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200, "threshold": 0.5},
                no_speech_threshold=self._no_speech_prob,
                condition_on_previous_text=False,  # no drift between chunks
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:
            logger.warning(f"[HybridSTT] Interim error: {exc}")
            return ""

    async def _transcribe_final(self, audio_bytes: bytes) -> str:
        """
        High-quality final pass over the complete, committed utterance
        (pre-roll + everything spoken). beam_size=5 for max accuracy.
        """
        if not self._is_long_enough(audio_bytes):
            return ""
        audio_f32 = self._pcm_to_float(audio_bytes)
        try:
            segments, _ = await asyncio.to_thread(
                self._model.transcribe,
                audio_f32,
                language=self._lang_str,
                beam_size=5,                    # accurate
                vad_filter=True,
                # Lower threshold → more aggressive at capturing long speech
                vad_parameters={"min_silence_duration_ms": 300, "threshold": 0.35},
                no_speech_threshold=self._no_speech_prob,
                # False: no drift/hallucination loops on long utterances.
                # We have the FULL audio buffer already — no need to condition on prior output.
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:
            logger.warning(f"[HybridSTT] Final error: {exc}")
            return ""

    # ── Background interim task ───────────────────────────────────────────────

    async def _interim_loop(self) -> None:
        """
        Fires every INTERIM_INTERVAL_SECS while user is speaking.
        Cancelled cleanly when VAD stop fires.
        """
        try:
            while True:
                await asyncio.sleep(self._interim_interval)

                snapshot = bytes(self._audio_buffer)   # snapshot under no lock — fine for asyncio
                if not snapshot:
                    continue

                text = await self._transcribe_fast(snapshot)

                if text and text != self._last_interim:
                    self._last_interim = text
                    logger.debug(f"[HybridSTT] 🔄 Interim: {text!r}")
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text=text,
                            user_id="",
                            timestamp=time_now_iso8601(),
                        )
                    )
        except asyncio.CancelledError:
            pass  # normal shutdown path

    # ── FrameProcessor interface ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # ── Speech confirmed ─────────────────────────────────────────────
            # Deliberately do NOT clear _audio_buffer here. It already holds
            # up to `preroll_secs` of audio that arrived before VAD finished
            # confirming speech had started — exactly the words that used to
            # get dropped. We just stop trimming and let it keep growing.
            self._last_interim = ""
            self._speaking = True

            if self._interim_task and not self._interim_task.done():
                self._interim_task.cancel()
            self._interim_task = asyncio.create_task(self._interim_loop())
            logger.debug("[HybridSTT] 🎤 Speech started — interim loop running")

        elif isinstance(frame, InputAudioRawFrame):
            # Always accumulate, speaking or not.
            self._audio_buffer.extend(frame.audio)

            # While idle, keep only the trailing `preroll_secs` of audio so
            # the buffer doesn't grow unbounded during long silences. Once
            # speech starts we stop trimming, so nothing spoken is lost.
            if not self._speaking and len(self._audio_buffer) > self._preroll_bytes:
                discard = len(self._audio_buffer) - self._preroll_bytes
                del self._audio_buffer[:discard]

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # ── Speech ended ───────────────────────────────────────────────
            self._speaking = False

            # Cancel interim loop immediately — we're about to do the final pass.
            if self._interim_task and not self._interim_task.done():
                self._interim_task.cancel()
                self._interim_task = None

            final_audio        = bytes(self._audio_buffer)
            # ── Clear AFTER snapshotting, NOT before ──────────────────────
            # Audio frames that arrive during _transcribe_final() (a blocking
            # thread call) will land in a fresh buffer. We also seed the new
            # buffer with the trailing `preroll_bytes` of the just-finished
            # utterance so the very first word of the next sentence is never
            # lost if the founder starts speaking again immediately.
            self._audio_buffer = bytearray(final_audio[-self._preroll_bytes:])
            self._last_interim = ""

            if final_audio and self._is_long_enough(final_audio):
                logger.debug("[HybridSTT] 🔇 VAD stop — running final high-quality pass")
                text = await self._transcribe_final(final_audio)
                if text:
                    logger.info(f"[HybridSTT] ✅ Final: {text!r}")
                    await self.push_frame(
                        TranscriptionFrame(
                            text=text,
                            user_id="",
                            timestamp=time_now_iso8601(),
                            language=self._lang_enum,
                        )
                    )
                else:
                    logger.debug("[HybridSTT] ⚠ Final pass: no speech detected")
            else:
                logger.debug("[HybridSTT] ⚠ Buffer too short — skipping final pass")

        # Always pass every frame downstream unchanged.
        await self.push_frame(frame, direction)


# ── Public factory ─────────────────────────────────────────────────────────────

async def create_whisper_stt_service(
    model: str = "tiny",
    language: str | None = "en",
    device: str = "auto",
    compute_type: str = "default",
    no_speech_prob: float = 0.4,
    interim_interval: float = INTERIM_INTERVAL_SECS,
    preroll_secs: float = PREROLL_SECS,
) -> HybridWhisperSTTProcessor:
    """
    Build and return a HybridWhisperSTTProcessor backed by the shared,
    process-wide Whisper model (see get_shared_whisper_model()).

    This is now an async function — call sites must `await` it. It produces:
      • InterimTranscriptionFrame every `interim_interval` seconds while speaking
      • TranscriptionFrame (high-accuracy) immediately after VAD stop, with the
        leading edge of the utterance preserved via the pre-roll buffer

    Parameters
    ----------
    model :
        faster-whisper model name: "tiny", "base", "small", "medium",
        "large-v3", "distil-medium.en", etc.
    language :
        BCP-47 code ("en", "hi") or None for auto-detect.
    device :
        "auto", "cpu", or "cuda".
    compute_type :
        "default", "int8", "int8_float16", "float16", etc.
    no_speech_prob :
        Segments with no_speech_prob above this value are discarded.
    interim_interval :
        Seconds between interim transcription updates while the user speaks.
        Lower = more responsive UI; higher = fewer inference calls.
        Recommended: 0.6–1.0 s.
    preroll_secs :
        Seconds of pre-speech audio retained so the first word(s) of an
        utterance are never lost to VAD confirmation delay. 1.0s matches
        Pipecat's own SegmentedSTTService default; lower it only if you've
        confirmed your VAD start_secs is comfortably smaller.
    """
    shared_model = await get_shared_whisper_model(model, device, compute_type)
    return HybridWhisperSTTProcessor(
        model=shared_model,
        language=language,
        no_speech_prob=no_speech_prob,
        interim_interval=interim_interval,
        preroll_secs=preroll_secs,
    )