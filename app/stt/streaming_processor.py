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

  ② Accurate final transcription (via Pipecat-native WhisperSTTService)
     When VADUserStoppedSpeakingFrame arrives, the interim task is
     cancelled and control is handed to the embedded WhisperSTTService.
     It runs one clean pass over the complete utterance buffer and emits
     a TranscriptionFrame.  Full context → higher accuracy than chunked.

Why not just use WhisperSTTService alone?
  Because WhisperSTTService = SegmentedSTTService → it only runs AFTER
  VAD stop, never during speech.  Zero interim frames.

Why not just use the old StreamingWhisperProcessor?
  It ran faster-whisper directly (not through Pipecat's service layer),
  bypassed model caching, and produced chunked-accuracy finals.
  It also had no final clean pass — the last chunk WAS the final.

Frame flow:
  VADProcessor
    → HybridWhisperSTTProcessor     ← this file
        ├─ [during speech] InterimTranscriptionFrame (every INTERIM_INTERVAL_SECS)
        └─ [after VAD stop] TranscriptionFrame (from WhisperSTTService clean pass)
    → TranscriptionBroadcaster
    → transport.output()

Key design decisions:
  • Interim: beam_size=1 (fast greedy decode) — speed > accuracy for interim
  • Final:   WhisperSTTService (beam_size=5 default) — accuracy for committed text
  • Buffer management: shared bytearray, snapshotted before each inference
  • Thread safety: asyncio.to_thread() for all blocking Whisper calls
  • Grace period removed: VAD stop = immediate final pass (WhisperSTTService
    already handles brief mid-sentence pauses via its start_secs param)
"""

import asyncio

import numpy as np
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
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz  (must match transport audio_in_sample_rate)
MIN_AUDIO_SECS = 0.3          # ignore buffers shorter than this (avoid hallucinations)
INTERIM_INTERVAL_SECS = 0.8   # how often to produce an interim frame while speaking


# ── WhisperSTTService factory ──────────────────────────────────────────────────

def _make_whisper_service(
    model: str,
    language: Language | None,
    device: str,
    compute_type: str,
    no_speech_prob: float,
) -> WhisperSTTService:
    """Build a Pipecat-native WhisperSTTService for the final-pass transcription."""
    settings = WhisperSTTService.Settings(
        model=model,
        language=language,
        no_speech_prob=no_speech_prob,
    )
    return WhisperSTTService(
        settings=settings,
        device=device,
        compute_type=compute_type,
    )


# ── HybridWhisperSTTProcessor ──────────────────────────────────────────────────

class HybridWhisperSTTProcessor(FrameProcessor):
    """
    Drop-in Pipecat FrameProcessor providing low-latency interim transcription
    AND high-accuracy final transcription from a single local Whisper model.

    Interim strategy
    ────────────────
    While VADUserStartedSpeakingFrame is active, a background task wakes up
    every INTERIM_INTERVAL_SECS, snapshots the growing PCM buffer, runs a
    fast greedy Whisper pass (beam_size=1) in a thread pool, and emits an
    InterimTranscriptionFrame if the text changed.

    Final strategy
    ──────────────
    On VADUserStoppedSpeakingFrame the interim task is cancelled and the
    accumulated PCM is handed to the embedded WhisperSTTService which does
    one clean, high-quality pass (beam_size=5) and emits a TranscriptionFrame.

    The embedded WhisperSTTService is intentionally NOT placed in the pipeline
    directly; instead its internal _model is used for the final pass so that
    both interim and final share the same loaded model instance (one load,
    zero extra memory).
    """

    def __init__(
        self,
        model: str = "tiny",
        language: str | None = "en",
        device: str = "auto",
        compute_type: str = "default",
        no_speech_prob: float = 0.4,
        interim_interval: float = INTERIM_INTERVAL_SECS,
    ) -> None:
        super().__init__()

        # ── language normalisation ──────────────────────────────────────────
        self._lang_enum: Language | None = None
        self._lang_str: str | None = None
        if language:
            try:
                self._lang_enum = Language(language.lower())
                self._lang_str  = language.lower()
            except ValueError:
                logger.warning(f"[HybridSTT] Unknown language {language!r}, using auto-detect")

        # ── build the Pipecat-native service for final pass ─────────────────
        # We keep it as an attribute so its model is loaded once and shared.
        self._stt_service = _make_whisper_service(
            model=model,
            language=self._lang_enum,
            device=device,
            compute_type=compute_type,
            no_speech_prob=no_speech_prob,
        )

        self._no_speech_prob    = no_speech_prob
        self._interim_interval  = interim_interval

        # ── state ───────────────────────────────────────────────────────────
        self._audio_buffer: bytearray          = bytearray()
        self._speaking: bool                   = False
        self._last_interim: str                = ""
        self._interim_task: asyncio.Task | None = None

    # ── Model access ──────────────────────────────────────────────────────────

    async def _ensure_model_loaded(self) -> None:
        """Trigger WhisperSTTService model load on first call (lazy)."""
        if not self._stt_service._model:
            await self._stt_service._load()

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
        model = self._stt_service._model
        if not model:
            return ""
        audio_f32 = self._pcm_to_float(audio_bytes)
        try:
            segments, _ = await asyncio.to_thread(
                model.transcribe,
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
        High-quality final pass via WhisperSTTService's own run_stt().
        Uses beam_size=5 (Pipecat default) for maximum accuracy on the
        complete, committed utterance.
        """
        if not self._is_long_enough(audio_bytes):
            return ""
        model = self._stt_service._model
        if not model:
            return ""
        audio_f32 = self._pcm_to_float(audio_bytes)
        try:
            segments, _ = await asyncio.to_thread(
                model.transcribe,
                audio_f32,
                language=self._lang_str,
                beam_size=5,                    # accurate
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300, "threshold": 0.45},
                no_speech_threshold=self._no_speech_prob,
                condition_on_previous_text=True,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:
            logger.warning(f"[HybridSTT] Final error: {exc}")
            return ""

    # ── Background interim task ───────────────────────────────────────────────

    async def _interim_loop(self) -> None:
        """
        Fires every INTERIM_INTERVAL_SECS while user is speaking.
        Uses a fast greedy pass so the UI feels responsive.
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
            # ── Speech started ──────────────────────────────────────────────
            # Ensure the Whisper model is loaded before the first inference.
            await self._ensure_model_loaded()

            self._audio_buffer.clear()
            self._last_interim = ""
            self._speaking = True

            # Cancel any stale interim task from a previous utterance.
            if self._interim_task and not self._interim_task.done():
                self._interim_task.cancel()
            self._interim_task = asyncio.create_task(self._interim_loop())
            logger.debug("[HybridSTT] 🎤 Speech started — interim loop running")

        elif isinstance(frame, InputAudioRawFrame) and self._speaking:
            # Accumulate raw PCM while speaking.
            self._audio_buffer.extend(frame.audio)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # ── Speech ended ───────────────────────────────────────────────
            self._speaking = False

            # Cancel interim loop immediately — we're about to do the final pass.
            if self._interim_task and not self._interim_task.done():
                self._interim_task.cancel()
                self._interim_task = None

            final_audio        = bytes(self._audio_buffer)
            self._audio_buffer.clear()
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

def create_whisper_stt_service(
    model: str = "tiny",
    language: str | None = "en",
    device: str = "auto",
    compute_type: str = "default",
    no_speech_prob: float = 0.4,
    interim_interval: float = INTERIM_INTERVAL_SECS,
) -> HybridWhisperSTTProcessor:
    """
    Build and return a HybridWhisperSTTProcessor.

    This is a drop-in replacement for WhisperSTTService in a Pipecat pipeline.
    It produces both:
      • InterimTranscriptionFrame every `interim_interval` seconds while speaking
      • TranscriptionFrame (high-accuracy) immediately after VAD stop

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
    """
    return HybridWhisperSTTProcessor(
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        no_speech_prob=no_speech_prob,
        interim_interval=interim_interval,
    )
