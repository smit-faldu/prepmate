"""
app/stt/streaming_processor.py — StreamingWhisperProcessor

Runs faster-whisper on a rolling audio buffer while the user is speaking,
emitting InterimTranscriptionFrames every CHUNK_INTERVAL_SECS seconds.

When VAD signals silence the processor does NOT immediately commit the
transcript.  Instead it enters a short "grace period" (COMMIT_GRACE_SECS).
If the user resumes talking within that window the grace timer is cancelled
and audio accumulation continues seamlessly — this handles the very common
case of a natural mid-sentence pause that Silero/Pipecat VAD sometimes
mistakes for end-of-speech.

Only after silence persists for the full grace period does the processor
run the final Whisper pass and emit a TranscriptionFrame.

This eliminates two bugs that appeared in production:
  1. A brief pause inside an utterance caused a premature final transcript.
  2. The remaining audio after that false stop leaked into the *next* turn,
     causing the AI to respond to a sentence fragment before the founder
     had finished speaking.
"""

import asyncio
import time

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


class StreamingWhisperProcessor(FrameProcessor):
    """
    Custom Pipecat FrameProcessor that provides streaming-like transcription
    from local faster-whisper by running chunked inference on a rolling buffer.

    Behaviour:
      • On VADUserStartedSpeakingFrame  → cancel any pending grace timer,
                                          resume / reset buffer, start interim loop
      • On InputAudioRawFrame           → append PCM bytes to buffer (always while
                                          speaking or during grace period)
      • Every CHUNK_INTERVAL_SECS       → transcribe current buffer, emit
                                          InterimTranscriptionFrame if text changed
      • On VADUserStoppedSpeakingFrame  → cancel interim loop, start
                                          COMMIT_GRACE_SECS countdown
      • Grace period elapses            → run final Whisper pass on buffer,
                                          emit TranscriptionFrame, clear buffer
      • Grace period interrupted        → discard snapshot, continue accumulating
      All frames are still passed downstream unchanged.
    """

    # How often (seconds) to run Whisper on the accumulated buffer mid-speech.
    CHUNK_INTERVAL_SECS: float = 1.5

    # How long (seconds) to wait after VAD stop before committing the transcript.
    # 0.8 s bridges natural between-phrase pauses without adding noticeable latency.
    COMMIT_GRACE_SECS: float = 0.8

    SAMPLE_RATE: int = 16_000

    def __init__(self, whisper_model, language: str | None = None):
        super().__init__()
        self._model    = whisper_model
        self._language = language

        # Audio accumulation buffer (raw Int16 PCM bytes from the browser)
        self._buffer: bytearray     = bytearray()
        self._speaking: bool        = False
        self._last_interim: str     = ""

        # Background task that fires interim transcriptions while speaking
        self._interim_task: asyncio.Task | None = None

        # Background task that commits the final transcript after grace period
        self._commit_task: asyncio.Task | None  = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _transcribe_buffer(self, audio_bytes: bytes) -> str:
        """
        Run faster-whisper inference on raw Int16 PCM bytes.
        Returns the concatenated transcript text (empty string on no speech).
        """
        if len(audio_bytes) < self.SAMPLE_RATE * 2 * 0.3:   # < 0.3 s of audio
            return ""

        # Convert Int16 PCM → float32 normalised [-1, 1] (Whisper format)
        pcm_i16   = np.frombuffer(audio_bytes, dtype=np.int16)
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
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            logger.warning(f"[Whisper] Transcription error: {e}")
            return ""

    async def _run_transcribe(self, audio_bytes: bytes) -> str:
        """Run blocking Whisper inference in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_buffer, audio_bytes)

    def _cancel_task(self, task: asyncio.Task | None) -> None:
        """Cancel a task if it is still running."""
        if task and not task.done():
            task.cancel()

    async def _interim_loop(self):
        """
        Background coroutine: every CHUNK_INTERVAL_SECS, transcribe the current
        buffer and emit an InterimTranscriptionFrame if the text changed.
        Cancelled cleanly when VAD fires stop or speech resumes.
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
                    await self.push_frame(InterimTranscriptionFrame(text=text, user_id="", timestamp=""))
        except asyncio.CancelledError:
            pass  # normal shutdown

    async def _commit_after_grace(self):
        """
        Background coroutine started when VAD fires stop.

        Waits COMMIT_GRACE_SECS.  If cancelled before the timer elapses
        (because the user resumed speaking) this is a no-op — the buffer
        is retained for continued accumulation.

        If the timer elapses we run final Whisper on the full current buffer
        and emit a TranscriptionFrame.
        """
        try:
            await asyncio.sleep(self.COMMIT_GRACE_SECS)

            # Grab whatever is in the buffer at commit time (may include a few
            # frames of silence that arrived during the grace period — that's fine).
            final_audio        = bytes(self._buffer)
            self._buffer.clear()
            self._last_interim = ""

            if final_audio:
                logger.debug("[Whisper] 🔇 Grace elapsed — running final transcription")
                t0         = time.monotonic()
                text       = await self._run_transcribe(final_audio)
                elapsed_ms = (time.monotonic() - t0) * 1000

                if text:
                    logger.info(f"[Whisper] ✅ Final ({elapsed_ms:.0f}ms): {text!r}")
                    await self.push_frame(TranscriptionFrame(text=text, user_id="", timestamp=""))
                else:
                    logger.debug(f"[Whisper] ⚠ No speech detected ({elapsed_ms:.0f}ms)")

        except asyncio.CancelledError:
            # Speech resumed during grace period — keep the buffer as-is.
            logger.debug("[Whisper] ↩ Grace cancelled — user resumed speaking, buffer retained")

    # ── FrameProcessor interface ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # ── Speech (re)started ────────────────────────────────────────────
            grace_was_active = self._commit_task and not self._commit_task.done()

            if grace_was_active:
                # User resumed before the grace period elapsed.
                # Cancel the commit — buffer keeps accumulating seamlessly.
                self._cancel_task(self._commit_task)
                self._commit_task = None
                logger.debug("[Whisper] 🎤 Speech resumed mid-grace — commit cancelled, buffer retained")
            else:
                # Fresh utterance or grace already elapsed — start clean.
                self._buffer.clear()
                self._last_interim = ""
                logger.debug("[Whisper] 🎤 Speech start — buffer reset")

            self._speaking = True

            # (Re)start the interim transcription loop
            self._cancel_task(self._interim_task)
            self._interim_task = asyncio.create_task(self._interim_loop())

        elif isinstance(frame, InputAudioRawFrame):
            # Accumulate during active speech AND during the grace period so
            # audio arriving in that window is never lost.
            grace_active = self._commit_task and not self._commit_task.done()
            if self._speaking or grace_active:
                self._buffer.extend(frame.audio)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # ── VAD detected silence ──────────────────────────────────────────
            self._speaking = False

            # Stop interim updates immediately
            self._cancel_task(self._interim_task)
            self._interim_task = None

            # Cancel any previously scheduled commit (guard against overlapping
            # VAD events which can occasionally happen).
            self._cancel_task(self._commit_task)

            if self._buffer:
                buffered_secs = len(self._buffer) / (self.SAMPLE_RATE * 2)
                logger.debug(
                    f"[Whisper] ⏳ VAD stop — starting {self.COMMIT_GRACE_SECS}s grace period "
                    f"({buffered_secs:.1f}s audio buffered)"
                )
                self._commit_task = asyncio.create_task(self._commit_after_grace())
            else:
                logger.debug("[Whisper] ⚠ VAD stop with empty buffer — nothing to commit")

        # Always pass every frame downstream unchanged
        await self.push_frame(frame, direction)
