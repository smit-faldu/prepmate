"""
app/websockets/vc_ws.py — VC Pitch WebSocket endpoint (/ws/vc) and VCBroadcaster.

Turn-based flow (strict: one speaker at a time):
  1. Sends 'vc_thinking' to browser → client gates mic
  2. Streams LLM tokens → forwards as 'vc_token' for live text display
  3. On 'final' event: sends 'vc_response' JSON, THEN synthesizes the full
     VC reply via ElevenLabs and streams audio as binary WebSocket frames.
  4. Sends 'tts_done' → browser re-enables mic for next founder turn.
  While Marcus is speaking, any transcript that arrives is DISCARDED (no queue).

── Why we use a commit-delay timer ────────────────────────────────────────────
Pipecat's VADController has TWO independent mechanisms that emit
VADUserStoppedSpeakingFrame:

  a) stop_secs silence detection  — fires when Silero VAD sees N seconds of
     quiet audio frames.  Tunable via VAD_STOP_SECS (now 1.2s).

  b) _audio_idle_handler          — fires when audio FRAMES STOP ARRIVING at
     all (browser WebSocket jitter / mic buffer gap) while VAD thinks the user
     is still speaking.  NOT tunable from our config; hardcoded in Pipecat.
     Mitigated by reducing PCM chunk size from 4096 → 1024 samples (64ms)
     so frames arrive 4× more frequently, preventing the idle gap.

Path (b) can still fire mid-sentence on slow connections, splitting a single
utterance into 2-3 TranscriptionFrames.

The fix: VCBroadcaster accumulates TranscriptionFrames in a buffer and starts
a COMMIT_DELAY_SECS (2.5s) countdown on each arrival.  If another fragment
arrives before the countdown expires it resets.  Only when the countdown fires
does the combined text commit to the LLM.  A MAX_BUFFER_AGE_SECS (10s) hard
cap ensures very long monologues are committed even if fragments keep arriving.
"""


import asyncio
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
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

from app.config import VAD_STOP_SECS, resolve_whisper_config
from app.stt.serializer import WhisperLiveSerializer
from app.stt.streaming_processor import create_whisper_stt_service
from app.tts.engine import TTSEngine
from app.vc import new_session, run_turn_streaming
from app.vision.context import build_vision_context_block
from app.vision.state import vision_state


class VCBroadcaster(FrameProcessor):
    """
    Replaces TranscriptionBroadcaster for the /ws/vc endpoint.

    Intercepts final transcripts → utterance grouping → feeds LangGraph VC
    agent → streams tokens to browser → synthesizes TTS audio → re-enables mic.

    Utterance grouping
    ──────────────────
    Each TranscriptionFrame is appended to _commit_buffer and a
    COMMIT_DELAY_SECS asyncio timer is (re-)started.  When the timer fires
    without any new fragment arriving, the combined text is committed to the
    LLM as one coherent founder turn.

    Processing guard
    ────────────────
    While an LLM turn is running, arriving fragments go into _pending_queue.
    After the turn + TTS complete, pending fragments are committed as one
    follow-up turn (not as multiple separate turns).
    """

    # ── Tuning knobs ──────────────────────────────────────────────────────────
    # Seconds to wait after the LAST fragment / VAD-stop event before committing
    # to the LLM.  The VAD-event-gating in process_frame means this timer only
    # ticks during confirmed silence — it is cancelled on every VADUserStarted
    # SpeakingFrame and restarted on every VADUserStoppedSpeakingFrame.
    # So there is no risk of "infinite deferral" and MAX_BUFFER_AGE is not needed.
    COMMIT_DELAY_SECS: float = 2.5

    # NOTE: MIN_WORDS_TO_COMMIT removed.
    # Any speech — even "Hello" — is a valid turn and should reach the LLM.
    # Word-count filtering was causing short but meaningful utterances to be
    # silently dropped, which felt broken.  VAD handles noise rejection upstream.

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        tts_engine: TTSEngine,
        vision_session_id: str = "",
    ):
        super().__init__()
        self._websocket         = websocket
        self._session_id        = session_id
        self._tts               = tts_engine
        self._vision_session_id = vision_session_id or session_id

        # WS health flag — flip to True on first send failure after close.
        self._ws_closed: bool = False

        # ── Utterance grouping state ─────────────────────────────────────────
        # Fragments arriving within COMMIT_DELAY_SECS of each other are batched.
        self._commit_buffer: list[str]      = []
        self._commit_task:   asyncio.Task | None = None
        # Cached direction for the delayed commit callback.
        self._last_direction: FrameDirection = FrameDirection.DOWNSTREAM

        # ── Processing guard ─────────────────────────────────────────────────
        # True while run_turn_streaming / TTS is running.
        # STRICT TURN-BASED: while _is_processing, ALL incoming transcriptions
        # are DISCARDED.  No queue.  The user must wait for Marcus to finish
        # before their next turn is accepted — one speaker at a time.
        self._is_processing: bool = False

    # ── WebSocket helpers ─────────────────────────────────────────────────────

    async def _safe_send(self, payload: str) -> bool:
        """Send a text frame; return True on success, False if WS is closed."""
        if self._ws_closed:
            return False
        try:
            await self._websocket.send_text(payload)
            return True
        except Exception as e:
            self._ws_closed = True
            logger.debug(f"[VCBroadcaster] WS closed, stopping sends: {e}")
            return False

    # ── Utterance grouping ────────────────────────────────────────────────────

    def _cancel_commit_timer(self) -> None:
        if self._commit_task and not self._commit_task.done():
            self._commit_task.cancel()
            self._commit_task = None

    async def _schedule_commit(self, direction: FrameDirection) -> None:
        """(Re-)start the commit countdown.  Called on every new fragment."""
        self._last_direction = direction
        self._cancel_commit_timer()
        self._commit_task = asyncio.create_task(self._commit_after_delay())

    async def _commit_after_delay(self) -> None:
        """
        Fires COMMIT_DELAY_SECS after the last VAD-stop or TranscriptionFrame.
        Drains the combined buffer and calls the LLM with the full utterance.

        No MAX_BUFFER_AGE here — the VAD-event-gating in process_frame ensures
        this timer ONLY runs during silence (cancelled on speech-start, restarted
        on speech-stop), so there is no risk of running forever.
        """
        try:
            await asyncio.sleep(self.COMMIT_DELAY_SECS)
        except asyncio.CancelledError:
            return  # reset — a VAD-start or new fragment arrived before we fired

        if not self._commit_buffer:
            return

        combined = " ".join(self._commit_buffer).strip()
        self._commit_buffer.clear()
        self._commit_task = None

        if not combined:
            return

        if self._is_processing:
            # Strict turn-based: Marcus has the floor — discard.
            logger.warning(
                f"[VC] LLM busy (strict turn-based) — DISCARDING transcript: {combined!r}"
            )
            return

        await self._run_llm_turn(combined, self._last_direction)

    # ── LLM turn execution ────────────────────────────────────────────────────

    async def _run_llm_turn(self, text: str, direction: FrameDirection) -> None:
        """Execute one complete founder→investor exchange (LLM + TTS)."""
        self._is_processing = True
        logger.info(f"[VC] ▶ Founder said: {text!r}")
        turn_start = time.monotonic()

        # 1. Gate the mic — tell the browser Marcus is thinking.
        await self._safe_send(json.dumps({"type": "vc_thinking", "founder_text": text}))

        # 1.5. Enrich with vision body-language context (secondary signal).
        vision_data  = await vision_state.get(self._vision_session_id)
        vision_block = build_vision_context_block(vision_data)
        enriched_text = text
        if vision_block:
            expr      = vision_data.get("expression", "")
            pose      = vision_data.get("pose", "")
            expr_conf = vision_data.get("expression_confidence", 0)
            pose_conf = vision_data.get("pose_confidence", 0)
            logger.debug(
                f"[VC] Vision context → expr={expr} pose={pose}"
            )
            if expr_conf > 0.35 or pose_conf > 0.35:
                enriched_text = (
                    f"{text}\n"
                    f"[Body Language Context — secondary signal, audio is primary: "
                    f"facial expression = {expr} ({expr_conf:.0%} confidence), "
                    f"body pose = {pose} ({pose_conf:.0%} confidence)]"
                )

        # 2. Stream LLM tokens.
        first_token_at = None
        final_vc_text  = ""
        final_event    = None

        try:
            async for event in run_turn_streaming(self._session_id, enriched_text):
                if event["type"] == "token":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                        logger.info(
                            f"[VC] First token in {first_token_at - turn_start:.2f}s"
                        )
                    await self._safe_send(
                        json.dumps({"type": "vc_token", "text": event["text"]})
                    )
                elif event["type"] == "final":
                    final_event   = event
                    final_vc_text = event.get("vc_response", "")
        except Exception as e:
            logger.error(f"[VC] Agent error: {e}")
            try:
                await self._websocket.send_text(
                    json.dumps({"type": "error", "message": str(e)})
                )
            except Exception:
                pass
            self._is_processing = False
            await self._safe_send(json.dumps({"type": "tts_done"}))
            return

        # 3. Send full metadata response.
        if final_event:
            total_s = time.monotonic() - turn_start
            logger.info(
                f"[VC] Turn complete in {total_s:.2f}s | "
                f"Stage={final_event['stage']} | "
                f"Exchange={final_event['exchange_count']} | "
                f"is_out={final_event['is_out']}"
            )
            await self._safe_send(json.dumps({
                "type":           "vc_response",
                "founder_text":   text,
                "vc_text":        final_vc_text,
                "stage":          final_event["stage"],
                "exchange_count": final_event["exchange_count"],
                "pitch_metrics":  final_event["pitch_metrics"],
                "is_out":         final_event["is_out"],
                "pitch_ended":    final_event["pitch_ended"],
                "latency_ms":     round(total_s * 1000),
            }))

        # 4. Stream TTS audio, then signal mic re-enable.
        await self._tts.synthesize_full_response(final_vc_text, self._websocket)

        # ── Turn complete: release lock ──────────────────────────────────────
        # STRICT TURN-BASED: no pending queue — the floor returns to the founder.
        # Any speech that occurred while Marcus was talking was discarded already.
        self._is_processing = False
        logger.debug("[VC] Turn complete — mic floor returned to founder")

    # ── FrameProcessor interface ──────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                if self._is_processing:
                    # Strict turn-based: Marcus has the floor — discard immediately.
                    logger.warning(
                        f"[VC] Discarding transcript (AI turn in progress): {text!r}"
                    )
                    # Still pass downstream for logging but do NOT buffer or schedule.
                    await self.push_frame(frame, direction)
                    return

                logger.debug(
                    f"[VC] Fragment received: {text!r} "
                    f"— buffering, timer reset to {self.COMMIT_DELAY_SECS}s"
                )
                self._commit_buffer.append(text)
                await self._schedule_commit(direction)
            # Always pass the raw frame downstream (for logging / other processors).
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterimTranscriptionFrame):
            await self._safe_send(json.dumps({"type": "interim", "text": frame.text}))

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            # ── User is speaking again: SUSPEND the commit timer ──────────────
            # The commit timer should only count down during silence.  If VAD
            # sees a new speech-start while the timer is running it means the
            # user paused mid-sentence but kept going — cancel the countdown so
            # we don't commit a half-sentence to the LLM.
            if self._commit_task and not self._commit_task.done():
                logger.debug(
                    "[VC] VAD speech-start — suspending commit timer "
                    "(user is still speaking)"
                )
                self._cancel_commit_timer()
            await self._safe_send(json.dumps({"type": "status", "status": "speaking"}))

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # ── User paused/stopped: ACTIVATE the commit timer ────────────────
            # If we already have buffered text and the timer isn't running yet
            # (it was cancelled by a speech-start above), restart it now.
            # This ensures the 2.5s countdown begins from the confirmed pause,
            # not from when the first fragment arrived.
            if self._commit_buffer and not self._is_processing:
                if not (self._commit_task and not self._commit_task.done()):
                    logger.debug(
                        "[VC] VAD speech-stop — activating commit timer "
                        f"({len(self._commit_buffer)} fragments buffered)"
                    )
                    await self._schedule_commit(direction)
            await self._safe_send(json.dumps({"type": "status", "status": "silence"}))

        await self.push_frame(frame, direction)



async def vc_websocket_endpoint(websocket: WebSocket):
    """Handler for the VC Pitch WebSocket (/ws/vc)."""
    await websocket.accept()

    session_id = websocket.query_params.get("session_id", new_session())
    logger.info(f"[VC] WebSocket connected. session_id={session_id}")

    tts_engine = TTSEngine()
    wcfg       = resolve_whisper_config()

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=VAD_STOP_SECS,
            start_secs=0.2,
            confidence=0.65,
            # min_volume: minimum RMS for the VAD to consider audio as speech.
            # 0.6 was too aggressive — brief quieter moments in natural speech
            # (breathing, word transitions, sentence starts) were falsely seen
            # as silence, triggering premature VAD stops mid-sentence.
            # 0.3 is more tolerant while still rejecting true background noise.
            min_volume=0.3,
        )
    )
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

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

    streaming_stt = await create_whisper_stt_service(
        model=wcfg["model"],
        language=wcfg.get("language", "en"),
        device=wcfg["device"],
        compute_type=wcfg["compute_type"],
        no_speech_prob=wcfg.get("no_speech_prob", 0.4),
    )
    vc_broadcaster = VCBroadcaster(websocket, session_id, tts_engine)

    pipeline = Pipeline(
        [transport.input(), vad_processor, streaming_stt, vc_broadcaster, transport.output()]
    )
    task   = PipelineWorker(pipeline, params=PipelineParams())
    runner = WorkerRunner()

    try:
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"[VC] Pipeline exception: {e}")
    finally:
        logger.info(f"[VC] WebSocket closed. session_id={session_id}")