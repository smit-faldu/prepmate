"""
app/websockets/vc_ws.py — VC Pitch WebSocket endpoint (/ws/vc) and VCBroadcaster.

Turn-based flow (strict: one speaker at a time):
  1. Sends 'vc_thinking' to browser → client gates mic
  2. Streams LLM tokens → forwards as 'vc_token' for live text display
  3. On 'final' event: sends 'vc_response' JSON, THEN synthesizes the full
     VC reply via ElevenLabs and streams audio as binary WebSocket frames.
  4. Sends 'tts_done' → browser re-enables mic for next founder turn.
  While the AI is speaking, any transcript that arrives is DISCARDED (no queue).

── Turn-taking latency design ────────────────────────────────────────────────
Previous version used a COMMIT_DELAY_SECS=2.5 timer to batch fragmented
TranscriptionFrames (caused by Pipecat's _audio_idle_handler firing on audio
frame gaps while VAD thought user was still speaking).

That approach added a mandatory 2.5s dead-time AFTER VAD already detected
speech end — unacceptable for a real-time pitch evaluator.

Fix strategy (belt-and-suspenders):
  a) PCM chunk size is 1024 samples = 64ms, so frames arrive frequently
     enough that _audio_idle_handler almost never fires mid-utterance.
  b) VADProcessor now has audio_idle_timeout=2.0s — only triggers if no
     audio frames arrive for 2 full seconds (browser tab hidden, network
     blip), so it can't split a normal utterance.
  c) HybridWhisperSTTProcessor emits exactly ONE TranscriptionFrame per
     utterance (on VADUserStoppedSpeakingFrame). No batching needed.
  d) VCBroadcaster now dispatches to the LLM immediately on each
     TranscriptionFrame, guarded by _is_processing for turn-safety.

Result: the LLM fires within ~100ms of VAD detecting speech end.
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

    Intercepts final transcripts → dispatches immediately to the LangGraph VC
    agent → streams tokens to browser → synthesizes TTS audio → re-enables mic.

    Turn-based (strict, no queue)
    ──────────────────────────────
    One TranscriptionFrame arrives per utterance (HybridWhisperSTTProcessor
    emits it synchronously on VADUserStoppedSpeakingFrame). We dispatch to
    the LLM immediately — no commit-delay timer needed.  While _is_processing
    is True, additional transcripts are DISCARDED so the AI has the floor.
    """

    # ── Tuning knobs ──────────────────────────────────────────────────────────
    # NOTE: COMMIT_DELAY_SECS removed. Dispatching to LLM is now immediate.
    # The old 2.5s delay was added to batch split TranscriptionFrames from
    # Pipecat's _audio_idle_handler, but with PCM chunk size at 1024 samples
    # (64ms) and audio_idle_timeout=2.0s on the VADProcessor, frame-gap splits
    # are eliminated without any commit-delay overhead.

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

        # ── Processing guard ─────────────────────────────────────────────────
        # True while run_turn_streaming / TTS is running.
        # STRICT TURN-BASED: while _is_processing, ALL incoming transcriptions
        # are DISCARDED.  The user must wait for the AI to finish before their
        # next turn is accepted — one speaker at a time.
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
                    # Strict turn-based: AI has the floor — discard immediately.
                    logger.warning(
                        f"[VC] Discarding transcript (AI turn in progress): {text!r}"
                    )
                    await self.push_frame(frame, direction)
                    return

                logger.info(f"[VC] Transcript received — dispatching immediately: {text!r}")
                # Fire-and-forget the LLM turn; don't await here so the frame
                # pipeline keeps flowing (VAD events can still pass through).
                asyncio.create_task(self._run_llm_turn(text, direction))

            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterimTranscriptionFrame):
            await self._safe_send(json.dumps({"type": "interim", "text": frame.text}))

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            # User is speaking — inform the browser UI.
            await self._safe_send(json.dumps({"type": "status", "status": "speaking"}))

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # User paused/stopped — inform the browser UI.
            # The final Whisper pass runs inside HybridWhisperSTTProcessor and
            # will emit a TranscriptionFrame shortly after this event.
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
            # start_secs: how long Silero must see speech before confirming start.
            # 0.1s = 100ms faster reaction vs old 0.2s — catches the first word sooner.
            start_secs=0.1,
            # confidence: 0.6 is slightly more sensitive than 0.65.
            # Still robust against background noise; lower would over-trigger.
            confidence=0.6,
            # min_volume: 0.2 catches quieter speech that 0.3 was rejecting.
            # VAD's confidence threshold does the real noise rejection — this is
            # just a fast energy gate to skip obviously-silent frames cheaply.
            min_volume=0.2,
        )
    )
    # audio_idle_timeout: if NO audio frames arrive for this many seconds while
    # VAD thinks the user is speaking, force a speech-end transition.
    # This is the fix for _audio_idle_handler: 2.0s is long enough to never
    # trigger mid-sentence (even with browser tab jitter at 1024-sample chunks),
    # but short enough to unstick the pipeline if the WebSocket truly goes quiet.
    # Without this, a frame gap could leave the pipeline stuck in SPEAKING state.
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer, audio_idle_timeout=2.0)

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