"""
app/websockets/vc_ws.py — VC Pitch WebSocket endpoint (/ws/vc) and VCBroadcaster.

Turn-based flow (investor does NOT get interrupted):
  1. Sends 'vc_thinking' to browser → client gates mic
  2. Streams LLM tokens → forwards as 'vc_token' for live text display
  3. On 'final' event: sends 'vc_response' JSON, THEN synthesizes the full
     VC reply via ElevenLabs and streams audio as binary WebSocket frames.
  4. Sends 'tts_done' → browser re-enables mic for next founder turn.
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
from app.stt.streaming_processor import create_whisper_stt_service
from app.tts.engine import TTSEngine
from app.vc import new_session, run_turn_streaming
from app.vision.context import build_vision_context_block
from app.vision.state import vision_state


class VCBroadcaster(FrameProcessor):
    """
    Replaces TranscriptionBroadcaster for the /ws/vc endpoint.

    Intercepts final transcripts → feeds LangGraph VC agent → streams tokens
    to the browser → synthesizes TTS audio → re-enables mic.
    """

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
        # Flip to True on first send failure so we stop attempting after WS close.
        self._ws_closed: bool   = False

    async def _safe_send(self, payload: str) -> bool:
        """Send a text frame; return True on success, False if the WS is closed."""
        if self._ws_closed:
            return False
        try:
            await self._websocket.send_text(payload)
            return True
        except Exception as e:
            self._ws_closed = True
            logger.debug(f"[VCBroadcaster] WS closed, stopping sends: {e}")
            return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        payload = None

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"[VC] Founder said: {text!r}")
                turn_start = time.monotonic()

                # 1. Tell browser Marcus is thinking (gate the mic)
                await self._safe_send(json.dumps({"type": "vc_thinking", "founder_text": text}))

                # 1.5 Inject vision body language context (secondary signal)
                vision_data  = await vision_state.get(self._vision_session_id)
                vision_block = build_vision_context_block(vision_data)
                if vision_block:
                    logger.debug(
                        f"[VC] Vision context injected → "
                        f"expr={vision_data.get('expression')} "
                        f"pose={vision_data.get('pose')}"
                    )

                # Enrich founder text with body language as bracketed annotation
                enriched_text = text
                if vision_block:
                    expr      = vision_data.get("expression", "")
                    pose      = vision_data.get("pose", "")
                    expr_conf = vision_data.get("expression_confidence", 0)
                    pose_conf = vision_data.get("pose_confidence", 0)
                    if expr_conf > 0.35 or pose_conf > 0.35:
                        enriched_text = (
                            f"{text}\n"
                            f"[Body Language Context — secondary signal, audio is primary: "
                            f"facial expression = {expr} ({expr_conf:.0%} confidence), "
                            f"body pose = {pose} ({pose_conf:.0%} confidence)]"
                        )

                # 2. Stream LLM tokens
                first_token_at = None
                final_vc_text  = ""
                final_event    = None

                try:
                    async for event in run_turn_streaming(self._session_id, enriched_text):
                        if event["type"] == "token":
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                                logger.info(f"[VC] First token in {first_token_at - turn_start:.2f}s")
                            try:
                                await self._safe_send(
                                    json.dumps({"type": "vc_token", "text": event["text"]})
                                )
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
                    await self._safe_send(json.dumps({"type": "tts_done"}))
                    await self.push_frame(frame, direction)
                    return

                # 3. Send full metadata response
                if final_event:
                    total_s = time.monotonic() - turn_start
                    logger.info(
                        f"[VC] Turn complete in {total_s:.2f}s | "
                        f"Stage={final_event['stage']} | "
                        f"Exchange={final_event['exchange_count']} | "
                        f"is_out={final_event['is_out']}"
                    )
                    try:
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
                    except Exception as e:
                        logger.warning(f"[VCBroadcaster] vc_response send error: {e}")

                # 4. Stream TTS audio, then signal mic re-enable
                await self._tts.synthesize_full_response(final_vc_text, self._websocket)

        elif isinstance(frame, InterimTranscriptionFrame):
            payload = json.dumps({"type": "interim", "text": frame.text})
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            payload = json.dumps({"type": "status", "status": "speaking"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            payload = json.dumps({"type": "status", "status": "silence"})

        if payload:
            await self._safe_send(payload)

        await self.push_frame(frame, direction)


async def vc_websocket_endpoint(websocket: WebSocket):
    """Handler for the VC Pitch WebSocket (/ws/vc)."""
    await websocket.accept()

    session_id = websocket.query_params.get("session_id", new_session())
    logger.info(f"[VC] WebSocket connected. session_id={session_id}")

    tts_engine    = TTSEngine()
    wcfg          = resolve_whisper_config()

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(stop_secs=VAD_STOP_SECS, start_secs=0.2, confidence=0.7, min_volume=0.6)
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

    pipeline = Pipeline([transport.input(), vad_processor, streaming_stt, vc_broadcaster, transport.output()])
    task     = PipelineWorker(pipeline, params=PipelineParams())
    runner   = WorkerRunner()

    try:
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"[VC] Pipeline exception: {e}")
    finally:
        logger.info(f"[VC] WebSocket closed. session_id={session_id}")