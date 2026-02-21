import asyncio
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from bot.llm_processor import LangGraphProcessor

async def run_bot_pipeline(sdp: str, type: str) -> dict:
    """Sets up and runs the WebRTC STT Pipeline for a new connection."""
    logger.info("Setting up WebRTC STT Pipeline...")

    webrtc_connection = SmallWebRTCConnection()
    await webrtc_connection.initialize(sdp=sdp, type=type)

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False, 
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=2.0)),
        )
    )

    logger.debug("Initializing WhisperSTTService...")
    stt = WhisperSTTService(model="tiny.en", device="auto", compute_type="int8")
    
    # Instantiate the LangGraph processor for STT->LLM
    llm_processor = LangGraphProcessor()

    pipeline = Pipeline([transport.input(), stt, llm_processor, transport.output()])
    task = PipelineTask(pipeline)

    answer = webrtc_connection.get_answer()
    
    runner = PipelineRunner(handle_sigint=False)
    bg_task = asyncio.create_task(runner.run(task))

    # IMPORTANT: Catch and log any silent errors from the background pipeline!
    def handle_pipeline_result(t):
        try:
            t.result()
            logger.info("Pipeline finished successfully.")
        except Exception as e:
            logger.error(f"🚨 Pipeline crashed: {e}")

    bg_task.add_done_callback(handle_pipeline_result)
    logger.info("Pipeline task started successfully.")

    return {"sdp": answer["sdp"], "type": answer["type"]}
