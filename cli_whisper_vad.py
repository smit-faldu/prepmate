import asyncio
import os
import sys
from dotenv import load_dotenv
from loguru import logger
import pyaudio

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
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
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioInputTransport, LocalAudioTransportParams

# Load config from .env
load_dotenv(override=True)

# Configure logger
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))

class ConsoleTranscriptionPrinter(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._is_speaking = True
            sys.stdout.write("\n🎤 [Listening...] ")
            sys.stdout.flush()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._is_speaking = False
            # We don't print a new line here, we wait for the final transcription frame to clean up
            sys.stdout.write(" 🛑 [Processing...] ")
            sys.stdout.flush()
        elif isinstance(frame, InterimTranscriptionFrame):
            # Print interim results (overwriting the current line)
            text = frame.text.strip()
            if text:
                # Clear line and print
                sys.stdout.write(f"\r🎤 [Speaking]: {text}...")
                sys.stdout.flush()
        elif isinstance(frame, TranscriptionFrame):
            # Print the final transcribed sentence
            text = frame.text.strip()
            if text:
                sys.stdout.write(f"\r💬 [Final]: {text}\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("\r\n")
                sys.stdout.flush()

        await self.push_frame(frame, direction)

async def main():
    model_name = os.getenv("WHISPER_MODEL", "tiny")
    logger.info(f"Starting Whisper VAD+STT local pipeline with model '{model_name}'...")

    # Initialize PyAudio
    py_audio = pyaudio.PyAudio()

    # Configure local audio input transport
    transport_params = LocalAudioTransportParams(
        input_device_index=None,  # Use system default mic
    )
    transport = LocalAudioInputTransport(py_audio=py_audio, params=transport_params)

    # Initialize Whisper STT Service
    stt_service = WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=model_name,
            language="en"
        )
    )

    # Initialize VAD Processor with Silero analyzer
    vad_analyzer = SileroVADAnalyzer()
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

    # Custom transcription printer
    printer = ConsoleTranscriptionPrinter()

    # Define the pipeline flow:
    # 1. Receive audio frames from mic
    # 2. Feed audio into VAD to segment user turns
    # 3. Transcribe audio segments with Whisper STT
    # 4. Print transcripts in console
    pipeline = Pipeline([
        transport.input(),
        vad_processor,
        stt_service,
        printer
    ])

    # Task and runner configuration
    task = PipelineTask(pipeline, PipelineParams(
        # We don't need automated response triggers here
    ))

    runner = PipelineRunner()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Local microphone input active. Start speaking!")

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user.")
    finally:
        py_audio.terminate()
        logger.info("Pipeline shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
