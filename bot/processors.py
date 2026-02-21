from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, OutputTransportMessageFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# --- 1. Custom Processor to Send Transcripts to WebRTC Data Channel ---
class TranscriptionSender(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.success(f"🎤 Transcription generated: {frame.text}")
            msg = {"type": "transcription", "text": frame.text}
            await self.push_frame(OutputTransportMessageFrame(message=msg), direction)
        else:
            await self.push_frame(frame, direction)
