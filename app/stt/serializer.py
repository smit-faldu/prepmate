"""
app/stt/serializer.py — WhisperLiveSerializer

Custom Pipecat FrameSerializer that bridges the browser's binary audio WebSocket
with the Pipecat pipeline's InputAudioRawFrame format, and converts pipeline
output frames to JSON text events for the browser.
"""

import json

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class WhisperLiveSerializer(FrameSerializer):
    """
    Converts pipeline frames to WebSocket text messages sent back to the browser
    client, and incoming binary audio blobs to InputAudioRawFrame.
    """

    def __init__(self):
        super().__init__()

    async def serialize(self, frame: Frame) -> bytes | str | None:
        """Converts pipeline frames to WebSocket text messages."""
        if isinstance(frame, InterimTranscriptionFrame):
            logger.debug(f"[Serializer] Interim: {frame.text!r}")
            return json.dumps({"type": "interim", "text": frame.text})
        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"[Serializer] Final transcript: {frame.text!r}")
            return json.dumps({"type": "final", "text": frame.text})
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug("[Serializer] VAD: User started speaking")
            return json.dumps({"type": "status", "status": "speaking"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.debug("[Serializer] VAD: User stopped speaking")
            return json.dumps({"type": "status", "status": "silence"})
        return None

    async def deserialize(self, data: bytes | str) -> Frame | None:
        """Converts incoming raw binary frames from browser WebSocket into Pipecat input frames."""
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
        return None
