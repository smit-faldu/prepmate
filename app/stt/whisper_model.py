"""
app/stt/whisper_model.py — Singleton faster-whisper model loader.

The model is loaded once at first connection and shared across all subsequent
WebSocket connections, avoiding repeated disk reads.
"""

import asyncio

from loguru import logger


_WHISPER_MODEL = None
_WHISPER_MODEL_LOCK = asyncio.Lock()


async def get_whisper_model(wcfg: dict):
    """Lazily load faster-whisper WhisperModel as a singleton."""
    global _WHISPER_MODEL
    async with _WHISPER_MODEL_LOCK:
        if _WHISPER_MODEL is None:
            from faster_whisper import WhisperModel
            logger.info(
                f"[Whisper] Loading model '{wcfg['model']}' "
                f"device={wcfg['device']} compute_type={wcfg['compute_type']} ..."
            )
            _WHISPER_MODEL = WhisperModel(
                wcfg["model"],
                device=wcfg["device"],
                compute_type=wcfg["compute_type"],
            )
            logger.info("[Whisper] Model ready.")
    return _WHISPER_MODEL
