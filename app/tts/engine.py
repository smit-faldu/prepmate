"""
app/tts/engine.py — TTSEngine (ElevenLabs HTTP streaming TTS)

Feeds the VC response text to ElevenLabs and yields raw MP3 audio chunks as
they stream back. Turn-based: the caller awaits the full audio stream before
re-enabling the mic.
"""

import json
import os
import time

from fastapi import WebSocket
from loguru import logger

from app.config import ELEVENLABS_API_KEY, ELEVENLABS_MODEL, ELEVENLABS_VOICE_ID, TTS_ENABLED


class TTSEngine:
    """
    Async wrapper around ElevenLabs HTTP streaming TTS.

    Usage:
        engine = TTSEngine()
        async for pcm_chunk in engine.synthesize(text):
            await websocket.send_bytes(pcm_chunk)
    """

    def __init__(self):
        self._api_key  = ELEVENLABS_API_KEY
        self._voice_id = ELEVENLABS_VOICE_ID
        self._model    = ELEVENLABS_MODEL
        self._enabled  = (
            TTS_ENABLED
            and bool(self._api_key)
            and self._api_key != "your_elevenlabs_api_key_here"
        )

        if not self._enabled:
            logger.warning("[TTS] Disabled — set ELEVENLABS_API_KEY and TTS_ENABLED=true in .env")
        else:
            logger.info(f"[TTS] ElevenLabs ready. voice={self._voice_id!r} model={self._model!r}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def synthesize(self, text: str):
        """
        Synthesize *text* via ElevenLabs streaming API.
        Yields raw MP3 audio bytes as they arrive from the API.
        """
        if not self._enabled or not text.strip():
            return

        import aiohttp

        url     = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
        headers = {
            "xi-api-key":   self._api_key,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        }
        payload = {
            "text":     text,
            "model_id": self._model,
            "voice_settings": {
                "stability":        0.45,
                "similarity_boost": 0.85,
                "style":            0.30,
                "use_speaker_boost": True,
            },
            "output_format": "mp3_44100_128",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"[TTS] ElevenLabs error {resp.status}: {body[:200]}")
                        return
                    async for chunk in resp.content.iter_chunked(4096):
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"[TTS] Synthesis error: {e}")

    async def synthesize_full_response(self, full_text: str, websocket: WebSocket):
        """
        Synthesize the complete VC response and stream audio binary frames to
        the browser. Sends a JSON 'tts_done' event when finished so the client
        knows to re-enable the mic.
        """
        if not self._enabled:
            try:
                await websocket.send_text(json.dumps({"type": "tts_done"}))
            except Exception:
                pass
            return

        # Strip any internal tags before synthesizing
        clean = full_text.replace("<END_PITCH>", "").strip()
        if not clean:
            await websocket.send_text(json.dumps({"type": "tts_done"}))
            return

        logger.info(f"[TTS] Synthesizing {len(clean)} chars for Marcus Reid...")
        t0          = time.monotonic()
        chunk_count = 0

        try:
            async for audio_chunk in self.synthesize(clean):
                await websocket.send_bytes(audio_chunk)
                chunk_count += 1
        except Exception as e:
            logger.error(f"[TTS] Stream send error: {e}")

        elapsed = time.monotonic() - t0
        logger.info(f"[TTS] Done — {chunk_count} chunks in {elapsed:.2f}s")

        try:
            await websocket.send_text(json.dumps({"type": "tts_done"}))
        except Exception:
            pass
