"""
app/config.py — Centralised configuration resolution from environment variables.
All other modules import from here; never call os.getenv() for these values elsewhere.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "DEBUG"))


# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_ENABLED: bool = os.getenv("TTS_ENABLED", "true").lower() == "true"


# ── Whisper / STT ─────────────────────────────────────────────────────────────
def resolve_whisper_config() -> dict:
    """
    Resolves Whisper device, compute_type, model, and language from environment.

    Auto-selects the optimal compute type:
      - CUDA  → float16  (native GPU half-precision, fastest + most accurate)
      - CPU   → int8     (~3× faster than float32, no GPU needed)

    Override compute type at any time via WHISPER_COMPUTE_TYPE in .env.

    Language:
      - "auto" or empty → multilingual auto-detect (set language=None)
      - "en", "hi", etc. → force a specific language (faster, more accurate)
    """
    device   = os.getenv("WHISPER_DEVICE", "auto")           # auto | cpu | cuda
    model    = os.getenv("WHISPER_MODEL", "tiny")             # tiny | base | large-v3 | …
    lang_env = os.getenv("WHISPER_LANGUAGE", "en").strip()    # en | hi | auto | ""

    # ── Compute type: auto-select unless explicitly set ─────────────────────
    compute_override = os.getenv("WHISPER_COMPUTE_TYPE", "").strip()
    if compute_override:
        compute_type = compute_override
    else:
        effective_device = device
        if device == "auto":
            try:
                import torch
                effective_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                effective_device = "cpu"  # torch not installed → assume CPU

        compute_type = "float16" if effective_device == "cuda" else "int8"

    # ── Language: None triggers Whisper multilingual auto-detect ────────────
    language: str | None = None if lang_env in ("", "auto") else lang_env

    logger.info(
        f"Whisper config → device={device!r} | model={model!r} "
        f"| compute_type={compute_type!r} "
        f"| language={'auto-detect' if language is None else language!r}"
    )
    return {"device": device, "model": model, "compute_type": compute_type, "language": language}


# ── VAD ───────────────────────────────────────────────────────────────────────
# VAD_STOP_SECS: how long Silero VAD must see silence before declaring speech ended.
# Triggers the Whisper final pass. 1.5s tolerates natural breathing pauses mid-sentence.
# The COMMIT_DELAY_SECS in VCBroadcaster handles batching split fragments — keep these independent.
VAD_STOP_SECS: float = float(os.getenv("VAD_STOP_SECS", "1.5"))


# ── VC / LLM ─────────────────────────────────────────────────────────────────
VC_GEMINI_MODEL: str = os.getenv("VC_GEMINI_MODEL", "gemini-2.5-flash")
VC_PERSONA_TIMEOUT_S: float = float(os.getenv("VC_PERSONA_TIMEOUT_S", "8"))
VC_ANALYST_TIMEOUT_S: float = float(os.getenv("VC_ANALYST_TIMEOUT_S", "12"))


def resolve_vc_thinking_kwargs() -> dict:
    """Returns the correct thinking kwarg dict for the configured Gemini model."""
    is_gemini3 = VC_GEMINI_MODEL.startswith("gemini-3")
    if is_gemini3:
        return {"thinking_level": os.getenv("VC_GEMINI_THINKING_LEVEL", "low")}
    return {"thinking_budget": int(os.getenv("VC_GEMINI_THINKING_BUDGET", "0"))}


# ── ElevenLabs / TTS ─────────────────────────────────────────────────────────
ELEVENLABS_API_KEY: str    = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str   = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
ELEVENLABS_MODEL: str      = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")


# ── Vision / MediaPipe ────────────────────────────────────────────────────────
VISION_PROCESS_EVERY_N: int = int(os.getenv("VISION_PROCESS_EVERY_N", "3"))
VISION_MIN_DETECT: float    = float(os.getenv("VISION_MIN_DETECT", "0.5"))
VISION_MIN_TRACK: float     = float(os.getenv("VISION_MIN_TRACK", "0.5"))
