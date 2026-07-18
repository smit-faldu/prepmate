"""
app/vc/llm.py — LLM setup for the VC agent pipeline.

Two models are instantiated at import time:
  _analyst_llm  — structured output, low temperature, runs concurrently in background
  _vc_llm       — streaming, higher temperature, directly user-facing
"""

from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import VC_GEMINI_MODEL, VC_ANALYST_TIMEOUT_S, VC_PERSONA_TIMEOUT_S, resolve_vc_thinking_kwargs
from app.vc.schemas import PitchAnalysis

_thinking_kwargs = resolve_vc_thinking_kwargs()

logger.info(f"[VC Agent] Model={VC_GEMINI_MODEL!r} | thinking_kwargs={_thinking_kwargs}")

# Analyst LLM: low temperature, structured output, runs in the background —
# its latency no longer sits on the user-facing critical path.
analyst_llm = ChatGoogleGenerativeAI(
    model=VC_GEMINI_MODEL,
    temperature=0.1,
    **_thinking_kwargs,
).with_structured_output(PitchAnalysis)

# VC Persona LLM: higher temperature — this is the one the user is waiting
# on, so it streams.
vc_llm = ChatGoogleGenerativeAI(
    model=VC_GEMINI_MODEL,
    temperature=0.75,
    streaming=True,
    **_thinking_kwargs,
)

# Timeouts: bound how long we'll wait before falling back, so one slow
# preview-model call (documented 0.6s-20s variance) can't stall a turn.
PERSONA_TIMEOUT_S = VC_PERSONA_TIMEOUT_S
ANALYST_TIMEOUT_S = VC_ANALYST_TIMEOUT_S
