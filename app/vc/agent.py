"""
app/vc/agent.py — Core agentic turn logic for the VC pitch evaluator.

Pipeline per turn (AGENTIC, CONCURRENT — not sequential):

  HumanMessage (founder speech)
      │
      ├────────────────────────────────┬──────────────────────────────────────┐
      ▼                                 ▼
  [vc_persona]                      [analyst]
  STREAMS the in-character           Runs concurrently in the background.
  VC reply immediately, using        Extracts structured pitch metrics from
  pitch_metrics from the             THIS turn's transcript via Pydantic
  PREVIOUS turn (already in          structured output. Feeds into the
  state) + the raw new               *next* turn's persona prompt — never
  transcript for context.            blocks the current user-facing reply.
      │
      ▼
  User hears the first token in ~1s instead of waiting for both calls.
"""

import asyncio
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from app.vc.graph import ensure_session_initialized, get_vc_graph
from app.vc.llm import ANALYST_TIMEOUT_S, PERSONA_TIMEOUT_S, analyst_llm, vc_llm
from app.vc.prompts import ANALYST_SYSTEM, build_persona_system_prompt
from app.vc.schemas import PitchAnalysis, _merge_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Normalize Gemini content (str or list-of-parts) into plain text."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content) if content else ""


def _compute_stage_and_exit(
    stage: str,
    exchange_count: int,
    clarity_score: int,
    existing_red_flags: list[str],
    new_red_flags: list[str],
) -> tuple[str, int, bool, list[str]]:
    """Pure deterministic stage-routing + exit-condition logic (no LLM call)."""
    exchange_count = exchange_count + 1
    all_red_flags  = list({*existing_red_flags, *new_red_flags})

    is_out = False
    if stage != "intro":
        if clarity_score < 3:
            is_out = True
        if len(all_red_flags) >= 4:
            is_out = True

    new_stage = stage
    if not is_out:
        if stage == "intro":
            if exchange_count >= 2 and clarity_score >= 4:
                new_stage = "deep_dive"
            elif exchange_count >= 3:
                new_stage = "deep_dive"
        elif stage == "deep_dive":
            if exchange_count >= 5:
                if len(all_red_flags) >= 3:
                    is_out = True
                else:
                    new_stage = "negotiation"
        elif stage == "negotiation":
            if exchange_count >= 7:
                new_stage = "decision"

    return new_stage, exchange_count, is_out, all_red_flags


# ─────────────────────────────────────────────────────────────────────────────
# Analyst — structured background extraction
# ─────────────────────────────────────────────────────────────────────────────

async def _run_analyst(founder_text: str) -> PitchAnalysis | None:
    """
    Async structured-output extraction, timeout-bounded. On timeout/error we
    return None and the caller carries forward last turn's metrics — the
    session keeps moving instead of hanging.
    """
    try:
        return await asyncio.wait_for(
            analyst_llm.ainvoke(
                [SystemMessage(content=ANALYST_SYSTEM), HumanMessage(content=founder_text)]
            ),
            timeout=ANALYST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[Analyst] Timed out after {ANALYST_TIMEOUT_S}s — carrying forward prior metrics.")
        return None
    except Exception as e:
        logger.error(f"[Analyst] Error: {e} — carrying forward prior metrics.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core turn function
# ─────────────────────────────────────────────────────────────────────────────

async def stream_vc_turn(session_id: str, founder_text: str) -> AsyncIterator[dict]:
    """
    Core agentic turn function — yields incremental events so the caller
    (WebSocket handler) can forward tokens to the browser as they arrive.

    Yields:
      {"type": "token", "text": "..."}            — streamed chunk
      {"type": "final", ...full turn result...}    — turn complete

    Concurrency model:
      - vc_persona starts streaming IMMEDIATELY using last turn's metrics —
        this is the only thing the user waits on.
      - analyst runs concurrently in the background using THIS turn's text;
        its output updates state for the NEXT turn (never blocks this one).
    """
    graph  = get_vc_graph()
    config = {"configurable": {"thread_id": session_id}}

    snapshot = graph.get_state(config)
    state    = snapshot.values if snapshot and snapshot.values else {}

    stage          = state.get("stage", "intro")
    metrics        = state.get("pitch_metrics", {})
    is_out_hint    = state.get("is_out", False)
    exchange_count = state.get("exchange_count", 0)
    history        = state.get("messages", [])

    system_prompt    = build_persona_system_prompt(stage, metrics, is_out_hint)
    persona_messages = [SystemMessage(content=system_prompt)] + list(history) + [
        HumanMessage(content=founder_text)
    ]

    # ── Launch analyst concurrently — does NOT block persona streaming ───────
    analyst_task = asyncio.create_task(_run_analyst(founder_text))

    # ── Stream the persona reply — this is what the user is waiting on ───────
    final_content = ""
    try:
        token_queue: asyncio.Queue = asyncio.Queue()

        async def _pump_stream():
            try:
                async for chunk in vc_llm.astream(persona_messages):
                    piece = _extract_text(chunk.content)
                    if piece:
                        await token_queue.put(piece)
            finally:
                await token_queue.put(None)  # sentinel: stream done

        pump_task = asyncio.create_task(_pump_stream())
        while True:
            piece = await asyncio.wait_for(token_queue.get(), timeout=PERSONA_TIMEOUT_S)
            if piece is None:
                break
            final_content += piece
            yield {"type": "token", "text": piece}
        await pump_task  # surface any exception raised inside the pump

    except asyncio.TimeoutError:
        logger.warning(f"[Persona] Streaming timed out after {PERSONA_TIMEOUT_S}s — falling back to sync call.")
        pump_task.cancel()
        try:
            response      = await asyncio.wait_for(vc_llm.ainvoke(persona_messages), timeout=PERSONA_TIMEOUT_S)
            final_content = _extract_text(response.content)
            yield {"type": "token", "text": final_content}
        except Exception as e:
            logger.error(f"[Persona] Fallback also failed: {e}")
            final_content = "Sorry, I'm having trouble responding right now — could you repeat that?"
            yield {"type": "token", "text": final_content}
    except Exception as e:
        logger.error(f"[Persona] Streaming error: {e}")
        final_content = "Sorry, I'm having trouble responding right now — could you repeat that?"
        yield {"type": "token", "text": final_content}

    # ── Detect "I'm out" ───────────────────────────────────────────────────
    resp_lower   = final_content.lower()
    declared_out = "i'm out" in resp_lower or "im out" in resp_lower or "i am out" in resp_lower
    pitch_ended  = declared_out or (stage == "decision")
    if declared_out and "<END_PITCH>" not in final_content:
        final_content = final_content.rstrip() + "\n\n<END_PITCH>"

    # ── Await analyst result (it's been running this whole time) ─────────────
    analysis = await analyst_task
    if analysis is not None:
        metrics_update = analysis.model_dump()
        clarity_score  = metrics_update.get("clarity_score", 5)
        new_red_flags  = metrics_update.get("red_flags", [])
    else:
        metrics_update = {}
        clarity_score  = metrics.get("clarity_score", 5)
        new_red_flags  = []

    new_stage, new_exchange_count, is_out, all_red_flags = _compute_stage_and_exit(
        stage=stage,
        exchange_count=exchange_count,
        clarity_score=clarity_score,
        existing_red_flags=metrics.get("red_flags", []),
        new_red_flags=new_red_flags,
    )
    is_out      = is_out or declared_out
    pitch_ended = pitch_ended or is_out

    if metrics_update:
        metrics_update["red_flags"] = all_red_flags

    merged_metrics = _merge_metrics(metrics, metrics_update)

    # ── Persist state for next turn via the graph's checkpointer ─────────────
    graph.update_state(
        config,
        {
            "messages":       [HumanMessage(content=founder_text), AIMessage(content=final_content)],
            "pitch_metrics":  metrics_update,
            "stage":          new_stage,
            "exchange_count": new_exchange_count,
            "is_out":         is_out,
            "vc_response":    final_content,
            "pitch_ended":    pitch_ended,
        },
    )

    yield {
        "type":           "final",
        "vc_response":    final_content,
        "stage":          new_stage,
        "exchange_count": new_exchange_count,
        "pitch_metrics":  merged_metrics,
        "is_out":         is_out,
        "pitch_ended":    pitch_ended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def run_turn_streaming(session_id: str, founder_text: str) -> AsyncIterator[dict]:
    """
    Async streaming entry point — use this from the WebSocket handler:

        async for event in run_turn_streaming(session_id, text):
            if event["type"] == "token":
                await websocket.send_text(json.dumps({"type": "vc_token", "text": event["text"]}))
            elif event["type"] == "final":
                await websocket.send_text(json.dumps({"type": "vc_response", **event}))
    """
    ensure_session_initialized(session_id)
    async for event in stream_vc_turn(session_id, founder_text):
        yield event


def run_turn(session_id: str, founder_text: str) -> dict:
    """
    Synchronous, non-streaming convenience wrapper (for CLI use).
    Internally drives the async streaming generator to completion and returns
    only the final result. Prefer run_turn_streaming() for WebSocket servers.
    """
    async def _collect():
        final_event = None
        async for event in run_turn_streaming(session_id, founder_text):
            if event["type"] == "final":
                final_event = event
        return final_event

    final_event = asyncio.run(_collect())
    return {
        "vc_response":    final_event.get("vc_response", ""),
        "stage":          final_event.get("stage", "intro"),
        "exchange_count": final_event.get("exchange_count", 0),
        "pitch_metrics":  final_event.get("pitch_metrics", {}),
        "is_out":         final_event.get("is_out", False),
        "pitch_ended":    final_event.get("pitch_ended", False),
    }
