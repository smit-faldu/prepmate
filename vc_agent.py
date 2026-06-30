"""
vc_agent.py — Concurrent 2-Agent LangGraph VC Pitch Evaluator (low-latency)
============================================================================
Pipeline per turn (AGENTIC, CONCURRENT — not sequential):

  HumanMessage (founder speech)
      │
      ├──────────────────────────────┬───────────────────────────────────┐
      ▼                               ▼
  [vc_persona]                   [analyst]
  STREAMS the in-character        Runs concurrently in the background.
  VC reply immediately, using     Extracts structured pitch metrics from
  pitch_metrics from the          THIS turn's transcript via Pydantic
  PREVIOUS turn (already in       structured output. Updates stage,
  state) + the raw new            red_flags, exit-condition checks.
  transcript for context.         Feeds into the *next* turn's persona
      │                            prompt — never blocks the current
      ▼                            user-facing reply.
  User hears the first
  tokens in ~1s instead of
  waiting for both calls to
  finish serially.
      │
      └──────────────────────────────┘
                    ▼
                   END  (state persisted to SQLite via SqliteSaver)

Why this design (and not a single merged call):
  A single call that forces both "extract structured facts" AND "be a punchy
  in-character VC" into one schema-constrained response measurably hurts
  persona quality — structured-output / function-calling mode biases models
  toward dutifully filling fields rather than free, voiced text, and it
  collapses two genuinely different responsibilities (analysis vs. delivery)
  into one. Running them CONCURRENTLY instead gets the latency win (the user
  is only ever waiting on ONE call — the streamed persona reply) without
  trading away analysis quality or persona quality. The analyst's output
  lags by one turn, which is imperceptible in a live spoken conversation but
  keeps every turn fully analyzed for stage advancement / exit conditions.

Latency root causes fixed from the previous version:
  1. gemini-3-flash-preview defaults to thinking_level="high" in LangChain
     when unset — full reasoning mode on every call. Now explicitly set to
     a low-latency thinking config (configurable via env).
  2. The two LLM calls were hard-sequenced (analyst -> persona) via a graph
     edge, so total latency was additive. Now they run concurrently via
     asyncio.
  3. .invoke() blocked until the full response was generated before sending
     anything to the user. Now the persona reply is streamed token-by-token.
  4. No timeout/fallback — a single slow Gemini call (preview models are
     known to have 0.6s-20s variance) stalled the whole turn with no
     recovery path. Now wrapped with timeouts + sync fallback.
"""

import asyncio
import os
import uuid
from typing import Annotated, AsyncIterator, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv(override=True)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schema (analyst node only — unchanged shape)
# ─────────────────────────────────────────────────────────────────────────────

class PitchAnalysis(BaseModel):
    """Structured extraction of the founder's latest pitch message."""

    product: str = Field(
        description="What the product/service does (1-2 sentences). Use 'not mentioned' if absent."
    )
    market_size: str = Field(
        description="Any market size claim made (e.g. '$10B TAM'). Use 'not mentioned' if absent."
    )
    team: str = Field(
        description="Founder/team background described. Use 'not mentioned' if absent."
    )
    ask: str = Field(
        description="Funding amount and equity % being offered. Use 'not mentioned' if absent."
    )
    traction: str = Field(
        description="Revenue, users, partnerships, or growth metrics mentioned. Use 'not mentioned' if absent."
    )
    clarity_score: int = Field(
        ge=1, le=10,
        description="Rate the founder's communication clarity 1-10. 1=incoherent/evasive, 10=crystal clear & confident."
    )
    clarity_reasoning: str = Field(
        description="One sentence justifying the clarity score."
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Logical flaws, contradictions, or missing critical info. Empty list [] if none found."
    )
    market_validity: str = Field(
        default="not_mentioned",
        description="One of: credible | inflated | unverified | not_mentioned"
    )
    market_assessment: str = Field(
        default="",
        description="One-sentence explanation of the market validity verdict. Empty string if market not mentioned."
    )


# ─────────────────────────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────────────────────────

def _merge_metrics(left: dict, right: dict) -> dict:
    """Custom reducer for pitch_metrics. Merges updates; accumulates unique red_flags."""
    if not right:
        return left
    merged = {**left}
    for k, v in right.items():
        if k == "red_flags" and isinstance(v, list) and isinstance(merged.get(k), list):
            seen = set(merged[k])
            merged[k] = merged[k] + [x for x in v if x not in seen]
        else:
            merged[k] = v
    return merged


class PitchState(TypedDict):
    messages: Annotated[list, add_messages]
    pitch_metrics: Annotated[dict, _merge_metrics]
    stage: str           # "intro" | "deep_dive" | "negotiation" | "decision"
    exchange_count: int
    is_out: bool
    vc_response: str
    pitch_ended: bool


# ─────────────────────────────────────────────────────────────────────────────
# LLM Setup — explicit low-latency thinking config (the #1 latency fix)
# ─────────────────────────────────────────────────────────────────────────────

_model_name = os.getenv("VC_GEMINI_MODEL", "gemini-2.5-flash")
_is_gemini3 = _model_name.startswith("gemini-3")

# Per-model thinking config. Gemini 2.5 uses thinking_budget (token count,
# 0 = disabled, fastest). Gemini 3+ uses thinking_level ("minimal"/"low" are
# the fast options). Both overridable via env without touching code, in case
# you switch models again later.
if _is_gemini3:
    _thinking_kwargs = {"thinking_level": os.getenv("VC_GEMINI_THINKING_LEVEL", "low")}
else:
    _thinking_kwargs = {"thinking_budget": int(os.getenv("VC_GEMINI_THINKING_BUDGET", "0"))}

logger.info(f"[VC Agent] Model={_model_name!r} | thinking_kwargs={_thinking_kwargs}")

# Analyst LLM: low temperature, structured output, runs in the background —
# its latency no longer sits on the user-facing critical path.
_analyst_llm = ChatGoogleGenerativeAI(
    model=_model_name,
    temperature=0.1,
    **_thinking_kwargs,
).with_structured_output(PitchAnalysis)

# VC Persona LLM: higher temperature — this is the one the user is waiting
# on, so it streams.
_vc_llm = ChatGoogleGenerativeAI(
    model=_model_name,
    temperature=0.75,
    streaming=True,
    **_thinking_kwargs,
)

# Timeouts: bound how long we'll wait before falling back, so one slow
# preview-model call (documented 0.6s-20s variance) can't stall a turn.
_PERSONA_TIMEOUT_S = float(os.getenv("VC_PERSONA_TIMEOUT_S", "8"))
_ANALYST_TIMEOUT_S = float(os.getenv("VC_ANALYST_TIMEOUT_S", "12"))


# ─────────────────────────────────────────────────────────────────────────────
# Analyst — extracts structured metrics. Called concurrently with the
# persona reply rather than as a blocking prior graph step.
# ─────────────────────────────────────────────────────────────────────────────

_ANALYST_SYSTEM = """You are a precise pitch analysis agent. Your job is to extract structured data
from the founder's latest message.

Extract ONLY what was explicitly stated. Do NOT invent or infer facts not mentioned.
Use 'not mentioned' for any field the founder did not address.
Set market_validity to 'not_mentioned' if no market size was stated."""


async def _run_analyst(founder_text: str) -> PitchAnalysis | None:
    """
    Async structured-output extraction, timeout-bounded. On timeout/error we
    return None and the caller carries forward last turn's metrics — the
    session keeps moving instead of hanging.
    """
    try:
        return await asyncio.wait_for(
            _analyst_llm.ainvoke(
                [SystemMessage(content=_ANALYST_SYSTEM), HumanMessage(content=founder_text)]
            ),
            timeout=_ANALYST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[Analyst] Timed out after {_ANALYST_TIMEOUT_S}s — carrying forward prior metrics.")
        return None
    except Exception as e:
        logger.error(f"[Analyst] Error: {e} — carrying forward prior metrics.")
        return None


def _compute_stage_and_exit(
    stage: str,
    exchange_count: int,
    clarity_score: int,
    existing_red_flags: list[str],
    new_red_flags: list[str],
) -> tuple[str, int, bool, list[str]]:
    """Pure deterministic stage-routing + exit-condition logic (no LLM call)."""
    exchange_count = exchange_count + 1
    all_red_flags = list({*existing_red_flags, *new_red_flags})

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
# VC Persona — the user-facing, streamed response
# ─────────────────────────────────────────────────────────────────────────────

_VC_BASE_PERSONA = """You are Marcus Reid — a high-profile, analytical, and seasoned AI Venture Capitalist
on the "PrepMate" platform. You manage a $400M fund focused on AI-first startups.
You are known for being brutally honest, sharp, and fair — like a Shark Tank investor.

YOUR ABSOLUTE RULES:
- Maximum 3 sentences per response. Keep latency low. Real investors are concise.
- Ask only ONE primary question per turn.
- DO NOT output TTS markup, phonetic spellings, or stage directions.
- Format for SPOKEN delivery: no bullet points, no markdown, no special characters.
- Adapt to the founder's energy: if nervous, press harder; if aggressive, be firmer.
- Evaluate not just the idea, but the FOUNDER's confidence, clarity, and defensibility.

I'M OUT PROTOCOL — trigger if ANY of these are true:
  • Founder cannot answer basic questions about their own business (Low Confidence)
  • Business model defies basic math or logic (Flawed Logic)
  • Founder becomes hostile or refuses constructive pushback (Defensiveness)
  • You have heard enough to determine this is not investable

When declaring "I'm out":
  1. Start the response with exactly: "I'm out."
  2. Give one sharp, brutal-but-educational reason (1-2 sentences)
  3. End with the tag: <END_PITCH>"""

_STAGE_INSTRUCTIONS = {
    "intro": """
STAGE: Introduction & The Pitch
Goal: Understand the core product, target audience, and the initial ask.
Action: Let the founder speak. If the pitch is vague, ask 1 clarifying question about core mechanics.
        If it's clear, push toward deeper questions.""",

    "deep_dive": """
STAGE: Deep Dive — Stress Test Mode
Goal: Poke holes. Break the business model.
Action: Pick the BIGGEST red flag from the pitch intelligence below and attack it with ONE sharp question.
        Do not accept vague answers. Press on unit economics, competitive moat, and user acquisition.""",

    "negotiation": """
STAGE: Negotiation & Valuation
Goal: Discuss equity, funding amount, and terms.
Action: Challenge their valuation head-on. Ask what the money will specifically fund.
        Offer a counter-term or probe their flexibility.""",

    "decision": """
STAGE: The Final Decision
Goal: Conclude the pitch.
Action: Based on EVERYTHING you've heard across this entire conversation, make your call.
        Either propose a final deal condition, OR declare 'I'm out' with a clear reason.
        This is the last exchange — be decisive.""",
}


def _build_persona_system_prompt(stage: str, metrics: dict, is_out_hint: bool) -> str:
    intel_lines = []
    if metrics.get("product"):           intel_lines.append(f"  Product:          {metrics['product']}")
    if metrics.get("market_size"):       intel_lines.append(f"  Market Claim:     {metrics['market_size']}")
    if metrics.get("market_validity") and metrics["market_validity"] != "not_mentioned":
        intel_lines.append(f"  Market Validity:  {metrics['market_validity']} — {metrics.get('market_assessment', '')}")
    if metrics.get("team"):              intel_lines.append(f"  Team:             {metrics['team']}")
    if metrics.get("ask"):               intel_lines.append(f"  Ask:              {metrics['ask']}")
    if metrics.get("traction"):          intel_lines.append(f"  Traction:         {metrics['traction']}")
    if metrics.get("clarity_score"):     intel_lines.append(f"  Clarity Score:    {metrics['clarity_score']}/10 — {metrics.get('clarity_reasoning', '')}")
    if metrics.get("red_flags"):         intel_lines.append(f"  Red Flags Found:  {'; '.join(metrics['red_flags'])}")

    intel_block = ""
    if intel_lines:
        intel_block = "\n\nPITCH INTELLIGENCE (use this to inform your response):\n" + "\n".join(intel_lines)

    exit_block = ""
    if is_out_hint:
        exit_block = (
            "\n\nEXIT SIGNAL: The analytical system has flagged serious issues with this pitch. "
            "Strongly consider declaring 'I'm out.' if what you've heard supports this."
        )

    return f"{_VC_BASE_PERSONA}{_STAGE_INSTRUCTIONS.get(stage, _STAGE_INSTRUCTIONS['intro'])}{intel_block}{exit_block}"


def _extract_text(content) -> str:
    """Normalize Gemini content (str or list-of-parts) into plain text."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content) if content else ""


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
    graph = get_vc_graph()
    config = {"configurable": {"thread_id": session_id}}

    snapshot = graph.get_state(config)
    state = snapshot.values if snapshot and snapshot.values else {}

    stage = state.get("stage", "intro")
    metrics = state.get("pitch_metrics", {})
    is_out_hint = state.get("is_out", False)
    exchange_count = state.get("exchange_count", 0)
    history = state.get("messages", [])

    system_prompt = _build_persona_system_prompt(stage, metrics, is_out_hint)
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
                async for chunk in _vc_llm.astream(persona_messages):
                    piece = _extract_text(chunk.content)
                    if piece:
                        await token_queue.put(piece)
            finally:
                await token_queue.put(None)  # sentinel: stream done (success or error)

        pump_task = asyncio.create_task(_pump_stream())
        while True:
            piece = await asyncio.wait_for(token_queue.get(), timeout=_PERSONA_TIMEOUT_S)
            if piece is None:
                break
            final_content += piece
            yield {"type": "token", "text": piece}
        await pump_task  # surface any exception raised inside the pump
    except asyncio.TimeoutError:
        logger.warning(f"[Persona] Streaming timed out after {_PERSONA_TIMEOUT_S}s — falling back to sync call.")
        pump_task.cancel()
        try:
            response = await asyncio.wait_for(_vc_llm.ainvoke(persona_messages), timeout=_PERSONA_TIMEOUT_S)
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
    resp_lower = final_content.lower()
    declared_out = "i'm out" in resp_lower or "im out" in resp_lower or "i am out" in resp_lower
    pitch_ended = declared_out or (stage == "decision")
    if declared_out and "<END_PITCH>" not in final_content:
        final_content = final_content.rstrip() + "\n\n<END_PITCH>"

    # ── Await analyst result (it's been running this whole time) ─────────────
    analysis = await analyst_task
    if analysis is not None:
        metrics_update = analysis.model_dump()
        clarity_score = metrics_update.get("clarity_score", 5)
        new_red_flags = metrics_update.get("red_flags", [])
    else:
        metrics_update = {}
        clarity_score = metrics.get("clarity_score", 5)
        new_red_flags = []

    new_stage, new_exchange_count, is_out, all_red_flags = _compute_stage_and_exit(
        stage=stage,
        exchange_count=exchange_count,
        clarity_score=clarity_score,
        existing_red_flags=metrics.get("red_flags", []),
        new_red_flags=new_red_flags,
    )
    is_out = is_out or declared_out
    pitch_ended = pitch_ended or is_out

    if metrics_update:
        metrics_update["red_flags"] = all_red_flags

    merged_metrics = _merge_metrics(metrics, metrics_update)

    # ── Persist state for next turn via the graph's checkpointer ─────────────
    graph.update_state(
        config,
        {
            "messages": [HumanMessage(content=founder_text), AIMessage(content=final_content)],
            "pitch_metrics": metrics_update,
            "stage": new_stage,
            "exchange_count": new_exchange_count,
            "is_out": is_out,
            "vc_response": final_content,
            "pitch_ended": pitch_ended,
        },
    )

    yield {
        "type": "final",
        "vc_response": final_content,
        "stage": new_stage,
        "exchange_count": new_exchange_count,
        "pitch_metrics": merged_metrics,
        "is_out": is_out,
        "pitch_ended": pitch_ended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly — kept solely as the SQLite-backed state store
# (get_state / update_state). stream_vc_turn() above drives execution
# directly so we control concurrency and streaming; the compiled graph's
# checkpointer is reused purely for state persistence across turns.
# ─────────────────────────────────────────────────────────────────────────────

def _noop_node(state: PitchState) -> dict:
    return {}


def build_vc_graph(db_path: str = "pitch_sessions.db"):
    import sqlite3

    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    graph = StateGraph(PitchState)
    graph.add_node("noop", _noop_node)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)

    return graph.compile(checkpointer=memory)


_vc_graph = None
_initialized_sessions: set[str] = set()


def get_vc_graph():
    global _vc_graph
    if _vc_graph is None:
        _vc_graph = build_vc_graph()
    return _vc_graph


def new_session() -> str:
    return str(uuid.uuid4())


def _ensure_session_initialized(session_id: str) -> None:
    """Seed default state for a brand-new session (idempotent)."""
    if session_id in _initialized_sessions:
        return
    graph = get_vc_graph()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        graph.update_state(
            config,
            {
                "messages": [],
                "pitch_metrics": {},
                "stage": "intro",
                "exchange_count": 0,
                "is_out": False,
                "vc_response": "",
                "pitch_ended": False,
            },
        )
    _initialized_sessions.add(session_id)


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
    _ensure_session_initialized(session_id)
    async for event in stream_vc_turn(session_id, founder_text):
        yield event


def run_turn(session_id: str, founder_text: str) -> dict:
    """
    Synchronous, non-streaming convenience wrapper (kept for backward
    compatibility / CLI use, e.g. cli_whisper_vad.py). Internally drives the
    async streaming generator to completion and returns only the final
    result. Prefer run_turn_streaming() for the WebSocket server — that's
    where the latency win comes from.
    """
    async def _collect():
        final_event = None
        async for event in run_turn_streaming(session_id, founder_text):
            if event["type"] == "final":
                final_event = event
        return final_event

    final_event = asyncio.run(_collect())
    return {
        "vc_response": final_event.get("vc_response", ""),
        "stage": final_event.get("stage", "intro"),
        "exchange_count": final_event.get("exchange_count", 0),
        "pitch_metrics": final_event.get("pitch_metrics", {}),
        "is_out": final_event.get("is_out", False),
        "pitch_ended": final_event.get("pitch_ended", False),
    }