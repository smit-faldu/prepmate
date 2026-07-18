"""
app/vc/schemas.py — Pydantic + TypedDict schemas for the VC agent pipeline.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schema (analyst node only)
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
        description="Rate the founder's communication clarity 1-10. 1=incoherent/evasive, 10=crystal clear & confident.",
    )
    clarity_reasoning: str = Field(
        description="One sentence justifying the clarity score."
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Logical flaws, contradictions, or missing critical info. Empty list [] if none found.",
    )
    market_validity: str = Field(
        default="not_mentioned",
        description="One of: credible | inflated | unverified | not_mentioned",
    )
    market_assessment: str = Field(
        default="",
        description="One-sentence explanation of the market validity verdict. Empty string if market not mentioned.",
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
