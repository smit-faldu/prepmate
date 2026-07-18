"""
app/vision/context.py — Vision context block builder for VC prompt injection.

build_vision_context_block() formats the latest vision state into a body
language intelligence block for insertion into the VC turn prompt.
"""

from typing import Optional


_EXPRESSION_LABELS = {
    "smiling":   "appears calm and confident (smiling)",
    "excited":   "appears highly energized and excited",
    "neutral":   "appears composed and neutral",
    "nervous":   "appears visibly nervous or stressed",
    "thinking":  "appears to be thinking or hesitating",
    "surprised": "appears surprised or caught off guard",
    "sad":       "appears downbeat or low-confidence",
    "angry":     "appears frustrated or tense",
    "unknown":   "expression could not be determined",
}

_POSE_LABELS = {
    "upright":         "sitting/standing upright — confident posture",
    "leaning_forward": "leaning forward — engaged or eager",
    "slouched":        "slouching — low energy or disengaged",
    "gesturing":       "using hand gestures — expressive",
    "tilted":          "head/body tilted — casual or uncertain",
    "unknown":         "pose could not be determined",
}


def build_vision_context_block(vision_data: Optional[dict]) -> str:
    """
    Formats the latest vision state into a body language intelligence block
    for injection into the VC turn. Returns empty string if no data or
    confidence is too low to be meaningful.

    Intentionally framed as SECONDARY context — audio is always primary.
    """
    if not vision_data:
        return ""

    expr      = vision_data.get("expression", "unknown")
    expr_conf = vision_data.get("expression_confidence", 0.0)
    pose      = vision_data.get("pose", "unknown")
    pose_conf = vision_data.get("pose_confidence", 0.0)

    # Only inject if at least one signal is confident enough
    if expr_conf < 0.3 and pose_conf < 0.3:
        return ""

    expr_desc = _EXPRESSION_LABELS.get(expr, expr)
    pose_desc = _POSE_LABELS.get(pose, pose)

    lines = ["\n\nBODY LANGUAGE INTELLIGENCE (secondary signal — audio takes priority):"]
    lines.append(f"  • Facial Expression: {expr_desc} (confidence: {expr_conf:.0%})")
    lines.append(f"  • Body Pose:         {pose_desc} (confidence: {pose_conf:.0%})")
    lines.append(
        "  ⚠ Use body language as soft context only. "
        "What the founder says (audio) is the primary basis for your evaluation. "
        "If body language contradicts speech, you MAY briefly note the discrepancy."
    )

    return "\n".join(lines)
