"""
app/vision/expression.py — Facial expression analysis from MediaPipe blendshapes.

Two strategies:
  1. _analyze_expression_from_blendshapes()  — primary, uses 52 ARKit blendshapes
  2. _analyze_expression_from_landmarks_fallback() — geometry-based fallback
"""

from typing import Optional

import numpy as np

from app.vision.smoothers import EMASmoother


# ─────────────────────────────────────────────────────────────────────────────
# Blendshape name registry (52 ARKit blendshapes, indices 0–51)
# Source: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
# ─────────────────────────────────────────────────────────────────────────────

BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "_neutral",
]

# Pre-compute index map for O(1) lookup
_IDX: dict[str, int] = {
    name: (BLENDSHAPE_NAMES.index(name) if name in BLENDSHAPE_NAMES else -1)
    for name in BLENDSHAPE_NAMES
}


def _analyze_expression_from_blendshapes(
    blendshapes_list,
    smoother: Optional[EMASmoother],
) -> tuple[str, float, dict]:
    """
    Derive dominant expression from FaceLandmarker blendshape output (Tasks API).
    Uses the 52 ARKit blendshape coefficients — far more accurate than landmarks.

    blendshapes_list: result.face_blendshapes (list of Category lists)
    smoother: per-session EMA smoother for temporal stability

    Returns (expression_label, confidence, score_dict).
    """
    if not blendshapes_list or len(blendshapes_list) == 0:
        return "unknown", 0.0, {}

    cats = blendshapes_list[0]  # first face
    raw  = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)
    for cat in cats:
        name = cat.category_name
        if name in _IDX and _IDX[name] >= 0:
            raw[_IDX[name]] = cat.score

    smooth = smoother.update(raw) if smoother is not None else raw

    def bs(name: str) -> float:
        idx = _IDX.get(name, -1)
        return float(smooth[idx]) if idx >= 0 else 0.0

    scores: dict[str, float] = {}

    # Smiling (AU6+12): cheek raise + lip corners up
    smile = (bs("mouthSmileLeft") + bs("mouthSmileRight")) / 2.0
    cheek = (bs("cheekSquintLeft") + bs("cheekSquintRight")) / 2.0
    scores["smiling"] = min(1.0, (smile * 0.7 + cheek * 0.3) * 1.6)

    # Excited: smiling + wide eyes + brow raise
    wide_eye = (bs("eyeWideLeft") + bs("eyeWideRight")) / 2.0
    brow_up  = bs("browInnerUp")
    scores["excited"] = min(1.0, (scores["smiling"] * 0.5 + wide_eye * 0.3 + brow_up * 0.2) * 1.5)

    # Surprised (AU1+2+5B+27): jaw drop + wide eyes + brow raise
    jaw_open = bs("jawOpen")
    scores["surprised"] = min(1.0, (jaw_open * 0.5 + wide_eye * 0.3 + brow_up * 0.2) * 1.8)

    # Nervous / Stressed (AU4+7+17): brow lower + eye squint + lip stretch
    brow_down = (bs("browDownLeft") + bs("browDownRight")) / 2.0
    eye_sqnt  = (bs("eyeSquintLeft") + bs("eyeSquintRight")) / 2.0
    lip_str   = (bs("mouthStretchLeft") + bs("mouthStretchRight")) / 2.0
    scores["nervous"] = min(1.0, (brow_down * 0.4 + eye_sqnt * 0.3 + lip_str * 0.3) * 2.0)

    # Thinking / Concentrating (AU1+AU64): inner brow up + pucker
    pucker = bs("mouthPucker")
    scores["thinking"] = min(1.0, (brow_up * 0.5 + pucker * 0.3 + eye_sqnt * 0.2) * 2.0)

    # Sad (AU1+15+17): inner brow up + lip corners down + lip roll
    frown = (bs("mouthFrownLeft") + bs("mouthFrownRight")) / 2.0
    shrug = (bs("mouthShrugLower") + bs("mouthShrugUpper")) / 2.0
    scores["sad"] = min(1.0, (brow_up * 0.3 + frown * 0.5 + shrug * 0.2) * 2.0)

    # Angry / Frustrated (AU4+9+23+24): brow lower + nose sneer + lip press
    nose_sneer = (bs("noseSneerLeft") + bs("noseSneerRight")) / 2.0
    lip_press  = (bs("mouthPressLeft") + bs("mouthPressRight")) / 2.0
    scores["angry"] = min(1.0, (brow_down * 0.4 + nose_sneer * 0.4 + lip_press * 0.2) * 2.5)

    # Neutral: confidence in "no expression"
    total_signal   = sum(scores.values())
    scores["neutral"] = max(0.0, 1.0 - total_signal * 0.5)

    dominant     = max(scores, key=scores.get)
    max_bs_score = float(np.max(smooth[:51]))  # ignore _neutral at index 51
    presence     = min(1.0, max_bs_score * 3.0 + 0.3)
    confidence   = round(min(1.0, presence), 3)

    return dominant, confidence, {k: round(v, 3) for k, v in scores.items()}


def _analyze_expression_from_landmarks_fallback(
    face_landmarks_list,
) -> tuple[str, float, dict]:
    """
    Fallback geometry-based expression analysis when blendshapes are unavailable.
    Less accurate but works without blendshapes output enabled.
    """
    if not face_landmarks_list:
        return "unknown", 0.0, {}

    lms = face_landmarks_list[0]
    n   = len(lms)

    def lm(idx):
        return lms[idx] if idx < n else None

    # Landmark indices (face_mesh numbering)
    _MOUTH_LEFT      = 61
    _MOUTH_RIGHT     = 291
    _UPPER_LIP       = 13
    _LOWER_LIP       = 14
    _LEFT_BROW_INNER = 107
    _RIGHT_BROW_INNER= 336
    _NOSE_TIP        = 1
    _LEFT_EYE_TOP    = 159
    _LEFT_EYE_BOTTOM = 145
    _RIGHT_EYE_TOP   = 386
    _RIGHT_EYE_BOTTOM= 374

    scores: dict[str, float] = {}

    nose = lm(_NOSE_TIP)
    ml   = lm(_MOUTH_LEFT)
    mr   = lm(_MOUTH_RIGHT)
    if nose and ml and mr:
        smile_score = ((nose.y - ml.y) + (nose.y - mr.y)) / 2.0
        scores["smiling"] = max(0.0, smile_score * 8.0)

    ul = lm(_UPPER_LIP)
    ll = lm(_LOWER_LIP)
    if ul and ll and ml and mr:
        mouth_h = abs(ll.y - ul.y)
        mouth_w = abs(mr.x - ml.x) + 1e-6
        scores["surprised"] = min(1.0, (mouth_h / mouth_w) * 4.0)

    lb  = lm(_LEFT_BROW_INNER)
    rb  = lm(_RIGHT_BROW_INNER)
    let = lm(_LEFT_EYE_TOP)
    ret = lm(_RIGHT_EYE_TOP)
    if lb and rb and let and ret:
        brow_gap = ((let.y - lb.y) + (ret.y - rb.y)) / 2.0
        scores["nervous"] = min(1.0, max(0.0, 0.06 - brow_gap) * 12.0)

    leb = lm(_LEFT_EYE_BOTTOM)
    reb = lm(_RIGHT_EYE_BOTTOM)
    if let and leb and ret and reb:
        ear = ((abs(let.y - leb.y)) + (abs(ret.y - reb.y))) / 2.0
        scores["thinking"] = min(1.0, max(0.0, 0.025 - ear) * 30.0)

    total_signal   = sum(scores.values())
    scores["neutral"] = max(0.0, 1.0 - total_signal)

    if scores.get("smiling", 0) > 0.3 and scores.get("surprised", 0) > 0.3:
        scores["excited"] = (scores["smiling"] + scores["surprised"]) / 2.0

    dominant = max(scores, key=scores.get)

    visibilities = [
        lm_pt.visibility
        for lm_pt in lms
        if hasattr(lm_pt, "visibility") and lm_pt.visibility is not None
    ]
    mean_vis = float(np.mean(visibilities)) if visibilities else 0.6

    return dominant, round(mean_vis, 3), {k: round(v, 3) for k, v in scores.items()}
