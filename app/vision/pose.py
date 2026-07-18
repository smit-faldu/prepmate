"""
app/vision/pose.py — Body pose analysis from MediaPipe PoseLandmarker results.
"""

import numpy as np


# Pose landmark indices (MediaPipe Pose 33-point model)
_NOSE_POSE      = 0
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16
_LEFT_HIP       = 23
_RIGHT_HIP      = 24


def analyze_pose_from_landmarks(pose_landmarks_list) -> tuple[str, float]:
    """
    Classify body pose from PoseLandmarker result (Tasks API).
    pose_landmarks_list is result.pose_landmarks (list of NormalizedLandmark lists).
    Returns (pose_label, confidence).
    """
    if not pose_landmarks_list:
        return "unknown", 0.0

    lms = pose_landmarks_list[0]   # first person
    n   = len(lms)

    def lm(idx):
        return lms[idx] if idx < n else None

    ls   = lm(_LEFT_SHOULDER)
    rs   = lm(_RIGHT_SHOULDER)
    lh   = lm(_LEFT_HIP)
    rh   = lm(_RIGHT_HIP)
    lw   = lm(_LEFT_WRIST)
    rw   = lm(_RIGHT_WRIST)
    nose = lm(_NOSE_POSE)

    visibilities = [
        pt.visibility for pt in lms
        if hasattr(pt, "visibility") and pt.visibility is not None
    ]
    mean_vis = float(np.mean(visibilities)) if visibilities else 0.0

    if ls is None or rs is None:
        return "unknown", round(mean_vis, 3)

    # Only trust results with decent visibility
    shoulder_vis = 0.0
    if hasattr(ls, "visibility") and hasattr(rs, "visibility"):
        shoulder_vis = (ls.visibility + rs.visibility) / 2.0
    if shoulder_vis < 0.3:
        return "unknown", round(mean_vis, 3)

    shoulder_mid_y = (ls.y + rs.y) / 2.0
    shoulder_level = abs(ls.y - rs.y)

    # Gesturing: wrist raised above shoulder line
    wrists_raised = 0
    if lw and getattr(lw, "visibility", 0) > 0.3 and lw.y < shoulder_mid_y:
        wrists_raised += 1
    if rw and getattr(rw, "visibility", 0) > 0.3 and rw.y < shoulder_mid_y:
        wrists_raised += 1
    if wrists_raised >= 1:
        return "gesturing", round(mean_vis, 3)

    # Slouched: head appears too close to shoulders
    if nose and lh and rh:
        hip_mid_y = (lh.y + rh.y) / 2.0
        torso_h   = hip_mid_y - shoulder_mid_y
        head_h    = shoulder_mid_y - nose.y
        if torso_h > 0 and head_h < torso_h * 0.3:
            return "slouched", round(mean_vis, 3)

    # Leaning forward: shoulders z-depth negative (closer to camera)
    if hasattr(ls, "z") and hasattr(rs, "z"):
        avg_z = (ls.z + rs.z) / 2.0
        if avg_z < -0.12:
            return "leaning_forward", round(mean_vis, 3)

    # Tilted: shoulders asymmetric in Y
    if shoulder_level > 0.05:
        return "tilted", round(mean_vis, 3)

    return "upright", round(mean_vis, 3)
