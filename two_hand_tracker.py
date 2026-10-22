"""
two_hand_tracker.py
Independent Two-Handed Gesture Controller for HandSynth.

Features:
1. LEFT HAND (Harmonic Control & Quality):
   - Diatonic Chord Selection via finger counts (1-5) and custom rock poses:
     * 1 finger (Index): Chord 1 (I)
     * 2 fingers (Index+Middle): Chord 2 (ii)
     * 3 fingers: Chord 3 (iii)
     * 4 fingers: Chord 4 (IV)
     * 5 fingers (Fully Open Palm): Chord 5 (V) - robust tilt-invariant detection!
     * 6 gesture (🤘 Rock sign: Index + Pinky): Chord 6 (vi)
     * 7 gesture (🤟 Rock+Thumb sign: Index + Pinky + Thumb): Chord 7 (vii°)
     * Fist: Instant Mute
   - Left-Hand Tilt mapped to Chord Quality:
     * Tilted LEFT  (< -0.15 rad / ~-9°): Forces MINOR quality (e.g. C min, D min)
     * Tilted RIGHT (> +0.15 rad / ~+9°): Forces MAJOR quality (e.g. C Maj, D Maj)
     * Upright (-0.15 to +0.15 rad): Standard Diatonic triad quality
   - Left-Hand Tilt Magnitude mapped to Chord Gain ([0.10, 1.0]).
   - ZERO pinch-to-play required on left hand.

2. RIGHT HAND (Chord Variations, Transposition & Dynamics):
   - Finger Pointing distinguishes Chord Variations / Extensions:
     * Index Pointed (or 1 finger)   -> 7th Chord (Maj7, Dom7, min7)
     * Middle Pointed (or 2 fingers) -> Diminished Chord (dim / dim7)
     * Ring Pointed (or 3 fingers)   -> Suspended Chord (sus4)
     * Pinky Pointed                 -> Augmented Chord (aug)
     * Open Hand / Thumb (4-5 fgrs)  -> 9th / Add9 Chord (Maj9, min9, add9)
     * Closed / Relaxed (0 fingers)  -> Standard Triad
   - Right-Hand Lateral Tilt mapped to continuous Pitch Transposition (-12 to +12 semitones).
   - Right-Hand Vertical Elevation mapped to Volume (0.0 to 1.0, up = loud, down = soft).
   - Standalone Playback: If right hand is absent or resting, left hand plays at comfortable volume.

3. DUAL-VCA AMPLITUDE & SAFETY:
   - Final Amplitude = LeftGain * RightVolume (or LeftGain * 0.75 when right hand absent).
   - Fist gesture immediately mutes audio.
   - Hand loss grace period (500ms timeout before safe release, zero stuck notes).
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any

# Scale-aware chord definitions for all supported musical scales (Degrees 1 to 7)
SCALE_CHORD_DEFINITIONS = {
    "MAJOR": {
        1: {"degree": 1, "roman": "I",    "quality": "Major",      "semitones": 0,  "intervals": [0, 4, 7, 12]},
        2: {"degree": 2, "roman": "ii",   "quality": "Minor",      "semitones": 2,  "intervals": [0, 3, 7, 12]},
        3: {"degree": 3, "roman": "iii",  "quality": "Minor",      "semitones": 4,  "intervals": [0, 3, 7, 12]},
        4: {"degree": 4, "roman": "IV",   "quality": "Major",      "semitones": 5,  "intervals": [0, 4, 7, 12]},
        5: {"degree": 5, "roman": "V",    "quality": "Major",      "semitones": 7,  "intervals": [0, 4, 7, 12]},
        6: {"degree": 6, "roman": "vi",   "quality": "Minor",      "semitones": 9,  "intervals": [0, 3, 7, 12]},
        7: {"degree": 7, "roman": "vii°", "quality": "Diminished", "semitones": 11, "intervals": [0, 3, 6, 12]},
    },
    "MINOR": {
        1: {"degree": 1, "roman": "i",    "quality": "Minor",      "semitones": 0,  "intervals": [0, 3, 7, 12]},
        2: {"degree": 2, "roman": "ii°",  "quality": "Diminished", "semitones": 2,  "intervals": [0, 3, 6, 12]},
        3: {"degree": 3, "roman": "III",  "quality": "Major",      "semitones": 3,  "intervals": [0, 4, 7, 12]},
        4: {"degree": 4, "roman": "iv",   "quality": "Minor",      "semitones": 5,  "intervals": [0, 3, 7, 12]},
        5: {"degree": 5, "roman": "v",    "quality": "Minor",      "semitones": 7,  "intervals": [0, 3, 7, 12]},
        6: {"degree": 6, "roman": "VI",   "quality": "Major",      "semitones": 8,  "intervals": [0, 4, 7, 12]},
        7: {"degree": 7, "roman": "VII",  "quality": "Major",      "semitones": 10, "intervals": [0, 4, 7, 12]},
    },
    "DORIAN": {
        1: {"degree": 1, "roman": "i",    "quality": "Minor",      "semitones": 0,  "intervals": [0, 3, 7, 12]},
        2: {"degree": 2, "roman": "ii",   "quality": "Minor",      "semitones": 2,  "intervals": [0, 3, 7, 12]},
        3: {"degree": 3, "roman": "III",  "quality": "Major",      "semitones": 3,  "intervals": [0, 4, 7, 12]},
        4: {"degree": 4, "roman": "IV",   "quality": "Major",      "semitones": 5,  "intervals": [0, 4, 7, 12]},
        5: {"degree": 5, "roman": "v",    "quality": "Minor",      "semitones": 7,  "intervals": [0, 3, 7, 12]},
        6: {"degree": 6, "roman": "vi°",  "quality": "Diminished", "semitones": 9,  "intervals": [0, 3, 6, 12]},
        7: {"degree": 7, "roman": "VII",  "quality": "Major",      "semitones": 10, "intervals": [0, 4, 7, 12]},
    },
    "BLUES": {
        1: {"degree": 1, "roman": "I7",   "quality": "Major",      "semitones": 0,  "intervals": [0, 4, 7, 10]},
        2: {"degree": 2, "roman": "bIII", "quality": "Major",      "semitones": 3,  "intervals": [0, 4, 7, 12]},
        3: {"degree": 3, "roman": "IV7",  "quality": "Major",      "semitones": 5,  "intervals": [0, 4, 7, 10]},
        4: {"degree": 4, "roman": "bV°",  "quality": "Diminished", "semitones": 6,  "intervals": [0, 3, 6, 9]},
        5: {"degree": 5, "roman": "V7",   "quality": "Major",      "semitones": 7,  "intervals": [0, 4, 7, 10]},
        6: {"degree": 6, "roman": "bVII", "quality": "Major",      "semitones": 10, "intervals": [0, 4, 7, 12]},
        7: {"degree": 7, "roman": "i",    "quality": "Minor",      "semitones": 0,  "intervals": [0, 3, 7, 10]},
    },
    "PENTATONIC": {
        1: {"degree": 1, "roman": "I",    "quality": "Major",      "semitones": 0,  "intervals": [0, 4, 7, 12]},
        2: {"degree": 2, "roman": "ii",   "quality": "Minor",      "semitones": 2,  "intervals": [0, 3, 7, 12]},
        3: {"degree": 3, "roman": "iii",  "quality": "Minor",      "semitones": 4,  "intervals": [0, 3, 7, 12]},
        4: {"degree": 4, "roman": "IV",   "quality": "Major",      "semitones": 5,  "intervals": [0, 4, 7, 12]},
        5: {"degree": 5, "roman": "V",    "quality": "Major",      "semitones": 7,  "intervals": [0, 4, 7, 12]},
        6: {"degree": 6, "roman": "vi",   "quality": "Minor",      "semitones": 9,  "intervals": [0, 3, 7, 12]},
        7: {"degree": 7, "roman": "Iadd9","quality": "Major",      "semitones": 0,  "intervals": [0, 2, 7, 12]},
    },
    "HIRAJOSHI": {
        1: {"degree": 1, "roman": "i",    "quality": "Minor",      "semitones": 0,  "intervals": [0, 3, 7, 12]},
        2: {"degree": 2, "roman": "ii°",  "quality": "Diminished", "semitones": 2,  "intervals": [0, 3, 6, 12]},
        3: {"degree": 3, "roman": "III",  "quality": "Major",      "semitones": 3,  "intervals": [0, 4, 7, 12]},
        4: {"degree": 4, "roman": "v",    "quality": "Minor",      "semitones": 7,  "intervals": [0, 3, 7, 12]},
        5: {"degree": 5, "roman": "VI",   "quality": "Major",      "semitones": 8,  "intervals": [0, 4, 7, 12]},
        6: {"degree": 6, "roman": "i7",   "quality": "Minor",      "semitones": 0,  "intervals": [0, 3, 7, 10]},
        7: {"degree": 7, "roman": "v7",   "quality": "Minor",      "semitones": 7,  "intervals": [0, 3, 7, 10]},
    },
}

DIATONIC_CHORD_DEFINITIONS = SCALE_CHORD_DEFINITIONS["MAJOR"]
MAJOR_DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

ROOT_KEYS_FREQS = {
    "C":  261.63,
    "C#": 277.18,
    "D":  293.66,
    "D#": 311.13,
    "E":  329.63,
    "F":  349.23,
    "F#": 369.99,
    "G":  392.00,
    "G#": 415.30,
    "A":  440.00,
    "A#": 466.16,
    "B":  493.88,
}

KEY_SEMITONES = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}

SHARP_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NOTE_NAMES  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def get_chord_root_note_name(
    key_name: str,
    degree: int,
    base_octave: Optional[int] = None,
    semitone_offset: Optional[int] = None,
) -> str:
    """
    Computes the exact root note name (e.g. 'C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb')
    for a given musical key, scale degree, and optional base octave.
    """
    clean_key = key_name.upper().replace("♯", "#").replace("♭", "b")
    tonic_st = KEY_SEMITONES.get(clean_key, 0)
    if semitone_offset is not None:
        deg_st = int(semitone_offset)
    else:
        deg_st = MAJOR_DEGREE_SEMITONES.get(degree, 0)

    total_st = tonic_st + deg_st
    semitone_mod = total_st % 12

    # Keys/scales that traditionally use flats: F, Bb, Eb, Ab, Db, or Minor scale with flat degrees
    if clean_key in ("F", "BB", "EB", "AB", "DB") or "b" in clean_key or deg_st in (1, 3, 6, 8, 10):
        note_name = FLAT_NOTE_NAMES[semitone_mod]
    else:
        note_name = SHARP_NOTE_NAMES[semitone_mod]

    if base_octave is not None:
        octave_num = base_octave + (total_st // 12)
        return f"{note_name}{octave_num}"
    return note_name


def is_finger_raised(landmarks: List[Dict[str, float]], pip_idx: int, tip_idx: int) -> bool:
    """
    Tilt-invariant check if a finger is extended relative to the palm orientation axis.
    Works consistently regardless of hand tilt angle (tilted left, right, or vertical).
    """
    if not landmarks or len(landmarks) < 21:
        return False
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    # Palm orientation vector (wrist -> middle MCP)
    hx = middle_mcp["x"] - wrist["x"]
    hy = middle_mcp["y"] - wrist["y"]
    hlen = math.hypot(hx, hy)
    if hlen < 1e-5:
        return False
    ux, uy = hx / hlen, hy / hlen

    # Vector from PIP to Tip projected along palm's orientation axis
    fx = tip["x"] - pip["x"]
    fy = tip["y"] - pip["y"]
    proj = fx * ux + fy * uy

    # Distance from wrist to tip vs wrist to pip
    d_tip = math.hypot(tip["x"] - wrist["x"], tip["y"] - wrist["y"])
    d_pip = math.hypot(pip["x"] - wrist["x"], pip["y"] - wrist["y"])

    # Extended along palm axis and further from wrist than PIP
    is_projected = (proj > 0.015) and (d_tip > d_pip * 1.10)

    # Secondary check: Tip vertically higher when mostly upright
    is_vertical_up = (tip["y"] < pip["y"]) and (d_tip > d_pip * 1.08)

    return is_projected or is_vertical_up


def is_thumb_extended(landmarks: List[Dict[str, float]], handedness: str = "Left") -> bool:
    """
    Tilt-invariant check if thumb is truly extended away from palm.
    Decomposes the thumb into rotation-invariant palm coordinates and anatomical distance ratios.
    Prevents false activations during closed fists / palms, and ensures reliable
    detection when thumb is stretched outward or upward across all hand angles.
    """
    if not landmarks or len(landmarks) < 21:
        return False

    wrist = landmarks[0]
    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    def d(p1, p2):
        return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])

    palm_len = d(wrist, middle_mcp)
    if palm_len < 1e-4:
        return False

    # Unit vector along palm length (Wrist -> Middle MCP)
    ux = (middle_mcp["x"] - wrist["x"]) / palm_len
    uy = (middle_mcp["y"] - wrist["y"]) / palm_len

    # Vector from Pinky MCP to Thumb MCP
    px = thumb_mcp["x"] - pinky_mcp["x"]
    py = thumb_mcp["y"] - pinky_mcp["y"]
    dot_p_u = px * ux + py * uy

    # Normal vector perpendicular to palm length, pointing strictly toward thumb side
    vx = px - dot_p_u * ux
    vy = py - dot_p_u * uy
    v_len = math.hypot(vx, vy)
    if v_len < 1e-4:
        return False
    vx /= v_len
    vy /= v_len

    # Thumb tip vector relative to thumb MCP
    dx = thumb_tip["x"] - thumb_mcp["x"]
    dy = thumb_tip["y"] - thumb_mcp["y"]

    ext_up = (dx * ux + dy * uy) / palm_len
    ext_out = (dx * vx + dy * vy) / palm_len

    # Invariant Euclidean distance ratios:
    # 1. Pinky ratio: distance(thumb_tip, pinky_mcp) / distance(thumb_mcp, pinky_mcp)
    # When curled into palm/fist, thumb tip moves toward pinky knuckle (ratio < 1.05).
    # When extended outward or upward, ratio is significantly higher (ratio > 1.15).
    d_tip_pinky = d(thumb_tip, pinky_mcp)
    d_mcp_pinky = d(thumb_mcp, pinky_mcp) or 1.0
    pinky_ratio = d_tip_pinky / d_mcp_pinky

    d_tip_idx = d(thumb_tip, index_mcp) / palm_len
    d_tip_wrist = d(thumb_tip, wrist)
    d_mcp_wrist = d(thumb_mcp, wrist) or 1.0
    wrist_ratio = d_tip_wrist / d_mcp_wrist

    # Extended outward (Abduction, e.g. Open Palm Chord 5, Rock+Thumb Chord 7):
    # Tip moves outward along thumb normal and away from pinky MCP
    is_outward = (ext_out > 0.25) and (pinky_ratio > 1.12) and (d_tip_idx > 0.18)

    # Extended upward (Elevation, e.g. Thumbs Up, Right Hand Octave Up):
    # Tip extends upward along palm axis, elevated above wrist and away from index MCP
    is_upward = (ext_up > 0.35) and (wrist_ratio > 1.15) and (pinky_ratio > 1.04) and (d_tip_idx > 0.20)

    return is_outward or is_upward


def classify_left_hand_chord(landmarks: List[Dict[str, float]], handedness: str = "Left") -> Dict[str, Any]:
    """
    Classifies Left Hand raised fingers into Diatonic Chords 1-7.
    Robust against hand tilt and immediately recognizes Chord 5 (open palm).
    """
    if not landmarks or len(landmarks) < 21:
        return {
            "chord_index": None,
            "chord_name": "--",
            "gesture_name": "No Hand",
            "is_fist": False,
        }

    thumb = is_thumb_extended(landmarks, handedness)
    index = is_finger_raised(landmarks, 6, 8)
    middle = is_finger_raised(landmarks, 10, 12)
    ring = is_finger_raised(landmarks, 14, 16)
    pinky = is_finger_raised(landmarks, 18, 20)

    # 1. Custom Rock Gestures:
    # Index & Pinky raised, Middle & Ring folded
    if index and pinky and not middle and not ring:
        if thumb:
            return {
                "chord_index": 7,
                "chord_name": "vii°",
                "gesture_name": "Rock+Thumb 🤟 (vii°)",
                "is_fist": False,
            }
        else:
            return {
                "chord_index": 6,
                "chord_name": "vi",
                "gesture_name": "Rock 🤘 (vi)",
                "is_fist": False,
            }

    main_count = sum([index, middle, ring, pinky])

    # 2. Main 4 fingers raised:
    if main_count == 4:
        if thumb:
            return {
                "chord_index": 5,
                "chord_name": "V",
                "gesture_name": "5 Fingers (Open Palm)",
                "is_fist": False,
            }
        else:
            return {
                "chord_index": 4,
                "chord_name": "IV",
                "gesture_name": "4 Fingers (IV)",
                "is_fist": False,
            }

    # 3. 3 Main fingers raised:
    if main_count == 3:
        return {
            "chord_index": 3,
            "chord_name": "iii",
            "gesture_name": "3 Fingers (iii)",
            "is_fist": False,
        }

    # 4. 2 Main fingers raised:
    if main_count == 2:
        return {
            "chord_index": 2,
            "chord_name": "ii",
            "gesture_name": "2 Fingers (ii)",
            "is_fist": False,
        }

    # 5. 1 Main finger raised:
    if main_count == 1:
        return {
            "chord_index": 1,
            "chord_name": "I",
            "gesture_name": "1 Finger (I)",
            "is_fist": False,
        }

    # 6. Only thumb extended (e.g. thumbs up):
    if main_count == 0 and thumb:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        palm_len = math.hypot(middle_mcp["x"] - wrist["x"], middle_mcp["y"] - wrist["y"]) or 1.0
        ux = (middle_mcp["x"] - wrist["x"]) / palm_len
        uy = (middle_mcp["y"] - wrist["y"]) / palm_len
        # Project along palm axis: thumb tip should be extended past index MCP along hand's length
        proj_tip = (thumb_tip["x"] - wrist["x"]) * ux + (thumb_tip["y"] - wrist["y"]) * uy
        proj_idx = (index_mcp["x"] - wrist["x"]) * ux + (index_mcp["y"] - wrist["y"]) * uy
        if proj_tip > proj_idx - 0.05 * palm_len:
            return {
                "chord_index": 1,
                "chord_name": "I",
                "gesture_name": "Thumbs Up (I)",
                "is_fist": False,
            }

    # 7. Closed fist:
    return {
        "chord_index": None,
        "chord_name": "MUTE",
        "gesture_name": "Fist (Mute)",
        "is_fist": True,
    }


def classify_right_hand_variation(landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Classifies Right Hand finger pointing to distinguish chord variations / extensions,
    and detects Right Hand thumb elevation for Octave UP (+1 Octave) transposition:
    - Thumb Raised Up                       -> Octave UP (+1 Octave, e.g. C4 -> C5)
    - Index pointed up (or 1 finger)        -> 7th Variation (Maj7 / m7 / Dom7)
    - Middle pointed up (or 2 fingers / ✌)  -> Diminished Variation (dim / dim7)
    - Ring pointed up (or 3 fingers)        -> Suspended Variation (sus4)
    - Pinky pointed up (🤙)                 -> Augmented Variation (aug)
    - Open Palm (4 fingers)                 -> 9th / Add9 Variation
    - Closed / Fist (0 fingers)             -> Standard Triad
    """
    if not landmarks or len(landmarks) < 21:
        return {
            "variation": "TRIAD",
            "variation_name": "Standard Triad",
            "finger_count": 0,
            "is_thumb_up": False,
            "octave_shift": 0.0,
        }

    thumb = is_thumb_extended(landmarks, "Right")
    index = is_finger_raised(landmarks, 6, 8)
    middle = is_finger_raised(landmarks, 10, 12)
    ring = is_finger_raised(landmarks, 14, 16)
    pinky = is_finger_raised(landmarks, 18, 20)

    main_count = sum([index, middle, ring, pinky])

    # 1. Index pointing up (7th Chord)
    if index and not middle and not ring and not pinky:
        var = "7TH"
        var_name = "7th Extension (Index 👆)"
    # 2. Middle finger pointing OR 2 fingers (✌ Peace / Index+Middle) -> Diminished
    elif (middle and not index and not ring and not pinky) or (main_count == 2):
        var = "DIMINISHED"
        var_name = "Diminished (Middle / 2 Fingers)"
    # 3. Ring finger pointing OR 3 fingers -> Sus4
    elif (ring and not index and not middle and not pinky) or (main_count == 3):
        var = "SUS4"
        var_name = "Suspended Sus4 (Ring / 3 Fingers)"
    # 4. Pinky finger pointing -> Augmented
    elif pinky and not index and not middle and not ring:
        var = "AUGMENTED"
        var_name = "Augmented (Pinky 🤙)"
    # 5. Open Palm (4 fingers) -> 9th Extension
    elif main_count == 4:
        var = "9TH"
        var_name = "Rich 9th (Open Palm)"
    # 6. Fallback 1 finger
    elif main_count == 1:
        var = "7TH"
        var_name = "7th Chord (1 Finger)"
    # 7. Closed hand / Fist -> Standard Triad
    else:
        var = "TRIAD"
        var_name = "Standard Triad"

    return {
        "variation": var,
        "variation_name": var_name,
        "finger_count": main_count,
        "is_thumb_up": thumb,
        "octave_shift": 12.0 if thumb else 0.0,
    }


def calculate_hand_tilt(landmarks: List[Dict[str, float]]) -> float:
    """Calculates hand tilt angle in radians from vertical axis (Wrist to Middle MCP)."""
    if not landmarks or len(landmarks) < 10:
        return 0.0
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    dx = middle_mcp["x"] - wrist["x"]
    dy = wrist["y"] - middle_mcp["y"]  # Invert y because screen coordinates point downward

    return math.atan2(dx, dy)


def map_left_tilt_to_quality(tilt_rad: float, current_quality: Optional[str] = None) -> Optional[str]:
    """
    Maps Left Hand tilt to Major or Minor chord quality with a stable hysteresis deadband:
    - Upright / Neutral (-0.25 to +0.25 rad / ~-15° to +15°): None (uses natural diatonic scale quality).
    - Distinct Tilt LEFT  (< -0.26 rad / ~-15°): Forces 'Minor'
    - Distinct Tilt RIGHT (> +0.26 rad / ~+15°): Forces 'Major'
    - Hysteresis release deadband: returning within +/-0.18 rad (~10°) returns to diatonic quality.
    """
    if current_quality == "Minor":
        if tilt_rad > -0.18:
            return None
        return "Minor"
    elif current_quality == "Major":
        if tilt_rad < 0.18:
            return None
        return "Major"
    else:
        if tilt_rad < -0.26:
            return "Minor"
        elif tilt_rad > 0.26:
            return "Major"
        return None


def map_left_tilt_to_gain(tilt_rad: float) -> float:
    """
    Returns clean, steady chord gain.
    Left hand tilt controls Major/Minor chord quality only, not volume, gain or tone!
    Right hand elevation controls dynamic volume.
    """
    return 0.85


def map_right_tilt_to_pitch(tilt_rad: float) -> float:
    """
    Maps Right Hand lateral tilt to pitch transposition (-12 to +12 semitones).
    Features a wide deadband (+/- 0.28 rad / ~16°) so hand resting or finger pointing
    NEVER causes accidental pitch modulation.
    """
    if abs(tilt_rad) < 0.28:
        return 0.0

    sign = 1.0 if tilt_rad > 0 else -1.0
    effective = abs(tilt_rad) - 0.28
    semitones = sign * (effective / 0.35) * 12.0
    return max(-12.0, min(12.0, float(semitones)))


def map_right_height_to_volume(landmarks: List[Dict[str, float]]) -> float:
    """Maps Right Hand vertical position to volume [0.0, 1.0] (top = loud, bottom = soft)."""
    if not landmarks or len(landmarks) < 10:
        return 0.75
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    avg_y = (wrist["y"] + middle_mcp["y"]) / 2.0

    # Range: y=0.15 (top, 100%) to y=0.85 (bottom, 0%)
    vol = (0.85 - avg_y) / 0.70
    return max(0.0, min(1.0, float(vol)))


def generate_chord_frequencies(
    chord_index: int,
    key_name: str = "C",
    scale_name: str = "MAJOR",
    pitch_shift_semitones: float = 0.0,
    quality_override: Optional[str] = None,
    variation: str = "TRIAD",
    base_octave: Optional[int] = None,
) -> Tuple[List[float], str, str]:
    """
    Generates scale-aware chord frequencies in the active musical key and scale,
    applying Left Hand quality override (Major/Minor), Right Hand variation (7th, Dim, Sus4, Aug, 9th),
    and Right Hand thumb octave transposition (+1 Octave).
    """
    scale_clean = str(scale_name).upper().strip()
    scale_dict = SCALE_CHORD_DEFINITIONS.get(scale_clean, SCALE_CHORD_DEFINITIONS["MAJOR"])
    def_info = scale_dict.get(chord_index)
    if not def_info:
        return [], "--", "--"

    tonic_freq = ROOT_KEYS_FREQS.get(key_name.upper(), 261.63)
    # Anchor tonic base octave to 150 - 300 Hz so all 7 scale degrees ascend smoothly in one octave
    while tonic_freq > 300.0:
        tonic_freq /= 2.0
    while tonic_freq < 150.0:
        tonic_freq *= 2.0

    degree_offset = def_info.get("semitones", MAJOR_DEGREE_SEMITONES.get(def_info["degree"], 0))
    degree_tonic = tonic_freq * math.pow(2.0, degree_offset / 12.0)

    # Determine base quality
    if quality_override in ("Major", "Minor"):
        quality = quality_override
    else:
        quality = def_info["quality"]

    degree = def_info["degree"]
    roman = def_info["roman"]

    # Build intervals based on quality and variation
    if variation == "7TH":
        if quality == "Major":
            if degree == 5:
                intervals = [0, 4, 7, 10, 12]  # Dominant 7th
                var_suffix = "7"
            else:
                intervals = [0, 4, 7, 11, 12]  # Major 7th
                var_suffix = "Maj7"
        elif quality == "Minor":
            intervals = [0, 3, 7, 10, 12]      # Minor 7th
            var_suffix = "min7"
        else:
            intervals = [0, 3, 6, 9, 12]       # Diminished 7th
            var_suffix = "dim7"

    elif variation == "DIMINISHED":
        intervals = [0, 3, 6, 9, 12]           # Full Diminished 7th
        var_suffix = "dim7"

    elif variation == "SUS4":
        intervals = [0, 5, 7, 12]              # Suspended 4th
        var_suffix = "sus4"

    elif variation == "AUGMENTED":
        intervals = [0, 4, 8, 12]              # Augmented triad
        var_suffix = "aug"

    elif variation == "9TH":
        if quality == "Major":
            intervals = [0, 4, 7, 11, 14]      # Major 9th
            var_suffix = "Maj9"
        elif quality == "Minor":
            intervals = [0, 3, 7, 10, 14]      # Minor 9th
            var_suffix = "min9"
        else:
            intervals = [0, 3, 6, 9, 14]       # Diminished 9th
            var_suffix = "dim9"

    else:
        # Standard Triad
        if quality == "Major":
            intervals = [0, 4, 7, 12]
            var_suffix = "Maj"
        elif quality == "Minor":
            intervals = [0, 3, 7, 12]
            var_suffix = "min"
        else:
            intervals = [0, 3, 6, 12]
            var_suffix = "dim"

    pitch_multiplier = math.pow(2.0, pitch_shift_semitones / 12.0)
    freqs = [
        float(degree_tonic * math.pow(2.0, interval / 12.0) * pitch_multiplier)
        for interval in intervals
    ]

    # Exact Chord Name with optional Octave (e.g. 'C4 Major', 'C5 Major', 'D4 Diminished')
    root_name = get_chord_root_note_name(key_name, degree, base_octave=base_octave, semitone_offset=degree_offset)

    if variation == "TRIAD":
        chord_label = f"{root_name} {quality}"
    elif variation == "7TH":
        if quality == "Major":
            chord_label = f"{root_name}7" if degree == 5 else f"{root_name} Maj7"
        elif quality == "Minor":
            chord_label = f"{root_name} min7"
        else:
            chord_label = f"{root_name} dim7"
    elif variation == "DIMINISHED":
        chord_label = f"{root_name} dim7"
    elif variation == "SUS4":
        chord_label = f"{root_name} sus4"
    elif variation == "AUGMENTED":
        chord_label = f"{root_name} aug"
    elif variation == "9TH":
        chord_label = f"{root_name} Maj9" if quality == "Major" else f"{root_name} min9"
    else:
        chord_label = f"{root_name} {quality}"

    chord_degree = f"{roman} ({quality})"

    return freqs, chord_label, chord_degree


class TwoHandSynthesizerState:
    """
    Maintains independent two-handed synthesizer state, parameter smoothing,
    and Dual-VCA final amplitude calculation.
    """

    def __init__(
        self,
        timeout_ms: float = 500.0,
        smoothing_alpha: float = 0.25,
        ai_classifier: Optional[Any] = None,
    ):
        self.timeout_sec = timeout_ms / 1000.0
        self.alpha = float(smoothing_alpha)
        self.ai_classifier = ai_classifier

        # Left Hand state
        self.active_chord_index: Optional[int] = None
        self.selected_chord_name = "--"
        self.selected_gesture_name = "Waiting for Left Hand..."
        self.left_tilt = 0.0
        self.quality_override: Optional[str] = None
        self._pending_quality: Optional[str] = None
        self._quality_debounce_count = 0
        self.left_gain = 0.75
        self.smoothed_left_gain = 0.75
        self.last_left_seen_time = 0.0
        self.is_left_present = False

        # Chord debouncing to prevent single-frame flickering
        self._pending_chord_index: Optional[int] = None
        self._chord_debounce_count = 0

        # Right Hand state
        self.right_tilt = 0.0
        self.pitch_shift_semitones = 0.0
        self.smoothed_pitch_shift = 0.0
        self.right_volume = 0.75
        self.smoothed_right_volume = 0.75
        self.right_variation = "TRIAD"
        self.right_variation_name = "Standard Triad"
        self._pending_variation = "TRIAD"
        self._variation_debounce_count = 0
        self.is_right_thumb_up = False
        self.smoothed_octave_shift = 0.0
        self._pending_octave_thumb = False
        self._octave_debounce_count = 0
        self.last_right_seen_time = 0.0
        self.is_right_present = False

        # Combined output
        self.final_amplitude = 0.0
        self.active_frequencies: List[float] = []

    def update(
        self,
        left_landmarks: Optional[List[Dict[str, float]]],
        right_landmarks: Optional[List[Dict[str, float]]],
        current_time: Optional[float] = None,
        key_name: str = "C",
        scale_name: str = "MAJOR",
    ) -> Dict[str, Any]:
        """Update state from live video frame hand detections."""
        if current_time is None:
            current_time = time.time()

        alpha = self.alpha

        # -------------------------------------------------------------
        # 1. LEFT HAND: Chord Selection & Major/Minor Quality Override
        # -------------------------------------------------------------
        was_left_present = self.is_left_present
        if left_landmarks and len(left_landmarks) == 21:
            self.last_left_seen_time = current_time
            self.is_left_present = True

            # Calculate tilt and dynamic Major/Minor quality override with 3-frame debounce
            self.left_tilt = calculate_hand_tilt(left_landmarks)
            raw_quality = map_left_tilt_to_quality(self.left_tilt, self.quality_override)
            if raw_quality == self.quality_override:
                self._pending_quality = raw_quality
                self._quality_debounce_count = 0
            else:
                if raw_quality == self._pending_quality:
                    self._quality_debounce_count += 1
                    if self._quality_debounce_count >= 3:
                        self.quality_override = raw_quality
                else:
                    self._pending_quality = raw_quality
                    self._quality_debounce_count = 1

            # Deterministic, tilt-invariant chord finger counting
            chord_res = classify_left_hand_chord(left_landmarks, "Left")
            if chord_res["is_fist"]:
                # Fist mute is immediate with 0 delay for safety
                self.active_chord_index = None
                self.selected_chord_name = "MUTE"
                self.selected_gesture_name = "Fist (Mute)"
                self._pending_chord_index = None
                self._chord_debounce_count = 0
            elif chord_res["chord_index"] is not None:
                raw_chord = chord_res["chord_index"]
                # If hand just appeared or matches active chord, accept immediately
                if not was_left_present or self.active_chord_index is None or raw_chord == self.active_chord_index:
                    self.active_chord_index = raw_chord
                    self.selected_chord_name = chord_res["chord_name"]
                    self.selected_gesture_name = chord_res["gesture_name"]
                    self._pending_chord_index = raw_chord
                    self._chord_debounce_count = 0
                else:
                    # Debounce chord change by 2 frames to eliminate landmark flickers
                    if raw_chord == self._pending_chord_index:
                        self._chord_debounce_count += 1
                        if self._chord_debounce_count >= 2:
                            self.active_chord_index = raw_chord
                            self.selected_chord_name = chord_res["chord_name"]
                            self.selected_gesture_name = chord_res["gesture_name"]
                    else:
                        self._pending_chord_index = raw_chord
                        self._chord_debounce_count = 1

            # Add quality badge to gesture display if tilted
            if self.active_chord_index is not None and self.quality_override:
                self.selected_gesture_name = f"{self.selected_gesture_name} [{self.quality_override}]"

            self.left_gain = map_left_tilt_to_gain(self.left_tilt)
            self.smoothed_left_gain = alpha * self.left_gain + (1.0 - alpha) * self.smoothed_left_gain

        else:
            # Check grace period timeout (500 ms)
            if current_time - self.last_left_seen_time >= self.timeout_sec:
                self.is_left_present = False
                self.smoothed_left_gain = 0.0
                self.active_chord_index = None
                self.selected_chord_name = "--"
                self.selected_gesture_name = "Left hand absent"
                self._pending_chord_index = None
                self._chord_debounce_count = 0
                self.quality_override = None
                self._pending_quality = None
                self._quality_debounce_count = 0

        # -------------------------------------------------------------
        # 2. RIGHT HAND: Finger Variations, Pitch & Volume
        # -------------------------------------------------------------
        was_right_present = self.is_right_present
        if right_landmarks and len(right_landmarks) == 21:
            self.last_right_seen_time = current_time
            self.is_right_present = True

            # Finger variation detection with 2-frame debounce stability
            var_res = classify_right_hand_variation(right_landmarks)
            raw_var = var_res["variation"]
            if not was_right_present or raw_var == self.right_variation:
                self.right_variation = raw_var
                self.right_variation_name = var_res["variation_name"]
                self._pending_variation = raw_var
                self._variation_debounce_count = 0
            else:
                if raw_var == self._pending_variation:
                    self._variation_debounce_count += 1
                    if self._variation_debounce_count >= 2:
                        self.right_variation = raw_var
                        self.right_variation_name = var_res["variation_name"]
                else:
                    self._pending_variation = raw_var
                    self._variation_debounce_count = 1

            # Right Hand Thumb: Octave UP (+1 Octave, 12 semitones) with 2-frame debounce
            raw_thumb_up = bool(var_res.get("is_thumb_up", False))
            if not was_right_present or raw_thumb_up == self.is_right_thumb_up:
                self.is_right_thumb_up = raw_thumb_up
                self._pending_octave_thumb = raw_thumb_up
                self._octave_debounce_count = 0
            else:
                if raw_thumb_up == self._pending_octave_thumb:
                    self._octave_debounce_count += 1
                    if self._octave_debounce_count >= 2:
                        self.is_right_thumb_up = raw_thumb_up
                else:
                    self._pending_octave_thumb = raw_thumb_up
                    self._octave_debounce_count = 1

            target_oct_shift = 12.0 if self.is_right_thumb_up else 0.0
            self.smoothed_octave_shift = alpha * target_oct_shift + (1.0 - alpha) * self.smoothed_octave_shift

            # Pitch transposition via tilt (with deadband)
            self.right_tilt = calculate_hand_tilt(right_landmarks)
            self.pitch_shift_semitones = map_right_tilt_to_pitch(self.right_tilt)
            if abs(self.pitch_shift_semitones) < 0.01:
                self.smoothed_pitch_shift = 0.0
            else:
                self.smoothed_pitch_shift = alpha * self.pitch_shift_semitones + (1.0 - alpha) * self.smoothed_pitch_shift

            # Volume via vertical height
            self.right_volume = map_right_height_to_volume(right_landmarks)
            self.smoothed_right_volume = alpha * self.right_volume + (1.0 - alpha) * self.smoothed_right_volume

        else:
            # Check grace period timeout (500 ms)
            if current_time - self.last_right_seen_time >= self.timeout_sec:
                self.is_right_present = False
                # When right hand absent, default variation to Triad, and default volume to 0.75
                self.right_variation = "TRIAD"
                self.right_variation_name = "Standard Triad"
                self._pending_variation = "TRIAD"
                self._variation_debounce_count = 0
                self.is_right_thumb_up = False
                self.smoothed_octave_shift = 0.0
                self.smoothed_pitch_shift = 0.0
                self.smoothed_right_volume = 0.75

        # -------------------------------------------------------------
        # 3. DUAL-VCA FINAL AMPLITUDE & CHORD SYNTHESIS
        # -------------------------------------------------------------
        if self.is_left_present:
            if self.is_right_present:
                self.final_amplitude = max(0.0, min(1.0, self.smoothed_left_gain * self.smoothed_right_volume))
            else:
                # Standalone left hand chord playing
                self.final_amplitude = max(0.0, min(1.0, self.smoothed_left_gain * 0.75))
        else:
            self.final_amplitude = 0.0

        # Fist emergency mute
        if self.selected_chord_name == "MUTE":
            self.final_amplitude = 0.0

        freqs = []
        chord_label = self.selected_chord_name
        chord_degree = "--"

        if self.active_chord_index is not None:
            active_base_octave = 5 if (self.is_right_present and self.is_right_thumb_up) else 4
            oct_semitones = 12.0 if (self.is_right_present and self.is_right_thumb_up) else 0.0
            effective_shift = (self.smoothed_pitch_shift + oct_semitones) if self.is_right_present else 0.0
            freqs, chord_label, chord_degree = generate_chord_frequencies(
                self.active_chord_index,
                key_name=key_name,
                scale_name=scale_name,
                pitch_shift_semitones=effective_shift,
                quality_override=self.quality_override,
                variation=self.right_variation,
                base_octave=active_base_octave,
            )

        self.active_frequencies = freqs

        return {
            "is_left_present": self.is_left_present,
            "chord_index": self.active_chord_index,
            "chord_name": chord_label,
            "chord_degree": chord_degree,
            "gesture_name": self.selected_gesture_name,
            "left_gain": self.smoothed_left_gain,
            "quality_override": self.quality_override,
            "is_right_present": self.is_right_present,
            "right_variation": self.right_variation,
            "right_variation_name": self.right_variation_name,
            "is_right_thumb_up": self.is_right_thumb_up,
            "octave_shift": 1 if self.is_right_thumb_up else 0,
            "pitch_shift_semitones": self.smoothed_pitch_shift,
            "right_volume": self.smoothed_right_volume,
            "final_amplitude": self.final_amplitude,
            "frequencies": self.active_frequencies,
        }
