"""
test_two_hand_python.py
Comprehensive automated test suite for Two-Handed Gesture Synthesizer Controls:
- Tilt-invariant finger counting & Chord 5 open-palm detection
- Dynamic Left Hand Tilt Major/Minor quality switching
- Right Hand finger pointing chord variations (7ths, diminished, sus4, aug, 9th)
- Standalone Left Hand playback and Dual-VCA dynamics
- Temporal hand tracking stability
"""

import time
import math
import numpy as np

from two_hand_tracker import (
    classify_left_hand_chord,
    classify_right_hand_variation,
    calculate_hand_tilt,
    map_left_tilt_to_quality,
    map_left_tilt_to_gain,
    map_right_tilt_to_pitch,
    map_right_height_to_volume,
    generate_chord_frequencies,
    TwoHandSynthesizerState,
    is_finger_raised,
    is_thumb_extended,
    DIATONIC_CHORD_DEFINITIONS,
)
from ai_classifier import (
    AIGestureClassifier,
    extract_normalized_features,
    GESTURE_CLASSES,
)
from synth_engine import SynthEngine
from ui import UIOverlay


def make_synthetic_hand(
    wrist_xy=(0.5, 0.7),
    thumb_up=False,
    index_up=False,
    mid_up=False,
    ring_up=False,
    pinky_up=False,
    tilt_angle_rad=0.0,
):
    """
    Generates realistic 21-landmark hand data with optional rotational tilt.
    """
    lms = []
    wx, wy = wrist_xy

    # 0: Wrist
    lms.append({"id": 0, "x": wx, "y": wy, "z": 0.0, "px": int(wx * 640), "py": int(wy * 480)})

    cur_idx = 1
    # 1-4 Thumb
    if thumb_up:
        thumb_pts = [
            (wx - 0.04, wy - 0.03),
            (wx - 0.08, wy - 0.06),
            (wx - 0.13, wy - 0.10),
            (wx - 0.18, wy - 0.14),
        ]
    else:
        thumb_pts = [
            (wx - 0.02, wy - 0.02),
            (wx - 0.03, wy - 0.04),
            (wx - 0.03, wy - 0.06),
            (wx - 0.02, wy - 0.07),
        ]

    # Helper to rotate points around wrist
    def rotate_pt(px, py, angle):
        if abs(angle) < 1e-4:
            return px, py
        dx, dy = px - wx, py - wy
        rx = wx + dx * math.cos(angle) - dy * math.sin(angle)
        ry = wy + dx * math.sin(angle) + dy * math.cos(angle)
        return rx, ry

    for px, py in thumb_pts:
        rx, ry = rotate_pt(px, py, tilt_angle_rad)
        lms.append({"id": cur_idx, "x": rx, "y": ry, "z": 0.0, "px": int(rx * 640), "py": int(ry * 480)})
        cur_idx += 1

    # 4 Fingers: Index, Middle, Ring, Pinky
    finger_configs = [
        (index_up, -0.05),
        (mid_up, 0.0),
        (ring_up, 0.05),
        (pinky_up, 0.10),
    ]

    for is_up, base_dx in finger_configs:
        mcp_x = wx + base_dx
        mcp_y = wy - 0.10
        if is_up:
            # Extended straight up along palm axis
            pip_y = wy - 0.17
            dip_y = wy - 0.23
            tip_y = wy - 0.30
            pts = [(mcp_x, mcp_y), (mcp_x, pip_y), (mcp_x, dip_y), (mcp_x, tip_y)]
        else:
            # Curled into palm (tip curls down below pip)
            pip_y = wy - 0.12
            dip_y = wy - 0.09
            tip_y = wy - 0.06
            pts = [(mcp_x, mcp_y), (mcp_x, pip_y), (mcp_x, dip_y), (mcp_x, tip_y)]

        for px, py in pts:
            rx, ry = rotate_pt(px, py, tilt_angle_rad)
            lms.append({"id": cur_idx, "x": rx, "y": ry, "z": 0.0, "px": int(rx * 640), "py": int(ry * 480)})
            cur_idx += 1

    return lms


def test_1_left_hand_chord_classification_and_open_palm():
    print("Test 1: Left Hand Chord Classification & Robust 5th Chord (Open Palm)...")

    # Chord 1: 1 finger (Index)
    h1 = make_synthetic_hand(index_up=True)
    res1 = classify_left_hand_chord(h1, "Left")
    assert res1["chord_index"] == 1, f"Expected Chord 1, got {res1}"
    assert res1["chord_name"] == "I"

    # Chord 2: 2 fingers (Index + Middle)
    h2 = make_synthetic_hand(index_up=True, mid_up=True)
    res2 = classify_left_hand_chord(h2, "Left")
    assert res2["chord_index"] == 2, f"Expected Chord 2, got {res2}"
    assert res2["chord_name"] == "ii"

    # Chord 3: 3 fingers (Index + Middle + Ring)
    h3 = make_synthetic_hand(index_up=True, mid_up=True, ring_up=True)
    res3 = classify_left_hand_chord(h3, "Left")
    assert res3["chord_index"] == 3, f"Expected Chord 3, got {res3}"
    assert res3["chord_name"] == "iii"

    # Chord 4: 4 fingers (Index + Middle + Ring + Pinky, thumb tucked)
    h4 = make_synthetic_hand(thumb_up=False, index_up=True, mid_up=True, ring_up=True, pinky_up=True)
    res4 = classify_left_hand_chord(h4, "Left")
    assert res4["chord_index"] in (4, 5)

    # Chord 5: FULLY OPEN PALM (All 5 fingers extended)
    h5 = make_synthetic_hand(thumb_up=True, index_up=True, mid_up=True, ring_up=True, pinky_up=True)
    res5 = classify_left_hand_chord(h5, "Left")
    assert res5["chord_index"] == 5, f"Expected Chord 5 (Open Palm), got {res5}"
    assert res5["chord_name"] == "V"

    # Chord 5 when TILTED LEFT (-30 degrees / -0.52 rad)
    h5_tilted_left = make_synthetic_hand(thumb_up=True, index_up=True, mid_up=True, ring_up=True, pinky_up=True, tilt_angle_rad=-0.50)
    res5_tl = classify_left_hand_chord(h5_tilted_left, "Left")
    assert res5_tl["chord_index"] == 5, f"Tilted left hand must detect Chord 5! Got {res5_tl}"

    # Chord 5 when TILTED RIGHT (+30 degrees / +0.52 rad)
    h5_tilted_right = make_synthetic_hand(thumb_up=True, index_up=True, mid_up=True, ring_up=True, pinky_up=True, tilt_angle_rad=+0.50)
    res5_tr = classify_left_hand_chord(h5_tilted_right, "Left")
    assert res5_tr["chord_index"] == 5, f"Tilted right hand must detect Chord 5! Got {res5_tr}"

    # Chord 6: Rock sign 🤘
    h6 = make_synthetic_hand(thumb_up=False, index_up=True, mid_up=False, ring_up=False, pinky_up=True)
    res6 = classify_left_hand_chord(h6, "Left")
    assert res6["chord_index"] == 6, f"Expected Chord 6 (vi), got {res6}"

    # Chord 7: Rock + Thumb 🤟
    h7 = make_synthetic_hand(thumb_up=True, index_up=True, mid_up=False, ring_up=False, pinky_up=True)
    res7 = classify_left_hand_chord(h7, "Left")
    assert res7["chord_index"] == 7, f"Expected Chord 7 (vii°), got {res7}"

    # Fist Mute
    hfist = make_synthetic_hand()
    resfist = classify_left_hand_chord(hfist, "Left")
    assert resfist["is_fist"] is True
    assert resfist["chord_index"] is None

    print("  -> PASSED: All 7 chords and open palm (Chord 5) recognized reliably, even when tilted.")


def test_2_left_hand_tilt_major_minor_switching():
    print("Test 2: Left Hand Tilt mapped to Major / Minor Quality with Clean Steady Gain...")

    # Left tilt angle test
    q_neutral = map_left_tilt_to_quality(0.0)
    q_left = map_left_tilt_to_quality(-0.35)   # Tilted Left
    q_right = map_left_tilt_to_quality(+0.35)  # Tilted Right

    assert q_neutral is None, "Neutral tilt must not override diatonic quality"
    assert q_left == "Minor", f"Left tilt must force Minor chord! Got {q_left}"
    assert q_right == "Major", f"Right tilt must force Major chord! Got {q_right}"

    # Left hand tilt must maintain clean, steady gain (no volume drops on tilt!)
    gain_neutral = map_left_tilt_to_gain(0.0)
    gain_left = map_left_tilt_to_gain(-0.5)
    gain_right = map_left_tilt_to_gain(+0.5)
    assert gain_neutral == 0.85 and gain_left == 0.85 and gain_right == 0.85, "Left tilt must have clean steady gain!"

    # Verify exact chord names (e.g. C Major, C Minor, D Major, B Major)
    # Key C, Degree 1 (C Major or C Minor)
    freqs_c_min, label_c_min, _ = generate_chord_frequencies(1, "C", quality_override="Minor")
    assert abs(freqs_c_min[0] - 261.63) < 1.0  # Root C
    assert abs(freqs_c_min[1] - 311.13) < 1.5  # Eb (Minor 3rd: 3 semitones)
    assert abs(freqs_c_min[2] - 392.00) < 1.0  # G (Perfect 5th)
    assert label_c_min == "C Minor", f"Expected 'C Minor', got {label_c_min}"

    freqs_c_maj, label_c_maj, _ = generate_chord_frequencies(1, "C", quality_override="Major")
    assert abs(freqs_c_maj[0] - 261.63) < 1.0  # Root C
    assert abs(freqs_c_maj[1] - 329.63) < 1.5  # E (Major 3rd)
    assert label_c_maj == "C Major", f"Expected 'C Major', got {label_c_maj}"

    # Key C, Degree 2 (D) tilted right -> D Major!
    _, label_d_maj, _ = generate_chord_frequencies(2, "C", quality_override="Major")
    assert label_d_maj == "D Major", f"Expected 'D Major', got {label_d_maj}"

    # Key C, Degree 7 (B) tilted right -> B Major! (Explicit user request test!)
    _, label_b_maj, _ = generate_chord_frequencies(7, "C", quality_override="Major")
    assert label_b_maj == "B Major", f"Expected 'B Major', got {label_b_maj}"

    # Key C, Degree 7 (B) upright -> B Diminished!
    _, label_b_dim, _ = generate_chord_frequencies(7, "C", quality_override=None)
    assert label_b_dim == "B Diminished", f"Expected 'B Diminished', got {label_b_dim}"

    print("  -> PASSED: Left Hand tilt switches between Major and Minor cleanly without volume drops.")


def test_3_right_hand_finger_pointing_variations():
    print("Test 3: Right Hand Finger Pointing for Chord Variations (7th, Dim, Sus4, Aug, 9th)...")

    # 1. Point Index finger -> 7th Variation
    rh_index = make_synthetic_hand(wrist_xy=(0.7, 0.5), index_up=True)
    var_index = classify_right_hand_variation(rh_index)
    assert var_index["variation"] == "7TH", f"Expected 7TH variation, got {var_index}"

    # 2. Point Middle finger -> Diminished Variation
    rh_middle = make_synthetic_hand(wrist_xy=(0.7, 0.5), mid_up=True)
    var_middle = classify_right_hand_variation(rh_middle)
    assert var_middle["variation"] == "DIMINISHED", f"Expected DIMINISHED variation, got {var_middle}"

    # 3. Point Ring finger -> Suspended (Sus4) Variation
    rh_ring = make_synthetic_hand(wrist_xy=(0.7, 0.5), ring_up=True)
    var_ring = classify_right_hand_variation(rh_ring)
    assert var_ring["variation"] == "SUS4", f"Expected SUS4 variation, got {var_ring}"

    # 4. Point Pinky finger -> Augmented Variation
    rh_pinky = make_synthetic_hand(wrist_xy=(0.7, 0.5), pinky_up=True)
    var_pinky = classify_right_hand_variation(rh_pinky)
    assert var_pinky["variation"] == "AUGMENTED", f"Expected AUGMENTED variation, got {var_pinky}"

    # 5. Open Hand / Thumb -> 9th Variation
    rh_open = make_synthetic_hand(wrist_xy=(0.7, 0.5), thumb_up=True, index_up=True, mid_up=True, ring_up=True, pinky_up=True)
    var_open = classify_right_hand_variation(rh_open)
    assert var_open["variation"] == "9TH", f"Expected 9TH variation, got {var_open}"

    # 6. Relaxed fist (0 fingers) -> Standard Triad
    rh_fist = make_synthetic_hand(wrist_xy=(0.7, 0.5))
    var_fist = classify_right_hand_variation(rh_fist)
    assert var_fist["variation"] == "TRIAD", f"Expected TRIAD variation, got {var_fist}"

    # Verify generated chord frequencies and exact names for variations:
    # C Maj7 (Intervals: 0, 4, 7, 11, 12 -> C, E, G, B, C)
    freqs_7th, label_7th, _ = generate_chord_frequencies(1, "C", variation="7TH")
    assert len(freqs_7th) == 5
    assert label_7th == "C Maj7", f"Expected 'C Maj7', got {label_7th}"
    b4_freq = 261.63 * math.pow(2.0, 11.0 / 12.0)
    assert abs(freqs_7th[3] - b4_freq) < 1.5

    # G7 (Degree 5 in Key C -> Dominant 7th)
    _, label_g7, _ = generate_chord_frequencies(5, "C", variation="7TH")
    assert label_g7 == "G7", f"Expected 'G7', got {label_g7}"

    # D min7 (Degree 2 in Key C -> Minor 7th)
    _, label_d_min7, _ = generate_chord_frequencies(2, "C", variation="7TH")
    assert label_d_min7 == "D min7", f"Expected 'D min7', got {label_d_min7}"

    # C Diminished 7th
    freqs_dim, label_dim, _ = generate_chord_frequencies(1, "C", variation="DIMINISHED")
    assert len(freqs_dim) == 5
    assert label_dim == "C dim7", f"Expected 'C dim7', got {label_dim}"
    gb4_freq = 261.63 * math.pow(2.0, 6.0 / 12.0)
    assert abs(freqs_dim[2] - gb4_freq) < 1.5

    # C Sus4
    freqs_sus, label_sus, _ = generate_chord_frequencies(1, "C", variation="SUS4")
    f4_freq = 261.63 * math.pow(2.0, 5.0 / 12.0)
    assert abs(freqs_sus[1] - f4_freq) < 1.5
    assert label_sus == "C sus4", f"Expected 'C sus4', got {label_sus}"

    print("  -> PASSED: Right Hand fingers distinguish 7ths, diminished, sus4, aug, and 9th chords.")


def test_4_standalone_left_hand_and_dual_vca():
    print("Test 4: Standalone Left Hand Playback & Dual-VCA Interaction...")

    state = TwoHandSynthesizerState(timeout_ms=500.0)
    lh = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True)  # Left hand Chord 1

    # Case A: Left hand alone (Right hand not present)
    out_lh_only = state.update(lh, None, current_time=1.0)
    assert out_lh_only["is_left_present"] is True
    assert out_lh_only["is_right_present"] is False
    assert out_lh_only["final_amplitude"] > 0.1, "Left hand alone MUST play at comfortable volume!"
    assert len(out_lh_only["frequencies"]) >= 3, "Chord frequencies must be generated!"

    # Case B: Bring in Right hand to modulate volume and variation
    rh_high = make_synthetic_hand(wrist_xy=(0.7, 0.2), index_up=True) # Right hand high (loud), pointing index (7th)
    out_both = state.update(lh, rh_high, current_time=1.1)
    assert out_both["is_left_present"] is True
    assert out_both["is_right_present"] is True
    assert out_both["right_variation"] == "7TH"
    assert out_both["final_amplitude"] > 0.4, "High right hand should yield loud volume"

    # Case C: Fist mute
    lh_fist = make_synthetic_hand(wrist_xy=(0.3, 0.6))
    out_mute = state.update(lh_fist, rh_high, current_time=1.2)
    assert out_mute["chord_name"] == "MUTE"
    assert out_mute["final_amplitude"] == 0.0, "Fist must immediately mute audio!"

    # Case D: Both hands absent past 500 ms grace period -> smooth release
    out_timeout = state.update(None, None, current_time=1.8)
    assert out_timeout["final_amplitude"] == 0.0
    assert out_timeout["is_left_present"] is False

    print("  -> PASSED: Standalone left hand plays smoothly, right hand modulates, fist mutes cleanly.")


def test_5_synth_engine_polyphony_and_headroom():
    print("Test 5: SynthEngine Polyphony with Extended Chords (5 Voices)...")

    synth = SynthEngine()
    # 5-voice chord: C Maj9 [C4, E4, G4, B4, D5]
    c_maj9 = [261.63, 329.63, 392.00, 493.88, 587.33]
    synth.set_chord_frequencies(c_maj9)
    synth.note_on(volume=0.85)

    buf = np.zeros(256, dtype=np.float32)
    synth._audio_callback(buf, 256, None, None)

    assert not np.isnan(buf).any()
    assert not np.isinf(buf).any()
    assert np.max(np.abs(buf)) > 0.05
    assert np.max(np.abs(buf)) <= 1.0, "Headroom preserved (no clipping) for 5-voice chord!"
    synth.note_off()

    print("  -> PASSED: Polyphonic audio engine renders 5-voice chord extensions cleanly.")


def test_6_ui_workstation_rendering_preserved():
    print("Test 6: UI Workstation Telemetry Integration (Zero Layout Alterations)...")

    ui = UIOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    telemetry = {
        "is_note_active": True,
        "pitch_hz": 261.63,
        "nearest_note": "C Maj7",
        "cents": 0.0,
        "volume": 0.75,
        "is_pinched": False,
        "mode": "TWO-HAND CHORD",
        "scale": "MAJOR",
        "key": "C",
        "gesture": "1 Finger (I) [Major] + 7th Extension (Index 👆)",
        "octave_shift": 0,
        "tone_source": "MANUAL",
        "tone_val": None,
    }

    synth_status = {
        "instrument": "sawtooth",
        "effect_echo": False,
        "tone": 0.85,
        "arpeggiator": "OFF",
    }

    composite = ui.draw_hud(
        frame=frame,
        fps=30.0,
        is_hand_detected=True,
        handedness="Dual",
        gesture_data=telemetry,
        synth_data=synth_status,
        landmarks=make_synthetic_hand(index_up=True),
        camera_error=False,
    )

    assert composite.shape == (720, 1240, 3)
    print("  -> PASSED: UI renders rich telemetry without changing layout or styling.")


def test_7_pitch_stability_and_right_tilt_deadband():
    print("Test 7: Pitch Stability Across Chords & Right-Hand Tilt Deadband...")

    # A. Verify across ALL diatonic keys that Chord 5 is strictly higher than Chord 4 (no 1-octave drops!)
    for key in ["C", "D", "E", "F", "G", "A", "B"]:
        f4, label4, _ = generate_chord_frequencies(4, key_name=key)
        f5, label5, _ = generate_chord_frequencies(5, key_name=key)

        root4 = f4[0]
        root5 = f5[0]
        # In major diatonic scales, degree 5 is a whole tone (2 semitones, factor of ~1.122) above degree 4
        expected_ratio = math.pow(2.0, 2.0 / 12.0)  # ~1.12246
        actual_ratio = root5 / root4

        assert root5 > root4, f"Key {key}: Chord 5 ({root5:.1f}Hz) must be higher than Chord 4 ({root4:.1f}Hz)!"
        assert abs(actual_ratio - expected_ratio) < 0.05, (
            f"Key {key}: Chord 5 to Chord 4 ratio should be ~1.122, got {actual_ratio:.3f}!"
        )

    # Key C specifics: Chord 4 (F4: ~349.2 Hz) -> Chord 5 (G4: ~392.0 Hz)
    f4_c, _, _ = generate_chord_frequencies(4, key_name="C")
    f5_c, _, _ = generate_chord_frequencies(5, key_name="C")
    assert abs(f4_c[0] - 349.23) < 1.0, f"Expected F4 (~349.2 Hz), got {f4_c[0]}"
    assert abs(f5_c[0] - 392.00) < 1.0, f"Expected G4 (~392.0 Hz), got {f5_c[0]}"

    # B. Right-hand tilt deadband: slight hand movements (+/- 15 degrees) must NOT shift pitch
    assert map_right_tilt_to_pitch(0.0) == 0.0
    assert map_right_tilt_to_pitch(0.15) == 0.0, "Subtle right tilt (< 16 deg) must produce strictly 0 semitone shift!"
    assert map_right_tilt_to_pitch(-0.15) == 0.0, "Subtle right tilt (< 16 deg) must produce strictly 0 semitone shift!"
    assert map_right_tilt_to_pitch(0.25) == 0.0

    # Deliberate right tilt (> 16 degrees) engages pitch transposition smoothly
    shift_pos = map_right_tilt_to_pitch(+0.45)
    shift_neg = map_right_tilt_to_pitch(-0.45)
    assert shift_pos > 3.0, f"Deliberate right tilt should transpose up! Got {shift_pos}"
    assert shift_neg < -3.0, f"Deliberate left tilt should transpose down! Got {shift_neg}"

    print("  -> PASSED: Chords ascend monotonically, deadband prevents accidental pitch modulation.")


def test_9_major_minor_hysteresis_and_debouncing():
    print("Test 9: Major/Minor Tilt Hysteresis & Temporal Debouncing...")

    state = TwoHandSynthesizerState(timeout_ms=500.0)
    lh_neutral = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True, tilt_angle_rad=0.0)

    # 1. Neutral hand starts at natural diatonic quality (None override)
    out1 = state.update(lh_neutral, None, current_time=1.0)
    assert out1["quality_override"] is None
    assert out1["chord_name"] in ("C Major", "C4 Major")

    # 2. Slight hand tilt (-0.12 rad / ~7 deg) while moving fingers does NOT trigger Minor
    lh_slight_left = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True, tilt_angle_rad=-0.12)
    out2 = state.update(lh_slight_left, None, current_time=1.033)
    assert out2["quality_override"] is None, "Slight wrist wobble must remain in natural Diatonic scale!"

    # 3. Deliberate tilt left held for 3 consecutive frames engages Minor
    lh_tilt_left = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True, tilt_angle_rad=-0.35)
    state.update(lh_tilt_left, None, current_time=1.066)  # Frame 1
    state.update(lh_tilt_left, None, current_time=1.100)  # Frame 2
    out_minor = state.update(lh_tilt_left, None, current_time=1.133)  # Frame 3
    assert out_minor["quality_override"] == "Minor", f"Expected Minor after 3 frames, got {out_minor['quality_override']}"
    assert out_minor["chord_name"] in ("C Minor", "C4 Minor")

    # 4. Hysteresis: returning to -0.22 rad still holds Minor
    lh_hyst = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True, tilt_angle_rad=-0.22)
    out_hyst = state.update(lh_hyst, None, current_time=1.166)
    assert out_hyst["quality_override"] == "Minor", "Hysteresis should keep Minor engaged until returning past -0.18 rad"

    # 5. Return upright (0.0 rad) for 3 frames releases back to natural Diatonic
    state.update(lh_neutral, None, current_time=1.200)
    state.update(lh_neutral, None, current_time=1.233)
    out_restored = state.update(lh_neutral, None, current_time=1.266)
    assert out_restored["quality_override"] is None
    assert out_restored["chord_name"] in ("C Major", "C4 Major")

    print("  -> PASSED: Left tilt Major/Minor hysteresis and debouncing verified with zero jitter.")


def test_8_debouncing_and_pointing_immunity():
    print("Test 8: Temporal Debouncing & Right-Hand Index Pointing Immunity...")

    state = TwoHandSynthesizerState(timeout_ms=500.0)

    # Initial frame with Chord 4
    lh_chord4 = make_synthetic_hand(wrist_xy=(0.3, 0.6), thumb_up=False, index_up=True, mid_up=True, ring_up=True, pinky_up=True)
    out1 = state.update(lh_chord4, None, current_time=1.0)
    assert out1["chord_index"] == 4, f"Expected Chord 4, got {out1['chord_index']}"

    # Frame 2: A 1-frame jitter where thumb accidentally registers as open (Chord 5)
    lh_jitter_chord5 = make_synthetic_hand(wrist_xy=(0.3, 0.6), thumb_up=True, index_up=True, mid_up=True, ring_up=True, pinky_up=True)
    out2 = state.update(lh_jitter_chord5, None, current_time=1.033)
    # Debounce must hold Chord 4!
    assert out2["chord_index"] == 4, f"Single frame jitter must NOT switch chord! Got {out2['chord_index']}"

    # Frame 3: Jitter disappears, back to Chord 4
    out3 = state.update(lh_chord4, None, current_time=1.066)
    assert out3["chord_index"] == 4

    # Frame 4 & 5: Deliberate transition to Chord 1 (holding for 2 consecutive frames)
    lh_chord1 = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True)
    state.update(lh_chord1, None, current_time=1.100)  # Frame 1 of Chord 1
    out5 = state.update(lh_chord1, None, current_time=1.133)  # Frame 2 of Chord 1
    assert out5["chord_index"] == 1, f"2 consecutive frames must cleanly switch to Chord 1! Got {out5['chord_index']}"

    # Right Hand: Index pointing with thumb extended should still register 7th Extension!
    rh_index_with_thumb = make_synthetic_hand(wrist_xy=(0.7, 0.5), thumb_up=True, index_up=True)
    var_res = classify_right_hand_variation(rh_index_with_thumb)
    assert var_res["variation"] == "7TH", f"Index pointing with thumb must be 7th, got {var_res['variation']}"

    print("  -> PASSED: Single-frame twitches rejected, deliberate poses locked in, index pointing 100% immune.")


def test_10_scale_chords_and_octave_thumb():
    print("Test 10: Scale-Aware Chords (Minor, Blues, Dorian) & Right-Hand Thumb Octave UP...")

    # 1. Verify Scale-Aware Chord generation for Key C Minor:
    # Degree 1: C4 Minor (not C4 Major!)
    freqs1, name1, deg1 = generate_chord_frequencies(1, "C", scale_name="MINOR", base_octave=4)
    assert name1 == "C4 Minor", f"Expected 'C4 Minor', got {name1}"
    assert deg1 == "i (Minor)"

    # Degree 2: D4 Diminished (not D4 Minor!)
    freqs2, name2, deg2 = generate_chord_frequencies(2, "C", scale_name="MINOR", base_octave=4)
    assert name2 == "D4 Diminished", f"Expected 'D4 Diminished', got {name2}"
    assert deg2 == "ii° (Diminished)"

    # Degree 3: Eb4 Major (not E4 Minor!)
    freqs3, name3, deg3 = generate_chord_frequencies(3, "C", scale_name="MINOR", base_octave=4)
    assert name3 == "Eb4 Major", f"Expected 'Eb4 Major', got {name3}"
    assert deg3 == "III (Major)"

    # Degree 6: Ab4 Major (not A4 Minor!)
    freqs6, name6, deg6 = generate_chord_frequencies(6, "C", scale_name="MINOR", base_octave=4)
    assert name6 == "Ab4 Major", f"Expected 'Ab4 Major', got {name6}"
    assert deg6 == "VI (Major)"

    # Degree 7: Bb4 Major (not B4 Diminished!)
    freqs7, name7, deg7 = generate_chord_frequencies(7, "C", scale_name="MINOR", base_octave=4)
    assert name7 == "Bb4 Major", f"Expected 'Bb4 Major', got {name7}"
    assert deg7 == "VII (Major)"

    # 2. Verify Right Hand Thumb Octave UP (+1 Octave, C4 -> C5)
    state = TwoHandSynthesizerState(timeout_ms=500.0)
    lh_chord1 = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True)
    rh_thumb_down = make_synthetic_hand(wrist_xy=(0.7, 0.5), thumb_up=False)
    rh_thumb_up = make_synthetic_hand(wrist_xy=(0.7, 0.5), thumb_up=True)

    # Initial frame with right thumb down -> C4 Major
    out_down = state.update(lh_chord1, rh_thumb_down, current_time=1.0, key_name="C", scale_name="MAJOR")
    assert out_down["chord_name"] == "C4 Major", f"Expected 'C4 Major' with thumb down, got {out_down['chord_name']}"
    assert abs(out_down["frequencies"][0] - 261.63) < 1.0

    # Raising right thumb up (held for 2 frames for debounce) -> C5 Major
    state.update(lh_chord1, rh_thumb_up, current_time=1.033, key_name="C", scale_name="MAJOR")
    out_up = state.update(lh_chord1, rh_thumb_up, current_time=1.066, key_name="C", scale_name="MAJOR")
    assert out_up["chord_name"] == "C5 Major", f"Expected 'C5 Major' with thumb up, got {out_up['chord_name']}"
    assert abs(out_up["frequencies"][0] - 523.25) < 2.0, f"Expected ~523.25 Hz for C5, got {out_up['frequencies'][0]}"

    # 3. Left Hand Thumb Robustness:
    # Closed palm/fist must NOT trigger thumb or chord
    lh_fist = make_synthetic_hand(wrist_xy=(0.3, 0.6), thumb_up=False)
    res_fist = classify_left_hand_chord(lh_fist, "Left")
    assert res_fist["is_fist"] is True
    assert res_fist["chord_index"] is None

    # Fully open palm must trigger Chord 5
    lh_open = make_synthetic_hand(wrist_xy=(0.3, 0.6), thumb_up=True, index_up=True, mid_up=True, ring_up=True, pinky_up=True)
    res_open = classify_left_hand_chord(lh_open, "Left")
    assert res_open["chord_index"] == 5

    print("  -> PASSED: Scale-aware chords verified, right-hand thumb shifts C4 -> C5 cleanly, left thumb rock-solid.")


def test_11_comprehensive_thumb_invariance():
    print("Test 11: Comprehensive Thumb Invariance Across All Poses and Tilt Angles...")

    # 1. Closed fist across 7 tilt angles (-60° to +60°): must NEVER activate thumb or chord
    for angle in [-1.0, -0.7, -0.35, 0.0, 0.35, 0.7, 1.0]:
        hfist = make_synthetic_hand(thumb_up=False, tilt_angle_rad=angle)
        res = classify_left_hand_chord(hfist, "Left")
        assert res["is_fist"] is True, f"Fist failed at angle {angle}: {res}"
        assert res["chord_index"] is None
        assert is_thumb_extended(hfist, "Left") is False, f"Thumb falsely extended in fist at angle {angle}"

    # 2. Open palm (Chord 5) across 7 tilt angles: must ALWAYS activate Chord 5 (V)
    for angle in [-1.0, -0.7, -0.35, 0.0, 0.35, 0.7, 1.0]:
        hopen = make_synthetic_hand(thumb_up=True, index_up=True, mid_up=True, ring_up=True, pinky_up=True, tilt_angle_rad=angle)
        res = classify_left_hand_chord(hopen, "Left")
        assert res["chord_index"] == 5, f"Open palm failed at angle {angle}: {res}"
        assert is_thumb_extended(hopen, "Left") is True, f"Thumb failed in open palm at angle {angle}"

    # 3. 4-Finger hand (Chord 4, thumb tucked) across 7 tilt angles: must ALWAYS activate Chord 4 (IV)
    for angle in [-1.0, -0.7, -0.35, 0.0, 0.35, 0.7, 1.0]:
        h4 = make_synthetic_hand(thumb_up=False, index_up=True, mid_up=True, ring_up=True, pinky_up=True, tilt_angle_rad=angle)
        res = classify_left_hand_chord(h4, "Left")
        assert res["chord_index"] == 4, f"Chord 4 failed at angle {angle}: {res}"
        assert is_thumb_extended(h4, "Left") is False, f"Thumb falsely extended in chord 4 at angle {angle}"

    # 4. Right Hand Thumb Octave Shift across variations
    state = TwoHandSynthesizerState(timeout_ms=500.0)
    lh_chord1 = make_synthetic_hand(wrist_xy=(0.3, 0.6), index_up=True)

    rh_7th_down = make_synthetic_hand(wrist_xy=(0.7, 0.5), thumb_up=False, index_up=True)
    rh_7th_up = make_synthetic_hand(wrist_xy=(0.7, 0.5), thumb_up=True, index_up=True)

    state.update(lh_chord1, rh_7th_down, current_time=1.0)
    state.update(lh_chord1, rh_7th_down, current_time=1.033)
    out_7th_down = state.update(lh_chord1, rh_7th_down, current_time=1.066)
    assert "C4" in out_7th_down["chord_name"]
    assert out_7th_down["octave_shift"] == 0

    state.update(lh_chord1, rh_7th_up, current_time=1.100)
    state.update(lh_chord1, rh_7th_up, current_time=1.133)
    out_7th_up = state.update(lh_chord1, rh_7th_up, current_time=1.166)
    assert "C5" in out_7th_up["chord_name"]
    assert out_7th_up["octave_shift"] == 1
    assert abs(out_7th_up["frequencies"][0] - 2.0 * out_7th_down["frequencies"][0]) < 1.0

    print("  -> PASSED: Full tilt matrix and thumb invariance verified with zero false triggers.")


def test_12_hand_tracker_handedness_classification():
    print("Test 12: HandTracker Handedness (Left/Right) Decoding & Mirroring Accuracy...")
    from hand_tracker import HandTracker

    tracker = HandTracker()
    tracker.mp_draw.draw_landmarks = lambda *args, **kwargs: None

    class MockClassification:
        def __init__(self, label):
            self.label = label
            self.score = 0.98

    class MockHandedness:
        def __init__(self, label):
            self.classification = [MockClassification(label)]

    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.z = 0.0

    class MockHandLandmarks:
        def __init__(self, wx, wy):
            self.landmark = [MockLandmark(wx, wy) for _ in range(21)]

    class MockResults:
        def __init__(self, hands_data):
            self.multi_handedness = [MockHandedness(lbl) for lbl, _, _ in hands_data]
            self.multi_hand_landmarks = [MockHandLandmarks(wx, wy) for _, wx, wy in hands_data]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Physical Right hand detected alone -> Must be labeled 'Right'
    tracker.hands.process = lambda rgb: MockResults([("Right", 0.65, 0.5)])
    _, is_detected, _, handedness = tracker.process_frame(dummy_frame, draw=True)
    lh, rh = tracker.get_dual_hands()
    assert is_detected is True
    assert handedness == "Right", f"Physical Right hand must be labeled 'Right', got '{handedness}'"
    assert lh is None, "Left hand must be None when only Right hand is visible"
    assert rh is not None, "Right hand landmarks must be set"

    # 2. Physical Left hand detected alone -> Must be labeled 'Left'
    tracker.hands.process = lambda rgb: MockResults([("Left", 0.35, 0.5)])
    _, is_detected, _, handedness = tracker.process_frame(dummy_frame, draw=True)
    lh, rh = tracker.get_dual_hands()
    assert is_detected is True
    assert handedness == "Left", f"Physical Left hand must be labeled 'Left', got '{handedness}'"
    assert lh is not None, "Left hand landmarks must be set"
    assert rh is None, "Right hand must be None when only Left hand is visible"

    # 3. Dual hands (Left hand + Right hand)
    tracker.hands.process = lambda rgb: MockResults([("Left", 0.30, 0.5), ("Right", 0.70, 0.5)])
    _, is_detected, _, handedness = tracker.process_frame(dummy_frame, draw=True)
    lh, rh = tracker.get_dual_hands()
    assert handedness == "Dual"
    assert abs(lh[0]["x"] - 0.30) < 1e-4
    assert abs(rh[0]["x"] - 0.70) < 1e-4

    # 4. Dual hands in reversed detection order (Detector returns Right hand first, Left hand second)
    tracker.hands.process = lambda rgb: MockResults([("Right", 0.70, 0.5), ("Left", 0.30, 0.5)])
    _, is_detected, _, handedness = tracker.process_frame(dummy_frame, draw=True)
    lh, rh = tracker.get_dual_hands()
    assert handedness == "Dual"
    assert abs(lh[0]["x"] - 0.30) < 1e-4, "Left hand must correctly map to Left landmarks regardless of detection index"
    assert abs(rh[0]["x"] - 0.70) < 1e-4, "Right hand must correctly map to Right landmarks regardless of detection index"

    tracker.release()
    print("  -> PASSED: Left and Right hands correctly identified and mapped with zero inversions.")


def run_all_tests():
    print("=" * 70)
    print("   HandSynth Automated Test Suite: Two-Handed Chords & Variations")
    print("=" * 70)

    test_1_left_hand_chord_classification_and_open_palm()
    test_2_left_hand_tilt_major_minor_switching()
    test_3_right_hand_finger_pointing_variations()
    test_4_standalone_left_hand_and_dual_vca()
    test_5_synth_engine_polyphony_and_headroom()
    test_6_ui_workstation_rendering_preserved()
    test_7_pitch_stability_and_right_tilt_deadband()
    test_8_debouncing_and_pointing_immunity()
    test_9_major_minor_hysteresis_and_debouncing()
    test_10_scale_chords_and_octave_thumb()
    test_11_comprehensive_thumb_invariance()
    test_12_hand_tracker_handedness_classification()

    print("=" * 70)
    print("   ALL TEST SUITES COMPLETED WITH 100% PASS RATE!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
