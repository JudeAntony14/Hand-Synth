"""
test_musical_mapping.py
Comprehensive Verification Suite for Musical Note Mapping and Key Selection.
Verifies:
1. Default Root Key is C and functional Key Selector.
2. Support for all 7 Major Scales (C, D, E, F, G, A, B).
3. Scale purity: 100% of generated notes stay strictly within the selected scale.
4. Correct note names and octave formatting (including sharps and flats like F#4, Bb4).
5. Pitch quantizer hysteresis: prevents flutter from micro-tremors.
6. Immediate note updating upon changing the selected key.
"""

import math
from gesture_controller import GestureController
from synth_engine import SynthEngine


def test_musical_mapping_suite():
    print("=" * 70)
    print("   HandSynth - Musical Note Mapping & Key Selector Test Suite   ")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Test 1: Key Selector Defaults and Key Switching
    # -----------------------------------------------------------------
    print("\n[Test 1/6] Verifying Default Musical Key and Key Switching...")
    gc = GestureController()
    assert gc.key == "C", f"Default key must be 'C', got '{gc.key}'"
    print("  -> Default Key is 'C': [VERIFIED]")

    for target_key in ["D", "E", "F", "G", "A", "B", "C"]:
        gc.set_key(target_key)
        assert gc.key == target_key, f"Failed to set key to '{target_key}'"
    print("  -> Key Selector supports all 7 diatonic root keys (C, D, E, F, G, A, B): [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 2: Scale Note Purity Across All 7 Major Scales
    # -----------------------------------------------------------------
    print("\n[Test 2/6] Verifying Scale Note Purity Across All 7 Major Scales...")
    gc.set_scale("MAJOR")

    MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]

    for root_key in ["C", "D", "E", "F", "G", "A", "B"]:
        gc.set_key(root_key)
        root_offset = gc.NOTE_NAMES.index(root_key)
        expected_semitones = {(root_offset + i) % 12 for i in MAJOR_INTERVALS}

        # Sweep 50 frequencies across 3 octaves
        for f in [130.81 * (2 ** (step / 12.0)) for step in range(37)]:
            snapped_freq, note_name, _ = gc.snap_to_scale(f)
            midi_round = round(69.0 + 12.0 * math.log2(snapped_freq / 440.0))
            pitch_class = midi_round % 12

            assert pitch_class in expected_semitones, (
                f"Key {root_key} Major played note outside scale: {note_name} (pitch class {pitch_class})"
            )
        print(f"  -> Key '{root_key}' Major Scale: 100% of swept notes are strictly in scale [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 3: Note Name and Octave Formatting
    # -----------------------------------------------------------------
    print("\n[Test 3/6] Verifying Note Name and Octave Display Formatting...")
    # Key C Major: MIDI 60 -> C4, 62 -> D4, 71 -> B4
    gc.set_key("C")
    gc.set_scale("MAJOR")
    assert gc.format_note_name(60) == "C4"
    assert gc.format_note_name(62) == "D4"
    assert gc.format_note_name(71) == "B4"

    # Key F Major: MIDI 70 must be Bb4 (not A#4)
    gc.set_key("F")
    assert gc.format_note_name(70) == "Bb4", f"Expected Bb4 in F major, got {gc.format_note_name(70)}"

    # Key G Major: MIDI 66 must be F#4
    gc.set_key("G")
    assert gc.format_note_name(66) == "F#4", f"Expected F#4 in G major, got {gc.format_note_name(66)}"
    print("  -> Correct enharmonic note names and octave formatting (e.g. C4, Bb4, F#4): [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 4: Pitch Quantizer Hysteresis (Deadband Stability)
    # -----------------------------------------------------------------
    print("\n[Test 4/6] Verifying Pitch Quantizer Hysteresis...")
    gc.reset()
    gc.set_key("C")
    gc.set_scale("MAJOR")

    # Start at C4 (MIDI 60 = 261.63 Hz)
    f_c4 = 261.63
    snapped, name, _ = gc.snap_to_scale(f_c4)
    assert name == "C4"

    # Fluctuate slightly upwards near boundary of 61.0 (e.g. MIDI 60.9 = 275.6 Hz)
    f_jitter_up = 440.0 * (2.0 ** ((60.90 - 69.0) / 12.0))
    _, name_jitter1, _ = gc.snap_to_scale(f_jitter_up)
    assert name_jitter1 == "C4", f"Hysteresis failed! Switched prematurely to {name_jitter1}"

    # Micro-fluctuation back down
    f_jitter_down = 440.0 * (2.0 ** ((60.75 - 69.0) / 12.0))
    _, name_jitter2, _ = gc.snap_to_scale(f_jitter_down)
    assert name_jitter2 == "C4"

    # Deliberate intentional movement across threshold (MIDI 61.3 = 282.1 Hz)
    f_cross = 440.0 * (2.0 ** ((61.30 - 69.0) / 12.0))
    _, name_cross, _ = gc.snap_to_scale(f_cross)
    assert name_cross == "D4", f"Intentional note transition failed, got {name_cross}"
    print("  -> Hysteresis deadband eliminates landmark jitter and note flickering: [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 5: Immediate Note & Audio Update on Key Change
    # -----------------------------------------------------------------
    print("\n[Test 5/6] Verifying Immediate Pitch Update on Key Change...")
    synth = SynthEngine(sample_rate=44100, block_size=256)
    synth.set_instrument("sine")

    # Hand at pitch 261.63 Hz in Key C -> plays C4 (261.63 Hz)
    gc.set_key("C")
    gc._current_raw_pitch = 261.63
    p1, n1, _ = gc.snap_to_scale(gc._current_raw_pitch)
    synth.set_frequency(p1)
    assert n1 == "C4"
    assert abs(synth._target_frequency - 261.63) < 0.1

    # Switch Key to D -> In D Major, C is not in scale; closest is D (293.66 Hz) or C# (277.18 Hz)
    gc.set_key("D")
    p2, n2, _ = gc.snap_to_scale(gc._current_raw_pitch)
    synth.set_frequency(p2)
    assert n2 in ("D4", "C#4")
    assert synth._target_frequency != 261.63
    print(f"  -> Changing key immediately shifted note from {n1} ({p1:.1f}Hz) to {n2} ({p2:.1f}Hz): [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 6: Gesture Controller Mock Landmark Pipeline
    # -----------------------------------------------------------------
    print("\n[Test 6/6] Verifying Real-Time Gesture Controller Loop with New Mapping...")
    def make_mock(x, y):
        lms = [{"id": i, "x": x, "y": y, "z": 0.0, "px": int(x * 640), "py": int(y * 480)} for i in range(21)]
        lms[4] = {"id": 4, "x": x - 0.015, "y": y, "z": 0.0, "px": int((x - 0.015) * 640), "py": int(y * 480)}
        lms[8] = {"id": 8, "x": x + 0.015, "y": y, "z": 0.0, "px": int((x + 0.015) * 640), "py": int(y * 480)}
        return lms

    gc.reset()
    gc.set_key("C")
    gc.set_scale("MAJOR")
    res = gc.process(make_mock(0.50, 0.30), is_hand_present=True)
    assert res["is_pinched"] is True
    assert res["key"] == "C"
    assert res["scale"] == "MAJOR"
    assert len(res["nearest_note"]) >= 2
    print(f"  -> Live Pipeline: {res['nearest_note']} ({res['pitch_hz']:.1f}Hz) Key={res['key']} [VERIFIED]")

    print("\n" + "=" * 70)
    print("ALL MUSICAL MAPPING & KEY SELECTION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    test_musical_mapping_suite()
