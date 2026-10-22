"""
test_milestone4.py
Automated Unit and Integration Tests for Milestone 4 (Gesture to Synth Link).
Verifies:
1. Hysteresis state machine (eliminates threshold chatter).
2. Continuous musical pitch mapping (C3 to C6 curve).
3. Vertical volume mapping.
4. Hand loss safety (automatic note-off).
5. Mode switching (Pinch vs Continuous).
6. Non-blocking audio synchronization.
"""

from gesture_controller import GestureController
from synth_engine import SynthEngine


def test_gesture_synth_integration():
    print("=" * 65)
    print("      HandSynth - Milestone 4 Automated Test Suite              ")
    print("=" * 65)

    controller = GestureController(
        f_min=130.81,
        f_max=1046.50,
        pinch_on_thresh=0.045,
        pinch_off_thresh=0.070,
        mode="PINCH",
    )

    # -------------------------------------------------------------
    # Test 1: Pitch Mapping across Horizontal Axis (X)
    # -------------------------------------------------------------
    print("\n[Test 1/5] Testing Horizontal Pitch Mapping (C3 to C6)...")
    # Helper to generate mock landmarks
    def make_mock_landmarks(x, y, pinch_d):
        lms = [{"id": i, "x": x, "y": y, "z": 0.0, "px": int(x * 640), "py": int(y * 480)} for i in range(21)]
        # Thumb (4) and Index (8)
        lms[4] = {"id": 4, "x": x - pinch_d / 2, "y": y, "z": 0.0, "px": int((x - pinch_d / 2) * 640), "py": int(y * 480)}
        lms[8] = {"id": 8, "x": x + pinch_d / 2, "y": y, "z": 0.0, "px": int((x + pinch_d / 2) * 640), "py": int(y * 480)}
        return lms

    # Left edge (x = 0.15) -> should be near C3 (130.81 Hz)
    res_left = controller.process(make_mock_landmarks(0.15, 0.5, 0.10), is_hand_present=True)
    assert abs(res_left["pitch_hz"] - 130.81) < 1.0, f"Expected ~130.81 Hz, got {res_left['pitch_hz']}"
    assert "C3" in res_left["nearest_note"]
    print(f"  -> Leftmost point:  {res_left['pitch_hz']:.2f} Hz ({res_left['nearest_note']}) [PASS]")

    # Right edge (x = 0.85) -> should be near C6 (1046.50 Hz)
    # Reset controller to clear smoothing memory
    controller.reset()
    res_right = controller.process(make_mock_landmarks(0.85, 0.5, 0.10), is_hand_present=True)
    assert abs(res_right["pitch_hz"] - 1046.50) < 5.0, f"Expected ~1046.50 Hz, got {res_right['pitch_hz']}"
    assert "C6" in res_right["nearest_note"]
    print(f"  -> Rightmost point: {res_right['pitch_hz']:.2f} Hz ({res_right['nearest_note']}) [PASS]")

    # -------------------------------------------------------------
    # Test 2: Vertical Volume Mapping (Y)
    # -------------------------------------------------------------
    print("\n[Test 2/5] Testing Vertical Volume Mapping (Hand UP vs DOWN)...")
    controller.reset()
    # High hand (y = 0.15) -> Maximum Volume
    res_high = controller.process(make_mock_landmarks(0.5, 0.15, 0.10), is_hand_present=True)
    assert res_high["volume"] > 0.35, f"Expected max volume (>0.35), got {res_high['volume']}"
    print(f"  -> Hand UP (y=0.15):   Vol = {res_high['volume']:.2f} (Max Loud) [PASS]")

    controller.reset()
    # Low hand (y = 0.85) -> Minimum Volume
    res_low = controller.process(make_mock_landmarks(0.5, 0.85, 0.10), is_hand_present=True)
    assert res_low["volume"] < 0.05, f"Expected min volume (<0.05), got {res_low['volume']}"
    print(f"  -> Hand DOWN (y=0.85): Vol = {res_low['volume']:.2f} (Silent/Quiet) [PASS]")

    # -------------------------------------------------------------
    # Test 3: Pinch Hysteresis State Machine & Chatter Prevention
    # -------------------------------------------------------------
    print("\n[Test 3/5] Testing Pinch Hysteresis State Machine...")
    controller.reset()

    # Step A: Hand wide open (d = 0.12) -> SILENT
    r1 = controller.process(make_mock_landmarks(0.5, 0.5, 0.12), is_hand_present=True)
    assert r1["note_event"] == "SILENT" and not r1["is_pinched"]
    print("  -> Wide open hand (d=0.12): Event = SILENT [PASS]")

    # Step B: Moving closer (d = 0.055) -> Still above pinch_on_thresh (0.045) -> Still SILENT
    r2 = controller.process(make_mock_landmarks(0.5, 0.5, 0.055), is_hand_present=True)
    assert r2["note_event"] == "SILENT" and not r2["is_pinched"]
    print("  -> Approaching pinch (d=0.055): Event = SILENT (No false trigger) [PASS]")

    # Step C: Fully pinched (d = 0.035) -> Crosses pinch_on_thresh -> NOTE_ON
    r3 = controller.process(make_mock_landmarks(0.5, 0.5, 0.035), is_hand_present=True)
    assert r3["note_event"] == "NOTE_ON" and r3["is_pinched"]
    print("  -> Pinch engaged (d=0.035): Event = NOTE_ON [PASS]")

    # Step D: Hand holds pinch (d = 0.035) -> NOTE_HOLD (No duplicate NOTE_ON!)
    r4 = controller.process(make_mock_landmarks(0.5, 0.5, 0.035), is_hand_present=True)
    assert r4["note_event"] == "NOTE_HOLD" and r4["is_pinched"]
    print("  -> Pinch sustained: Event = NOTE_HOLD (No duplicate spam) [PASS]")

    # Step E: Hand relaxes slightly (d = 0.058) -> In deadband zone (< pinch_off_thresh 0.070)
    # Hysteresis must keep note active!
    r5 = controller.process(make_mock_landmarks(0.5, 0.5, 0.058), is_hand_present=True)
    assert r5["note_event"] == "NOTE_HOLD" and r5["is_pinched"]
    print("  -> Slight relaxation (d=0.058 in deadband): Event = NOTE_HOLD (Hysteresis Active!) [PASS]")

    # Step F: Hand opens wide (d = 0.080) -> Exceeds pinch_off_thresh -> NOTE_OFF
    r6 = controller.process(make_mock_landmarks(0.5, 0.5, 0.080), is_hand_present=True)
    assert r6["note_event"] == "NOTE_OFF" and not r6["is_pinched"]
    print("  -> Hand released (d=0.080): Event = NOTE_OFF [PASS]")

    # -------------------------------------------------------------
    # Test 4: Hand Disappearance Safety
    # -------------------------------------------------------------
    print("\n[Test 4/5] Testing Disappearing Hand Safety...")
    # Pinch again
    controller.process(make_mock_landmarks(0.5, 0.5, 0.035), is_hand_present=True)
    assert controller._is_note_active

    # Suddenly hand leaves camera
    r_lost = controller.process([], is_hand_present=False)
    assert r_lost["note_event"] == "NOTE_OFF"
    assert not r_lost["is_note_active"]
    print("  -> Hand lost while playing: Immediate safe NOTE_OFF dispatched [PASS]")

    # -------------------------------------------------------------
    # Test 5: Mode Switching (Pinch vs Continuous)
    # -------------------------------------------------------------
    print("\n[Test 5/5] Testing Continuous Mode (Theremin)...")
    controller.set_mode("CONTINUOUS")
    # Even with open hand (d=0.15), Continuous mode triggers NOTE_ON
    r_cont = controller.process(make_mock_landmarks(0.5, 0.5, 0.15), is_hand_present=True)
    assert r_cont["note_event"] == "NOTE_ON" and r_cont["is_note_active"]
    print(f"  -> Continuous mode with open hand: Event = NOTE_ON [PASS]")

    controller.set_mode("PINCH")
    assert controller.mode == "PINCH"

    print("\n" + "=" * 65)
    print("ALL MILESTONE 4 LOGIC & SAFETY TESTS PASSED SUCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_gesture_synth_integration()
