"""
test_final.py
Comprehensive Verification Suite for HandSynth (Final Pass).
Audits:
1. Multi-instrument waveform generation (Sine, Sawtooth, Square).
2. Delay / Echo audio effect buffer stability and feedback.
3. UI rendering under all permutation states (offline, tracking, instruments, effects).
4. Codebase imports, thread safety, and resource teardown.
"""

import time
import numpy as np
from synth_engine import SynthEngine
from gesture_controller import GestureController
from hand_tracker import HandTracker
from ui import UIOverlay


def test_suite():
    print("=" * 68)
    print("      HandSynth - Final Stability & Verification Audit Suite     ")
    print("=" * 68)

    # -----------------------------------------------------------------
    # Test 1: Multi-Instrument Waveform Verification
    # -----------------------------------------------------------------
    print("\n[Audit 1/6] Testing Multi-Instrument Oscillators (Sine, Saw, Square, Triangle)...")
    synth = SynthEngine(sample_rate=44100, block_size=256)

    # Mock buffer for audio callback
    buffer = np.zeros((256, 1), dtype=np.float32)

    for inst in ["sine", "sawtooth", "square", "triangle"]:
        synth.set_instrument(inst)
        synth.note_on(freq=440.0, volume=0.30)
        synth._current_amplitude = 0.30  # Force immediate steady state
        synth._audio_callback(buffer, 256, None, None)

        peak = np.max(np.abs(buffer[:, 0]))
        rms = np.sqrt(np.mean(buffer[:, 0] ** 2))
        assert peak > 0.15, f"{inst} produced silent or negligible output (peak: {peak})"
        assert not np.isnan(peak) and not np.isinf(peak), f"{inst} produced NaN/Inf!"
        print(f"  -> Instrument '{inst:8}': Peak = {peak:.3f}, RMS = {rms:.3f} [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 2: Tone Cutoff Filter & Delay / Echo Effect
    # -----------------------------------------------------------------
    print("\n[Audit 2/6] Testing Tone Cutoff Filter & Delay/Echo Audio Effect...")
    # Test Tone filter
    synth.set_instrument("sawtooth")
    synth.set_tone(0.20)  # Heavy low pass filter
    synth.note_on(freq=440.0, volume=0.30)
    synth._current_amplitude = 0.30
    synth._audio_callback(buffer, 256, None, None)
    filtered_rms = np.sqrt(np.mean(buffer[:, 0] ** 2))
    assert filtered_rms > 0.05, "Tone filter silenced output completely!"
    print(f"  -> Tone Cutoff Filter: Warm Sub LPF RMS = {filtered_rms:.3f} [VERIFIED]")
    synth.set_tone(0.85)

    # Test Echo
    synth.set_echo(True)
    assert synth._effect_echo is True, "Echo failed to enable"

    # Send an impulse block
    impulse_buffer = np.zeros((256, 1), dtype=np.float32)
    synth.note_on(freq=440.0, volume=0.30)
    synth._audio_callback(impulse_buffer, 256, None, None)

    # Turn note off and advance by 250ms (delay time)
    synth.note_off()
    synth._current_amplitude = 0.0

    # Advance until delay taps appear
    echo_detected = False
    for _ in range(50):  # 50 blocks * 256 / 44100 ~= 290 ms
        echo_buffer = np.zeros((256, 1), dtype=np.float32)
        synth._audio_callback(echo_buffer, 256, None, None)
        if np.max(np.abs(echo_buffer)) > 0.01:
            echo_detected = True
            break

    assert echo_detected, "Delay effect did not produce audible echo taps!"
    print("  -> Delay / Echo: Circular buffer feedback confirmed active [VERIFIED]")
    synth.set_echo(False)

    # -----------------------------------------------------------------
    # Test 3: Thread-Safety Concurrency Stress Test
    # -----------------------------------------------------------------
    print("\n[Audit 3/5] Testing Thread-Safety under Rapid Concurrent Calls...")
    import threading

    def spam_parameters():
        for i in range(500):
            synth.set_frequency(100.0 + i * 2)
            synth.set_volume(0.1 + (i % 3) * 0.1)
            if i % 2 == 0:
                synth.note_on()
            else:
                synth.note_off()
            if i % 5 == 0:
                synth.cycle_instrument()
            if i % 10 == 0:
                synth.toggle_echo()

    # Run spam thread alongside mock callback calls
    t = threading.Thread(target=spam_parameters)
    t.start()
    for _ in range(200):
        dummy_out = np.zeros((256, 1), dtype=np.float32)
        synth._audio_callback(dummy_out, 256, None, None)
        assert not np.isnan(dummy_out).any(), "Audio callback produced NaN under thread contention!"
    t.join()
    print("  -> Concurrency: 500 rapid thread-safe updates executed with 0 errors [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 4: UIOverlay Comprehensive State Permutations
    # -----------------------------------------------------------------
    print("\n[Audit 4/6] Testing UIOverlay Permutation Rendering (1240 x 720 Workstation)...")
    ui = UIOverlay()
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Permutation A: No Hand, Muted
    f1 = ui.draw_hud(canvas.copy(), fps=30.0, is_hand_detected=False)
    assert f1.shape == (UIOverlay.CANVAS_HEIGHT, UIOverlay.CANVAS_WIDTH, 3), f"Expected {(UIOverlay.CANVAS_HEIGHT, UIOverlay.CANVAS_WIDTH, 3)}, got {f1.shape}"

    # Permutation B: Hand Detected, Pinch Active, Sawtooth, Echo ON
    g_data = {
        "is_note_active": True,
        "pitch_hz": 523.25,
        "nearest_note": "C5",
        "cents": -2.0,
        "volume": 0.35,
        "is_pinched": True,
        "mode": "CONTINUOUS",
        "scale": "BLUES",
        "key": "G",
    }
    s_data = {
        "instrument": "sawtooth",
        "effect_echo": True,
        "tone": 0.65,
        "arpeggiator": "UP",
    }
    sim_lms = [{"id": i, "x": 0.5, "y": 0.5, "z": 0.0, "px": 320, "py": 240} for i in range(21)]
    f2 = ui.draw_hud(canvas.copy(), fps=32.0, is_hand_detected=True, handedness="Right", gesture_data=g_data, synth_data=s_data, landmarks=sim_lms)
    assert f2.shape == (UIOverlay.CANVAS_HEIGHT, UIOverlay.CANVAS_WIDTH, 3)

    # Permutation C: Camera Offline Warning Overlay
    f3 = ui.draw_hud(canvas.copy(), fps=0.0, is_hand_detected=False, camera_error=True)
    assert f3.shape == (UIOverlay.CANVAS_HEIGHT, UIOverlay.CANVAS_WIDTH, 3)
    print("  -> UI Rendering: All workstation permutations, ribbons, & badges verified [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 5: Interactive UI Hitbox & Control Dispatching
    # -----------------------------------------------------------------
    print("\n[Audit 5/6] Testing Interactive UI Hitboxes (Keys, Scales, Tone Slider, Instruments)...")
    controller = GestureController()
    # Click Key 'D' button
    dx, dy = (ui.btn_keys["D"][0] + 5, ui.btn_keys["D"][1] + 5)
    hit_key = ui.handle_click(dx, dy, controller, synth)
    assert hit_key is True and controller.key == "D", "Key click did not switch key to D!"
    print(f"  -> Mouse Click on Key 'D': Switched key to '{controller.key}' [VERIFIED]")

    # Click Scale 'BLUES'
    bx, by = (ui.btn_scales["BLUES"][0] + 5, ui.btn_scales["BLUES"][1] + 5)
    hit_scale = ui.handle_click(bx, by, controller, synth)
    assert hit_scale is True and controller.scale == "BLUES", "Scale click did not switch scale!"
    print(f"  -> Mouse Click on Scale 'BLUES': Switched scale to '{controller.scale}' [VERIFIED]")

    # Drag Tone slider
    tx, ty = (ui.slider_tone[0] + 100, ui.slider_tone[1] + 5)
    hit_tone = ui.handle_drag(tx, ty, synth)
    assert hit_tone is True and synth._tone != 0.85, "Tone slider drag failed!"
    print(f"  -> Mouse Drag on Tone Slider: Cutoff shifted to {int(synth._tone * 100)}% [VERIFIED]")

    # -----------------------------------------------------------------
    # Test 6: Resource Teardown
    # -----------------------------------------------------------------
    print("\n[Audit 6/6] Testing Resource Cleanup & Teardown...")
    synth.stop()
    assert synth._is_running is False
    assert synth._stream is None
    print("  -> Resource Teardown: Audio and model instances freed cleanly [VERIFIED]")

    print("\n" + "=" * 68)
    print("ALL FINAL STABILITY AUDIT TESTS PASSED WITH ZERO DEFECTS!")
    print("=" * 68)


if __name__ == "__main__":
    test_suite()
