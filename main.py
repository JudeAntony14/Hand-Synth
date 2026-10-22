"""
main.py
Main Application Coordinator for HandSynth.
Integrates:
- Live 21-landmark MediaPipe Hand Tracking (mirror mode, CPU-optimized)
- Gesture controller with scale quantization (Free, Chromatic, Major, Minor, Pentatonic, Blues)
- Multi-gesture recognition (Pinch trigger, Peace sign octave shift, Fist mute)
- Multi-instrument real-time DSP synthesizer (Sine, Sawtooth, Square)
- Real-time Delay / Echo audio effect
- Comprehensive OpenCV HUD with visual telemetry and keyboard hotkeys
- Resilient error handling (missing camera, audio recovery, graceful shutdown)
"""

import sys
import time
from typing import Optional
import cv2
import numpy as np

from hand_tracker import HandTracker
from gesture_controller import GestureController
from synth_engine import SynthEngine
from ui import UIOverlay
from two_hand_tracker import TwoHandSynthesizerState
from ai_classifier import AIGestureClassifier


def initialize_camera(preferred_idx: int = 0, width: int = 640, height: int = 480) -> Optional[cv2.VideoCapture]:
    """
    Safely initialize and configure the default webcam with fallback support.

    Returns:
        Configured cv2.VideoCapture instance, or None if camera cannot be accessed.
    """
    print(f"[HandSynth] Attempting to open webcam index {preferred_idx}...")
    # On Windows, try cv2.CAP_DSHOW first for instant hardware startup (< 0.4s)
    cap = cv2.VideoCapture(preferred_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(preferred_idx)

    # Fallback to index 1 if default index fails
    if not cap.isOpened():
        print(f"[HandSynth] Camera index {preferred_idx} unavailable. Attempting index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[HandSynth] Warning: No physical webcam could be accessed.", file=sys.stderr)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[HandSynth] Webcam active: {actual_w}x{actual_h} @ 30 FPS")
    return cap


def main():
    """Main execution loop for HandSynth."""
    print("=" * 70)
    print("   HandSynth: Real-Time Hand-Gesture Musical Synthesizer       ")
    print("=" * 70)
    print("Quick Controls & Gestures Guide:")
    print(" - Two-Hand Mode [C]: Toggle Two-Hand Chords <-> Single-Hand Solo")
    print(" - Fullscreen   [F]: Toggle Fullscreen / Windowed Mode (resizes cleanly)")
    print("   * LEFT HAND   : 1-5 fingers -> Chords 1-5 | Rock 🤘 -> Chord 6 | Rock+Thumb 🤟 -> Chord 7")
    print("   * LEFT TILT   : Smooth Chord Gain (0.05 to 1.0) without changing chords")
    print("   * RIGHT HAND  : Continuous Pitch Transposition (Tilt -12 to +12 st) & Volume (Height)")
    print("   * DUAL-VCA    : Final Amplitude = LeftGain * RightVolume")
    print(" - AI Recording [R]: Record current left hand pose into custom dataset")
    print(" - AI Training  [G]: Train neural network gesture classifier")
    print(" - Root Key     [K]: Cycle Key (C -> D -> E -> F -> G -> A -> B)")
    print(" - Sharp        [#]: Toggle Sharp / Flat Transposition")
    print(" - Scale        [S]: Cycle scale (Pentatonic -> Blues -> Major -> Minor -> Chromatic)")
    print(" - Tone Presets [T]: Cycle manual presets (Sub -> Warm -> Crisp -> Bright)")
    print(" - Tone Source  [O]: Toggle Tone Mode (MANUAL Slider <-> GESTURE Hand Tilt)")
    print(" - Arp          [A]: Cycle Arpeggiator (OFF -> UP -> BOUNCE)")
    print(" - Solo Mode    [M]: Toggle Pinch-Triggered vs Continuous Theremin Mode")
    print(" - Instruments     : [1] Sine  |  [2] Sawtooth  |  [3] Square  |  [4] Triangle")
    print(" - Effect       [E]: Toggle Delay / Echo effect ON / OFF")
    print(" - Backup  [SPACE] : Manual note toggle for keyboard testing")
    print(" - Mouse Controls  : Direct click on any button or drag Tone slider!")
    print(" - Exit    [Q/ESC] : Safely release all camera and audio hardware")
    print("-" * 70)

    cap: Optional[cv2.VideoCapture] = None
    tracker: Optional[HandTracker] = None
    synth: Optional[SynthEngine] = None
    ui: Optional[UIOverlay] = None

    try:
        # 1. Initialize Camera (with graceful offline fallback)
        cap = initialize_camera(preferred_idx=0)
        camera_error = (cap is None)

        # 2. Initialize Hand Tracker (Dual-Hand support)
        tracker = HandTracker(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # 2b. Initialize AI Classifier & Two-Hand State Coordinator
        ai_classifier = AIGestureClassifier()
        two_hand_state = TwoHandSynthesizerState(
            timeout_ms=500.0,
            smoothing_alpha=0.22,
            ai_classifier=ai_classifier,
        )
        performance_mode = "TWO_HAND_CHORD"  # "TWO_HAND_CHORD" or "SINGLE_HAND_SOLO"

        # 3. Initialize Gesture Controller
        controller = GestureController(
            f_min=130.81,        # C3
            f_max=1046.50,       # C6
            pinch_on_thresh=0.045,
            pinch_off_thresh=0.070,
            smoothing_alpha=0.35,
            mode="PINCH",
            scale="PENTATONIC",  # Default to Pentatonic (instantly musical!)
        )

        # 4. Initialize Real-Time Audio Engine (with graceful audio fallback)
        audio_available = True
        try:
            synth = SynthEngine(sample_rate=44100, block_size=256, fade_time=0.015)
            synth.start()
        except Exception as audio_err:
            print(f"[HandSynth] Audio Warning: Could not initialize audio output ({audio_err}).", file=sys.stderr)
            print("[HandSynth] Running in visual-only mode.", file=sys.stderr)
            audio_available = False

        # 5. Initialize UI Overlay
        ui = UIOverlay()

        window_name = "HandSynth - Gesture Musical Synthesizer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1240, 720)

        # Mouse Interaction Callback (Buttons, Toggles, and Sliders)
        def on_mouse(event, x, y, flags, param):
            nonlocal performance_mode
            if event == cv2.EVENT_LBUTTONDOWN:
                mx1, my1, mx2, my2 = ui.btn_mode_header
                if mx1 <= x <= mx2 and my1 <= y <= my2:
                    if performance_mode == "TWO_HAND_CHORD":
                        performance_mode = "SINGLE_HAND_SOLO"
                    else:
                        performance_mode = "TWO_HAND_CHORD"
                    print(f"[HandSynth] Performance Mode: {performance_mode}")
                    if audio_available and synth:
                        synth.note_off()
                    return
                ui.handle_click(x, y, controller, synth)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                ui.handle_drag(x, y, synth)

        cv2.setMouseCallback(window_name, on_mouse)

        manual_note_override = False
        is_fullscreen = False
        consecutive_frame_drops = 0

        print("[HandSynth] System active and ready. Enjoy playing HandSynth!\n")

        # Main Real-Time Loop
        while True:
            # Handle camera frame grab
            if cap and cap.isOpened() and not camera_error:
                success, raw_frame = cap.read()
                if not success:
                    consecutive_frame_drops += 1
                    if consecutive_frame_drops > 10:
                        camera_error = True
                    time.sleep(0.01)
                    continue
                consecutive_frame_drops = 0
            else:
                raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                camera_error = True
                time.sleep(0.03)

            # 1. Hand Tracking Pipeline (Mirror flip + 21-landmark inference on up to 2 hands)
            annotated_frame, is_detected, landmarks, handedness = tracker.process_frame(
                raw_frame, draw=True
            )
            left_lms, right_lms = tracker.get_dual_hands()

            # 2. Performance Mode & Gesture Interpretation Pipeline
            if performance_mode == "TWO_HAND_CHORD":
                two_hand_out = two_hand_state.update(
                    left_lms,
                    right_lms,
                    current_time=time.time(),
                    key_name=controller.key,
                    scale_name=controller.scale,
                )
                final_amp = two_hand_out["final_amplitude"]
                ch_freqs = two_hand_out["frequencies"]

                # Thread-Safe Audio Updates
                if audio_available and synth:
                    if manual_note_override:
                        if ch_freqs:
                            synth.set_chord_frequencies(ch_freqs)
                        synth.set_volume(0.25)
                        synth.note_on()
                    elif final_amp > 0.001 and ch_freqs:
                        synth.set_chord_frequencies(ch_freqs)
                        synth.set_volume(final_amp)
                        synth.note_on()
                    else:
                        synth.note_off()

                gesture_label = two_hand_out["gesture_name"]
                if two_hand_out.get("is_right_present"):
                    if two_hand_out.get("is_right_thumb_up"):
                        gesture_label = f"{gesture_label} [8va Up]"
                    if two_hand_out.get("right_variation") != "TRIAD":
                        gesture_label = f"{gesture_label} + {two_hand_out['right_variation_name']}"

                gesture = {
                    "is_note_active": (final_amp > 0.01) or manual_note_override,
                    "pitch_hz": ch_freqs[0] if ch_freqs else 261.63,
                    "nearest_note": two_hand_out["chord_name"],
                    "cents": two_hand_out["pitch_shift_semitones"] * 100.0,
                    "volume": 0.25 if manual_note_override else final_amp,
                    "is_pinched": False,
                    "mode": "TWO-HAND CHORD",
                    "scale": controller.scale,
                    "key": controller.key,
                    "gesture": gesture_label,
                    "octave_shift": 0,
                    "tone_source": controller.tone_source,
                    "tone_val": None,
                }
            else:
                # Single-Hand Solo Mode
                gesture = controller.process(landmarks, is_detected)

                if audio_available and synth:
                    synth.set_frequency(gesture["pitch_hz"])
                    if gesture.get("tone_val") is not None:
                        synth.set_tone(gesture["tone_val"])

                    if manual_note_override:
                        synth.set_volume(0.25)
                        synth.note_on()
                    else:
                        synth.set_volume(gesture["volume"])
                        if gesture["note_event"] == "NOTE_ON":
                            synth.note_on()
                        elif gesture["note_event"] == "NOTE_OFF":
                            synth.note_off()

            # 4. Query Synth Telemetry for HUD
            synth_status = synth.get_status() if (audio_available and synth) else {
                "instrument": "sine",
                "effect_echo": False,
            }

            # 5. Render Comprehensive HUD Overlay
            display_frame = ui.draw_hud(
                frame=annotated_frame,
                fps=tracker.fps,
                is_hand_detected=is_detected,
                handedness=handedness,
                gesture_data=gesture,
                synth_data=synth_status,
                landmarks=landmarks,
                camera_error=camera_error,
            )

            # 6. Display to Window
            cv2.imshow(window_name, display_frame)

            # 7. Keyboard Shortcuts Handling
            key = cv2.waitKey(1) & 0xFF

            # Exit on 'q', 'Q', or ESC (27)
            if key in (ord("q"), ord("Q"), 27):
                print("[HandSynth] Exit request received.")
                break

            # Root Key Cycling on 'k' / 'K'
            elif key in (ord("k"), ord("K")):
                new_key = controller.cycle_key()
                print(f"[HandSynth] Musical Key: {new_key}")
                if audio_available and synth and hasattr(controller, "_current_raw_pitch"):
                    new_pitch, _, _ = controller.snap_to_scale(controller._current_raw_pitch)
                    synth.set_frequency(new_pitch)

            # Sharp / Flat Transpose on '#'
            elif key == ord("#"):
                cur_key = controller.key
                base = cur_key.replace("#", "")
                new_key = base if cur_key.endswith("#") else f"{base}#"
                controller.set_key(new_key)
                print(f"[HandSynth] Musical Key: {controller.key}")
                if audio_available and synth and hasattr(controller, "_current_raw_pitch"):
                    new_pitch, _, _ = controller.snap_to_scale(controller._current_raw_pitch)
                    synth.set_frequency(new_pitch)

            # Scale Cycling on 's' / 'S'
            elif key in (ord("s"), ord("S")):
                new_scale = controller.cycle_scale()
                print(f"[HandSynth] Musical Scale: {new_scale}")
                if audio_available and synth and hasattr(controller, "_current_raw_pitch"):
                    new_pitch, _, _ = controller.snap_to_scale(controller._current_raw_pitch)
                    synth.set_frequency(new_pitch)

            # Tone Filter Cycling on 't' / 'T'
            elif key in (ord("t"), ord("T")):
                if audio_available and synth:
                    new_tone = synth.cycle_tone()
                    print(f"[HandSynth] Tone / Cutoff Filter: {int(new_tone * 100)}%")

            # Tone Modulation Source Toggle on 'o' / 'O'
            elif key in (ord("o"), ord("O")):
                new_src = controller.cycle_tone_source()
                print(f"[HandSynth] Tone Control Source: {new_src}")

            # Arpeggiator Cycling on 'a' / 'A'
            elif key in (ord("a"), ord("A")):
                if audio_available and synth:
                    new_arp = synth.cycle_arpeggiator()
                    print(f"[HandSynth] Arpeggiator: {new_arp}")

            # Mode Toggle: 'm' / 'M'
            elif key in (ord("m"), ord("M")):
                new_mode = controller.toggle_mode()
                print(f"[HandSynth] Switched to {new_mode} mode.")
                if new_mode == "PINCH" and audio_available and synth:
                    synth.note_off()

            # Instrument Selectors: '1', '2', '3', '4'
            elif key == ord("1"):
                if audio_available and synth:
                    inst = synth.set_instrument("sine")
                    print(f"[HandSynth] Instrument: {inst.upper()} (Pure Warm Tone)")
            elif key == ord("2"):
                if audio_available and synth:
                    inst = synth.set_instrument("sawtooth")
                    print(f"[HandSynth] Instrument: {inst.upper()} (Brassy Rich Harmonics)")
            elif key == ord("3"):
                if audio_available and synth:
                    inst = synth.set_instrument("square")
                    print(f"[HandSynth] Instrument: {inst.upper()} (Hollow 8-Bit Pulse)")
            elif key == ord("4"):
                if audio_available and synth:
                    inst = synth.set_instrument("triangle")
                    print(f"[HandSynth] Instrument: {inst.upper()} (Warm Electric Piano / Flute)")

            # Audio Effect Toggle: 'e' / 'E' (Echo / Delay)
            elif key in (ord("e"), ord("E")):
                if audio_available and synth:
                    state = synth.toggle_echo()
                    state_str = "ENABLED" if state else "DISABLED"
                    print(f"[HandSynth] Audio Effect: Delay/Echo {state_str}")

            # Manual Note Override: SPACEBAR
            elif key == ord(" "):
                manual_note_override = not manual_note_override
                if audio_available and synth:
                    if manual_note_override:
                        synth.note_on(volume=0.25)
                        print("[HandSynth] [SPACE] Manual note override: ACTIVE")
                    else:
                        synth.note_off()
                        print("[HandSynth] [SPACE] Manual note override: RELEASED")

            # Toggle Fullscreen: 'f' / 'F'
            elif key in (ord("f"), ord("F")):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("[HandSynth] Fullscreen: ENABLED (Press [F] to toggle)")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1240, 720)
                    print("[HandSynth] Fullscreen: DISABLED")

            # Toggle Performance Mode: 'c' / 'C'
            elif key in (ord("c"), ord("C")):
                if performance_mode == "TWO_HAND_CHORD":
                    performance_mode = "SINGLE_HAND_SOLO"
                else:
                    performance_mode = "TWO_HAND_CHORD"
                print(f"[HandSynth] Performance Mode: {performance_mode}")
                if audio_available and synth:
                    synth.note_off()

            # Record Hand Pose to AI Dataset: 'r' / 'R'
            elif key in (ord("r"), ord("R")):
                active_lms = left_lms or landmarks
                if active_lms and len(active_lms) == 21:
                    chord_idx = two_hand_state.active_chord_index
                    if chord_idx is not None and 1 <= chord_idx <= 7:
                        class_names = [
                            "CHORD_1_INDEX", "CHORD_2_PEACE", "CHORD_3_THREE",
                            "CHORD_4_FOUR", "CHORD_5_OPEN_PALM", "CHORD_6_ROCK", "CHORD_7_ROCK_THUMB"
                        ]
                        target_class = class_names[chord_idx - 1]
                        ai_classifier.dataset.add_sample(target_class, active_lms)
                        ai_classifier.dataset.save()
                        print(f"[HandSynth AI] Recorded sample for {target_class}! (Total: {ai_classifier.dataset.total_count()})")
                    else:
                        print("[HandSynth AI] Form a valid chord gesture (1-7) before pressing [R].")
                else:
                    print("[HandSynth AI] No hand visible to record.")

            # Retrain AI Neural Network Classifier: 'g' / 'G'
            elif key in (ord("g"), ord("G")):
                print("[HandSynth AI] Training neural network gesture classifier...")
                try:
                    metrics = ai_classifier.train()
                    print(f"[HandSynth AI] Training Complete! Accuracy: {metrics['accuracy']:.1%}, F1-Score: {metrics['f1_score']:.1%}")
                except Exception as train_err:
                    print(f"[HandSynth AI] Training failed: {train_err}")

            # Check if user clicked window close button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[HandSynth] Video window closed.")
                break

    except KeyboardInterrupt:
        print("\n[HandSynth] Interrupted by keyboard.")
    except Exception as exc:
        print(f"\n[HandSynth] Unhandled Runtime Exception: {exc}", file=sys.stderr)
        raise
    finally:
        # Guaranteed Resource Cleanup
        print("\n[HandSynth] Initiating guaranteed resource teardown...")
        if synth:
            try:
                synth.stop()
            except Exception as e:
                print(f"[HandSynth] Error stopping synth: {e}")
        if tracker:
            try:
                tracker.release()
            except Exception as e:
                print(f"[HandSynth] Error releasing tracker: {e}")
        if cap and cap.isOpened():
            try:
                cap.release()
            except Exception as e:
                print(f"[HandSynth] Error releasing camera: {e}")
        cv2.destroyAllWindows()
        print("[HandSynth] All audio streams, camera handles, and windows closed cleanly.")


if __name__ == "__main__":
    main()
