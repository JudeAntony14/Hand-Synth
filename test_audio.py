"""
test_audio.py
Independent Audio Verification Test for HandSynth (Milestone 3).
Tests real-time audio playback, portamento pitch glide, and click-free note on/off fading.
Runs without requiring webcam input.
"""

import time
import sys
from synth_engine import SynthEngine


def run_audio_test():
    print("=" * 65)
    print("      HandSynth - Milestone 3: Real-Time Audio Engine Test      ")
    print("=" * 65)
    print("Testing Sound Device output and NumPy synthesis...")

    synth = SynthEngine(sample_rate=44100, block_size=256, fade_time=0.015)

    try:
        # Step 1: Start Stream
        print("\n[Test 1/4] Starting audio stream (non-blocking callback)...")
        synth.start()
        time.sleep(0.1)
        assert synth._is_running, "Audio stream failed to enter running state"
        print("  -> Stream started successfully.")

        # Step 2: Play Constant Pitch (A4 = 440 Hz)
        print("\n[Test 2/4] Triggering Note-On: 440.0 Hz (Concert A4) at 25% volume...")
        print("  -> Listen for a smooth sine tone without any initial 'click' or 'pop'.")
        synth.note_on(freq=440.0, volume=0.25)
        time.sleep(1.2)

        # Step 3: Test Smooth Frequency Glide (Portamento Pitch Sweep)
        print("\n[Test 3/4] Sweeping pitch from 440.0 Hz up to 880.0 Hz (A5)...")
        print("  -> Listen for continuous, seamless pitch slide without distortion.")
        start_time = time.time()
        sweep_duration = 1.5
        while time.time() - start_time < sweep_duration:
            elapsed = time.time() - start_time
            progress = elapsed / sweep_duration
            # Exponential pitch sweep for natural musical octave curve
            current_f = 440.0 * (2.0 ** progress)
            synth.set_frequency(current_f)
            time.sleep(0.01)

        print(f"  -> Arrived at {synth._target_frequency:.1f} Hz.")
        time.sleep(0.5)

        # Step 4: Test Note-Off Fade-out (De-clicking verification)
        print("\n[Test 4/4] Triggering Note-Off (Release fade-out)...")
        print("  -> Listen for a smooth release fade ending in absolute silence.")
        synth.note_off()
        time.sleep(0.3)
        assert not synth.is_playing(), "Audio failed to fade out completely"
        print("  -> Tone silenced cleanly with no release click.")

        # Brief second note test (Middle C = 261.63 Hz)
        print("\n[Bonus Check] Triggering Middle C (261.6 Hz) for 0.8s...")
        synth.note_on(freq=261.63, volume=0.25)
        time.sleep(0.8)
        synth.note_off()
        time.sleep(0.3)

    except Exception as e:
        print(f"\n[Test Error] Audio test encountered an exception: {e}", file=sys.stderr)
        raise
    finally:
        print("\n[Teardown] Stopping audio stream...")
        synth.stop()
        print("  -> Audio device released cleanly.")

    print("\n" + "=" * 65)
    print("STATUS: ALL SYNTH ENGINE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_audio_test()
