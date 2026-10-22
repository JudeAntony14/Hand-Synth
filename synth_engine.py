"""
synth_engine.py
Real-Time Software Synthesizer Engine for HandSynth.
Features:
- Non-blocking audio stream with sounddevice callback architecture.
- 3 distinct waveforms: Sine, Sawtooth, and Square.
- Stereo/Mono real-time Delay / Echo audio effect with feedback.
- Phase-continuous synthesis to eliminate phase distortion.
- Anti-click 15ms amplitude envelope attack/release.
- Portamento pitch frequency smoothing (glissando).
- Thread-safe parameter exchange using threading.Lock.
"""

import math
import threading
from typing import Optional, Dict, List
import numpy as np
import sounddevice as sd


class SynthEngine:
    """
    Real-time software synthesizer with multi-waveform synthesis and audio effects.
    DSP computation runs in a high-priority background audio thread.
    """

    SAMPLE_RATE = 44100
    BLOCK_SIZE = 256            # ~5.8 ms latency at 44.1 kHz
    DEFAULT_FADE_TIME = 0.015   # 15 ms envelope ramp to eliminate audio clicks
    DEFAULT_PITCH_SLEW = 0.050  # 50 ms portamento smoothing for natural glissando

    AVAILABLE_INSTRUMENTS = ["sine", "sawtooth", "square", "triangle"]
    ARP_MODES = ["OFF", "UP", "BOUNCE"]

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        fade_time: float = DEFAULT_FADE_TIME,
        **kwargs,
    ):
        """
        Initialize the synthesizer parameters, oscillators, and effect buffers.

        Args:
            sample_rate: Audio sampling frequency in Hz (default 44100).
            block_size: Buffer size per audio callback frame (default 256 samples).
            fade_time: Duration in seconds for note attack and release fades.
        """
        self.sample_rate = int(sample_rate)
        self.block_size = int(kwargs.get("blocksize", block_size))
        self.fade_time = float(fade_time)

        # Thread synchronization lock
        self._lock = threading.Lock()

        # Shared synthesis parameters (controlled by main thread)
        self._target_frequency = 440.0   # Hz (A4)
        self._chord_frequencies: List[float] = [] # Polyphonic chord frequencies
        self._target_volume = 0.25       # 0.0 to 1.0 (comfortable default)
        self._is_note_on = False         # Gate: True = sound on, False = fade to silent
        self._instrument = "sine"        # 'sine', 'sawtooth', 'square', 'triangle'
        self._effect_echo = False        # Delay / Echo effect enabled state
        self._tone = 0.75                # Cutoff filter: 0.0 (Warm/Sub) to 1.0 (Bright/Crisp)
        self._arpeggiator_mode = "OFF"   # 'OFF', 'UP', 'BOUNCE'

        # Internal DSP state (mutated only within the audio callback thread)
        self._current_phase = 0.0        # Continuous phase accumulator (0.0 to 2*pi)
        self._current_frequency = 440.0   # Interpolated frequency
        self._current_chord_phases = np.zeros(6, dtype=np.float32)
        self._current_chord_freqs = np.ones(6, dtype=np.float32) * 440.0
        self._current_amplitude = 0.0    # Interpolated envelope gain (0.0 = silent)
        self._filter_state_1 = 0.0       # 2-Pole Cascaded Low-Pass Filter Stage 1
        self._filter_state_2 = 0.0       # 2-Pole Cascaded Low-Pass Filter Stage 2
        self._filter_alpha = 1.0         # Slewed filter coefficient

        # Arpeggiator internal state
        self._arp_sample_counter = 0
        self._arp_step = 0
        self._arp_samples_per_step = int(self.sample_rate * 0.125)  # 125 ms (120 BPM 8th notes)

        # Delay / Echo Effect circular buffer (250 ms delay time)
        self.delay_samples = int(self.sample_rate * 0.250)
        self._delay_buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self._delay_write_ptr = 0
        self.delay_feedback = 0.40       # Decay factor of echoes
        self.delay_wet = 0.35            # Wet mix level

        # sounddevice OutputStream handle
        self._stream: Optional[sd.OutputStream] = None
        self._is_running = False

    # -------------------------------------------------------------------------
    # Public Thread-Safe Control API (Called from Main / Camera Thread)
    # -------------------------------------------------------------------------

    def set_frequency(self, freq: float):
        """Set target pitch frequency in Hz (thread-safe)."""
        clamped_freq = max(40.0, min(4000.0, float(freq)))
        with self._lock:
            self._target_frequency = clamped_freq
            self._chord_frequencies = []

    def set_chord_frequencies(self, freqs: list):
        """Set target chord frequencies in Hz for polyphonic synthesis (thread-safe)."""
        with self._lock:
            if freqs and len(freqs) > 0:
                self._chord_frequencies = [max(40.0, min(4000.0, float(f))) for f in freqs]
                self._target_frequency = self._chord_frequencies[0]
            else:
                self._chord_frequencies = []

    def set_volume(self, volume: float):
        """Set target output volume normalized 0.0 to 1.0 (thread-safe)."""
        clamped_vol = max(0.0, min(1.0, float(volume)))
        with self._lock:
            self._target_volume = clamped_vol

    def note_on(self, freq: Optional[float] = None, volume: Optional[float] = None):
        """Trigger note attack with click-free fade-in (thread-safe)."""
        with self._lock:
            if freq is not None:
                self._target_frequency = max(40.0, min(4000.0, float(freq)))
            if volume is not None:
                self._target_volume = max(0.0, min(1.0, float(volume)))
            self._is_note_on = True

    def note_off(self):
        """Trigger note release with smooth fade-out to silence (thread-safe)."""
        with self._lock:
            self._is_note_on = False

    def set_instrument(self, name: str) -> str:
        """
        Switch active synthesized instrument sound.
        Options: 'sine', 'sawtooth', 'square'.
        """
        clean_name = str(name).lower().strip()
        if clean_name in self.AVAILABLE_INSTRUMENTS:
            with self._lock:
                self._instrument = clean_name
        return self._instrument

    def cycle_instrument(self) -> str:
        """Cycle to the next instrument sound in sequence."""
        with self._lock:
            idx = self.AVAILABLE_INSTRUMENTS.index(self._instrument)
            self._instrument = self.AVAILABLE_INSTRUMENTS[(idx + 1) % len(self.AVAILABLE_INSTRUMENTS)]
            return self._instrument

    def toggle_echo(self) -> bool:
        """Toggle the real-time Delay / Echo audio effect."""
        with self._lock:
            self._effect_echo = not self._effect_echo
            return self._effect_echo

    def set_echo(self, enabled: bool):
        """Explicitly set Delay / Echo audio effect state."""
        with self._lock:
            self._effect_echo = bool(enabled)

    def set_tone(self, tone: float):
        """Set tone cutoff filter parameter: 0.0 (Warm Sub) to 1.0 (Bright Lead)."""
        clamped = max(0.05, min(1.0, float(tone)))
        with self._lock:
            self._tone = clamped

    def get_tone(self) -> float:
        """Get current tone parameter."""
        with self._lock:
            return self._tone

    def cycle_tone(self) -> float:
        """Cycle tone through presets: 0.25 (Sub) -> 0.50 (Warm) -> 0.75 (Balanced) -> 1.00 (Bright)."""
        with self._lock:
            presets = [0.25, 0.50, 0.75, 1.00]
            # find next preset
            for p in presets:
                if p > self._tone + 0.05:
                    self._tone = p
                    return self._tone
            self._tone = presets[0]
            return self._tone

    def set_arpeggiator(self, mode: str) -> str:
        """Set arpeggiator mode ('OFF', 'UP', 'BOUNCE')."""
        clean = str(mode).upper().strip()
        if clean in self.ARP_MODES:
            with self._lock:
                self._arpeggiator_mode = clean
                self._arp_step = 0
        return self._arpeggiator_mode

    def cycle_arpeggiator(self) -> str:
        """Cycle arpeggiator mode (OFF -> UP -> BOUNCE -> OFF)."""
        with self._lock:
            idx = self.ARP_MODES.index(self._arpeggiator_mode)
            self._arpeggiator_mode = self.ARP_MODES[(idx + 1) % len(self.ARP_MODES)]
            self._arp_step = 0
            return self._arpeggiator_mode

    def is_playing(self) -> bool:
        """Return True if a note is triggered or currently sounding."""
        with self._lock:
            note_gate = self._is_note_on
        return note_gate or (self._current_amplitude > 0.001)

    def get_status(self) -> Dict[str, any]:
        """Query current synthesis state for telemetry and UI display."""
        with self._lock:
            return {
                "target_frequency": self._target_frequency,
                "current_frequency": self._current_frequency,
                "target_volume": self._target_volume,
                "is_note_on": self._is_note_on,
                "is_running": self._is_running,
                "instrument": self._instrument,
                "effect_echo": self._effect_echo,
                "tone": self._tone,
                "arpeggiator": self._arpeggiator_mode,
                "is_audible": self._current_amplitude > 0.001,
            }

    # -------------------------------------------------------------------------
    # Stream Lifecycle Management
    # -------------------------------------------------------------------------

    def start(self):
        """Open and start the asynchronous audio output stream."""
        if self._is_running and self._stream and self._stream.active:
            return

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._is_running = True
        print(f"[SynthEngine] Audio stream started ({self.sample_rate} Hz, block: {self.block_size} samples)")

    def stop(self):
        """Stop and close the audio stream safely."""
        if self._stream:
            try:
                with self._lock:
                    self._is_note_on = False
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"[SynthEngine] Note during stream stop: {e}")
            finally:
                self._stream = None

        self._is_running = False
        self._current_amplitude = 0.0
        self._current_phase = 0.0
        self._filter_state_1 = 0.0
        self._filter_state_2 = 0.0
        self._filter_alpha = 1.0
        self._delay_buffer.fill(0.0)
        print("[SynthEngine] Audio stream stopped cleanly.")

    # -------------------------------------------------------------------------
    # Audio Callback (High-Priority C/C++ PortAudio Thread)
    # Lightweight DSP only: No file I/O, no OpenCV, no memory allocations!
    # -------------------------------------------------------------------------

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Compute synthesis and effects on the audio thread."""
        # 1. Thread-safe snapshot of control parameters (microsecond lock)
        with self._lock:
            target_freq = self._target_frequency
            chord_freqs = list(self._chord_frequencies) if self._chord_frequencies else []
            target_amp = self._target_volume if self._is_note_on else 0.0
            instrument = self._instrument
            echo_active = self._effect_echo
            tone = self._tone
            arp_mode = self._arpeggiator_mode

        # Arpeggiator note sequencing
        if arp_mode != "OFF" and target_amp > 0.0:
            if arp_mode == "UP":
                arp_steps = [1.0, 1.25992, 1.49831, 2.0]
            else:  # BOUNCE
                arp_steps = [1.0, 1.25992, 1.49831, 2.0, 1.49831, 1.25992]

            self._arp_sample_counter += frames
            if self._arp_sample_counter >= self._arp_samples_per_step:
                self._arp_sample_counter = 0
                self._arp_step = (self._arp_step + 1) % len(arp_steps)

            target_freq *= arp_steps[self._arp_step % len(arp_steps)]
        else:
            self._arp_step = 0
            self._arp_sample_counter = 0

        # Fast path: If silent, no active note, and delay line is empty
        delay_energy = np.max(np.abs(self._delay_buffer[:frames])) if echo_active else 0.0
        if self._current_amplitude <= 0.0001 and target_amp == 0.0 and delay_energy <= 0.0001:
            outdata.fill(0.0)
            self._current_amplitude = 0.0
            self._filter_state_1 = 0.0
            self._filter_state_2 = 0.0
            return

        # 2. Smooth amplitude envelope ramp to eliminate clicks
        amp_step = 1.0 / (self.sample_rate * max(0.005, self.fade_time))
        max_delta_amp = amp_step * frames

        if abs(target_amp - self._current_amplitude) <= max_delta_amp:
            end_amp = target_amp
        elif target_amp > self._current_amplitude:
            end_amp = self._current_amplitude + max_delta_amp
        else:
            end_amp = self._current_amplitude - max_delta_amp

        amplitude_ramp = np.linspace(
            self._current_amplitude, end_amp, frames, endpoint=False, dtype=np.float32
        )
        self._current_amplitude = end_amp

        # 3. Waveform Generation (Polyphonic Chords or Monophonic Pitch)
        if len(chord_freqs) > 1 and arp_mode == "OFF":
            num_v = len(chord_freqs)
            raw_samples = np.zeros(frames, dtype=np.float32)

            if len(self._current_chord_phases) < num_v:
                self._current_chord_phases = np.zeros(num_v, dtype=np.float32)
                self._current_chord_freqs = np.array(chord_freqs, dtype=np.float32)

            for i in range(num_v):
                vf = chord_freqs[i]
                cur_f = float(self._current_chord_freqs[i])
                freq_step = 1.0 / (self.sample_rate * self.DEFAULT_PITCH_SLEW)
                max_delta = abs(vf - cur_f) * min(1.0, freq_step * frames * 10)

                if abs(vf - cur_f) <= max_delta:
                    end_f = vf
                elif vf > cur_f:
                    end_f = cur_f + max_delta
                else:
                    end_f = cur_f - max_delta
                self._current_chord_freqs[i] = end_f

                f_ramp = np.linspace(cur_f, end_f, frames, endpoint=False, dtype=np.float32)
                phase_inc = (2.0 * np.pi / self.sample_rate) * f_ramp
                cumsum = np.empty(frames, dtype=np.float32)
                cumsum[0] = self._current_chord_phases[i]
                if frames > 1:
                    cumsum[1:] = self._current_chord_phases[i] + np.cumsum(phase_inc[:-1])
                self._current_chord_phases[i] = float((cumsum[-1] + phase_inc[-1]) % (2.0 * np.pi))

                # Warm, Anti-Aliased Analog Synthesis (Gentle on ears, zero digital harshness)
                if instrument == "sine":
                    # Warm ambient sine with subtle warm 2nd harmonic
                    s = 0.92 * np.sin(cumsum, dtype=np.float32) + 0.08 * np.sin(2.0 * cumsum, dtype=np.float32)
                elif instrument == "sawtooth":
                    # Band-limited warm poly saw (Hann-windowed harmonics eliminate aliasing screech)
                    s = np.zeros(frames, dtype=np.float32)
                    for k in range(1, 25):
                        w = float(np.cos((k - 1) * np.pi / 48.0))
                        amp = float(((-1.0) ** (k + 1) / k) * w)
                        s += amp * np.sin(k * cumsum, dtype=np.float32)
                    s *= 0.72
                elif instrument == "square":
                    # Warm vintage pulse shaper (smooth tanh waveshaping, zero step discontinuities)
                    s = np.tanh(2.4 * np.sin(cumsum, dtype=np.float32)) * 0.72
                elif instrument == "triangle":
                    # Silky Rhodes / Warm Flute triangle
                    s = 0.88 * np.sin(cumsum, dtype=np.float32) - 0.10 * np.sin(3.0 * cumsum, dtype=np.float32) + 0.02 * np.sin(5.0 * cumsum, dtype=np.float32)
                else:
                    s = np.sin(cumsum, dtype=np.float32)

                raw_samples += s

            # Headroom normalization to prevent clipping across polyphonic chord voices
            raw_samples *= (1.0 / math.sqrt(num_v))

        else:
            # Monophonic Smooth Frequency Transition
            freq_step = 1.0 / (self.sample_rate * self.DEFAULT_PITCH_SLEW)
            max_delta_freq = abs(target_freq - self._current_frequency) * min(1.0, freq_step * frames * 10)

            if abs(target_freq - self._current_frequency) <= max_delta_freq:
                end_freq = target_freq
            elif target_freq > self._current_frequency:
                end_freq = self._current_frequency + max_delta_freq
            else:
                end_freq = self._current_frequency - max_delta_freq

            freq_ramp = np.linspace(
                self._current_frequency, end_freq, frames, endpoint=False, dtype=np.float32
            )
            self._current_frequency = end_freq

            # Phase Accumulator Calculation
            phase_increments = (2.0 * np.pi / self.sample_rate) * freq_ramp
            cumsum_phases = np.empty(frames, dtype=np.float32)
            cumsum_phases[0] = self._current_phase
            if frames > 1:
                cumsum_phases[1:] = self._current_phase + np.cumsum(phase_increments[:-1])

            # Advance accumulator and wrap around [0, 2*pi)
            self._current_phase = float((cumsum_phases[-1] + phase_increments[-1]) % (2.0 * np.pi))

            # Multi-Waveform Generation (Warm Analog Models)
            if instrument == "sine":
                raw_samples = 0.92 * np.sin(cumsum_phases, dtype=np.float32) + 0.08 * np.sin(2.0 * cumsum_phases, dtype=np.float32)
            elif instrument == "sawtooth":
                raw_samples = np.zeros(frames, dtype=np.float32)
                for k in range(1, 25):
                    w = float(np.cos((k - 1) * np.pi / 48.0))
                    amp = float(((-1.0) ** (k + 1) / k) * w)
                    raw_samples += amp * np.sin(k * cumsum_phases, dtype=np.float32)
                raw_samples *= 0.72
            elif instrument == "square":
                raw_samples = np.tanh(2.4 * np.sin(cumsum_phases, dtype=np.float32)) * 0.72
            elif instrument == "triangle":
                raw_samples = 0.88 * np.sin(cumsum_phases, dtype=np.float32) - 0.10 * np.sin(3.0 * cumsum_phases, dtype=np.float32) + 0.02 * np.sin(5.0 * cumsum_phases, dtype=np.float32)
            else:
                raw_samples = np.sin(cumsum_phases, dtype=np.float32)

        # Apply envelope gain
        dry_signal = raw_samples * amplitude_ramp

        # 5b. Tone 2-Pole Resonant Low-Pass Filter (LPF)
        if tone < 0.98 and self._current_amplitude > 0.0001:
            # Musical exponential cutoff from 90 Hz (warm sub) to 14,000 Hz (bright open lead)
            fc = 90.0 * ((14000.0 / 90.0) ** tone)
            target_alpha = float(min(1.0, 1.0 - np.exp(-2.0 * np.pi * fc / self.sample_rate)))
            # Slew smooth filter coefficient to eliminate digital zipper noise
            alpha = self._filter_alpha + (target_alpha - self._filter_alpha) * 0.35
            self._filter_alpha = alpha

            s1 = self._filter_state_1
            s2 = self._filter_state_2
            for i in range(frames):
                s1 += alpha * (dry_signal[i] - s1)
                s2 += alpha * (s1 - s2)
                dry_signal[i] = s2
            self._filter_state_1 = s1
            self._filter_state_2 = s2
        else:
            self._filter_alpha = 1.0
            self._filter_state_1 = 0.0
            self._filter_state_2 = 0.0

        # 5c. Transparent Soft Limiter (Warm saturation headroom, prevents digital clipping)
        dry_signal = np.tanh(dry_signal * 1.05).astype(np.float32)

        # 6. Audio Effect: Circular Delay Line / Echo
        if echo_active:
            # Read delayed samples from circular buffer
            buffer_len = self.delay_samples
            read_indices = (np.arange(frames) + self._delay_write_ptr) % buffer_len
            delayed_signal = self._delay_buffer[read_indices]

            # Combine dry signal + delayed wet signal
            wet_signal = delayed_signal * self.delay_wet
            mixed_output = dry_signal + wet_signal

            # Write current signal + feedback back into delay buffer
            feedback_signal = dry_signal + delayed_signal * self.delay_feedback
            self._delay_buffer[read_indices] = feedback_signal
            self._delay_write_ptr = (self._delay_write_ptr + frames) % buffer_len

            # Soft clip limiter to prevent digital clipping
            limited = np.clip(mixed_output, -1.0, 1.0)
            if outdata.ndim == 2:
                outdata[:, 0] = limited
            else:
                outdata[:] = limited
        else:
            if outdata.ndim == 2:
                outdata[:, 0] = dry_signal
            else:
                outdata[:] = dry_signal

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
