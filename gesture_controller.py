"""
gesture_controller.py
Advanced Gesture Recognition & Musical Intelligence for HandSynth.
Features:
- Multi-Scale Quantization (FREE, CHROMATIC, MAJOR, MINOR, PENTATONIC, BLUES)
- Seamless Snap-to-Scale vs Free-Glissando pitch control
- Dual-Threshold Hysteresis Pinch Trigger (Thumb-Index)
- Multi-Gesture Recognition:
    * PINCH: Note Trigger
    * PEACE / V-SIGN (Index + Middle extended): +1 Octave Transpose
    * FIST (All fingers curled): Emergency Mute / Stop
    * OPEN PALM: Full expression & dynamic volume control
- Exponential pitch scale mapping across 3 octaves (C3 to C6)
- Inverted Y-axis volume mapping with safety cap
- Low-pass Exponential Moving Average (EMA) coordinate smoothing
"""

import math
from typing import Dict, List, Optional, Tuple


class GestureController:
    """Interprets raw hand landmarks into synthesized musical events and parameters."""

    # 12-Tone Equal Temperament note names
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    ROOT_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Enharmonically correct scale spellings for all 7 diatonic Major scales (and sharp keys)
    MAJOR_SCALE_SPELLINGS = {
        "C":  ["C", "D", "E", "F", "G", "A", "B"],
        "D":  ["D", "E", "F#", "G", "A", "B", "C#"],
        "E":  ["E", "F#", "G#", "A", "B", "C#", "D#"],
        "F":  ["F", "G", "A", "Bb", "C", "D", "E"],
        "G":  ["G", "A", "B", "C", "D", "E", "F#"],
        "A":  ["A", "B", "C#", "D", "E", "F#", "G#"],
        "B":  ["B", "C#", "D#", "E", "F#", "G#", "A#"],
        "C#": ["C#", "D#", "E#", "F#", "G#", "A#", "B#"],
        "D#": ["D#", "E#", "F##", "G#", "A#", "B#", "C##"],
        "F#": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
        "G#": ["G#", "A#", "B#", "C#", "D#", "E#", "F##"],
        "A#": ["A#", "B#", "C##", "D#", "E#", "F##", "G##"],
    }

    # Musical Scales (semitone offsets from tonic root)
    SCALES = {
        "PENTATONIC": [0, 2, 4, 7, 9],    # Harmonious, impossible to hit wrong notes
        "MAJOR": [0, 2, 4, 5, 7, 9, 11],  # Classical, bright, cheerful melodies
        "MINOR": [0, 2, 3, 5, 7, 8, 10],  # Melancholic, emotional ballads
        "BLUES": [0, 3, 5, 6, 7, 10],     # Blues / rock solos & riffs
        "DORIAN": [0, 2, 3, 5, 7, 9, 10], # Funk, Daft Punk, medieval jazz groove
        "HIRAJOSHI": [0, 2, 3, 7, 8],     # Traditional Japanese / Anime / Zen
        "CHROMATIC": list(range(12)),     # Full 12-note piano keyboard
        "FREE": None,                     # Continuous fretless (Theremin / Violin)
    }
    SCALE_KEYS = list(SCALES.keys())

    # Playable spatial bounding zone (normalized 0.0 to 1.0)
    X_MIN, X_MAX = 0.15, 0.85
    Y_MIN, Y_MAX = 0.15, 0.85

    def __init__(
        self,
        f_min: float = 130.81,           # C3
        f_max: float = 1046.50,          # C6 (3 full octaves)
        pinch_on_thresh: float = 0.045,  # Trigger pinch when distance < 0.045
        pinch_off_thresh: float = 0.070, # Release pinch only when distance > 0.070
        smoothing_alpha: float = 0.35,   # Low-pass filter weight
        mode: str = "PINCH",             # "PINCH" or "CONTINUOUS"
        scale: str = "PENTATONIC",       # Default to Pentatonic (instantly musical!)
        key: str = "C",                  # Default musical root key (Defaults to C!)
        pitch_hysteresis: float = 0.15,  # Hysteresis deadband in semitones (~15 cents)
        tracking_loss_grace_frames: int = 0, # Grace frames for brief camera blinks (configurable)
    ):
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.pinch_on_thresh = float(pinch_on_thresh)
        self.pinch_off_thresh = float(pinch_off_thresh)
        self.alpha = float(smoothing_alpha)
        self.mode = mode.upper()
        self.scale = scale.upper() if scale.upper() in self.SCALES else "PENTATONIC"
        self.key = key.upper() if key.upper() in self.ROOT_KEYS else "C"
        self.pitch_hysteresis_semitones = float(pitch_hysteresis)

        # Configurable gesture mappings
        self.pitch_axis = "X"               # "X" = Horizontal, "Y" = Vertical
        self.volume_axis = "Y"              # "Y" = Vertical, "X" = Horizontal
        self.tone_source = "MANUAL"         # "MANUAL" = Slider, "GESTURE_TILT" = Hand tilt/roll
        self.octave_gesture_enabled = True  # V-sign enables +1 octave
        self.fist_mute_enabled = True       # Fist enables emergency mute
        self.tracking_loss_grace_frames = int(tracking_loss_grace_frames) # Default 0 (instant)

        # Internal state
        self._is_pinched = False
        self._is_note_active = False
        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None
        self._smoothed_tilt: float = 0.5
        self._missing_hand_counter = 0
        self._current_freq = 440.0
        self._current_raw_pitch = 440.0
        self._locked_midi: Optional[int] = None
        self._current_volume = 0.25
        self._active_gesture = "NONE"
        self._octave_shift = 0           # 0 = normal, +1 = octave up (V-sign)

    def set_mapping_config(self, **kwargs) -> Dict[str, any]:
        """Configure gesture-to-note and gesture-to-tone parameter mappings."""
        if "pitch_axis" in kwargs and kwargs["pitch_axis"] in ("X", "Y"):
            self.pitch_axis = kwargs["pitch_axis"]
        if "volume_axis" in kwargs and kwargs["volume_axis"] in ("X", "Y"):
            self.volume_axis = kwargs["volume_axis"]
        if "tone_source" in kwargs and kwargs["tone_source"] in ("MANUAL", "GESTURE_TILT"):
            self.tone_source = kwargs["tone_source"]
        if "octave_gesture_enabled" in kwargs:
            self.octave_gesture_enabled = bool(kwargs["octave_gesture_enabled"])
        if "fist_mute_enabled" in kwargs:
            self.fist_mute_enabled = bool(kwargs["fist_mute_enabled"])
        if "tracking_loss_grace_frames" in kwargs:
            self.tracking_loss_grace_frames = max(0, int(kwargs["tracking_loss_grace_frames"]))
        return self.get_mapping_config()

    def get_mapping_config(self) -> Dict[str, any]:
        """Query currently active gesture mapping configuration."""
        return {
            "pitch_axis": self.pitch_axis,
            "volume_axis": self.volume_axis,
            "tone_source": self.tone_source,
            "octave_gesture_enabled": self.octave_gesture_enabled,
            "fist_mute_enabled": self.fist_mute_enabled,
            "tracking_loss_grace_frames": self.tracking_loss_grace_frames,
            "key": self.key,
            "scale": self.scale,
            "mode": self.mode,
        }

    def set_tone_source(self, source: str) -> str:
        """Set tone control source ('MANUAL' or 'GESTURE_TILT')."""
        clean = str(source).upper().strip()
        if clean in ("MANUAL", "GESTURE_TILT"):
            self.tone_source = clean
        return self.tone_source

    def cycle_tone_source(self) -> str:
        """Toggle tone source between MANUAL and GESTURE_TILT."""
        self.tone_source = "GESTURE_TILT" if self.tone_source == "MANUAL" else "MANUAL"
        return self.tone_source

    def set_key(self, key_name: str) -> str:
        """Explicitly set musical root key (e.g. 'C', 'D', 'E', 'F', 'G', 'A', 'B')."""
        clean = str(key_name).upper().strip()
        # Handle enharmonic equivalents (e.g. Db -> C#, Eb -> D#, Bb -> A#)
        enharmonics = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
        clean = enharmonics.get(clean, clean)
        if clean in self.ROOT_KEYS:
            self.key = clean
            self._locked_midi = None  # Immediately unlock previous note to snap to new key
        return self.key

    def cycle_key(self, direction: int = 1) -> str:
        """Cycle musical root key in sequence."""
        idx = self.ROOT_KEYS.index(self.key)
        self.key = self.ROOT_KEYS[(idx + direction) % len(self.ROOT_KEYS)]
        self._locked_midi = None
        return self.key

    def toggle_mode(self) -> str:
        """Toggle between PINCH and CONTINUOUS mode."""
        self.mode = "CONTINUOUS" if self.mode == "PINCH" else "PINCH"
        return self.mode

    def set_mode(self, mode: str) -> str:
        """Explicitly set operating mode ('PINCH' or 'CONTINUOUS')."""
        clean = str(mode).upper().strip()
        if clean in ("PINCH", "CONTINUOUS"):
            self.mode = clean
        return self.mode

    def cycle_scale(self) -> str:
        """Cycle through musical scales."""
        idx = self.SCALE_KEYS.index(self.scale)
        self.scale = self.SCALE_KEYS[(idx + 1) % len(self.SCALE_KEYS)]
        self._locked_midi = None
        return self.scale

    def set_scale(self, scale_name: str) -> str:
        """Explicitly set scale."""
        clean = str(scale_name).upper().strip()
        if clean in self.SCALES:
            self.scale = clean
            self._locked_midi = None
        return self.scale

    def reset(self):
        """Reset internal filter states and release active note."""
        self._is_pinched = False
        self._is_note_active = False
        self._smoothed_x = None
        self._smoothed_y = None
        self._smoothed_tilt = 0.5
        self._active_gesture = "NONE"
        self._octave_shift = 0
        self._locked_midi = None
        self._missing_hand_counter = 0

    def _calculate_hand_tilt(self, landmarks: List[Dict[str, float]]) -> float:
        """
        Calculate continuous lateral hand tilt (roll angle) for expressive tone modulation.
        Uses vector from Wrist (0) to Middle MCP knuckle (9).
        Returns normalized tone value in [0.05, 0.98] (warm sub to bright lead).
        """
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        dx = middle_mcp["x"] - wrist["x"]
        dy = middle_mcp["y"] - wrist["y"]
        # Normal upright hand has dx ~ 0, dy < 0
        angle = math.atan2(dx, -dy)  # radians (~0 vertical, negative left, positive right)
        # Map angle from [-0.5, +0.5] radians (~ -30 deg to +30 deg) to [0.05, 0.98]
        norm_tilt = max(0.05, min(0.98, (angle + 0.5) / 1.0))
        return norm_tilt

    def calculate_distance(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Calculate 3D Euclidean distance between two landmarks."""
        dx = p1["x"] - p2["x"]
        dy = p1["y"] - p2["y"]
        dz = p1.get("z", 0.0) - p2.get("z", 0.0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @classmethod
    def freq_to_musical_note(cls, freq: float) -> Tuple[str, float, float]:
        """Convert frequency (Hz) to note name, cents offset, and exact tuned pitch."""
        if freq <= 0:
            return "---", 0.0, 0.0
        midi_num = 69.0 + 12.0 * math.log2(freq / 440.0)
        midi_round = round(midi_num)
        note_idx = midi_round % 12
        octave = (midi_round // 12) - 1
        note_name = f"{cls.NOTE_NAMES[note_idx]}{octave}"

        exact_freq = 440.0 * (2.0 ** ((midi_round - 69.0) / 12.0))
        cents = 1200.0 * math.log2(freq / exact_freq)
        return note_name, cents, exact_freq

    def format_note_name(self, midi_num: int) -> str:
        """Format note name with accidental and octave (e.g. 'C4', 'Bb4', 'F#5')."""
        pitch_class = midi_num % 12
        octave = (midi_num // 12) - 1
        root_offset = self.NOTE_NAMES.index(self.key)
        degree = (pitch_class - root_offset) % 12

        if self.scale == "MAJOR" and self.key in self.MAJOR_SCALE_SPELLINGS:
            major_intervals = [0, 2, 4, 5, 7, 9, 11]
            if degree in major_intervals:
                note_str = self.MAJOR_SCALE_SPELLINGS[self.key][major_intervals.index(degree)]
                return f"{note_str}{octave}"

        return f"{self.NOTE_NAMES[pitch_class]}{octave}"

    def snap_to_scale(self, raw_freq: float) -> Tuple[float, str, float]:
        """
        Quantize raw continuous frequency to nearest note in the selected scale.
        Uses pitch hysteresis to prevent rapid flutter at note boundaries.
        If scale is 'FREE', returns the raw continuous frequency.
        """
        if raw_freq <= 0:
            return 0.0, "---", 0.0

        if self.scale == "FREE":
            self._locked_midi = None
            note_name, cents, _ = self.freq_to_musical_note(raw_freq)
            return raw_freq, note_name, cents

        root_offset = self.NOTE_NAMES.index(self.key)
        allowed = [(root_offset + s) % 12 for s in self.SCALES[self.scale]]
        midi_float = 69.0 + 12.0 * math.log2(raw_freq / 440.0)
        base_midi = round(midi_float)

        # Search nearest allowed semitone
        best_midi = base_midi
        min_dist = 999.0
        for offset in range(-6, 7):
            cand = base_midi + offset
            if (cand % 12) in allowed:
                dist = abs(midi_float - cand)
                if dist < min_dist:
                    min_dist = dist
                    best_midi = cand

        # Apply pitch quantizer hysteresis to prevent rapid flickering from hand micro-tremors
        if self._locked_midi is None or abs(best_midi - self._locked_midi) > 3:
            self._locked_midi = best_midi
        elif best_midi != self._locked_midi:
            midpoint = (self._locked_midi + best_midi) / 2.0
            H = self.pitch_hysteresis_semitones
            if best_midi > self._locked_midi and midi_float > (midpoint + H):
                self._locked_midi = best_midi
            elif best_midi < self._locked_midi and midi_float < (midpoint - H):
                self._locked_midi = best_midi

        active_midi = self._locked_midi
        snapped_freq = 440.0 * (2.0 ** ((active_midi - 69.0) / 12.0))
        note_name = self.format_note_name(active_midi)
        cents = 1200.0 * math.log2(raw_freq / snapped_freq) if raw_freq > 0 else 0.0
        return snapped_freq, note_name, cents

    def _classify_hand_gesture(self, landmarks: List[Dict[str, float]]) -> str:
        """
        Detect discrete hand shapes:
        - FIST: All fingers curled into palm
        - V_SIGN: Index and Middle fingers extended (Peace Sign) -> +1 Octave Shift
        - OPEN: All fingers spread
        """
        wrist = landmarks[0]

        # Check finger extension by comparing tip distance to wrist vs pip distance to wrist
        def is_extended(tip_idx, pip_idx):
            tip_d = self.calculate_distance(landmarks[tip_idx], wrist)
            pip_d = self.calculate_distance(landmarks[pip_idx], wrist)
            return tip_d > pip_d * 1.15

        index_ext = is_extended(8, 6)
        middle_ext = is_extended(12, 10)
        ring_ext = is_extended(16, 14)
        pinky_ext = is_extended(20, 18)

        # Fist: None of the 4 fingers extended
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "FIST"

        # Peace / V-Sign: Index & Middle extended, Ring & Pinky curled
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "PEACE"

        # Open Hand: All fingers extended
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return "OPEN"

        return "NORMAL"

    def process(
        self,
        landmarks: Optional[List[Dict[str, float]]],
        is_hand_present: bool,
    ) -> Dict[str, any]:
        """Process hand landmarks and determine musical parameters."""
        if not is_hand_present or not landmarks or len(landmarks) < 21:
            self._missing_hand_counter += 1
            if self._missing_hand_counter <= self.tracking_loss_grace_frames and self._is_note_active:
                # Brief tracking glitch: hold note safely without stuttering
                return {
                    "note_event": "NOTE_HOLD",
                    "is_note_active": True,
                    "pitch_hz": self._current_freq,
                    "nearest_note": self.format_note_name(self._locked_midi) if self._locked_midi else "---",
                    "cents": 0.0,
                    "volume": self._current_volume,
                    "pinch_distance": 0.04,
                    "is_pinched": True,
                    "cursor_px": (0, 0),
                    "mode": self.mode,
                    "scale": self.scale,
                    "key": self.key,
                    "gesture": "TRACKING_RECOVERY",
                    "octave_shift": self._octave_shift,
                    "tone_val": self._smoothed_tilt if self.tone_source == "GESTURE_TILT" else None,
                    "tone_source": self.tone_source,
                }

            # Sustained hand absence: cleanly release note and reset
            prev_active = self._is_note_active
            self.reset()
            return {
                "note_event": "NOTE_OFF" if prev_active else "SILENT",
                "is_note_active": False,
                "pitch_hz": self._current_freq,
                "nearest_note": "---",
                "cents": 0.0,
                "volume": 0.0,
                "pinch_distance": 1.0,
                "is_pinched": False,
                "cursor_px": (0, 0),
                "mode": self.mode,
                "scale": self.scale,
                "key": self.key,
                "gesture": "NONE",
                "octave_shift": 0,
                "tone_val": None,
                "tone_source": self.tone_source,
            }

        # Hand is actively tracked: reset missing counter
        self._missing_hand_counter = 0

        thumb = landmarks[4]
        index = landmarks[8]

        # Midpoint of thumb and index as control cursor
        raw_x = (thumb["x"] + index["x"]) * 0.5
        raw_y = (thumb["y"] + index["y"]) * 0.5
        cursor_px = (int((thumb["px"] + index["px"]) * 0.5), int((thumb["py"] + index["py"]) * 0.5))

        # 1. Low-Pass Exponential Smoothing (Jitter reduction)
        if self._smoothed_x is None:
            self._smoothed_x = raw_x
            self._smoothed_y = raw_y
        else:
            self._smoothed_x = self.alpha * raw_x + (1.0 - self.alpha) * self._smoothed_x
            self._smoothed_y = self.alpha * raw_y + (1.0 - self.alpha) * self._smoothed_y

        # 2. Hand Gesture Pose Recognition & Tilt Extraction
        gesture_shape = self._classify_hand_gesture(landmarks)
        self._active_gesture = gesture_shape

        raw_tilt = self._calculate_hand_tilt(landmarks)
        self._smoothed_tilt = 0.25 * raw_tilt + 0.75 * self._smoothed_tilt
        active_tone = self._smoothed_tilt if self.tone_source == "GESTURE_TILT" else None

        # Octave Shift: V-Sign transposes +1 octave up if enabled
        if self.octave_gesture_enabled and gesture_shape == "PEACE":
            self._octave_shift = 1
        else:
            self._octave_shift = 0

        # 3. Configurable Pitch Mapping Axis
        if self.pitch_axis == "X":
            norm_pitch = (self._smoothed_x - self.X_MIN) / (self.X_MAX - self.X_MIN)
        else:  # "Y"
            norm_pitch = 1.0 - ((self._smoothed_y - self.Y_MIN) / (self.Y_MAX - self.Y_MIN))
        norm_pitch = max(0.0, min(1.0, norm_pitch))
        base_pitch = self.f_min * ((self.f_max / self.f_min) ** norm_pitch)

        # Apply octave shift if active (+1 octave = frequency * 2)
        if self._octave_shift > 0:
            base_pitch *= 2.0

        self._current_raw_pitch = base_pitch

        # 4. Scale Quantization with Hysteresis (Snaps to selected key & scale)
        pitch_hz, nearest_note, cents = self.snap_to_scale(base_pitch)
        self._current_freq = pitch_hz

        # 5. Configurable Volume Mapping Axis
        if self.volume_axis == "Y":
            norm_vol = 1.0 - ((self._smoothed_y - self.Y_MIN) / (self.Y_MAX - self.Y_MIN))
        else:  # "X"
            norm_vol = (self._smoothed_x - self.X_MIN) / (self.X_MAX - self.X_MIN)
        norm_vol = max(0.0, min(1.0, norm_vol))
        volume = norm_vol * 0.38  # Safe volume cap
        self._current_volume = volume

        # 6. Pinch Hysteresis State Machine
        pinch_dist = self.calculate_distance(thumb, index)
        if not self._is_pinched:
            if pinch_dist < self.pinch_on_thresh:
                self._is_pinched = True
        else:
            if pinch_dist > self.pinch_off_thresh:
                self._is_pinched = False

        # 7. Note Event Dispatch
        note_event = "SILENT"

        # Fist gesture forces immediate mute if enabled
        if self.fist_mute_enabled and gesture_shape == "FIST":
            if self._is_note_active:
                note_event = "NOTE_OFF"
                self._is_note_active = False
            else:
                note_event = "SILENT"
        elif self.mode == "PINCH":
            if self._is_pinched:
                if not self._is_note_active:
                    note_event = "NOTE_ON"
                    self._is_note_active = True
                else:
                    note_event = "NOTE_HOLD"
            else:
                if self._is_note_active:
                    note_event = "NOTE_OFF"
                    self._is_note_active = False
                else:
                    note_event = "SILENT"
        elif self.mode == "CONTINUOUS":
            if not self._is_note_active:
                note_event = "NOTE_ON"
                self._is_note_active = True
            else:
                note_event = "NOTE_HOLD"

        return {
            "note_event": note_event,
            "is_note_active": self._is_note_active,
            "pitch_hz": pitch_hz,
            "nearest_note": nearest_note,
            "cents": cents,
            "volume": volume,
            "pinch_distance": pinch_dist,
            "is_pinched": self._is_pinched,
            "cursor_px": cursor_px,
            "mode": self.mode,
            "scale": self.scale,
            "key": self.key,
            "gesture": gesture_shape,
            "octave_shift": self._octave_shift,
            "tone_val": active_tone,
            "tone_source": self.tone_source,
        }
