"""
ui.py
User Interface and Real-Time Music Synthesizer Workstation for HandSynth.
Features:
- Clean, modern, dark Digital Audio Workstation (DAW) layout (1240 x 720).
- High-definition typography using TrueType anti-aliased font rendering (Pillow).
- Framed live webcam feed with high-contrast dual-hand landmark tracking overlays.
- Interactive Musical Key Selector (defaults to C, supports C through B and sharp transpose).
- Simplified high-precision Tone Cutoff Filter slider (warm sub <-> bright sizzle lead).
- Real-time Note & Pitch Telemetry with clean scale degree badges and gate indicators.
- Live Detected Gesture Status with clear dual-hand chord & variation display.
- Compact Instrument & Audio Effects Rack (Sine, Saw, Square, Triangle, Echo, Arpeggiator).
- 100% mouse click & drag interactivity + full keyboard shortcut accessibility.
"""

import math
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UIOverlay:
    """Renders a sleek, dark modern synthesizer workstation."""

    CANVAS_WIDTH = 1240
    CANVAS_HEIGHT = 720

    # Modern Dark DAW Palette (RGB)
    COLOR_BG = (13, 15, 20)              # Deep obsidian background
    COLOR_CARD = (22, 25, 34)            # Surface card slate
    COLOR_CARD_BORDER = (38, 44, 59)     # Subtle card border
    COLOR_TEXT_PRIMARY = (248, 250, 252) # Crisp white text
    COLOR_TEXT_MUTED = (142, 155, 174)   # Sleek silver grey
    COLOR_ACCENT_CYAN = (0, 229, 255)    # Vibrant cyan (active / lead)
    COLOR_ACCENT_EMERALD = (16, 185, 129)# Rich emerald green (scales / active gate)
    COLOR_ACCENT_AMBER = (245, 158, 11)  # Warm amber (warnings / sharps)
    COLOR_ACCENT_ROSE = (239, 68, 68)    # Crimson red (mute / errors)
    COLOR_ACCENT_PURPLE = (168, 85, 247) # Electric purple (effects / arp)
    COLOR_ACCENT_SKY = (56, 189, 248)    # Sky blue (subtle data)
    COLOR_BTN_INACTIVE = (30, 34, 46)    # Inactive button fill
    COLOR_BTN_BORDER = (48, 56, 74)      # Inactive button outline

    # Natural root keys
    KEYS_LIST = ["C", "D", "E", "F", "G", "A", "B"]

    def __init__(self):
        """Initialize typography fonts and button hitboxes."""
        self._load_fonts()
        self._init_hitboxes()

    def _load_fonts(self):
        """Load anti-aliased TrueType fonts from Windows system font cache."""
        font_dir = "C:/Windows/Fonts/"
        try:
            self.font_title = ImageFont.truetype(font_dir + "segoeuib.ttf", 19)
            self.font_note_big = ImageFont.truetype(font_dir + "segoeuib.ttf", 34)
            self.font_card_head = ImageFont.truetype(font_dir + "segoeuib.ttf", 11)
            self.font_body = ImageFont.truetype(font_dir + "segoeui.ttf", 13)
            self.font_body_bold = ImageFont.truetype(font_dir + "segoeuib.ttf", 13)
            self.font_small = ImageFont.truetype(font_dir + "segoeui.ttf", 11)
            self.font_small_bold = ImageFont.truetype(font_dir + "segoeuib.ttf", 11)
            self.font_mono = ImageFont.truetype(font_dir + "consola.ttf", 11)
        except Exception:
            # Fallback for systems without specific font files
            default = ImageFont.load_default()
            self.font_title = default
            self.font_note_big = default
            self.font_card_head = default
            self.font_body = default
            self.font_body_bold = default
            self.font_small = default
            self.font_small_bold = default
            self.font_mono = default

    def _init_hitboxes(self):
        """Precompute interactive mouse click hitboxes matching rack layout."""
        self.btn_keys: Dict[str, Tuple[int, int, int, int]] = {}
        for i, k in enumerate(self.KEYS_LIST):
            x1 = 760 + i * 55
            self.btn_keys[k] = (x1, 262, x1 + 46, 294)
        self.btn_sharp = (1150, 262, 1210, 294)

        # Scale buttons (2 rows of 4)
        scale_names_r1 = ["PENTATONIC", "BLUES", "MAJOR", "MINOR"]
        scale_names_r2 = ["DORIAN", "HIRAJOSHI", "CHROMATIC", "FREE"]
        self.btn_scales: Dict[str, Tuple[int, int, int, int]] = {}

        for i, s in enumerate(scale_names_r1):
            x1 = 760 + i * 114
            self.btn_scales[s] = (x1, 345, x1 + 106, 373)
        for i, s in enumerate(scale_names_r2):
            x1 = 760 + i * 114
            self.btn_scales[s] = (x1, 380, x1 + 106, 408)

        # Tone slider rect (interactive scrub zone)
        self.slider_tone = (760, 456, 1210, 482)
        self.btn_tone_source = (0, 0, 0, 0)  # Deprecated toggle removed from UI

        # Instrument buttons
        inst_names = ["sine", "sawtooth", "square", "triangle"]
        self.btn_instruments: Dict[str, Tuple[int, int, int, int]] = {}
        for i, inst in enumerate(inst_names):
            x1 = 760 + i * 114
            self.btn_instruments[inst] = (x1, 532, x1 + 106, 560)

        # FX & Mode buttons
        self.btn_echo = (760, 572, 980, 600)
        self.btn_arp = (990, 572, 1210, 600)
        self.btn_mode_header = (830, 10, 1065, 38)

    # -------------------------------------------------------------------------
    # Mouse Interaction Handlers
    # -------------------------------------------------------------------------

    def handle_click(self, x: int, y: int, controller, synth) -> bool:
        """
        Handle mouse click events and dispatch state changes to controller and synth.
        Returns True if an interactive control was hit.
        """
        # 1. Root Key Buttons
        for key_name, (x1, y1, x2, y2) in self.btn_keys.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                cur_key = getattr(controller, "key", "C")
                is_sharp = cur_key.endswith("#")
                target = f"{key_name}#" if (is_sharp and key_name not in ("B", "E")) else key_name
                controller.set_key(target)
                if synth and hasattr(controller, "_current_raw_pitch"):
                    new_pitch, _, _ = controller.snap_to_scale(controller._current_raw_pitch)
                    synth.set_frequency(new_pitch)
                return True

        # Sharp Toggle Button
        sx1, sy1, sx2, sy2 = self.btn_sharp
        if sx1 <= x <= sx2 and sy1 <= y <= sy2:
            cur_key = getattr(controller, "key", "C")
            base = cur_key.replace("#", "")
            if base in ("B", "E"):
                return True
            new_key = base if cur_key.endswith("#") else f"{base}#"
            controller.set_key(new_key)
            if synth and hasattr(controller, "_current_raw_pitch"):
                new_pitch, _, _ = controller.snap_to_scale(controller._current_raw_pitch)
                synth.set_frequency(new_pitch)
            return True

        # 2. Scale Selector Buttons
        for scale_name, (x1, y1, x2, y2) in self.btn_scales.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                controller.set_scale(scale_name)
                if synth and hasattr(controller, "_current_raw_pitch"):
                    new_pitch, _, _ = controller.snap_to_scale(controller._current_raw_pitch)
                    synth.set_frequency(new_pitch)
                return True

        # 3. Tone Slider Scrub
        tx1, ty1, tx2, ty2 = self.slider_tone
        if tx1 <= x <= tx2 and ty1 <= y <= ty2:
            val = max(0.05, min(1.0, (x - tx1) / (tx2 - tx1)))
            synth.set_tone(val)
            return True

        # 4. Instrument Buttons
        for inst_name, (x1, y1, x2, y2) in self.btn_instruments.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                synth.set_instrument(inst_name)
                return True

        # 5. Echo Button
        ex1, ey1, ex2, ey2 = self.btn_echo
        if ex1 <= x <= ex2 and ey1 <= y <= ey2:
            synth.toggle_echo()
            return True

        # 6. Arpeggiator Button
        ax1, ay1, ax2, ay2 = self.btn_arp
        if ax1 <= x <= ax2 and ay1 <= y <= ay2:
            synth.cycle_arpeggiator()
            return True

        # 7. Mode Header Button
        mx1, my1, mx2, my2 = self.btn_mode_header
        if mx1 <= x <= mx2 and my1 <= y <= my2:
            controller.toggle_mode()
            return True

        return False

    def handle_drag(self, x: int, y: int, synth) -> bool:
        """Handle continuous dragging on slider controls."""
        tx1, ty1, tx2, ty2 = self.slider_tone
        if (tx1 - 10) <= x <= (tx2 + 10) and (ty1 - 15) <= y <= (ty2 + 15):
            val = max(0.05, min(1.0, (x - tx1) / max(1, (tx2 - tx1))))
            synth.set_tone(val)
            return True
        return False

    # -------------------------------------------------------------------------
    # Main Render Pipeline
    # -------------------------------------------------------------------------

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        is_hand_detected: bool,
        handedness: str = "None",
        gesture_data: Optional[Dict[str, any]] = None,
        synth_data: Optional[Dict[str, any]] = None,
        landmarks: Optional[List[Dict[str, float]]] = None,
        camera_error: bool = False,
    ) -> np.ndarray:
        """
        Compose and render the complete workstation display frame.
        Returns a clean 1240x720 BGR composite image ready for display.
        """
        if not gesture_data:
            gesture_data = {
                "is_note_active": False,
                "pitch_hz": 261.63,
                "nearest_note": "---",
                "cents": 0.0,
                "volume": 0.0,
                "is_pinched": False,
                "mode": "TWO-HAND CHORD",
                "scale": "PENTATONIC",
                "key": "C",
                "gesture": "NONE",
                "octave_shift": 0,
            }

        if not synth_data:
            synth_data = {
                "instrument": "sine",
                "effect_echo": False,
                "tone": 0.85,
                "arpeggiator": "OFF",
            }

        # 1. Base Canvas
        canvas_pil = Image.new("RGB", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), self.COLOR_BG)
        draw = ImageDraw.Draw(canvas_pil)

        # 2. Top Header Bar
        self._render_header(draw, fps, gesture_data.get("mode", "TWO-HAND CHORD"))

        # 3. Framed Webcam Feed
        self._embed_webcam_feed(
            canvas_pil,
            frame,
            is_hand_detected,
            handedness,
            gesture_data,
            landmarks,
            camera_error,
        )

        # 4. Right Rack Control Cards
        self._render_control_rack(draw, gesture_data, synth_data, is_hand_detected, handedness)

        # 5. Bottom Footer
        self._render_footer(draw)

        # Convert PIL RGB to OpenCV BGR
        composite_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
        return composite_bgr

    # -------------------------------------------------------------------------
    # Modular UI Section Renderers
    # -------------------------------------------------------------------------

    def _render_header(self, draw: ImageDraw.Draw, fps: float, mode: str):
        """Draw clean top header bar with branding, mode pill, and engine telemetry."""
        draw.rectangle([0, 0, self.CANVAS_WIDTH, 48], fill=(10, 12, 16))
        draw.line([0, 48, self.CANVAS_WIDTH, 48], fill=self.COLOR_CARD_BORDER, width=1)

        # Clean Musical Logo
        draw.text((22, 13), "♫ HANDSYNTH", font=self.font_title, fill=self.COLOR_ACCENT_CYAN)
        draw.text((160, 18), "GESTURAL AUDIO WORKSTATION", font=self.font_small, fill=self.COLOR_TEXT_MUTED)

        # Active Mode Pill (Clickable & Dynamic)
        mode_str = str(mode).upper()
        if "CHORD" in mode_str:
            mode_label = "TWO-HAND CHORD MODE"
            mode_bg = (18, 42, 52)
            mode_border = self.COLOR_ACCENT_CYAN
        elif "PINCH" in mode_str:
            mode_label = "SOLO: PINCH-TO-PLAY"
            mode_bg = (38, 22, 48)
            mode_border = self.COLOR_ACCENT_PURPLE
        else:
            mode_label = "SOLO: CONTINUOUS THEREMIN"
            mode_bg = (48, 24, 38)
            mode_border = self.COLOR_ACCENT_AMBER

        mx1, my1, mx2, my2 = self.btn_mode_header
        draw.rounded_rectangle([mx1, my1, mx2, my2], radius=6, fill=mode_bg, outline=mode_border, width=1)

        # Center text inside mode pill to prevent any truncation
        text_w = self.font_small_bold.getlength(mode_label)
        text_x = mx1 + int((mx2 - mx1 - text_w) / 2)
        draw.text((text_x, my1 + 6), mode_label, font=self.font_small_bold, fill=self.COLOR_TEXT_PRIMARY)

        # Engine Stats Pill
        draw.rounded_rectangle([1080, 10, 1215, 38], radius=6, fill=self.COLOR_CARD, outline=self.COLOR_CARD_BORDER, width=1)
        fps_color = self.COLOR_ACCENT_EMERALD if fps >= 25 else self.COLOR_ACCENT_AMBER
        draw.text((1094, 15), f"{fps:.0f} FPS  ·  44.1k", font=self.font_mono, fill=fps_color)

    def _embed_webcam_feed(
        self,
        canvas_pil: Image.Image,
        raw_frame: np.ndarray,
        is_hand_detected: bool,
        handedness: str,
        gesture_data: Dict[str, any],
        landmarks: Optional[List[Dict[str, float]]],
        camera_error: bool,
    ):
        """Prepare and frame the live camera stream with in-camera overlay."""
        cam_card_x, cam_card_y = 20, 58
        cam_card_w, cam_card_h = 700, 604

        draw_canvas = ImageDraw.Draw(canvas_pil)
        draw_canvas.rounded_rectangle(
            [cam_card_x, cam_card_y, cam_card_x + cam_card_w, cam_card_y + cam_card_h],
            radius=8,
            fill=self.COLOR_CARD,
            outline=self.COLOR_CARD_BORDER,
            width=1,
        )

        vw, vh = 676, 507
        vx = cam_card_x + 12
        vy = cam_card_y + 12

        if camera_error or raw_frame is None or raw_frame.size == 0:
            draw_canvas.rounded_rectangle([vx, vy, vx + vw, vy + vh], radius=6, fill=(16, 18, 24))
            draw_canvas.text((vx + 220, vy + 240), "[ CAMERA ACCESS OFFLINE ]", font=self.font_body_bold, fill=self.COLOR_ACCENT_AMBER)
        else:
            resized_cam = cv2.resize(raw_frame, (vw, vh), interpolation=cv2.INTER_LINEAR)
            self._draw_incam_hud(resized_cam, vw, vh, is_hand_detected, handedness, gesture_data, landmarks)
            cam_rgb = cv2.cvtColor(resized_cam, cv2.COLOR_BGR2RGB)
            cam_pil = Image.fromarray(cam_rgb)
            canvas_pil.paste(cam_pil, (vx, vy))

        # Bottom pitch ribbon guide
        self._render_pitch_ribbon(draw_canvas, vx, vy + vh + 12, vw, gesture_data)

    def _draw_incam_hud(
        self,
        cam_bgr: np.ndarray,
        vw: int,
        vh: int,
        is_hand_detected: bool,
        handedness: str,
        gesture_data: Dict[str, any],
        landmarks: Optional[List[Dict[str, float]]],
    ):
        """Draw minimal, modern in-camera tracking overlay directly on video feed."""
        # Top-left Status Badge
        if is_hand_detected:
            h_upper = handedness.upper()
            if h_upper == "DUAL":
                status_text = "TRACKING: DUAL HANDS ACTIVE"
                dot_color = (255, 220, 0)  # Cyan
            elif h_upper == "LEFT":
                status_text = "TRACKING: LEFT HAND (CHORD)"
                dot_color = (100, 225, 50) # Emerald
            else:
                status_text = f"TRACKING: {h_upper} HAND"
                dot_color = (40, 180, 255) # Amber
        else:
            status_text = "SEARCHING FOR HANDS..."
            dot_color = (120, 130, 150)   # Silver

        cv2.rectangle(cam_bgr, (14, 14), (250, 40), (16, 18, 24), -1)
        cv2.rectangle(cam_bgr, (14, 14), (250, 40), (45, 52, 68), 1)
        cv2.circle(cam_bgr, (27, 27), 4, dot_color, -1)
        cv2.putText(cam_bgr, status_text, (40, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)

        # Right Vertical Volume Indicator
        vol = max(0.0, min(1.0, gesture_data.get("volume", 0.0)))
        bar_x = vw - 20
        bar_y1, bar_y2 = 35, vh - 35
        bar_h = bar_y2 - bar_y1
        cv2.rectangle(cam_bgr, (bar_x, bar_y1), (bar_x + 8, bar_y2), (22, 25, 34), -1)
        cv2.rectangle(cam_bgr, (bar_x, bar_y1), (bar_x + 8, bar_y2), (48, 56, 74), 1)

        fill_h = int(bar_h * vol)
        if fill_h > 0:
            cv2.rectangle(cam_bgr, (bar_x, bar_y2 - fill_h), (bar_x + 8, bar_y2), (240, 200, 40), -1)

        # Hand Skeleton Drawing
        if is_hand_detected and landmarks and len(landmarks) >= 21:
            CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
                (9, 13), (13, 14), (14, 15), (15, 16), # Ring
                (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky & Base
            ]
            for p1, p2 in CONNECTIONS:
                pt1 = (int(landmarks[p1]["x"] * vw), int(landmarks[p1]["y"] * vh))
                pt2 = (int(landmarks[p2]["x"] * vw), int(landmarks[p2]["y"] * vh))
                # Cyan accent on index/thumb, emerald on others
                c = (255, 230, 0) if (p1 in (2, 3, 4, 6, 7, 8) and p2 in (2, 3, 4, 6, 7, 8)) else (90, 200, 110)
                cv2.line(cam_bgr, pt1, pt2, c, 2, cv2.LINE_AA)

            for j in [4, 8, 12, 16, 20]:
                pt = (int(landmarks[j]["x"] * vw), int(landmarks[j]["y"] * vh))
                cv2.circle(cam_bgr, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def _render_pitch_ribbon(self, draw: ImageDraw.Draw, x: int, y: int, w: int, gesture_data: Dict[str, any]):
        """Draw clean pitch ribbon beneath the camera frame."""
        draw.rectangle([x, y, x + w, y + 46], fill=(16, 19, 26), outline=self.COLOR_CARD_BORDER, width=1)

        octaves = [("C3", 0.0), ("C4", 0.333), ("C5", 0.666), ("C6", 1.0)]
        axis_x1 = x + 35
        axis_x2 = x + w - 35
        axis_w = axis_x2 - axis_x1
        axis_y = y + 23

        draw.line([axis_x1, axis_y, axis_x2, axis_y], fill=(48, 56, 74), width=2)

        for name, frac in octaves:
            ox = int(axis_x1 + frac * axis_w)
            draw.line([ox, axis_y - 5, ox, axis_y + 5], fill=self.COLOR_TEXT_MUTED, width=2)
            draw.text((ox - 9, axis_y - 18), name, font=self.font_small, fill=self.COLOR_TEXT_MUTED)

        pitch_hz = gesture_data.get("pitch_hz", 261.63)
        f_min, f_max = 130.81, 1046.50
        if f_min <= pitch_hz <= f_max:
            frac = max(0.0, min(1.0, math.log2(pitch_hz / f_min) / math.log2(f_max / f_min)))
            cur_x = int(axis_x1 + frac * axis_w)
            cursor_color = self.COLOR_ACCENT_CYAN if gesture_data.get("is_note_active", False) else self.COLOR_TEXT_MUTED
            draw.ellipse([cur_x - 5, axis_y - 5, cur_x + 5, axis_y + 5], fill=cursor_color)
            draw.text((cur_x - 14, axis_y + 7), f"{pitch_hz:.0f}Hz", font=self.font_small, fill=cursor_color)

    def _render_control_rack(
        self,
        draw: ImageDraw.Draw,
        gesture_data: Dict[str, any],
        synth_data: Dict[str, any],
        is_hand_detected: bool,
        handedness: str,
    ):
        """Render all DAW control cards in the right rack panel."""
        rx = 740
        rw = 480

        # Card 1: Note & Pitch Telemetry
        self._card_note_telemetry(draw, rx, 58, rw, 94, gesture_data)

        # Card 2: Detected Gesture Status
        self._card_gesture_status(draw, rx, 160, rw, 62, gesture_data, is_hand_detected, handedness)

        # Card 3: Musical Key Selector
        self._card_key_selector(draw, rx, 230, rw, 76, gesture_data)

        # Card 4: Scale Selector
        self._card_scale_selector(draw, rx, 314, rw, 102, gesture_data)

        # Card 5: Tone & Filter Control (Cleaned & Simplified)
        self._card_tone_control(draw, rx, 424, rw, 72, synth_data, gesture_data)

        # Card 6: Waveform & Audio Effects
        self._card_instrument_fx(draw, rx, 504, rw, 114, synth_data)

    def _card_note_telemetry(self, draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, g: Dict[str, any]):
        """Card 1: Active chord / note name, pitch frequency, tuning cents, and gate status."""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=self.COLOR_CARD, outline=self.COLOR_CARD_BORDER, width=1)
        draw.text((x + 14, y + 8), "ACTIVE HARMONY & NOTE", font=self.font_card_head, fill=self.COLOR_TEXT_MUTED)

        is_active = g.get("is_note_active", False)
        note_name = g.get("nearest_note", "---")
        freq = g.get("pitch_hz", 0.0)
        cents = g.get("cents", 0.0)

        # Prominent Note / Chord Label
        note_color = self.COLOR_ACCENT_CYAN if is_active else self.COLOR_TEXT_MUTED
        draw.text((x + 18, y + 30), note_name, font=self.font_note_big, fill=note_color)

        # Frequency & Tuning Deviation
        draw.text((x + 180, y + 34), f"PITCH: {freq:.1f} Hz", font=self.font_body_bold, fill=self.COLOR_TEXT_PRIMARY)
        cents_sign = "+" if cents >= 0 else ""
        draw.text((x + 180, y + 54), f"TUNING: {cents_sign}{cents:.1f} Cents", font=self.font_small, fill=self.COLOR_TEXT_MUTED)

        # Gate status badge
        gate_bg = (16, 48, 36) if is_active else (26, 30, 40)
        gate_border = self.COLOR_ACCENT_EMERALD if is_active else self.COLOR_CARD_BORDER
        gate_text = "● PLAYING" if is_active else "○ STANDBY"
        draw.rounded_rectangle([x + w - 116, y + 36, x + w - 16, y + 66], radius=5, fill=gate_bg, outline=gate_border, width=1)
        draw.text((x + w - 105, y + 44), gate_text, font=self.font_small_bold, fill=self.COLOR_TEXT_PRIMARY)

    def _card_gesture_status(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        w: int,
        h: int,
        g: Dict[str, any],
        is_hand_detected: bool,
        handedness: str,
    ):
        """Card 2: Live detected gesture and active chord quality display."""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=self.COLOR_CARD, outline=self.COLOR_CARD_BORDER, width=1)
        draw.text((x + 14, y + 8), "LIVE GESTURE TRACKING", font=self.font_card_head, fill=self.COLOR_TEXT_MUTED)

        raw_gesture = str(g.get("gesture", "NONE")).strip()

        if not is_hand_detected or raw_gesture in ("NONE", "No Hand"):
            pill_text = "SEARCHING FOR HANDS..."
            pill_color = self.COLOR_TEXT_MUTED
            pill_bg = (24, 28, 36)
        elif is_hand_detected and "Left hand absent" in raw_gesture:
            pill_text = "RIGHT HAND (Show Left Hand for Chords)"
            pill_color = (40, 200, 255)
            pill_bg = (28, 20, 16)
        elif "MUTE" in raw_gesture.upper() or "FIST" in raw_gesture.upper():
            pill_text = "FIST (EMERGENCY MUTE)"
            pill_color = self.COLOR_ACCENT_ROSE
            pill_bg = (55, 18, 22)
        else:
            pill_text = raw_gesture
            pill_color = self.COLOR_ACCENT_EMERALD if "[MAJOR]" in pill_text.upper() else (
                self.COLOR_ACCENT_CYAN if "[MINOR]" in pill_text.upper() else self.COLOR_ACCENT_CYAN
            )
            pill_bg = (16, 36, 44)

        # Truncate text cleanly if very long
        if len(pill_text) > 42:
            pill_text = pill_text[:39] + "..."

        draw.rounded_rectangle([x + 14, y + 26, x + 340, y + 52], radius=5, fill=pill_bg, outline=pill_color, width=1)
        draw.text((x + 22, y + 32), pill_text, font=self.font_small_bold, fill=pill_color)

        hand_label = f"{handedness.upper()} HAND" if is_hand_detected else "IDLE"
        draw.text((x + w - 110, y + 33), hand_label, font=self.font_small, fill=self.COLOR_TEXT_MUTED)

    def _card_key_selector(self, draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, g: Dict[str, any]):
        """Card 3: Musical Key Selector defaulting to C."""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=self.COLOR_CARD, outline=self.COLOR_CARD_BORDER, width=1)

        current_key = str(g.get("key", "C")).upper()
        draw.text((x + 14, y + 8), f"ROOT KEY: {current_key} MAJOR", font=self.font_card_head, fill=self.COLOR_TEXT_MUTED)

        base_key = current_key.replace("#", "")
        is_sharp = current_key.endswith("#")

        # 7 Key Buttons
        for key_name in self.KEYS_LIST:
            bx1, by1, bx2, by2 = self.btn_keys[key_name]
            is_active = (key_name == base_key)

            bg = self.COLOR_ACCENT_CYAN if is_active else self.COLOR_BTN_INACTIVE
            border = self.COLOR_ACCENT_CYAN if is_active else self.COLOR_BTN_BORDER
            text_color = (10, 14, 20) if is_active else self.COLOR_TEXT_PRIMARY

            draw.rounded_rectangle([bx1, by1, bx2, by2], radius=5, fill=bg, outline=border, width=1)
            draw.text((bx1 + 17, by1 + 7), key_name, font=self.font_body_bold, fill=text_color)

        # Sharp toggle button
        sx1, sy1, sx2, sy2 = self.btn_sharp
        s_bg = self.COLOR_ACCENT_AMBER if is_sharp else self.COLOR_BTN_INACTIVE
        s_border = self.COLOR_ACCENT_AMBER if is_sharp else self.COLOR_BTN_BORDER
        s_text_color = (10, 14, 20) if is_sharp else self.COLOR_TEXT_PRIMARY

        draw.rounded_rectangle([sx1, sy1, sx2, sy2], radius=5, fill=s_bg, outline=s_border, width=1)
        draw.text((sx1 + 13, sy1 + 7), "# SHARP", font=self.font_small_bold, fill=s_text_color)

    def _card_scale_selector(self, draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, g: Dict[str, any]):
        """Card 4: Scale & Tune Selector."""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=self.COLOR_CARD, outline=self.COLOR_CARD_BORDER, width=1)

        current_scale = str(g.get("scale", "PENTATONIC")).upper()
        draw.text((x + 14, y + 8), f"SCALE & TUNING: {current_scale}", font=self.font_card_head, fill=self.COLOR_TEXT_MUTED)

        for scale_name, (bx1, by1, bx2, by2) in self.btn_scales.items():
            is_active = (scale_name == current_scale)
            bg = self.COLOR_ACCENT_EMERALD if is_active else self.COLOR_BTN_INACTIVE
            border = self.COLOR_ACCENT_EMERALD if is_active else self.COLOR_BTN_BORDER
            text_color = (10, 14, 20) if is_active else self.COLOR_TEXT_PRIMARY

            label = scale_name.capitalize()
            draw.rounded_rectangle([bx1, by1, bx2, by2], radius=5, fill=bg, outline=border, width=1)
            draw.text((bx1 + 14, by1 + 6), label, font=self.font_small, fill=text_color)

    def _card_tone_control(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        w: int,
        h: int,
        synth_data: Dict[str, any],
        gesture_data: Optional[Dict[str, any]] = None,
    ):
        """Card 5: Simplified Tone & Low-Pass Filter Slider."""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=self.COLOR_CARD, outline=self.COLOR_CARD_BORDER, width=1)

        tone_val = float(synth_data.get("tone", 0.85))
        pct = int(tone_val * 100)
        fc_hz = int(150.0 * ((14000.0 / 150.0) ** tone_val))

        draw.text((x + 14, y + 8), "TONE / CUTOFF FILTER", font=self.font_card_head, fill=self.COLOR_TEXT_MUTED)
        draw.text((x + w - 170, y + 8), f"CUTOFF: {fc_hz} Hz ({pct}%)", font=self.font_mono, fill=self.COLOR_ACCENT_SKY)

        # Slider Track
        tx1, ty1, tx2, ty2 = self.slider_tone
        track_y = ty1 + (ty2 - ty1) // 2
        track_w = tx2 - tx1

        # Groove background
        draw.line([tx1, track_y, tx2, track_y], fill=(36, 42, 56), width=6)

        # Filled active portion
        thumb_x = int(tx1 + tone_val * track_w)
        draw.line([tx1, track_y, thumb_x, track_y], fill=self.COLOR_ACCENT_CYAN, width=6)

        # Knob handle
        draw.ellipse([thumb_x - 8, track_y - 8, thumb_x + 8, track_y + 8], fill=self.COLOR_TEXT_PRIMARY, outline=self.COLOR_ACCENT_CYAN, width=2)

        # Tone descriptors
        draw.text((tx1, track_y + 9), "WARM SUB", font=self.font_small, fill=self.COLOR_TEXT_MUTED)
        draw.text((tx2 - 76, track_y + 9), "BRIGHT SIZZLE", font=self.font_small, fill=self.COLOR_TEXT_MUTED)

    def _card_instrument_fx(self, draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, synth_data: Dict[str, any]):
        """Card 6: Waveform Selector and Audio Effects (Echo & Arpeggiator)."""
        draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=self.COLOR_CARD, outline=self.COLOR_CARD_BORDER, width=1)
        draw.text((x + 14, y + 8), "OSCILLATOR WAVEFORMS & AUDIO EFFECTS", font=self.font_card_head, fill=self.COLOR_TEXT_MUTED)

        active_inst = str(synth_data.get("instrument", "sine")).lower()

        inst_labels = {
            "sine": "1: SINE",
            "sawtooth": "2: SAW",
            "square": "3: SQUARE",
            "triangle": "4: TRIANGLE",
        }
        for inst_key, (bx1, by1, bx2, by2) in self.btn_instruments.items():
            is_active = (inst_key == active_inst)
            bg = self.COLOR_ACCENT_CYAN if is_active else self.COLOR_BTN_INACTIVE
            border = self.COLOR_ACCENT_CYAN if is_active else self.COLOR_BTN_BORDER
            text_color = (10, 14, 20) if is_active else self.COLOR_TEXT_PRIMARY

            draw.rounded_rectangle([bx1, by1, bx2, by2], radius=5, fill=bg, outline=border, width=1)
            draw.text((bx1 + 16, by1 + 6), inst_labels[inst_key], font=self.font_small, fill=text_color)

        # FX: Echo Button
        echo_active = synth_data.get("effect_echo", False)
        ex1, ey1, ex2, ey2 = self.btn_echo
        e_bg = (18, 48, 34) if echo_active else self.COLOR_BTN_INACTIVE
        e_border = self.COLOR_ACCENT_EMERALD if echo_active else self.COLOR_BTN_BORDER
        e_text = "ECHO / DELAY: [ON]" if echo_active else "ECHO / DELAY: [OFF]"
        draw.rounded_rectangle([ex1, ey1, ex2, ey2], radius=5, fill=e_bg, outline=e_border, width=1)
        draw.text((ex1 + 45, ey1 + 6), e_text, font=self.font_body_bold, fill=self.COLOR_TEXT_PRIMARY)

        # FX: Arpeggiator Button
        arp_mode = synth_data.get("arpeggiator", "OFF")
        ax1, ay1, ax2, ay2 = self.btn_arp
        a_active = (arp_mode != "OFF")
        a_bg = (48, 22, 58) if a_active else self.COLOR_BTN_INACTIVE
        a_border = self.COLOR_ACCENT_PURPLE if a_active else self.COLOR_BTN_BORDER
        a_text = f"ARPEGGIATOR: [{arp_mode}]"
        draw.rounded_rectangle([ax1, ay1, ax2, ay2], radius=5, fill=a_bg, outline=a_border, width=1)
        draw.text((ax1 + 40, ay1 + 6), a_text, font=self.font_body_bold, fill=self.COLOR_TEXT_PRIMARY)

    def _render_footer(self, draw: ImageDraw.Draw):
        """Draw bottom workstation footer with clean hotkey cheat sheet and mouse indicator."""
        fy = self.CANVAS_HEIGHT - 38
        draw.rectangle([0, fy, self.CANVAS_WIDTH, self.CANVAS_HEIGHT], fill=(10, 12, 16))
        draw.line([0, fy, self.CANVAS_WIDTH, fy], fill=self.COLOR_CARD_BORDER, width=1)

        hotkeys = "[K] Key · [S] Scale · [1-4] Wave · [T] Tone · [E] Echo · [A] Arp · [C] Mode · [F] Fullscreen · [SPACE] Test · [Q] Exit"
        draw.text((20, fy + 12), hotkeys, font=self.font_mono, fill=self.COLOR_TEXT_MUTED)

        # Mouse interaction badge on the right (guaranteed zero overlap)
        bx1, by1 = self.CANVAS_WIDTH - 330, fy + 6
        bx2, by2 = self.CANVAS_WIDTH - 20, fy + 32
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=4, fill=(18, 24, 34), outline=(38, 48, 68), width=1)
        draw.text((bx1 + 14, by1 + 5), "● MOUSE: CLICK CONTROLS OR DRAG TONE", font=self.font_small_bold, fill=self.COLOR_ACCENT_CYAN)
