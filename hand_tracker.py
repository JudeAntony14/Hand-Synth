"""
hand_tracker.py
MediaPipe Hand Tracking Module for HandSynth (Milestone 2).
Encapsulates MediaPipe Hands detection and landmark extraction.
Optimized for low CPU usage and real-time responsiveness.
"""

import math
import time
from typing import List, Tuple, Dict, Optional
import cv2
import mediapipe as mp


class HandTracker:
    """Encapsulates MediaPipe Hands tracking with CPU optimizations."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Hands.

        Args:
            max_num_hands: Number of hands to detect (default 2 for two-handed control).
            min_detection_confidence: Confidence threshold for initial palm detection.
            min_tracking_confidence: Confidence threshold for landmark tracking between frames.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # static_image_mode=False enables tracking across frames instead of redetecting every frame
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.results = None
        self.prev_time = time.time()
        self.fps = 0.0

        # Cached dual-hand tracking data and temporal positions
        self.left_landmarks: Optional[List[Dict[str, float]]] = None
        self.right_landmarks: Optional[List[Dict[str, float]]] = None
        self.prev_left_pos: Optional[Tuple[float, float]] = None
        self.prev_right_pos: Optional[Tuple[float, float]] = None

    def process_frame(
        self, frame: cv2.Mat, draw: bool = True
    ) -> Tuple[cv2.Mat, bool, List[Dict[str, float]], str]:
        """
        Process a video frame: flips horizontally for mirror mode, runs inference,
        and extracts 21 landmarks.

        Args:
            frame: Raw BGR frame from OpenCV VideoCapture.
            draw: Whether to draw the 21 landmarks and connections on the frame.

        Returns:
            Tuple of:
                - processed_frame: Flipped BGR frame with optional landmark annotations
                - is_hand_detected: True if a hand is detected in this frame
                - landmarks: List of 21 landmark dictionaries [{'id', 'x', 'y', 'z', 'px', 'py'}, ...]
                - handedness: 'Right', 'Left', or 'None'
        """
        # 1. Flip horizontally for natural mirror interaction
        mirrored_frame = cv2.flip(frame, 1)
        h, w, _ = mirrored_frame.shape

        # 2. Performance optimization: set writeable flag to False to avoid memory copy
        rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # 3. Calculate moving average FPS
        curr_time = time.time()
        dt = curr_time - self.prev_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
        self.prev_time = curr_time

        # 4. Check hand presence and extract landmarks
        is_hand_detected = bool(self.results and self.results.multi_hand_landmarks)
        landmarks = []
        handedness = "None"
        self.left_landmarks = None
        self.right_landmarks = None

        if is_hand_detected:
            num_hands = len(self.results.multi_hand_landmarks)
            all_hands_lms = []
            mp_handedness_labels = []

            for i in range(num_hands):
                hand_lms = self.results.multi_hand_landmarks[i]
                cur_lms = []
                for idx, lm in enumerate(hand_lms.landmark):
                    cur_lms.append({
                        "id": idx,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "px": int(lm.x * w),
                        "py": int(lm.y * h),
                    })
                all_hands_lms.append(cur_lms)

                # Decode MediaPipe handedness on mirrored frames
                # In mirrored mode (cv2.flip(frame, 1)), MediaPipe's model natively assumes selfie-view:
                # raw_lbl == 'Left' is the user's physical Left Hand
                # raw_lbl == 'Right' is the user's physical Right Hand
                label = "None"
                if self.results.multi_handedness and i < len(self.results.multi_handedness):
                    raw_lbl = self.results.multi_handedness[i].classification[0].label
                    label = raw_lbl if raw_lbl in ("Left", "Right") else "None"
                mp_handedness_labels.append(label)

                # Draw landmark skeleton on all detected hands
                if draw:
                    self.mp_draw.draw_landmarks(
                        mirrored_frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style(),
                    )

            # Assign Left and Right hands with anatomical & temporal stability
            if num_hands >= 2:
                h0_label = mp_handedness_labels[0] if len(mp_handedness_labels) > 0 else "None"
                h1_label = mp_handedness_labels[1] if len(mp_handedness_labels) > 1 else "None"

                if h0_label == "Left" and h1_label == "Right":
                    self.left_landmarks = all_hands_lms[0]
                    self.right_landmarks = all_hands_lms[1]
                elif h0_label == "Right" and h1_label == "Left":
                    self.left_landmarks = all_hands_lms[1]
                    self.right_landmarks = all_hands_lms[0]
                else:
                    # Spatial fallback in mirrored mode: user's left hand is on screen left (smaller x)
                    if all_hands_lms[0][0]["x"] < all_hands_lms[1][0]["x"]:
                        self.left_landmarks = all_hands_lms[0]
                        self.right_landmarks = all_hands_lms[1]
                    else:
                        self.left_landmarks = all_hands_lms[1]
                        self.right_landmarks = all_hands_lms[0]

                self.prev_left_pos = (self.left_landmarks[0]["x"], self.left_landmarks[0]["y"])
                self.prev_right_pos = (self.right_landmarks[0]["x"], self.right_landmarks[0]["y"])
                landmarks = self.left_landmarks or self.right_landmarks
                handedness = "Dual"

            else:
                # Single hand detected
                single_hand = all_hands_lms[0]
                wx = single_hand[0]["x"]
                wy = single_hand[0]["y"]
                detected_as = mp_handedness_labels[0] if mp_handedness_labels else "None"

                # Check temporal proximity to last known hand positions
                dist_to_left = 999.0
                dist_to_right = 999.0
                if self.prev_left_pos:
                    dist_to_left = math.hypot(wx - self.prev_left_pos[0], wy - self.prev_left_pos[1])
                if self.prev_right_pos:
                    dist_to_right = math.hypot(wx - self.prev_right_pos[0], wy - self.prev_right_pos[1])

                is_left = False
                if detected_as in ("Left", "Right"):
                    is_left = (detected_as == "Left")
                elif self.prev_left_pos and not self.prev_right_pos:
                    is_left = True
                elif self.prev_right_pos and not self.prev_left_pos:
                    is_left = False
                elif dist_to_left < dist_to_right:
                    is_left = True
                else:
                    is_left = (wx < 0.50)

                if is_left:
                    self.left_landmarks = single_hand
                    self.prev_left_pos = (wx, wy)
                    handedness = "Left"
                else:
                    self.right_landmarks = single_hand
                    self.prev_right_pos = (wx, wy)
                    handedness = "Right"

                landmarks = single_hand

            # Draw clear floating labels on hands in camera feed
            if draw:
                if self.left_landmarks:
                    lw = self.left_landmarks[0]
                    lwx, lwy = lw["px"], lw["py"]
                    lb_x = max(10, min(w - 180, lwx - 60))
                    lb_y = lwy + 26 if lwy < h - 45 else lwy - 26
                    cv2.rectangle(mirrored_frame, (lb_x, lb_y - 15), (lb_x + 160, lb_y + 7), (16, 24, 20), -1)
                    cv2.rectangle(mirrored_frame, (lb_x, lb_y - 15), (lb_x + 160, lb_y + 7), (50, 200, 100), 1)
                    cv2.putText(mirrored_frame, "LEFT HAND [CHORD]", (lb_x + 8, lb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 140), 1, cv2.LINE_AA)

                if self.right_landmarks:
                    rw = self.right_landmarks[0]
                    rwx, rwy = rw["px"], rw["py"]
                    rb_x = max(10, min(w - 180, rwx - 60))
                    rb_y = rwy + 26 if rwy < h - 45 else rwy - 26
                    cv2.rectangle(mirrored_frame, (rb_x, rb_y - 15), (rb_x + 160, rb_y + 7), (28, 20, 16), -1)
                    cv2.rectangle(mirrored_frame, (rb_x, rb_y - 15), (rb_x + 160, rb_y + 7), (40, 180, 255), 1)
                    cv2.putText(mirrored_frame, "RIGHT HAND [VARIATION]", (rb_x + 8, rb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 200, 255), 1, cv2.LINE_AA)

        return mirrored_frame, is_hand_detected, landmarks, handedness

    def get_dual_hands(self) -> Tuple[Optional[List[Dict[str, float]]], Optional[List[Dict[str, float]]]]:
        """Return (left_landmarks, right_landmarks) from most recent frame."""
        return self.left_landmarks, self.right_landmarks

    def release(self):
        """Cleanly release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
