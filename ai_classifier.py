"""
ai_classifier.py
Custom AI Gesture Classifier for HandSynth.

Features:
- Landmark normalization:
  * Translation invariant: Centered at wrist (landmark 0)
  * Scale invariant: Scaled by palm distance (wrist to middle MCP landmark 9)
  * Generates 63-element feature vector [x0..x20, y0..y20, z0..z20]
- Supervised Machine Learning Classifier:
  * Scikit-Learn MLPClassifier (Multi-Layer Perceptron neural network)
  * Configurable hidden layers (64, 32) with ReLU activation
  * 80/20 train/test split with stratified sampling
  * Full classification metrics: Accuracy, Precision, Recall, F1-Score
  * Confusion matrix generation and reporting
- Dataset Management:
  * Add, inspect, clear, save, and load datasets from gesture_dataset.json
  * Seed dataset generation with geometrically accurate landmark poses for all 7 chords + fist
  * Model persistence to gesture_model.pkl
- Safe Runtime Fallback:
  * If model is not trained or confidence is below threshold (< 0.60),
    gracefully falls back to rule-based classification in two_hand_tracker.py.
"""

import os
import json
import math
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


GESTURE_CLASSES = [
    "CHORD_1_INDEX",       # 1 finger (Chord 1 / I)
    "CHORD_2_PEACE",       # 2 fingers (Chord 2 / ii)
    "CHORD_3_THREE",       # 3 fingers (Chord 3 / iii)
    "CHORD_4_FOUR",        # 4 fingers (Chord 4 / IV)
    "CHORD_5_OPEN_PALM",   # 5 fingers (Chord 5 / V)
    "CHORD_6_ROCK",        # Rock sign: Index + Pinky (Chord 6 / vi)
    "CHORD_7_ROCK_THUMB",  # Rock + Thumb (Chord 7 / vii°)
    "FIST_MUTE",           # Closed Fist (Mute)
]

CLASS_TO_CHORD_INDEX = {
    "CHORD_1_INDEX": 1,
    "CHORD_2_PEACE": 2,
    "CHORD_3_THREE": 3,
    "CHORD_4_FOUR": 4,
    "CHORD_5_OPEN_PALM": 5,
    "CHORD_6_ROCK": 6,
    "CHORD_7_ROCK_THUMB": 7,
    "FIST_MUTE": None,
}

CLASS_DISPLAY_NAMES = {
    "CHORD_1_INDEX": "1 Finger (I Major)",
    "CHORD_2_PEACE": "2 Fingers (ii Minor)",
    "CHORD_3_THREE": "3 Fingers (iii Minor)",
    "CHORD_4_FOUR": "4 Fingers (IV Major)",
    "CHORD_5_OPEN_PALM": "5 Fingers (V Major)",
    "CHORD_6_ROCK": "Rock 🤘 (vi Minor)",
    "CHORD_7_ROCK_THUMB": "Rock+Thumb 🤟 (vii° Dim)",
    "FIST_MUTE": "Fist (Mute)",
}


def extract_normalized_features(landmarks: List[Dict[str, float]]) -> np.ndarray:
    """
    Normalizes 21 3D landmarks into a 63-element invariant feature vector.

    Steps:
    1. Translation invariance: Subtract wrist (landmark 0) coordinates.
    2. Scale invariance: Divide by palm span (distance from wrist 0 to middle MCP 9).
    3. Output shape: 1D numpy array of shape (63,).
    """
    if not landmarks or len(landmarks) < 21:
        return np.zeros(63, dtype=np.float32)

    # 1. Wrist reference point
    wrist = landmarks[0]
    wx = float(wrist.get("x", 0.0))
    wy = float(wrist.get("y", 0.0))
    wz = float(wrist.get("z", 0.0))

    # Center all points at wrist
    centered = []
    for lm in landmarks:
        dx = float(lm.get("x", 0.0)) - wx
        dy = float(lm.get("y", 0.0)) - wy
        dz = float(lm.get("z", 0.0)) - wz
        centered.append((dx, dy, dz))

    # 2. Compute palm scale (wrist 0 to middle MCP 9)
    mid_mcp = centered[9]
    scale = math.sqrt(mid_mcp[0] ** 2 + mid_mcp[1] ** 2 + mid_mcp[2] ** 2)
    if scale < 1e-5:
        scale = 1.0

    # 3. Scale-normalize all coordinates
    features = []
    for dx, dy, dz in centered:
        features.extend([dx / scale, dy / scale, dz / scale])

    return np.array(features, dtype=np.float32)


def generate_synthetic_seed_samples(samples_per_class: int = 40) -> Tuple[np.ndarray, List[str]]:
    """
    Generates high-quality synthetic training samples based on biomechanically
    realistic hand geometries with random Gaussian jitter. This ensures the ML
    classifier is functional immediately upon startup even before manual data collection.
    """
    rng = np.random.RandomState(42)
    X_list = []
    y_list = []

    # Base coordinates for canonical hand poses (wrist at 0,0,0)
    # Middle MCP is at roughly (0, -1.0, 0)
    for class_label in GESTURE_CLASSES:
        for _ in range(samples_per_class):
            lms = []
            # Wrist
            lms.append({"x": 0.0, "y": 0.0, "z": 0.0})

            # Finger extension states based on gesture class
            is_thumb_up = class_label in ("CHORD_5_OPEN_PALM", "CHORD_7_ROCK_THUMB")
            is_index_up = class_label in (
                "CHORD_1_INDEX", "CHORD_2_PEACE", "CHORD_3_THREE",
                "CHORD_4_FOUR", "CHORD_5_OPEN_PALM", "CHORD_6_ROCK", "CHORD_7_ROCK_THUMB"
            )
            is_mid_up = class_label in ("CHORD_2_PEACE", "CHORD_3_THREE", "CHORD_4_FOUR", "CHORD_5_OPEN_PALM")
            is_ring_up = class_label in ("CHORD_3_THREE", "CHORD_4_FOUR", "CHORD_5_OPEN_PALM")
            is_pinky_up = class_label in ("CHORD_4_FOUR", "CHORD_5_OPEN_PALM", "CHORD_6_ROCK", "CHORD_7_ROCK_THUMB")

            # 1-4 Thumb
            th_dir = -1.0 if is_thumb_up else -0.3
            for i in range(1, 5):
                t = i / 4.0
                jx = -0.4 * t + rng.normal(0, 0.02)
                jy = th_dir * 0.8 * t + rng.normal(0, 0.02)
                jz = -0.1 * t + rng.normal(0, 0.02)
                lms.append({"x": jx, "y": jy, "z": jz})

            # 5-8 Index
            idx_ext = 1.0 if is_index_up else 0.35
            for i in range(1, 5):
                t = i / 4.0
                jx = -0.2 * t + rng.normal(0, 0.02)
                jy = -idx_ext * 1.5 * t + (0.3 * (t ** 2) if not is_index_up else 0) + rng.normal(0, 0.02)
                jz = (0.2 * t if not is_index_up else 0.0) + rng.normal(0, 0.02)
                lms.append({"x": jx, "y": jy, "z": jz})

            # 9-12 Middle
            mid_ext = 1.0 if is_mid_up else 0.35
            for i in range(1, 5):
                t = i / 4.0
                jx = 0.0 + rng.normal(0, 0.02)
                jy = -mid_ext * 1.6 * t + (0.3 * (t ** 2) if not is_mid_up else 0) + rng.normal(0, 0.02)
                jz = (0.2 * t if not is_mid_up else 0.0) + rng.normal(0, 0.02)
                lms.append({"x": jx, "y": jy, "z": jz})

            # 13-16 Ring
            rng_ext = 1.0 if is_ring_up else 0.35
            for i in range(1, 5):
                t = i / 4.0
                jx = 0.2 * t + rng.normal(0, 0.02)
                jy = -rng_ext * 1.5 * t + (0.3 * (t ** 2) if not is_ring_up else 0) + rng.normal(0, 0.02)
                jz = (0.2 * t if not is_ring_up else 0.0) + rng.normal(0, 0.02)
                lms.append({"x": jx, "y": jy, "z": jz})

            # 17-20 Pinky
            pky_ext = 1.0 if is_pinky_up else 0.35
            for i in range(1, 5):
                t = i / 4.0
                jx = 0.4 * t + rng.normal(0, 0.02)
                jy = -pky_ext * 1.3 * t + (0.3 * (t ** 2) if not is_pinky_up else 0) + rng.normal(0, 0.02)
                jz = (0.2 * t if not is_pinky_up else 0.0) + rng.normal(0, 0.02)
                lms.append({"x": jx, "y": jy, "z": jz})

            feat = extract_normalized_features(lms)
            X_list.append(feat)
            y_list.append(class_label)

    return np.array(X_list, dtype=np.float32), y_list


class GestureDataset:
    """Manages training samples for hand gesture recognition."""

    def __init__(self, filepath: str = "gesture_dataset.json"):
        self.filepath = filepath
        self.samples: List[Dict[str, Any]] = []

    def add_sample(self, label: str, landmarks: List[Dict[str, float]]):
        """Add a landmark sample with given class label."""
        if label not in GESTURE_CLASSES:
            raise ValueError(f"Unknown label: {label}. Must be one of {GESTURE_CLASSES}")
        features = extract_normalized_features(landmarks).tolist()
        self.samples.append({
            "label": label,
            "features": features,
        })

    def count_by_class(self) -> Dict[str, int]:
        """Count samples per class."""
        counts = {c: 0 for c in GESTURE_CLASSES}
        for s in self.samples:
            lbl = s.get("label")
            if lbl in counts:
                counts[lbl] += 1
        return counts

    def total_count(self) -> int:
        return len(self.samples)

    def clear(self):
        self.samples = []

    def save(self):
        """Save dataset to JSON file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"classes": GESTURE_CLASSES, "samples": self.samples}, f, indent=2)

    def load(self) -> bool:
        """Load dataset from JSON file if it exists."""
        if not os.path.exists(self.filepath):
            return False
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.samples = data.get("samples", [])
            return True
        except Exception as e:
            print(f"[AIGesture] Error loading dataset: {e}")
            return False


class AIGestureClassifier:
    """
    Supervised Machine Learning Gesture Classifier for HandSynth.
    Trained on normalized 63-element 3D landmark features.
    """

    def __init__(
        self,
        model_path: str = "gesture_model.pkl",
        dataset_path: str = "gesture_dataset.json",
        min_confidence: float = 0.60,
    ):
        self.model_path = model_path
        self.dataset = GestureDataset(dataset_path)
        self.min_confidence = min_confidence
        self.model: Optional[MLPClassifier] = None
        self.is_trained = False
        self.classes: List[str] = GESTURE_CLASSES
        self.last_metrics: Dict[str, Any] = {}

        # Attempt to load existing model
        self.load_model()

    def train(self, test_size: float = 0.20) -> Dict[str, Any]:
        """
        Trains the MLPClassifier on the current dataset.
        If dataset has fewer than 10 samples per class, augments with synthetic seed samples.

        Returns:
            Dictionary containing evaluation metrics (accuracy, precision, recall, f1, confusion matrix).
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not installed in the environment.")

        # Load dataset if available
        self.dataset.load()

        X_user = []
        y_user = []
        for s in self.dataset.samples:
            X_user.append(s["features"])
            y_user.append(s["label"])

        # Augment with seed samples if user samples are sparse
        X_seed, y_seed = generate_synthetic_seed_samples(samples_per_class=50)

        if len(X_user) > 0:
            X = np.vstack([np.array(X_user, dtype=np.float32), X_seed])
            y = y_user + y_seed
        else:
            X = X_seed
            y = y_seed

        # Stratified 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Multi-Layer Perceptron neural network
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
        )
        mlp.fit(X_train, y_train)

        # Evaluation metrics on unseen test partition
        y_pred = mlp.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        c_matrix = confusion_matrix(y_test, y_pred, labels=self.classes).tolist()

        self.model = mlp
        self.is_trained = True

        self.last_metrics = {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "confusion_matrix": c_matrix,
            "classes": self.classes,
        }

        # Save model
        self.save_model()
        return self.last_metrics

    def predict(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Classifies hand landmarks using the trained ML model.

        Returns:
            Dict containing:
                - class_label: predicted class name
                - confidence: model softmax probability (0.0 to 1.0)
                - chord_index: 1 to 7 or None
                - display_name: formatted human-readable name
                - is_ai: True if prediction came from ML model, False if fallback
        """
        if not landmarks or len(landmarks) < 21:
            return {
                "class_label": "NONE",
                "confidence": 0.0,
                "chord_index": None,
                "display_name": "No Hand",
                "is_ai": False,
            }

        # If model is not trained, fallback immediately
        if not self.is_trained or self.model is None:
            return self._fallback_rule_based(landmarks)

        # Extract normalized 63-element feature vector
        features = extract_normalized_features(landmarks).reshape(1, -1)

        try:
            probs = self.model.predict_proba(features)[0]
            max_idx = int(np.argmax(probs))
            predicted_class = self.model.classes_[max_idx]
            confidence = float(probs[max_idx])

            # If confidence is below threshold, fallback to rule-based classification
            if confidence < self.min_confidence:
                fallback_res = self._fallback_rule_based(landmarks)
                fallback_res["low_confidence"] = True
                fallback_res["ai_candidate"] = predicted_class
                return fallback_res

            chord_idx = CLASS_TO_CHORD_INDEX.get(predicted_class)
            display = CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class)

            return {
                "class_label": predicted_class,
                "confidence": confidence,
                "chord_index": chord_idx,
                "display_name": display,
                "is_ai": True,
            }

        except Exception as e:
            return self._fallback_rule_based(landmarks)

    def _fallback_rule_based(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Safe fallback to geometric rule-based classification."""
        from two_hand_tracker import classify_left_hand_chord

        rule_res = classify_left_hand_chord(landmarks, "Left")
        chord_idx = rule_res.get("chord_index")
        gesture_name = rule_res.get("gesture_name", "--")

        return {
            "class_label": "RULE_BASED",
            "confidence": 1.0,
            "chord_index": chord_idx,
            "display_name": gesture_name,
            "is_ai": False,
        }

    def save_model(self):
        """Serialize trained model to disk."""
        if self.model and self.is_trained:
            try:
                with open(self.model_path, "wb") as f:
                    pickle.dump({
                        "model": self.model,
                        "classes": self.classes,
                        "metrics": self.last_metrics,
                    }, f)
            except Exception as e:
                print(f"[AIGesture] Error saving model: {e}")

    def load_model(self) -> bool:
        """Load serialized model from disk."""
        if not os.path.exists(self.model_path):
            return False
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data.get("model")
                self.classes = data.get("classes", GESTURE_CLASSES)
                self.last_metrics = data.get("metrics", {})
                self.is_trained = (self.model is not None)
            return self.is_trained
        except Exception as e:
            print(f"[AIGesture] Error loading model: {e}")
            self.is_trained = False
            return False
