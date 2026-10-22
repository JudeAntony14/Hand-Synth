# HandSynth: Setup & Environment Guide

This guide covers the prerequisites, installation steps, and hardware checks required to run HandSynth on Windows 11.

---

## 1. System Requirements

- **Operating System**: Windows 11 (or Windows 10, 64-bit)
- **Python**: Python 3.11.x (Python 3.11.9 recommended)
- **Webcam**: Standard integrated laptop camera or USB webcam (supporting 640x480 @ 30 FPS)
- **Audio Output**: Any standard speakers or 3.5mm / Bluetooth headphones
- **Processor**: Dual-core or quad-core modern CPU (CPU-only execution; no GPU required)

---

## 2. Pinned Dependencies

The exact tested versions are recorded in [`requirements.txt`](file:///c:/Users/judea/VsCode/HandSynth/requirements.txt):

| Package | Version | Purpose |
| :--- | :--- | :--- |
| **`mediapipe`** | `0.10.21` | CPU-accelerated 21-landmark hand detection |
| **`opencv-python`** | `4.10.0.84` | Webcam video capture and real-time HUD rendering |
| **`numpy`** | `1.26.4` | DSP audio buffer calculation and phase math |
| **`sounddevice`** | `0.5.1` | Non-blocking PortAudio callback audio streaming |

---

## 3. Fresh Installation Instructions

If setting up on a new machine or re-creating the environment:

### Step 1: Clone or Navigate to Project Folder
```powershell
cd C:\Users\judea\VsCode\HandSynth
```

### Step 2: Create a Dedicated Virtual Environment
Using a local virtual environment isolates HandSynth from global Python and other projects:
```powershell
python -m venv .venv
```

### Step 3: Upgrade pip & Install Core Packages
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install "numpy==1.26.4" "sounddevice==0.5.1" "opencv-python==4.10.0.84"
```

### Step 4: Install MediaPipe & Runtime Support
```powershell
.\.venv\Scripts\pip.exe install "protobuf<5,>=4.25.3" "absl-py" "flatbuffers" "attrs" "matplotlib" "sentencepiece"
.\.venv\Scripts\pip.exe install "mediapipe==0.10.21" --no-deps
```

*(Alternatively, run `.\.venv\Scripts\pip.exe install -r requirements.txt`)*

---

## 4. Hardware Verification Checks

Before running the full application, verify that your camera and audio output devices are functional.

### Camera Quick Test
Run this single command in PowerShell:
```powershell
.\.venv\Scripts\python.exe -c "import cv2; cap = cv2.VideoCapture(0); print('Camera functional:', cap.isOpened()); cap.release()"
```
**Expected Output**: `Camera functional: True`

### Audio Output Quick Test
Run the standalone audio test script:
```powershell
.\.venv\Scripts\python.exe test_audio.py
```
**Expected Output**: Plays a 440 Hz test tone, sweeps to 880 Hz, and finishes with a clean fade-out.

---

## 5. Troubleshooting Common Issues

### Issue 1: "Could not open webcam"
- **Cause**: Another application is using the camera (Zoom, Teams, Discord, or a web browser).
- **Fix**: Close all video conferencing software, or check Windows Privacy Settings:
  `Settings` $\rightarrow$ `Privacy & security` $\rightarrow$ `Camera` $\rightarrow$ `Let desktop apps access your camera` (set to **ON**).

### Issue 2: Audio Stutter or Glitches
- **Cause**: CPU throttling or high buffer load.
- **Fix**: In [`synth_engine.py`](file:///c:/Users/judea/VsCode/HandSynth/synth_engine.py), increase `block_size` from `256` to `512` samples.

### Issue 3: Inverted Movement
- HandSynth automatically applies horizontal mirror mode (`cv2.flip(frame, 1)`). Moving your physical hand right moves the cursor right.
