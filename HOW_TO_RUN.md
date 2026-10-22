# HandSynth: How to Run & Performance Guide

This document explains how to launch HandSynth, master the hand gestures, play songs using musical scales, and run a demonstration for your college project evaluation.

---

## 1. Quick Launch Command

### Launch from Workspace Folder:
```powershell
cd C:\Users\judea\VsCode\HandSynth
.\.venv\Scripts\python.exe main.py
```

### Launch from ANY Folder / Terminal in Windows:
```powershell
& "C:\Users\judea\VsCode\HandSynth\.venv\Scripts\python.exe" "C:\Users\judea\VsCode\HandSynth\main.py"
```

---

## 2. Two-Handed Synthesizer Performance Mode

HandSynth features an **independent two-handed performance engine** (toggle with **`C`**):

### LEFT HAND: Chord Selection & Quality (Major / Minor)
- **Finger-Count Chord Selection** (ZERO pinch required):
  * **1 Finger** (Index): **Chord 1** ($I$, e.g. $C\text{ Maj}$)
  * **2 Fingers** (Index + Middle): **Chord 2** ($ii$, e.g. $D\text{ min}$)
  * **3 Fingers** (Index + Middle + Ring): **Chord 3** ($iii$, e.g. $E\text{ min}$)
  * **4 Fingers** (Index + Middle + Ring + Pinky, thumb tucked): **Chord 4** ($IV$, e.g. $F\text{ Maj}$)
  * **5 Fingers** (Fully Open Palm): **Chord 5** ($V$, e.g. $G\text{ Maj}$) - *instant, robust detection with relaxed thumb separation!*
  * **6 Gesture** (🤘 Rock sign: Index + Pinky, Middle & Ring folded): **Chord 6** ($vi$, e.g. $A\text{ min}$)
  * **7 Gesture** (🤟 Rock+Thumb sign: Index + Pinky + Thumb, Middle & Ring folded): **Chord 7** ($vii^\circ$, e.g. $B\text{ dim}$)
  * **Fist Gesture** (Closed fist): **Instant Emergency Mute** (0 ms latency)
- **Smooth Monotonic Pitch Progression**:
  * Scale degrees 1 through 7 ascend monotonically in a single register ($C4 \rightarrow D4 \rightarrow E4 \rightarrow F4 \rightarrow G4 \rightarrow A4 \rightarrow B4$). **Chord 4 ($F4$) to Chord 5 ($G4$) transitions smoothly up by a whole tone with ZERO octave drops!**
- **Dynamic Major / Minor Quality on Left Hand Tilt Only**:
  * **Tilt LEFT** ($< -15^\circ$ / $-0.26\text{ rad}$ held for 3 frames): Forces the chord to **MINOR** ($C\text{ min}$, $D\text{ min}$, $G\text{ min}$).
  * **Tilt RIGHT** ($> +15^\circ$ / $+0.26\text{ rad}$ held for 3 frames): Forces the chord to **MAJOR** ($C\text{ Maj}$, $D\text{ Maj}$, $B\text{ Maj}$).
  * **Upright / Neutral** ($-15^\circ \longleftrightarrow +15^\circ$): Preserves natural diatonic triad quality ($I$ Maj, $ii$ min, $iii$ min, $IV$ Maj, $V$ Maj, $vi$ min, $vii^\circ$ dim). Moving fingers to change chords inside this zone never alters chord quality.
  * **Hysteresis deadband**: Tilt must return past $\pm 10^\circ$ to release back to diatonic scale quality.
  * *Left hand tilt controls Major/Minor chord quality ONLY (does not alter tone or gain).*
- **Anti-Glitch Debouncing**:
  * Features temporal debouncing that rejects single-frame landmark twitches, locking chords securely during movement while keeping Fist mute instantaneous.

### RIGHT HAND: Chord Variations (7ths, Diminished, etc.), Octave & Dynamics
- **Right Hand Thumb Octave Transposition (+1 Octave)**:
  * **Thumb DOWN / Folded**: Base Octave 4 (**`C4 Major`**).
  * **Thumb RAISED / Extended Up**: **+1 Octave Transposition** (**`C5 Major`**). Instantly shifts the active chord up by a full musical octave (+12 semitones) with 2-frame stability debounce!
- **Finger Pointing Chord Variations / Extensions**:
  * **Point INDEX Finger** (👆 or 1 finger): **7th Variation** ($Maj7$, $m7$, Dominant $7$, e.g. $C4\text{ Maj7}$ / $C5\text{ Maj7}$)
  * **Point MIDDLE Finger or 2 Fingers (✌)**: **Diminished Variation** ($dim7$)
  * **Point RING Finger or 3 Fingers**: **Suspended Variation** ($sus4$)
  * **Point PINKY Finger** (🤙): **Augmented Variation** ($aug$)
  * **OPEN PALM** (4 fingers): **9th Extension** ($Maj9$, $min9$, $add9$)
  * **Relaxed / Fist** (0 fingers): **Standard Triad**
- **Pitch Transposition Deadband**: Lateral tilt features a $\pm 16^\circ$ deadband so ordinary hand resting or finger pointing has strictly **$0.0$ transposition**. Deliberate tilt past $\pm 16^\circ$ smoothly glides pitch.
- **Vertical Volume Elevation**: Move your right hand **UP** (top of screen = 100% volume) and **DOWN** (bottom of screen = 0% volume).

### DUAL-VCA AMPLITUDE & STANDALONE PLAYBACK
- **Standalone Left Hand Mode**: If the right hand is absent or resting, the left hand plays chords independently at a comfortable default volume (75%)!
- **Dual-VCA Combination**: When both hands are active: $\text{Amplitude} = \text{LeftGain} \times \text{RightVolume}$.
- **Glitch-Free Tracking**: Temporal nearest-neighbor position caching and mirrored-handedness decoding prevent hand flickering or accidental swaps.
- **Grace Period (500 ms)**: Temporary tracking drops are smoothed for 500 ms; past 500 ms, the system performs a safe release with **zero stuck notes**.

---

## 3. Custom AI Gesture Classifier

HandSynth integrates a real **Machine Learning Gesture Classifier** trained on normalized 3D hand landmarks:
- **Mathematical Invariance**: 63-element feature vector ($21 \times 3$) centered at the wrist (landmark 0) and normalized by the palm span (wrist to middle MCP landmark 9). Scale- and distance-invariant.
- **Supervised ML Architecture**: Scikit-Learn Multi-Layer Perceptron (`MLPClassifier`) neural network with stratified 80/20 train/validation split.
- **Record Live Samples**: Press **`R`** to capture the active hand pose into `gesture_dataset.json`.
- **Train Classifier On-The-Fly**: Press **`G`** to retrain the neural network; outputs training accuracy, F1-score, and confusion matrix.
- **Fault-Tolerant Fallback**: If the model is not yet trained or confidence is low, it seamlessly falls back to the deterministic geometric classifier.

---

## 4. Hand Gestures & Control Reference (Single-Hand Solo Mode)

HandSynth tracks **21 hand landmarks** in real time and classifies both continuous coordinate positions and discrete hand poses:

| Gesture / Movement | Visual Feedback | Synthesis Behavior |
| :--- | :--- | :--- |
| **Move Hand Left $\leftrightarrow$ Right** | Ball slides along bottom Octave Ribbon | **Continuous Pitch Glide** across 3 full octaves (C3 to C6: 130 - 1046 Hz). |
| **Move Hand Up $\updownarrow$ Down** | Vertical `VOL` bar rises and falls | **Dynamic Volume**: Moving hand **UP** increases volume to 100%; moving **DOWN** softens to 0%. |
| **Thumb-Index Pinch** | Rings around Thumb (#4) & Index (#8) turn **Bright Green** | **Note-On Trigger**: Engages attack fade-in. Sound continues while held. |
| **Pinch Release** | Rings turn **Amber** | **Note-Off Release**: Smooth 15 ms fade-out to absolute silence (no pop). |
| **Peace / V-Sign**<br/>*(Index + Middle extended)* | Screen displays `[GESTURE: V-SIGN (+1 OCT)]` in yellow | **+1 Octave Transpose**: Shifts pitch up by a full musical octave (+12 semitones) instantly. |
| **Fist Gesture**<br/>*(All fingers curled)* | Screen displays `[GESTURE: FIST (MUTE)]` in red | **Instant Choke / Mute**: Safely silences any active note immediately. |
| **Open Hand** | Rings display `PINCH TO PLAY` | Hand position continues tracking pitch and volume without triggering sound. |
| **Hand Leaves View** | Status switches to `SEARCHING FOR HAND...` | **Safety Kill**: Automatically silences any sound to prevent hanging notes. |

---

## 3. Controls Reference (Mouse & Keyboard)

Every control on screen is **100% interactive**: you can either **click/drag directly with your mouse** or use **keyboard shortcuts**:

| Control | Mouse Action | Hotkey | Function & Details |
| :--- | :--- | :--- | :--- |
| **Root Key Selector** | Click `[C]`, `[D]`, `[E]`, `[F]`, `[G]`, `[A]`, `[B]` or `[#]` | **`K`** / **`#`** | Transposes all notes to any musical key. **Defaults to C**. |
| **Scale & Tunes** | Click any scale pill in the rack | **`S`** | Cycles: `PENTATONIC` $\rightarrow$ `BLUES` $\rightarrow$ `MAJOR` $\rightarrow$ `MINOR` $\rightarrow$ `DORIAN` $\rightarrow$ `HIRAJOSHI` $\rightarrow$ `CHROMATIC` $\rightarrow$ `FREE` |
| **Tone / Cutoff Filter** | **Click or drag** slider bar horizontally | **`T`** | Low-pass filter: **Warm Sub** (mellow, sub-bass) $\longleftrightarrow$ **Bright Sizzle** (crisp lead). |
| **Arpeggiator** | Click `[ARPEGGIATOR]` button | **`A`** | Automatically arpeggiates chords at 120 BPM: `OFF` $\rightarrow$ `UP` $\rightarrow$ `BOUNCE` |
| **Waveforms** | Click `[SINE]`, `[SAW]`, `[SQUARE]`, `[TRIANGLE]` | **`1` - `4`** | `1`: Sine (ambient) · `2`: Saw (warm poly Juno) · `3`: Square (Moog reed) · `4`: Triangle (Rhodes) |
| **Delay / Echo FX** | Click `[ECHO / DELAY]` button | **`E`** | 250 ms circular delay buffer with 40% feedback for spatial reverberant depth |
| **Performance Mode** | Click header mode pill | **`C` / `M`** | `TWO-HAND CHORDS` $\longleftrightarrow$ `SOLO MODE` (`PINCH` or `CONTINUOUS THEREMIN`) |
| **Fullscreen** | Press `[F]` or click maximize | **`F`** | Smoothly scales the entire workstation to fill your monitor resolution |
| **Audition Note** | Press Spacebar | **`SPACE`** | Manual sound audition toggle without webcam pinch |
| **Safe Exit** | Press `Q` or `ESC` | **`Q` / `ESC`** | Guaranteed cleanup of camera, audio stream, and windows |

---

## 4. How to Play Real Songs (Musical Flexibility)

HandSynth is a true expressive instrument, not a soundboard playing canned tunes.

### Recommended Beginner Mode: `SCALE: PENTATONIC`
Press **`S`** until the top header reads `SCALE: PENTATONIC`.
- In Pentatonic mode, every note on the horizontal ribbon snaps to notes in the Pentatonic scale (`C - D - E - G - A`).
- **Why it's great**: There are no dissonant half-steps. Whatever you play will sound musical and harmonious!

### Tutorial: Playing "Twinkle Twinkle Little Star"
1. Make sure `SCALE: MAJOR` or `PENTATONIC` is selected.
2. Locate the note name on the HUD (e.g. `NOTE: C4`, `NOTE: G4`).
3. Follow this horizontal position sequence by sliding your hand and pinching:
   - **C4 - C4** *(Twin-kle)*: Hand on left side, pinch twice.
   - **G4 - G4** *(twin-kle)*: Slide hand right to G4, pinch twice.
   - **A4 - A4** *(lit-tle)*: Slide slightly right to A4, pinch twice.
   - **G4** *(star)*: Slide back to G4, hold pinch.
   - **F4 - F4 - E4 - E4 - D4 - D4 - C4**: Step gradually to the left.

### Tutorial: Space Theremin Solo
1. Press **`M`** to switch to `MODE: CONTINUOUS`.
2. Press **`S`** to switch to `SCALE: FREE`.
3. Press **`2`** for Sawtooth or **`1`** for Sine.
4. Press **`E`** to turn `ECHO: ON`.
5. Wave your hand freely in the air:
   - Move smoothly left and right for sci-fi pitch sweeps.
   - Slightly shake your wrist for natural, expressive **vibrato**.
   - Move your hand up and down to create swelling dynamic crescendos.

---

## 6. College Project Demo Script (Presentation Sequence)

Follow this 2-minute sequence to demonstrate the full technical breadth of HandSynth:

1. **Launch**:
   Run `.\.venv\Scripts\python.exe main.py`. Point out that the system starts at 30 FPS with dual-hand tracking.
2. **Two-Handed Chords & Dual-VCA (Key Feature)**:
   - Hold up your **LEFT hand** with 1 finger: HUD displays **`C Major`**.
   - Show 2 fingers: HUD displays **`D Minor`**. Tilt right to switch to **`D Major`**!
   - Show 5 fingers (Open Palm): HUD displays **`G Major`**.
   - Flash the **Rock Sign 🤘**: HUD displays **`A Minor`**.
   - Flash the **Rock+Thumb 🤟**: HUD displays **`B Diminished`**. Tilt right to switch to **`B Major`**!
   - Raise your **RIGHT hand**: Point your index finger to add a 7th (**`G7`**, **`C Maj7`**), point middle finger for diminished (**`dim7`**), move your hand up and down to swell volume dynamically!
3. **AI Gesture Classifier (Machine Learning Component)**:
   - Form a gesture with your left hand and press **`R`**: Terminal records the 63-element feature vector.
   - Press **`G`**: Show real-time ML neural network training, printing test accuracy, F1 score, and the confusion matrix in the console.
4. **Tone Filter Modulation (Warm Sub <-> Bright Lead)**:
   - Drag the on-screen **Tone Slider** to show the 2-pole low-pass filter altering harmonic brightness (6.7x harmonic boost).
   - Press **`O`** to switch to `GESTURE TILT`: Tilt your hand to sweep the filter dynamically with natural wrist roll!
5. **Multi-Instrument Synthesis & Delay Effect**:
   - Press **`1`**, **`2`**, **`3`**, **`4`** to switch between Sine, Sawtooth, Square, and Triangle instruments.
   - Press **`E`** to toggle Delay / Echo effect for rich acoustic reverberation.
6. **Safety & Teardown**:
   - Make a **Fist** with your left hand: Sound instantly chokes to complete silence.
   - Press **`Q`**: All audio streams, camera handles, and windows close cleanly.
