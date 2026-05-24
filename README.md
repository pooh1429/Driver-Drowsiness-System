# Real-Time Driver Drowsiness Detection System

A low-latency, edge-based computer vision application designed to mitigate automotive risks by tracking driver fatigue in real-time. This system utilizes facial geometric analytics to evaluate eyelid blinking states and triggers instant localized audio alerts upon detection of prolonged eye closure.

---

## How It Works: Eye Aspect Ratio (EAR)

Instead of routing heavy pixel arrays through resource-heavy deep neural networks, the system leverages facial land-marking coordinates to calculate structural geometry.



The **Eye Aspect Ratio (EAR)** is computed using the Euclidean distance between vertical eye landmarks relative to horizontal landmark boundaries:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

* **Eyes Open:** The ratio remains stable (typically between `0.25` and `0.30`).
* **Eyes Closed:** The vertical coordinates converge toward zero, causing the EAR value to drop sharply below the detection threshold (`0.23`). If this state persists continuously for more than `2.0 seconds`, the alert sequence is initialized.

---

## Tech Stack & Architecture

* **Core Language:** Python 3.9+
* **Computer Vision Processing:** OpenCV (Webcam capture loop handling)
* **Facial Coordinate Mesh Engine:** Google MediaPipe Face Mesh (Traces 468 distinct landmarks)
* **Mathematical Infrastructure:** SciPy (Vector distance analysis) & NumPy
* **Native Audio Processing:** Multi-threaded system integration utilizing macOS `afplay`

---

## Local Deployment Setup

### 1. Environment Sandbox Initialization
Clone down the workspace and set up a clean, isolated virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
