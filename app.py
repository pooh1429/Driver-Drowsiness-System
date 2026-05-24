import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import sys
import os
import threading

# --- MediaPipe Safe Import Block ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe as mp
    sys.path.append('./venv/lib/python3.9/site-packages')
    mp_face_mesh = mp.solutions.face_mesh

# --- Tuning Threshold Configuration Bounds ---
EAR_THRESHOLD = 0.23  
DROWSY_TIME_LIMIT = 2.0  

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --- Native Mac Audio Setup ---
ALARM_PATH = "alarm.mp3"  # Ensure your file matches this filename!
alarm_playing = False

def play_mac_siren():
    """Uses Mac's native system architecture to fire off audio blasts."""
    global alarm_playing
    while alarm_playing:
        # Check if file exists before running system call
        if os.path.exists(ALARM_PATH):
            # afplay is Mac's native tool; it runs perfectly without libraries
            os.system(f"afplay {ALARM_PATH}")
        else:
            # Fallback to standard internal Mac audio terminal bell if file missing
            print("\a")
            time.sleep(0.5)

def calculate_ear(eye_points):
    v1 = dist.euclidean(eye_points[1], eye_points[5])
    v2 = dist.euclidean(eye_points[2], eye_points[4])
    h = dist.euclidean(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
eyes_closed_start_time = None

print("System Active. Native Mac Audio Engaged. Press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_ear = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark])
            
            left_eye_pts = landmarks[LEFT_EYE]
            right_eye_pts = landmarks[RIGHT_EYE]
            current_ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0

            for pt in left_eye_pts: cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
            for pt in right_eye_pts: cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)

    if current_ear is not None:
        cv2.putText(frame, f"EAR: {current_ear:.2f}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if current_ear < EAR_THRESHOLD:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            else:
                if time.time() - eyes_closed_start_time >= DROWSY_TIME_LIMIT:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    # If alarm is not currently playing, spin it up in a thread
                    if not alarm_playing:
                        alarm_playing = True
                        t = threading.Thread(target=play_mac_siren)
                        t.daemon = True
                        t.start()
        else:
            eyes_closed_start_time = None
            alarm_playing = False  # Instantly turns off sound when eyes open

    cv2.imshow('Driver Drowsiness Monitor', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()