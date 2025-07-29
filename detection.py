import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, timedelta

# Overwrite yawn_log.txt at start of detection.py's execution (when the server starts)
# This clears the log for a new session.
with open("yawn_log.txt", "w") as logf:
    logf.write(f"--- New session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

# Facial landmark indices for eyes and mouth based on MediaPipe Face Mesh
# These are fixed and generally do not need modification.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth landmarks for MAR calculation (indices for top lip, bottom lip, and corners)
# These indices are approximate and can be fine-tuned if MAR is not accurate.
MOUTH_TOP_LIP = 13
MOUTH_BOTTOM_LIP = 14
MOUTH_LEFT_CORNER = 78
MOUTH_RIGHT_CORNER = 308

# Thresholds and counters for drowsiness detection
EAR_THRESHOLD = 0.25      # Eye Aspect Ratio threshold for drowsiness
MAR_THRESHOLD = 0.6       # Mouth Aspect Ratio threshold for yawning
CONSEC_EYE_FRAMES = 50    # Number of consecutive frames eyes must be below EAR_THRESHOLD
YAWN_CONSEC_FRAMES = 15   # Number of consecutive frames mouth must be above MAR_THRESHOLD to count as a yawn
YAWN_TIME_WINDOW = timedelta(minutes=30) # Time window for counting yawns (e.g., 3 yawns in 30 minutes)
YAWN_ALERT_COOLDOWN = timedelta(minutes=10) # Cooldown before a new yawn alert can be triggered (not critical for current app.py SMS logic)

class DrowsinessDetector:
    def __init__(self):
        self.eye_counter = 0            # Counts consecutive frames eyes are closed
        self.yawn_frame_counter = 0     # Counts consecutive frames mouth is open (for yawning)
        self.yawn_timestamps = []       # Stores timestamps of detected yawns
        self.ear_alert_active = False   # Flag if an EAR-based alert is currently active
        self.yawn_alert_active = False  # Flag if a YAWN-based alert is currently active
        self.last_yawn_alert_time = None # Timestamp of last yawn alert (for cooldown)
        
        # Initialize MediaPipe Face Mesh
        # max_num_faces=1: only detect one face (the driver)
        # min_detection_confidence: lower this if face not detected easily
        # min_tracking_confidence: lower this if face tracking is lost often
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Improves accuracy of facial landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_ear(self, eye_points, landmarks):
        # Calculate Euclidean distances for vertical eye landmarks
        p1 = np.array(landmarks[eye_points[1]]) # Vertical 1
        p2 = np.array(landmarks[eye_points[5]]) # Vertical 2
        A = np.linalg.norm(p2 - p1)

        p3 = np.array(landmarks[eye_points[2]]) # Vertical 3
        p4 = np.array(landmarks[eye_points[4]]) # Vertical 4
        B = np.linalg.norm(p4 - p3)

        # Calculate Euclidean distance for horizontal eye landmarks
        p5 = np.array(landmarks[eye_points[0]]) # Horizontal 1
        p6 = np.array(landmarks[eye_points[3]]) # Horizontal 2
        C = np.linalg.norm(p6 - p5)

        # Eye Aspect Ratio (EAR) formula
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_mar(self, landmarks):
        # Extract mouth landmarks
        top_lip = np.array(landmarks[MOUTH_TOP_LIP])
        bottom_lip = np.array(landmarks[MOUTH_BOTTOM_LIP])
        left_corner = np.array(landmarks[MOUTH_LEFT_CORNER])
        right_corner = np.array(landmarks[MOUTH_RIGHT_CORNER])

        # Calculate vertical distance between top and bottom lip
        vertical_dist = np.linalg.norm(top_lip - bottom_lip)
        # Calculate horizontal distance between mouth corners
        horizontal_dist = np.linalg.norm(left_corner - right_corner)

        # Mouth Aspect Ratio (MAR) formula
        mar = vertical_dist / horizontal_dist
        return mar

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        # Convert BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        current_time = datetime.now()

        ear = None
        mar = None
        alert_text = "" # Text to be displayed on the frame if drowsiness is detected

        if result.multi_face_landmarks:
            # If face landmarks are detected
            mesh_points = result.multi_face_landmarks[0]
            # Convert normalized landmark coordinates to pixel coordinates
            landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh_points.landmark]

            # Calculate EAR for left and right eyes and average them
            left_ear = self.calculate_ear(LEFT_EYE, landmarks)
            right_ear = self.calculate_ear(RIGHT_EYE, landmarks)
            ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR for mouth
            mar = self.calculate_mar(landmarks)

            # --- EAR (Eye Drowsiness) Logic ---
            if ear < EAR_THRESHOLD:
                self.eye_counter += 1
                if self.eye_counter >= CONSEC_EYE_FRAMES:
                    self.ear_alert_active = True
                    alert_text = "DROWSINESS DETECTED via EYES"
            else:
                self.eye_counter = 0
                self.ear_alert_active = False

            # --- MAR (Yawn) Logic ---
            if mar > MAR_THRESHOLD:
                self.yawn_frame_counter += 1
            else:
                # If mouth was open for enough consecutive frames, it's a yawn
                if self.yawn_frame_counter >= YAWN_CONSEC_FRAMES:
                    self.yawn_timestamps.append(current_time)
                    # Log the yawn to the file
                    with open("yawn_log.txt", "a") as logf:
                        logf.write(f"Yawn detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"[DEBUG] Yawn detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.yawn_frame_counter = 0 # Reset yawn frame counter

            # Remove yawns older than YAWN_TIME_WINDOW (e.g., 30 minutes)
            self.yawn_timestamps = [t for t in self.yawn_timestamps if current_time - t <= YAWN_TIME_WINDOW]
            
            # For debugging/logging, you can log the current yawn count to file
            # with open("yawn_log.txt", "a") as logf:
            #     logf.write(f"Yawns in last {YAWN_TIME_WINDOW.total_seconds()/60} min (at {current_time.strftime('%H:%M:%S')}): {len(self.yawn_timestamps)}\n")
            # print(f"[DEBUG] Yawns in last {YAWN_TIME_WINDOW.total_seconds()/60} min: {len(self.yawn_timestamps)}")

            # --- Yawn Alert Handling (if more than 3 yawns in time window) ---
            if len(self.yawn_timestamps) > 3: # If more than 3 yawns in the defined time window
                # Apply cooldown to avoid rapid alerts for persistent yawning
                if self.last_yawn_alert_time is None or \
                   (current_time - self.last_yawn_alert_time) > YAWN_ALERT_COOLDOWN:
                    self.yawn_alert_active = True
                    self.last_yawn_alert_time = current_time # Update last alert time
                    # If EAR alert is not already active, set MAR alert text
                    if not self.ear_alert_active: # Prioritize EAR alert text if both are active
                        alert_text = "DROWSINESS DETECTED via YAWNS"
                    with open("yawn_log.txt", "a") as logf:
                        logf.write(f"DROWSINESS DETECTED via YAWN LIMIT ({len(self.yawn_timestamps)} in {YAWN_TIME_WINDOW.total_seconds()/60}min) at {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"[ALERT] DROWSINESS DETECTED via YAWNS at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    self.yawn_alert_active = False # Cooldown active, no new alert
            else:
                self.yawn_alert_active = False # Not enough yawns to trigger alert

            # If both EAR and YAWN alerts are active, combine text or prioritize EAR
            if self.ear_alert_active and self.yawn_alert_active:
                alert_text = "DROWSINESS DETECTED (EYES & YAWNS)"
            elif self.ear_alert_active:
                alert_text = "DROWSINESS DETECTED (EYES)"
            elif self.yawn_alert_active:
                alert_text = "DROWSINESS DETECTED (YAWNS)"
            else:
                alert_text = "" # No alert

        # Return the calculated EAR, MAR, the alert text, and the MediaPipe result for potential drawing (though not used in app.py)
        return ear, mar, alert_text, result