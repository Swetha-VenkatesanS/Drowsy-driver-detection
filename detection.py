import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, timedelta


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP_LIP = 13
MOUTH_BOTTOM_LIP = 14
MOUTH_LEFT_CORNER = 78
MOUTH_RIGHT_CORNER = 308

EAR_THRESHOLD = 0.25      
MAR_THRESHOLD = 0.6       
CONSEC_EYE_FRAMES = 50    
YAWN_CONSEC_FRAMES = 15   
YAWN_TIME_WINDOW = timedelta(minutes=30) 
YAWN_ALERT_COOLDOWN = timedelta(minutes=10) 

class DrowsinessDetector:
    def __init__(self):
        self.eye_counter = 0            
        self.yawn_frame_counter = 0     
        self.yawn_timestamps = []       
        self.ear_alert_active = False   
        self.yawn_alert_active = False  
        self.last_yawn_alert_time = None 
        
       
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_ear(self, eye_points, landmarks):
        
        p1 = np.array(landmarks[eye_points[1]]) 
        p2 = np.array(landmarks[eye_points[5]])
        A = np.linalg.norm(p2 - p1)

        p3 = np.array(landmarks[eye_points[2]]) 
        p4 = np.array(landmarks[eye_points[4]]) 
        B = np.linalg.norm(p4 - p3)

        p5 = np.array(landmarks[eye_points[0]])
        p6 = np.array(landmarks[eye_points[3]]) 
        C = np.linalg.norm(p6 - p5)

        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_mar(self, landmarks):
        
        top_lip = np.array(landmarks[MOUTH_TOP_LIP])
        bottom_lip = np.array(landmarks[MOUTH_BOTTOM_LIP])
        left_corner = np.array(landmarks[MOUTH_LEFT_CORNER])
        right_corner = np.array(landmarks[MOUTH_RIGHT_CORNER])

        vertical_dist = np.linalg.norm(top_lip - bottom_lip)
        horizontal_dist = np.linalg.norm(left_corner - right_corner)

        mar = vertical_dist / horizontal_dist
        return mar

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        current_time = datetime.now()

        ear = None
        mar = None
        alert_text = "" 

        if result.multi_face_landmarks:
            mesh_points = result.multi_face_landmarks[0]
            landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh_points.landmark]

            left_ear = self.calculate_ear(LEFT_EYE, landmarks)
            right_ear = self.calculate_ear(RIGHT_EYE, landmarks)
            ear = (left_ear + right_ear) / 2.0
            
            mar = self.calculate_mar(landmarks)

            if ear < EAR_THRESHOLD:
                self.eye_counter += 1
                if self.eye_counter >= CONSEC_EYE_FRAMES:
                    self.ear_alert_active = True
                    alert_text = "DROWSINESS DETECTED via EYES"
            else:
                self.eye_counter = 0
                self.ear_alert_active = False

            if mar > MAR_THRESHOLD:
                self.yawn_frame_counter += 1
            else:
                if self.yawn_frame_counter >= YAWN_CONSEC_FRAMES:
                    self.yawn_timestamps.append(current_time)
                self.yawn_frame_counter = 0 
            self.yawn_timestamps = [t for t in self.yawn_timestamps if current_time - t <= YAWN_TIME_WINDOW]
            
            if len(self.yawn_timestamps) > 3: 
                if self.last_yawn_alert_time is None or \
                   (current_time - self.last_yawn_alert_time) > YAWN_ALERT_COOLDOWN:
                    self.yawn_alert_active = True
                    self.last_yawn_alert_time = current_time 
                    if not self.ear_alert_active: 
                        alert_text = "DROWSINESS DETECTED via YAWNS"
                else:
                    self.yawn_alert_active = False 
            else:
                self.yawn_alert_active = False

            if self.ear_alert_active and self.yawn_alert_active:
                alert_text = "DROWSINESS DETECTED (EYES & YAWNS)"
            elif self.ear_alert_active:
                alert_text = "DROWSINESS DETECTED (EYES)"
            elif self.yawn_alert_active:
                alert_text = "DROWSINESS DETECTED (YAWNS)"
            else:
                alert_text = "" 
        return ear, mar, alert_text, result
