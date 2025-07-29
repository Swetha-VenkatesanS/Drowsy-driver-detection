from flask import Flask, render_template, Response
import cv2
import threading
import winsound
from detection import DrowsinessDetector

app = Flask(__name__)
detector = DrowsinessDetector()

alarm_playing = False

def sound_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        winsound.Beep(1000, 1000)  
        alarm_playing = False

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        ear, mar, alert_text, result = detector.process_frame(frame)

        if alert_text:
            threading.Thread(target=sound_alarm).start()
            cv2.putText(frame, alert_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if ear:
            cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if mar:
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
