Drowsy Driver Detection

This project provides a real-time driver monitoring system that uses webcam video, facial landmark detection, and audio-visual notifications to detect signs of drowsiness. The system works by analyzing the Eye Aspect Ratio (EAR) and the Mouth Aspect Ratio (MAR) to issue warnings when the driver appears fatigued or sleepy.

## Features

- Real-time monitoring through the user's webcam to assess driver alertness
- Automatic detection of prolonged eye closure (EAR), which indicates drowsiness
- Yawning recognition through the mouth aspect ratio (MAR), suggesting fatigue
- Visual warnings overlaid on the video feed and audible alarms when drowsiness is detected
- A session log that records yawn events for later review

## Demo
The web application streams video from your webcam and identifies the key facial landmarks using MediaPipe. For each frame:

- It calculates the EAR and MAR values using the detected landmarks.
- If the eye aspect ratio drops below a set threshold for several consecutive frames, a drowsiness alert is triggered.
- If yawning is detected frequently within a short period, another alert is issued.
- Alerts are displayed as text on the video stream, and a sound alarm is played to draw your attention.

## Project Structure

- `app.py` is the Flask server that manages video streaming, alert notifications, and the web interface.
- `detection.py` contains all core detection logic, including EAR and MAR calculations and drowsiness detection.
- `templates/index.html` provides the front-end dashboard where the real-time monitoring and alerts are displayed.

## Installation and Usage

Before you start, make sure you have Python 3.7 or newer, along with MediaPipe, OpenCV, and Flask installed.

To set up:

1. Clone this repository to your local machine and navigate into the project directory.
2. Install dependencies by running  
   `pip install -r requirements.txt`
3. Start the application with  
   `python app.py`
4. Open your web browser and go to `http://127.0.0.1:5000/` to access the dashboard.

You can adjust the EAR and MAR thresholds in `detection.py` to make the detection more or less sensitive. For more advanced alerting, such as email or SMS notifications, you can extend the logic in `app.py`.

