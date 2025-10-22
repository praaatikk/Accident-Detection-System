from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
from datetime import datetime
from threading import Lock
from detection import AccidentDetectionModel  # Make sure this file exists

app = Flask(__name__)

# Initialize model
model = AccidentDetectionModel("model_fixed.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

# Global variables
video_path = os.path.join(os.getcwd(), "test_video.mp4")
accident_log = []
log_lock = Lock()  # thread-safe logging


def generate_frames():
    """Generator function that yields processed video frames."""
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")

        # Send error frame to stream
        error_frame = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Error: Cannot open video file",
                    (30, 150), font, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    try:
        while True:
            ret, frame = video.read()
            if not ret or frame is None:
                # Loop video if end reached
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Convert to RGB (assuming model expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(rgb_frame, (250, 250))

            # Model prediction
            pred, prob = model.predict_accident(roi[np.newaxis, :, :])

            # Interpret probabilities safely
            try:
                if pred == "Accident":
                    prob_val = round(float(prob[0][0]) * 100, 2)
                    color = (0, 0, 255)
                else:
                    prob_val = round(float(prob[0][1]) * 100, 2)
                    color = (0, 255, 0)
            except Exception:
                prob_val = 0.0
                color = (255, 255, 255)

            # Draw overlay
            cv2.rectangle(frame, (0, 0), (300, 45), color, -1)
            cv2.putText(frame, f"{pred} {prob_val}%", (20, 30),
                        font, 1, (255, 255, 255), 2)

            # Log high-confidence accidents
            if pred == "Accident" and prob_val > 80:
                log_accident(prob_val)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except GeneratorExit:
        video.release()
        print("[INFO] Video stream stopped.")
    finally:
        video.release()


def log_accident(probability):
    """Record accident detection events with timestamp."""
    global accident_log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with log_lock:
        if (not accident_log or
            (datetime.now() - datetime.strptime(
                accident_log[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")).seconds > 2):

            accident_log.append({
                "timestamp": timestamp,
                "probability": probability
            })

            # Limit log size
            if len(accident_log) > 50:
                accident_log.pop(0)


@app.route('/')
def index():
    """Render main dashboard page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/accident_logs')
def get_accident_logs():
    """Return JSON list of logged accidents."""
    with log_lock:
        return jsonify(accident_log)


@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """Clear accident logs."""
    global accident_log
    with log_lock:
        accident_log = []
    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
