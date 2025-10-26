import cv2
import numpy as np
from keras.models import model_from_json
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage


# -----------------------------
# Gmail configuration
# -----------------------------
SENDER_EMAIL = "pratikksecondaryacc@gmail.com"
RECEIVER_EMAIL = "pratikksecondaryacc@gmail.com"
APP_PASSWORD = "xtbzqqbisteqltzw"

def send_email_alert(image_path, prob):
    msg = EmailMessage()
    msg['Subject'] = f'Accident Detected Alert ({prob}%)'
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg.set_content(f'Accident detected with probability {prob}%.\nSee attached image.')

    with open(image_path, 'rb') as f:
        img_data = f.read()
    msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            print("[EMAIL] Alert sent successfully!")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

# -----------------------------
# Load Keras model from JSON + weights
# -----------------------------
with open("model.json", "r") as json_file:
    model_json_str = json_file.read()

model = model_from_json(model_json_str)

# Build model with dummy input to load weights
dummy_input = np.zeros((1, 250, 250, 3))
model(dummy_input)
model.load_weights("model_weights.h5")

font = cv2.FONT_HERSHEY_SIMPLEX

ACCIDENT_FOLDER = "accident_records"
os.makedirs(ACCIDENT_FOLDER, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None
recording = False

def start_webcam():
    global video_writer, recording

    video = cv2.VideoCapture(0)  # 0 = default webcam
    if not video.isOpened():
        print("Error: Cannot access webcam")
        return

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            print("Cannot read frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        roi = np.expand_dims(roi, axis=0)

        predictions = model.predict(roi)
        pred_idx = np.argmax(predictions, axis=1)[0]
        classes = ["No Accident", "Accident"]
        pred_label = classes[pred_idx]
        prob = round(predictions[0][pred_idx]*100, 2)

        if pred_label == "Accident" and prob >= 90:
            cv2.rectangle(frame, (0, 0), (360, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"[ALERT] {pred_label} ({prob}%)", (10, 30), font, 0.8, (0, 255, 255), 2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ACCIDENT_FOLDER}/accident_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[ALERT] Accident detected ({prob}%) — saved to {filename}")

            # Send Gmail alert
            send_email_alert(filename, prob)

            if not recording:
                video_writer = cv2.VideoWriter(f"{ACCIDENT_FOLDER}/accident_{timestamp}.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                recording = True

        if recording:
            video_writer.write(frame)

        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    if recording:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_webcam()
