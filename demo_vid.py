import cv2
import numpy as np
from keras.models import model_from_json
import os

# -----------------------------
# Load Keras model from JSON + weights
# -----------------------------
with open("model.json", "r") as json_file:
    model_json_str = json_file.read()

model = model_from_json(model_json_str)
model.build((None, 250, 250, 3))  # Build the model before loading weights
model.load_weights("model_weights.h5")

font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    # Use video file instead of camera
    video_path = 'test_video.mp4'  # Make sure this file exists in the same folder or give full path
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            print("End of video or cannot read frame.")
            break  # Exit loop when video ends or frame is invalid

        # Preprocess the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension

        # Prediction
        predictions = model.predict(roi)
        pred_idx = np.argmax(predictions, axis=1)[0]
        classes = ["No Accident", "Accident"]
        pred_label = classes[pred_idx]
        prob = round(predictions[0][pred_idx]*100, 2)

        if pred_label == "Accident":
            # Optional beep alert
            # if prob > 90:
            #     os.system("say beep")  # macOS; on Windows you could use winsound.Beep

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred_label} {prob}%", (20, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
