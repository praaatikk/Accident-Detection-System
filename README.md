Accident Detection System (Enhanced Version)

Original Project by: @krishrustagi

Modified and Enhanced by: Pratik Gadekar
Last Updated: October 2025



--- Overview

This enhanced Accident Detection System uses Computer Vision (OpenCV) and Deep Learning (Keras + CNN) to detect accidents in real-time from a webcam or video footage.

The system identifies accidents, records evidence, and automatically sends an alert email with a screenshot of the detected accident when the confidence level exceeds 90%.




--- Demonstration

(You can later upload your own demo GIF or image here.)





--- Features

 Real-time accident detection from webcam or video
 Email alert system with attached screenshot
 Accident confidence threshold (90%)
 Automatic screenshot storage (organized neatly inside accident_records/)
 Alarm alert sound for immediate attention
 Clean, modular Python code compatible with TensorFlow 2.x





---Technologies Used

OpenCV – (Open Source Computer Vision Library) for image processing

Keras / TensorFlow – for deep learning model loading and prediction

NumPy & Pandas – for data manipulation

SMTPLib – (Simple Mail Transfer Protocol Library) for sending alert emails

Flask – for backend integration (optional, if you build a web interface later)





---Folder Structure

Accident-Detection-System/
│
├── accident-classification.ipynb   # Model training notebook
├── model.json                      # Model architecture
├── model_weights.h5                # Trained model weights
├── camera.py                       # Video-based detection
├── cam_webcam.py                   # Webcam-based detection with email alerts
├── requirements.txt
├── accident_records/               # Automatically stores accident screenshots
└── README.md





---How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/praaatikk/Accident-Detection-System.git
cd Accident-Detection-System

2️⃣ Install requirements
pip install -r requirements.txt

3️⃣ Run the detection

To use a webcam:

python cam_webcam.py


To use a video file:

python camera.py

==Email Alert Setup (Important)

Enable 2-Step Verification on your Gmail account.

Create an App Password under Security → App passwords.

Copy the 16-character password.

Replace it in the code where EMAIL_PASSWORD is defined.

Use your email for SENDER_EMAIL and RECEIVER_EMAIL.




---Future Enhancements

   Integration with emergency services (automated SOS calls)
   Real-time GPS-based accident location tracking
   Web dashboard for visualizing accident statistics




---Conclusion

This project demonstrates how deep learning and computer vision can help improve road safety by automating accident detection and enabling faster emergency response.
The added alert system ensures that no critical event goes unnoticed — helping save valuable time and lives.





---Credits

This project is based on the original work by @krishrustagi
, with modifications and additional features (webcam integration, email alerts, auto-recording, and sound alerts) developed by Pratik Gadekar.