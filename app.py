import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Mood Mirror", layout="wide")

st.title("😊 Real-Time Mood-Based Color Feedback Mirror")

start = st.button("Start Camera")
stop = st.button("Stop Camera")

frame_placeholder = st.empty()

# Load Haar Cascade Models
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        mood = "Neutral "
        color = (255, 0, 0)  # Blue (BGR)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect Smile
            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(20, 20)
            )

            # Detect Eyes
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(smiles) > 0:
                mood = "Happy "
                color = (0, 255, 0)  # Green

            elif len(eyes) == 0:
                mood = "Sad "
                color = (0, 255, 255)  # Yellow

            else:
                mood = "Neutral "
                color = (255, 0, 0)  # Blue

        # Create color overlay
        overlay = np.full(frame.shape, color, dtype=np.uint8)
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        cv2.putText(blended, mood, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb)

    cap.release()
