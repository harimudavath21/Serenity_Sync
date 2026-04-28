import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =====================================================
# LOAD FACE DETECTOR
# =====================================================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# =====================================================
# LOAD PRETRAINED MODEL
# =====================================================
emotion_model = load_model(
    "models/fer2013_mini_XCEPTION.102-0.66.hdf5",
    compile=False
)

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# =====================================================
# IMPROVED PREDICTION FUNCTION
# =====================================================
def predict_emotion_from_frame(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None, None, None

    (x, y, w, h) = faces[0]

    face_roi = gray[y:y+h, x:x+w]

    # 🔥 Improve lighting
    face_roi = cv2.equalizeHist(face_roi)

    # 🔥 Reduce noise
    face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)

    # Resize to model input size
    face_roi = cv2.resize(face_roi, (64, 64))

    # Normalize
    face_roi = face_roi.astype("float32") / 255.0

    # Add channel dimension
    face_roi = np.expand_dims(face_roi, axis=-1)

    # Add batch dimension
    face_roi = np.expand_dims(face_roi, axis=0)

    preds = emotion_model.predict(face_roi, verbose=0)[0]

    emotion_index = np.argmax(preds)
    confidence = float(preds[emotion_index])
    emotion = emotion_labels[emotion_index]

    # Low-confidence predictions should still return the detected emotion,
    # but the app may warn the user if the score is low.
    return emotion, confidence, (x, y, w, h)
