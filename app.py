import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from face_emotion_predict import predict_emotion_from_frame

# Optional HF model
try:
    from text_emotion_hf import predict_text_emotion as hf_predict
except:
    hf_predict = None

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Serenity Sync – Emotion Detection", layout="centered")
st.title("🎨 Serenity Sync – Emotion Detection System")

# =====================================================
# LOAD TEXT MODEL
# =====================================================
@st.cache_resource
def load_text_assets():
    model = load_model("models/emotion_model.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

text_model, tokenizer, label_encoder = load_text_assets()

# =====================================================
# COLOR MAP
# =====================================================
EMOTION_COLORS = {
    "angry": "#E63946",
    "disgust": "#2A9D8F",
    "fear": "#6A4C93",
    "happy": "#F4A261",
    "sad": "#457B9D",
    "surprise": "#E9C46A",
    "neutral": "#8D99AE",
}

def get_color(emotion):
    return EMOTION_COLORS.get(emotion.lower(), "#000000")

# =====================================================
# MODE
# =====================================================
mode = st.radio("Select Mode", ["Text Emotion", "Face Image", "Live Camera"])

# =====================================================
# TEXT MODE (FIXED PROPERLY)
# =====================================================
if mode == "Text Emotion":

    text = st.text_area("Enter text")

    if st.button("Analyze Text"):

        if text.strip() == "":
            st.warning("Please enter text")
        else:
            emotion = None
            confidence = 0.0

            # Try HF model first
            if hf_predict:
                try:
                    emotion, confidence = hf_predict(text)
                except:
                    emotion, confidence = None, 0.0

            # Fallback to local model on failure or low-confidence HF output
            if emotion is None or confidence < 0.30:
                seq = tokenizer.texts_to_sequences([text.strip()])
                padded = pad_sequences(seq, maxlen=20)

                preds = text_model.predict(padded, verbose=0)[0]

                idx = int(np.argmax(preds))

                emotion = label_encoder.inverse_transform([idx])[0]
                emotion = emotion.strip().lower()

                confidence = float(preds[idx])

            # ✅ FIXED LOGIC (NO FORCED NEUTRAL)
            if confidence < 0.35:
                st.warning("Low confidence prediction")

            color = get_color(emotion)

            st.success(f"Emotion: {emotion}")
            st.write(f"Confidence: {confidence:.2%}")

            st.markdown(
                f"""
                <div style="
                    background:{color};
                    padding:20px;
                    color:white;
                    text-align:center;
                    font-size:22px;
                    border-radius:10px;">
                    {emotion.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================
# FACE IMAGE MODE (IMPROVED)
# =====================================================
elif mode == "Face Image":

    uploaded = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        frame = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        emotion, confidence, box = predict_emotion_from_frame(frame)

        if emotion:
            emotion = emotion.lower()

            # Draw bounding box
            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame)

            if confidence < 0.35:
                st.warning("Low confidence prediction")

            color = get_color(emotion)

            st.success(f"Emotion: {emotion}")
            st.write(f"Confidence: {confidence:.2%}")

            st.markdown(
                f"""
                <div style="
                    background:{color};
                    padding:20px;
                    color:white;
                    text-align:center;
                    font-size:22px;
                    border-radius:10px;">
                    {emotion.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No face detected")

# =====================================================
# LIVE CAMERA MODE (STABLE)
# =====================================================
elif mode == "Live Camera":

    image = st.camera_input("Capture Image")

    if image is not None:

        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        emotion, confidence, box = predict_emotion_from_frame(frame)

        if emotion:
            emotion = emotion.lower()

            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame)

            if confidence < 0.35:
                st.warning("Low confidence prediction")

            color = get_color(emotion)

            st.success(f"Emotion: {emotion}")
            st.write(f"Confidence: {confidence:.2%}")

            st.markdown(
                f"""
                <div style="
                    background:{color};
                    padding:20px;
                    color:white;
                    text-align:center;
                    font-size:22px;
                    border-radius:10px;">
                    {emotion.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No face detected")