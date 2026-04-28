from transformers import pipeline

# Load a compact emotion classifier from Hugging Face
_hf_pipeline = None
try:
    _hf_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )
except Exception:
    _hf_pipeline = None


_LABEL_MAP = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
}


def predict_text_emotion(text):
    """Return (emotion_label, confidence) using HF pipeline.

    If the pipeline fails or isn't available, returns (None, 0.0).
    """
    if _hf_pipeline is None:
        return None, 0.0

    try:
        out = _hf_pipeline(text)
        if not out:
            return None, 0.0
        res = out[0]
        label = res.get("label", "").lower()
        score = float(res.get("score", 0.0))
        mapped = _LABEL_MAP.get(label, label)
        return mapped, score
    except Exception:
        return None, 0.0
