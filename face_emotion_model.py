import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================
# 1. CREATE REQUIRED DIRECTORIES (IF NOT EXISTS)
# =====================================================
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# =====================================================
# 2. LOAD DATASET
# =====================================================
DATA_PATH = os.path.join("data", "emotions.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("data/emotions.csv not found")

data = pd.read_csv(DATA_PATH)


# =====================================================
# 3. VALIDATE DATASET COLUMNS
# =====================================================
if "text" not in data.columns or "emotion" not in data.columns:
    raise ValueError("CSV file must contain 'text' and 'emotion' columns")


# =====================================================
# 4. ENCODE EMOTION LABELS
# =====================================================
label_encoder = LabelEncoder()
data["emotion"] = label_encoder.fit_transform(data["emotion"])

with open(os.path.join("models", "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)


# =====================================================
# 5. TOKENIZE TEXT DATA
# =====================================================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data["text"])

with open(os.path.join("models", "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)


# =====================================================
# 6. CONVERT TEXT TO SEQUENCES & PAD
# =====================================================
sequences = tokenizer.texts_to_sequences(data["text"])
X = pad_sequences(sequences, maxlen=20)
y = data["emotion"].values


# =====================================================
# 7. SPLIT DATA INTO TRAIN & TEST (SAFE, ERROR-FREE)
# =====================================================
num_classes = len(np.unique(y))
total_samples = len(y)

# Ensure at least one sample per class in test set
min_test_samples = num_classes
test_samples = int(total_samples * 0.2)

if test_samples < min_test_samples:
    test_size = min_test_samples
else:
    test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)


# =====================================================
# 8. BUILD NEURAL NETWORK MODEL
# =====================================================
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=20),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])


# =====================================================
# 9. COMPILE MODEL
# =====================================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# 10. TRAIN MODEL
# =====================================================
model.fit(
    X_train,
    y_train,
    epochs=15,
    validation_data=(X_test, y_test),
    verbose=1
)


# =====================================================
# 11. SAVE TRAINED MODEL
# =====================================================
model.save(os.path.join("models", "emotion_model.h5"))

print("Emotion model trained and saved successfully.")
