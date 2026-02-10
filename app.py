import os
import sys
import streamlit as st
import joblib
import pandas as pd

# Ensure the folder containing app.py is on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Must be imported BEFORE joblib.load
from model_utils import AudioTextCombiner  # noqa: F401

model = joblib.load(os.path.join(HERE, "best_multimodal_model.joblib"))
le = joblib.load(os.path.join(HERE, "label_encoder.joblib"))

st.title("Spotify Genre Predictor")
st.write("Type a word and predict the music genre.")

# ---- single word input ----
word = st.text_input("Enter a word (e.g., love, party, sad)")

# Default audio feature values (neutral / mean-like)
default_audio = {
    "danceability": 0.5,
    "energy": 0.5,
    "valence": 0.5,
    "speechiness": 0.1,
    "acousticness": 0.5,
    "instrumentalness": 0.0,
    "loudness": -10.0,
    "mode": 1,
    "duration_ms": 180000,
    "tempo": 120.0,
    "liveness": 0.2,
    "key": 5
}

if st.button("Predict") and word.strip():
    X_input = pd.DataFrame([{**default_audio, "lyrics": word}])
    pred = model.predict(X_input)[0]
    genre = le.inverse_transform([pred])[0]
    st.success(f"Predicted genre: {genre}")
