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
st.write("Predict main genre from lyrics + audio features.")

# ---- inputs (example) ----
lyrics = st.text_area("Lyrics", value="")

# IMPORTANT: audio feature names must match training exactly
audio_features = ["danceability","energy","valence","speechiness","acousticness",
                  "instrumentalness","loudness","mode","duration_ms","tempo","liveness","key"]

vals = {}
for f in audio_features:
    vals[f] = st.number_input(f, value=0.0)

if st.button("Predict"):
    X_input = pd.DataFrame([{**vals, "lyrics": lyrics}])
    pred = model.predict(X_input)[0]
    st.success(f"Predicted genre: {le.inverse_transform([pred])[0]}")
