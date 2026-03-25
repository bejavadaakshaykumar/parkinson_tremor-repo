import streamlit as st
import numpy as np
import pickle
import parselmouth
import tempfile

# Load model
model = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Parkinson Disease Detection using Voice")

# Upload audio
audio_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])

# Feature extraction function
def extract_features(file_path):
    sound = parselmouth.Sound(file_path)

    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]

    fo = np.mean(pitch_values)
    fhi = np.max(pitch_values)
    flo = np.min(pitch_values)

    harmonicity = sound.to_harmonicity()
    hnr = harmonicity.values.mean()

    # ⚠️ Simplified (replace later with full 22 features)
    features = [
        fo, fhi, flo,
        0.005, 0.00003, 0.002, 0.004, 0.006,
        0.03, 0.35, 0.015, 0.02, 0.018, 0.045,
        0.015, hnr, 0.35, 0.75,
        -5.2, 0.20, 2.0, 0.20
    ]

    return np.array(features).reshape(1, -1)


# Prediction
if audio_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        file_path = tmp.name

    features = extract_features(file_path)

    features = scaler.transform(features)

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    st.subheader("Result")

    if prediction[0] == 1:
        st.error("Parkinson Detected")
    else:
        st.success("Healthy")

    st.write(f"Healthy Probability: {probability[0][0]*100:.2f}%")
    st.write(f"Parkinson Probability: {probability[0][1]*100:.2f}%")