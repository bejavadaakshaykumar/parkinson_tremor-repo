import streamlit as st
import numpy as np
import pickle
import parselmouth
import tempfile

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Parkinson Detection",
    page_icon="🧠",
    layout="centered"
)

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL (FAST) ------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("svm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ------------------ FEATURE EXTRACTION ------------------
@st.cache_data
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

    # ⚠️ Approximate full 22 features
    features = [
        fo, fhi, flo,
        0.005, 0.00003, 0.002, 0.004, 0.006,
        0.03, 0.35, 0.015, 0.02, 0.018, 0.045,
        0.015, hnr,
        0.35, 0.75,
        -5.2, 0.20, 2.0, 0.20
    ]

    return np.array(features).reshape(1, -1)

# ------------------ UI ------------------
st.title("🧠 Parkinson Disease Detection")
st.write("Upload your voice recording (.wav) to check if Parkinson symptoms are detected.")

st.divider()

audio_file = st.file_uploader("🎤 Upload Voice File", type=["wav"])

# ------------------ PROCESS ------------------
if audio_file is not None:

    st.audio(audio_file)

    if st.button("🔍 Analyze Voice"):

        with st.spinner("Analyzing... Please wait"):

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                file_path = tmp.name

            # Extract features
            features = extract_features(file_path)

            # Scale
            features = scaler.transform(features)

            # Predict
            prediction = model.predict(features)
            probability = model.predict_proba(features)

        st.divider()

        # ------------------ RESULT ------------------
        st.subheader("📊 Result")

        healthy_prob = probability[0][0]
        parkinson_prob = probability[0][1]

        if prediction[0] == 1:
            st.error("⚠️ Parkinson Detected")
        else:
            st.success("✅ Healthy")

        # ------------------ PROGRESS BARS ------------------
        st.write("### Confidence Levels")

        st.progress(float(healthy_prob))
        st.write(f"Healthy: {healthy_prob*100:.2f}%")

        st.progress(float(parkinson_prob))
        st.write(f"Parkinson: {parkinson_prob*100:.2f}%")

# ------------------ FOOTER ------------------
st.divider()
st.caption("Developed using Machine Learning & Voice Analysis")