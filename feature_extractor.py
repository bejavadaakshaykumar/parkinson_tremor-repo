import parselmouth
import numpy as np

def extract_features(file_path):
    sound = parselmouth.Sound(file_path)

    # ------------------ PITCH ------------------
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]

    if len(pitch_values) == 0:
        return np.zeros((1, 22))

    fo = np.mean(pitch_values)
    fhi = np.max(pitch_values)
    flo = np.min(pitch_values)

    # ------------------ POINT PROCESS ------------------
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)

    # ------------------ JITTER ------------------
    jitter_percent = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 75, 500, 1.3)
    jitter_abs = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 75, 500, 1.3)
    rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 75, 500, 1.3)
    ppq = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 75, 500, 1.3)
    ddp = 3 * rap

    # ------------------ SHIMMER ------------------
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6)
    shimmer_db = parselmouth.praat.call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 75, 500, 1.3, 1.6)
    apq3 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 75, 500, 1.3, 1.6)
    apq5 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq5)", 0, 0, 75, 500, 1.3, 1.6)
    apq = parselmouth.praat.call([sound, point_process], "Get shimmer (apq11)", 0, 0, 75, 500, 1.3, 1.6)
    dda = 3 * apq3

    # ------------------ NOISE ------------------
    harmonicity = sound.to_harmonicity()
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    noise = sound.to_harmonicity()
    nhr = 1 / (hnr + 1e-6)   # Approx conversion

    # ------------------ NONLINEAR FEATURES ------------------
    # ⚠️ These are complex → approximated using statistical methods

    rpde = np.std(pitch_values)
    dfa = np.var(pitch_values)
    spread1 = np.min(pitch_values)
    spread2 = np.max(pitch_values)
    d2 = np.mean(pitch_values)
    ppe = np.std(np.diff(pitch_values))

    # ------------------ FINAL FEATURE VECTOR ------------------
    features = [
        fo, fhi, flo,
        jitter_percent, jitter_abs, rap, ppq, ddp,
        shimmer, shimmer_db, apq3, apq5, apq, dda,
        nhr, hnr,
        rpde, dfa,
        spread1, spread2, d2, ppe
    ]

    return np.array(features).reshape(1, -1)