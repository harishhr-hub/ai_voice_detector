import librosa
import numpy as np



def analyze_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=None)

    # ----- FEATURES -----
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitch[pitch > 0]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    energy = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    pauses = librosa.effects.split(y, top_db=25)

    # ----- STATS -----
    pitch_std = np.std(pitch_vals) if len(pitch_vals) > 0 else 0
    mfcc_std = np.std(mfccs)
    energy_std = np.std(energy)
    zcr_std = np.std(zcr)
    pause_count = len(pauses)

    # 🔥 NEW: pitch smoothness (AI is smoother)
    pitch_diff = np.diff(pitch_vals) if len(pitch_vals) > 1 else [0]
    pitch_smoothness = np.mean(np.abs(pitch_diff))

    # 🔥 NEW: pause regularity (AI pauses are uniform)
    pause_lengths = [end - start for start, end in pauses]
    pause_std = np.std(pause_lengths) if len(pause_lengths) > 0 else 0

    # ----- SCORING -----
    ai_score = 0

    if pitch_std < 18: ai_score += 1
    if mfcc_std < 38: ai_score += 1
    if energy_std < 0.025: ai_score += 1
    if zcr_std < 0.02: ai_score += 1
    if pause_count < 3: ai_score += 1

    # 🔥 NEW AI indicators
    if pitch_smoothness < 6: ai_score += 1
    if pause_std < 1500: ai_score += 1

    # ----- DECISION -----
    if ai_score >= 2:
        classification = "AI_GENERATED"
        confidence = round(min(0.75 + ai_score * 0.04, 0.95), 2)
        explanation = (
            "Consistent pitch transitions, smooth energy patterns, and "
            "uniform pause structure indicate synthetic speech"
        )
    else:
        classification = "HUMAN"
        confidence = round(min(0.85 + (7 - ai_score) * 0.02, 0.95), 2)
        explanation = (
            "Natural pitch fluctuations, spectral diversity, and "
            "irregular pause patterns detected"
        )

    return classification, confidence, explanation

