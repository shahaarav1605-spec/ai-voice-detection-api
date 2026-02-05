import librosa
import numpy as np
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def extract_features(audio_path, return_raw=False):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    features = np.hstack([
        mfcc_mean,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zcr,
        pitch
    ])

    if return_raw:
        return features, {
            "pitch": pitch,
            "zcr": zcr,
            "spectral_centroid": spectral_centroid
        }

    return features


def load_model():
    return joblib.load(MODEL_PATH)

