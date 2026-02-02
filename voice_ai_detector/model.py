import librosa
import numpy as np
import pickle
import os

MODEL_PATH = "voice_ai_detector/model.pkl"


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(zcr),
        np.mean(sc)
    ])

    return features.reshape(1, -1)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


