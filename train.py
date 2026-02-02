import os
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
HUMAN_DIR = os.path.join(DATA_DIR, "human")
AI_DIR = os.path.join(DATA_DIR, "ai")

MODEL_PATH = os.path.join(BASE_DIR, "voice_ai_detector", "model.pkl")

def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(zcr),
        np.mean(sc)
    ])

def load_data(folder, label):
    X, y = [], []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)
                try:
                    X.append(extract_features(path))
                    y.append(label)
                    print("Loaded:", path)
                except Exception as e:
                    print("Skipped:", path, e)
    return X, y

print("Loading human samples...")
Xh, yh = load_data(HUMAN_DIR, 0)

print("Loading AI samples...")
Xa, ya = load_data(AI_DIR, 1)

X = Xh + Xa
y = yh + ya

print("Training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved at:", MODEL_PATH)
