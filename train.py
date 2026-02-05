import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from voice_ai_detector.model import extract_features

X = []
y = []

DATASET = [
    (1, "data/ai"),
    (0, "data/human")
]

for label, base_folder in DATASET:
    for lang in os.listdir(base_folder):
        lang_path = os.path.join(base_folder, lang)

        if not os.path.isdir(lang_path):
            continue

        print(f"Processing {lang_path}")

        for file in os.listdir(lang_path):
            if not file.lower().endswith((".wav", ".mp3")):
                continue

            path = os.path.join(lang_path, file)

            try:
                features = extract_features(path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print("Error:", path, e)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise RuntimeError("❌ No audio files found")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=30,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "voice_ai_detector/model.pkl")
print("✅ Model trained and saved")

