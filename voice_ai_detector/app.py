from fastapi import FastAPI, Header, HTTPException
import base64
import tempfile
import os

from voice_ai_detector.model import extract_features, load_model

# ================= CONFIG =================
API_KEY = "my_voice_api_123"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
# ==========================================

app = FastAPI(title="AI-Generated Voice Detection API")


@app.post("/api/voice-detection")
def detect_voice(payload: dict, x_api_key: str = Header(None)):

    print("DEBUG HEADER x_api_key =", x_api_key)
    print("DEBUG SERVER API_KEY =", API_KEY)


    # -------- API KEY CHECK --------
    if x_api_key != API_KEY:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    try:
        # -------- READ INPUT --------
        language = payload["language"]
        audio_format = payload["audioFormat"]
        audio_base64 = payload["audioBase64"]

        # -------- VALIDATION --------
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language")

        if audio_format.lower() != "mp3":
            raise ValueError("Only MP3 format is supported")

        # -------- BASE64 → FILE --------
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        # -------- FEATURE EXTRACTION --------
        features = extract_features(temp_path)

        os.remove(temp_path)

        # -------- MODEL PREDICTION --------
        model = load_model()
        prediction = model.predict(features)[0]
        confidence = float(model.predict_proba(features)[0].max())

        # -------- RESPONSE --------
        return {
            "status": "success",
            "language": language,
            "classification": "AI_GENERATED" if prediction == 1 else "HUMAN",
            "confidenceScore": round(confidence, 2),
            "explanation": (
                "Unnatural pitch consistency and robotic speech patterns detected"
                if prediction == 1
                else "Natural speech variations and human-like prosody detected"
            )
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    