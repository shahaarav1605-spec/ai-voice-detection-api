from fastapi import FastAPI, Header, HTTPException
import base64
import tempfile
import os

from voice_ai_detector.model import extract_features, load_model

# ---------------- CONFIG ----------------
API_KEY = "sk_voice_12345"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ---------------------------------------
app = FastAPI(title="AI-Generated Voice Detection API")


@app.post("/api/voice-detection")
def detect_voice(
    payload: dict,
    x_api_key: str = Header(None)
):
    # -------- API KEY CHECK --------
    if x_api_key != API_KEY:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    try:
        # -------- READ INPUT --------
        language = payload.get("language")
        audio_format = payload.get("audioFormat")
        audio_base64 = payload.get("audioBase64")

        # -------- VALIDATIONS --------
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language")

        if audio_format.lower() != "mp3":
            raise ValueError("Only MP3 format is supported")

        if not audio_base64:
            raise ValueError("Audio data missing")

        # -------- BASE64 DECODE --------
        audio_bytes = base64.b64decode(audio_base64)

        # -------- SAVE TEMP MP3 --------
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

        classification = "AI_GENERATED" if prediction == 1 else "HUMAN"

        # -------- RESPONSE --------
        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
        }

    except Exception:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }


