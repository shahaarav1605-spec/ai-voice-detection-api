from fastapi import FastAPI, Header
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
def detect_voice(
    payload: dict,
    x_api_key: str = Header(None, alias="x-api-key")  # 🔴 THIS IS THE FIX
):
    # -------- DEBUG (KEEP THIS FOR NOW) --------
    print("HEADER RECEIVED:", x_api_key)
    print("SERVER API KEY:", API_KEY)
    # -------------------------------------------

    # -------- API KEY CHECK --------
    if x_api_key != API_KEY:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    # -------- PAYLOAD VALIDATION --------
    language = payload.get("language")
    audio_format = payload.get("audioFormat")
    audio_base64 = payload.get("audioBase64")

    if not language or not audio_format or not audio_base64:
        return {
            "status": "error",
            "message": "Missing required fields"
        }

    if language not in SUPPORTED_LANGUAGES:
        return {
            "status": "error",
            "message": "Unsupported language"
        }

    if audio_format.lower() != "mp3":
        return {
            "status": "error",
            "message": "Only mp3 format supported"
        }

    # -------- AUDIO PROCESSING --------
    try:
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        features = extract_features(audio_path)
        os.remove(audio_path)

        model = load_model()
        prediction = model.predict(features)[0]
        confidence = float(model.predict_proba(features)[0].max())

        return {
            "status": "success",
            "language": language,
            "classification": "AI_GENERATED" if prediction == 1 else "HUMAN",
            "confidenceScore": round(confidence, 2),
            "explanation": "Unnatural pitch consistency detected"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
