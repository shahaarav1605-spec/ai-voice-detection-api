import base64
import tempfile
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from voice_ai_detector.model import extract_features, load_model

app = FastAPI(title="AI Generated Voice Detection API")

model = load_model()


class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


def generate_reason(label, raw):
    reasons = []

    if raw["pitch"] < 80:
        reasons.append("very stable pitch")

    if raw["zcr"] < 0.04:
        reasons.append("low zero-crossing rate")

    if raw["spectral_centroid"] < 1500:
        reasons.append("limited spectral variation")

    if not reasons:
        reasons.append("natural vocal variations")

    if label == "AI_GENERATED":
        return (
            "AI-generated speech characteristics detected: "
            + ", ".join(reasons)
            + ". These patterns are common in synthesized voices."
        )

    return (
        "Human speech characteristics detected: "
        + ", ".join(reasons)
        + ". Natural voice fluctuations indicate a real human speaker."
    )


@app.post("/api/voice-detection")
def detect_voice(
    req: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != "my_voice_api_123":
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        audio_bytes = base64.b64decode(req.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    with tempfile.NamedTemporaryFile(suffix="." + req.audioFormat, delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    features, raw = extract_features(audio_path, return_raw=True)
    features = np.array(features).reshape(1, -1)

    probs = model.predict_proba(features)[0]
    ai_conf = probs[1]
    human_conf = probs[0]

    label = "AI_GENERATED" if ai_conf > human_conf else "HUMAN"
    confidence = float(max(ai_conf, human_conf))
    reason = generate_reason(label, raw)

    return {
        "prediction": label,
        "confidence": round(confidence, 3),
        "reason": reason
    }
