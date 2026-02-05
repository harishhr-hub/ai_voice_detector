from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os
from fastapi import Request

from app.utils import analyze_audio

API_KEY ="YOURSECRETKEY"  # change this

app = FastAPI(title="AI Generated Voice Detection API")


# -------- Request Model --------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# -------- API Endpoint --------
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(..., alias="x-api-key")
):
    print("RECEIVED HEADER:", x_api_key)
    print("EXPECTED KEY:", API_KEY)
    


    # 🔐 API Key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key or malformed request")
    

    # ✅ Validate input
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    # 🔊 Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 💾 Save temp MP3 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        # 🧠 Analyze audio
        classification, confidence, explanation = analyze_audio(temp_audio_path)
    finally:
        os.remove(temp_audio_path)

    # 📤 Response
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
