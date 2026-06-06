import os
import torch
import whisperx
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# =====================================
# SETUP
# =====================================

app = FastAPI(title="GD Arena API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Change model loading to this
model = whisperx.load_model(
    "large-v3",
    device=device,
    compute_type="int8",
    asr_options={"beam_size": 1}  # reduces memory usage
)

# =====================================
# ALIGNMENT CONFIG
# =====================================

SUPPORTED_ALIGN_LANGUAGES = [
    "en", "fr", "de", "es", "it",
    "ja", "zh", "nl", "uk", "pt"
]

CUSTOM_ALIGN_MODELS = {
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "pa": "harveen-chadha/wav2vec2-large-xlsr-53-punjabi",
    "ur": "kingabzpro/wav2vec2-large-xlsr-53-urdu",
}

# =====================================
# ALIGNMENT FUNCTION
# =====================================

def run_alignment(result, audio, device):
    language = result["language"]
    try:
        if language in SUPPORTED_ALIGN_LANGUAGES:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
        elif language in CUSTOM_ALIGN_MODELS:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                model_name=CUSTOM_ALIGN_MODELS[language],
                device=device
            )
        else:
            print(f"⚠️ No alignment for {language} — skipping")
            return result

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device
        )

        del model_a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"⚠️ Alignment failed: {e}")
        return result

# =====================================
# API — TRANSCRIBE
# =====================================

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:

        # Step 1 — save uploaded webm file
        input_path = f"temp_{file.filename}"
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Step 2 — convert webm to wav
        wav_path = "converted_audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, wav_path],
            check=True,
            capture_output=True
        )

        # Step 3 — load audio
        audio = whisperx.load_audio(wav_path)

        # Step 4 — detect language
        result_original = model.transcribe(
            audio,
            language=None,
            batch_size=16
        )
        detected_language = result_original["language"]
        print(f"✅ Detected Language: {detected_language}")

        # Step 5 — translate to English if needed
        if detected_language != "en":
            print(f"🔄 Translating {detected_language} to English...")
            result = model.transcribe(
                audio,
                language=None,
                batch_size=16,
                task="translate"
            )
            result["language"] = "en"
            print("✅ Translation Complete")
        else:
            result = result_original

        # Step 6 — alignment
        result = run_alignment(result, audio, device)

        # Step 7 — build transcript
        transcript = " ".join([
            seg["text"] for seg in result["segments"]
        ])

        # Cleanup
        os.remove(input_path)
        os.remove(wav_path)

        return {
            "status": "success",
            "detected_language": detected_language,
            "transcript": transcript
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)