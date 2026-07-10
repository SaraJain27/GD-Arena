import os
import time
import torch
import whisperx
import subprocess
import librosa
import numpy as np
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

model = whisperx.load_model(
    "large-v3",
    device=device,
    compute_type="int8",
    asr_options={"beam_size": 1}
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


# VOICE FEATURES FUNCTION


def get_voice_features(wav_path, segments):
    try:
        # Load audio with librosa
        y, sr = librosa.load(wav_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # ── 1. Speech Rate (WPM) ──
        total_words = sum(
            len(seg["text"].split()) for seg in segments
        )
        total_speech_time = sum(
            seg["end"] - seg["start"] for seg in segments
        )
        speech_rate_wpm = int(
            (total_words / total_speech_time * 60)
            if total_speech_time > 0 else 0
        )

        # ── 2. Pause Ratio ──
        pause_duration = duration - total_speech_time
        pause_ratio = round(
            pause_duration / duration if duration > 0 else 0, 2
        )

        # ── 3. Filler Count ──
        fillers = ["um", "uh", "like", "you know", "hmm", "er", "ah"]
        full_text = " ".join(
            seg["text"].lower() for seg in segments
        )
        filler_count = sum(
            full_text.count(f) for f in fillers
        )

        # ── 4. Energy Stability ──
        rms = librosa.feature.rms(y=y)[0]
        energy_stability = round(
            float(1 - (np.std(rms) / (np.mean(rms) + 1e-6))), 2
        )
        energy_stability = max(0.0, min(1.0, energy_stability))

        # ── 5. Pitch Variation ──
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_variation = round(
            float(np.std(pitch_values)) if len(pitch_values) > 0 else 0, 2
        )

        # ── 6. Delivery Confidence Score ──
        # Speech Rate Score (ideal: 110-160 WPM)
        if 110 <= speech_rate_wpm <= 160:
            speech_rate_score = 100
        elif speech_rate_wpm < 110:
            speech_rate_score = max(0, int(speech_rate_wpm / 110 * 100))
        else:
            speech_rate_score = max(0, int((200 - speech_rate_wpm) / 40 * 100))

        # Pause Score (ideal: < 20%)
        pause_score = max(0, int((1 - pause_ratio / 0.20) * 100)) if pause_ratio < 0.20 else 0

        # Filler Score (less fillers = better)
        filler_score = max(0, 100 - (filler_count * 10))

        # Energy Score
        energy_score = int(energy_stability * 100)

        # Pitch Score (moderate variation is good)
        if 10 <= pitch_variation <= 50:
            pitch_score = 100
        elif pitch_variation < 10:
            pitch_score = int(pitch_variation / 10 * 100)
        else:
            pitch_score = max(0, int((100 - pitch_variation) / 50 * 100))

        # Final delivery confidence score
        delivery_confidence_score = int(
            speech_rate_score * 0.30 +
            pause_score       * 0.25 +
            filler_score      * 0.20 +
            energy_score      * 0.15 +
            pitch_score       * 0.10
        )

        return {
            "delivery_confidence_score": delivery_confidence_score,
            "speech_rate_wpm": speech_rate_wpm,
            "pause_ratio": pause_ratio,
            "filler_count": filler_count,
            "energy_stability": energy_stability,
            "pitch_variation": pitch_variation
        }

    except Exception as e:
        print(f"⚠️ Voice features failed: {e}")
        return {
            "delivery_confidence_score": 0,
            "speech_rate_wpm": 0,
            "pause_ratio": 0,
            "filler_count": 0,
            "energy_stability": 0,
            "pitch_variation": 0
        }


# API — TRANSCRIBE


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        # Step 1 — save uploaded file
        input_path = f"temp_{file.filename}"
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Step 2 — convert to wav
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
        translated_language = detected_language
        if detected_language != "en":
            print(f"🔄 Translating {detected_language} to English...")
            result = model.transcribe(
                audio,
                language=None,
                batch_size=16,
                task="translate"
            )
            result["language"] = "en"
            translated_language = "en"
            print("✅ Translation Complete")
        else:
            result = result_original

        # Step 6 — alignment
        result = run_alignment(result, audio, device)

        # Step 7 — build segments
        segments = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]

        # Step 8 — get voice features
        voice_features = get_voice_features(wav_path, segments)

        # Step 9 — calculate processing time
        processing_time = round(time.time() - start_time, 2)

        # Cleanup
        os.remove(input_path)
        os.remove(wav_path)

        return {
            "success": True,
            "message": "Audio transcribed successfully.",
            "data": {
                "detected_language": detected_language,
                "translated_language": translated_language,
                "processing_time": processing_time,
                "segments": segments,
                "voice_features": voice_features
            }
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "data": None
        }

# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
