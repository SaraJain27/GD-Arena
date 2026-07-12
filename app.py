import os
import torch
import whisperx
import gradio as gr
import yt_dlp

# =====================================
# GPU SETUP
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing Device: {device}")

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)

# =====================================
# LOAD WHISPERX MODEL
# =====================================

model = whisperx.load_model(
    "large-v3",
    device=device,
    compute_type="int8"
)

# =====================================
# ALIGNMENT LANGUAGE CONFIG
# =====================================

SUPPORTED_ALIGN_LANGUAGES = [
    "en", "fr", "de", "es", "it",
    "ja", "zh", "nl", "uk", "pt"
]

CUSTOM_ALIGN_MODELS = {
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "pa": "harveen-chadha/wav2vec2-large-xlsr-53-punjabi",
    "ur": "kingabzpro/wav2vec2-large-xlsr-53-urdu",
    "ar": "othrif/wav2vec2-large-xlsr-arabic",
    "ru": "bond005/wav2vec2-large-xlsr-53-russian",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "bn": "arijitx/wav2vec2-large-xlsr-bengali"
}

# =====================================
# YOUTUBE AUDIO DOWNLOAD
# =====================================

def download_youtube_audio(url):
    output_name = "downloaded_audio"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_name,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3"
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_name + ".mp3"

# =====================================
# ALIGNMENT FUNCTION
# =====================================

def run_alignment(result, audio, device):
    language = result["language"]
    try:
        if language in SUPPORTED_ALIGN_LANGUAGES:
            print(f"Running built-in alignment for: {language}")
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
        elif language in CUSTOM_ALIGN_MODELS:
            print(f"Running custom alignment for: {language}")
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                model_name=CUSTOM_ALIGN_MODELS[language],
                device=device
            )
        else:
            print(f"⚠️ No alignment model for '{language}' — skipping")
            return result

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device
        )
        print(f"✅ Alignment Complete for: {language}")

        del model_a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"⚠️ Alignment failed: {str(e)} — continuing without alignment")
        return result

# =====================================
# MAIN PROCESSING FUNCTION
# =====================================

def process_audio(audio_file, youtube_url):
    try:

        # ---------------------------------
        # GET AUDIO FILE
        # ---------------------------------

        if youtube_url and youtube_url.strip():
            print("Downloading YouTube Audio...")
            audio_path = download_youtube_audio(youtube_url)

        elif audio_file is not None:
            audio_path = audio_file.name

        else:
            return "❌ Please upload a file or enter a YouTube URL."

        print("\nAudio Path:", audio_path)

        # ---------------------------------
        # LOAD AUDIO
        # ---------------------------------

        audio = whisperx.load_audio(audio_path)

        # ---------------------------------
        # TRANSCRIPTION + AUTO TRANSLATION
        # ---------------------------------

        print("\nStarting Transcription...")

        # Step 1 — detect language
        result_original = model.transcribe(
            audio,
            language=None,
            batch_size=16
        )

        detected_language = result_original["language"]
        print(f"✅ Detected Language: {detected_language}")

        # Step 2 — translate to English if not English
        if detected_language != "en":
            print(f"🔄 Translating from {detected_language} to English...")
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
            print("✅ Already English — skipping translation")

        # ---------------------------------
        # ALIGNMENT
        # ---------------------------------

        print("\nStarting Alignment...")
        result = run_alignment(result, audio, device)

        # ---------------------------------
        # BUILD TRANSCRIPT
        # ---------------------------------

        transcript = " ".join([
            seg["text"] for seg in result["segments"]
        ])

        print("✅ Transcript Ready")

        # ---------------------------------
        # CLEANUP
        # ---------------------------------

        if youtube_url and os.path.exists(audio_path):
            os.remove(audio_path)

        return transcript

    except Exception as e:
        return f"❌ Error:\n{str(e)}"

# =====================================
# GRADIO INTERFACE
# =====================================

app = gr.Interface(
    fn=process_audio,

    inputs=[
        gr.File(
            label="Upload Audio/Video File",
            file_types=[
                ".mp3", ".wav", ".mp4",
                ".m4a", ".ogg", ".flac",
                ".webm", ".mpeg"
            ]
        ),
        gr.Textbox(
            label="OR Paste YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        ),
    ],

    outputs=gr.Textbox(
        label="Transcript",
        lines=20,
        show_copy_button=True
    ),

    title="🏟️ GD Arena",
    description="Upload audio and get the transcript!"
)

# =====================================
# START APP
# =====================================

if __name__ == "__main__":
    app.launch()
