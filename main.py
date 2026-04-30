import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import PlainTextResponse
from openai import OpenAI

app = FastAPI(title="Minute Mail Private Transcription API")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACTION_SECRET = os.environ.get("ACTION_SECRET")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

client = OpenAI(api_key=OPENAI_API_KEY)


@app.get("/privacy", response_class=PlainTextResponse)
def privacy():
    return """
Minute Mail Private Transcription API - Privacy Policy

This private API is used only to transcribe audio files uploaded by Julien Trezza for meeting minutes and follow-up email drafting.

The API does not send emails.
The API does not access Gmail, calendars, CRM, Google Drive, Salesforce or any external business system.
The API does not store audio files permanently.
Audio files are processed temporarily for transcription and deleted after processing.
The transcript is returned to ChatGPT for draft-only meeting minutes and follow-up emails.
"""


@app.get("/", response_class=PlainTextResponse)
def root():
    return "Minute Mail Private Transcription API is running."


@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    x_action_secret: str | None = Header(default=None),
):
    if ACTION_SECRET and x_action_secret != ACTION_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    allowed_extensions = (".mp3", ".m4a", ".wav", ".mp4", ".mpeg", ".mpga", ".webm")
    filename = audio_file.filename or ""

    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Please upload mp3, m4a, wav, mp4, mpeg, mpga or webm.",
        )

    tmp_path = None

    try:
        suffix = os.path.splitext(filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await audio_file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                response_format="text",
            )

        return {
            "transcript": transcription,
            "language": "To be inferred by GPT from transcript",
            "confidence_note": "Raw transcript generated from uploaded audio. Speaker diarization not guaranteed."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
