import os
import tempfile
import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from pydantic import BaseModel

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Minute Mail Private Transcription API is running."
    }


class EchoPayload(BaseModel):
    file_url: str | None = None
    filename: str | None = None


@app.post("/echo")
def echo(payload: EchoPayload):
    return {
        "status": "ok",
        "received_file_url": payload.file_url,
        "received_filename": payload.filename
    }


class TranscribePayload(BaseModel):
    file_url: str
    filename: str | None = None


@app.post("/transcribe")
async def transcribe_audio(payload: TranscribePayload):
    tmp_path = None

    try:
        response = requests.get(payload.file_url, timeout=60)
        response.raise_for_status()

        filename = payload.filename or "audio.mp3"
        suffix = os.path.splitext(filename)[1] or ".mp3"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
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
            "confidence_note": "Raw transcript generated from audio URL. Speaker diarization not guaranteed."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
