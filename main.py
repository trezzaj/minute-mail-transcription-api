import os
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
async def transcribe_audio(
    payload: TranscribePayload,
    x_action_secret: str | None = Header(default=None),
):
    if ACTION_SECRET and x_action_secret != ACTION_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return {
        "transcript": "TEST ONLY - transcribeAudio received the file reference successfully.",
        "language": "test",
        "confidence_note": f"Received file_url={payload.file_url}, filename={payload.filename}"
    }
