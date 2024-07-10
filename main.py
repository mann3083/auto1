from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from openai import OpenAI
import os, logging
from pathlib import Path
from dotenv import load_dotenv

from InsuranceAssistant import InsuranceAssistant

app = FastAPI()
logging.basicConfig(level=logging.INFO)
load_dotenv()

IA = InsuranceAssistant()
# Load OpenAI API key from environment
api_key = os.getenv("O_API_KEY")


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):

    try:

        receivedAudio = await file.read()
        with open("audio/utterance.webm", "wb") as f:
            f.write(receivedAudio)

        # Call OpenAI Whisper model
        transcription = IA.client.audio.transcriptions.create(
            model="whisper-1",
            language="ja",
            temperature=0.1,
            file=open("audio/utterance.webm", "rb"),
            response_format="json",
        )
        logging.info(transcription.text)
        rawText = transcription.text
        ## CALL THE EXTRACT KEY VAL CONCEPT TO EXTRACT DETAILS
        japPII = IA.extract_PII_Japanese_Text(rawText)
        logging.info(japPII)
        return {"transcription": str(japPII)}

    except Exception as e:
        return {"Error": str(e)}


@app.post("/tts/")
async def convert_text_to_speech(request: Request, text: str = Form(...)):
    urlPath = "static/speech.mp3"
    speech_file_path = Path(urlPath)

    try:
        response = IA.client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        with open(speech_file_path, "wb") as f:
            f.write(response.content)

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "audio_url": "/static/speech.mp3", "text": text},
        )
    except Exception as e:
        logging.error(f"Error during TTS: {e}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": str(e)}
        )
