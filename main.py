from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
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

LANGUAGE = 'en' # 'ja'

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    #logging.info(f"STT {transcription.text}")
    try:

        receivedAudio = await file.read()
        with open("audio/utterance.webm", "wb") as f:
            f.write(receivedAudio)

        # Call OpenAI Whisper model
        transcription = IA.client.audio.transcriptions.create(
            model="whisper-1",
            language=LANGUAGE,
            temperature=0.1,
            file=open("audio/utterance.webm", "rb"),
            response_format="json",
        )
        rawText = transcription.text
        logging.info(f"TRANSCRIBED RAW text is {rawText}")
        
        ## CALL THE EXTRACT KEY VAL CONCEPT TO EXTRACT DETAILS
        if LANGUAGE == 'ja':
            japPII = IA.extract_PII_Japanese_Text_JP(rawText)
        else:
            japPII = IA.extract_PII_Japanese_Text_EN(rawText)
            
        logging.info(f"EXTRACTED text is {japPII}")

        return JSONResponse(content={"transcription": str(japPII)})

    except Exception as e:
        return JSONResponse(content={"Error": str(e)}, status_code=500)


@app.post("/tts/")
async def convert_text_to_speech(request: Request, text: str = Form(...)):
    logging.info("TTS " + text)
    urlPath = "static/speech.mp3"
    speech_file_path = Path(urlPath)

    try:
        response = IA.client.audio.speech.create(
            model="tts-1", voice="alloy", input=text
        )
        with open(speech_file_path, "wb") as f:
            f.write(response.content)

        return JSONResponse(content={"audio_url": "/static/speech.mp3"})
        

    except Exception as e:
        logging.error(f"Error during TTS: {e}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": str(e)}
        )
