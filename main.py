from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from openai import OpenAI
import os,logging
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

        #with open("audio/utterance.webm", "rb") as f:
        #    audio_content = f.read()

                # Determine file format based on file extension
        #file_extension =  os.path.splitext(file.filename)[1][1:].strip().lower()
        #logging.info("File Extension "+file_extension)
                # Optionally, save the audio file
                # Example: Save the file to a directory
        #save_path = f"audio/{file.filename}"
        #with open(save_path, "wb") as f:
        #    f.write(audio_content)

        client = OpenAI(api_key=api_key)
        # Call OpenAI Whisper model
        transcription = client.audio.transcriptions.create(
                model="whisper-1",language='ja',temperature=.1,
                file = open("audio/utterance.webm", "rb"),response_format="json")
        logging.info(transcription.text)
        rawText = transcription.text
        ## CALL THE EXTRACT KEY VAL CONCEPT TO EXTRACT DETAILS
        japPII = IA.extract_PII_Japanese_Text(rawText)
        logging.info(japPII)
        return {"transcription": str(japPII)}
    
    except Exception as e:
        return {"Error":str(e)} 

