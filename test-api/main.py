from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import load_model, predict_long_text
from config import MODEL_SAVE_PATH
import logging
import os
import docx
import PyPDF2
from io import BytesIO

app = FastAPI(title="Values Classifier Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    tokenizer, model, device = load_model(MODEL_SAVE_PATH)
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise Exception(f"Не удалось загрузить модель: {e}")

app.mount("/static", StaticFiles(directory="static"), name="static")

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/analyze")
async def analyze_text(input: TextInput):
    try:
        text = input.text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")

        top_values = predict_long_text(text, tokenizer, model, device)

        response = {
            "categories": [
                {"name": category, "probability": float(prob)} for category, prob in top_values
            ]
        }
        return response
    except Exception as e:
        logger.error(f"Ошибка анализа текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        contents = await file.read()

        if filename.endswith(".txt"):
            text = contents.decode("utf-8", errors="ignore")

        elif filename.endswith(".docx"):
            text = extract_text_from_docx(contents)

        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(contents)

        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только .txt, .docx и .pdf файлы")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Файл не содержит текста для анализа")

        top_values = predict_long_text(text, tokenizer, model, device)

        response = {
            "categories": [
                {"name": category, "probability": float(prob)} for category, prob in top_values
            ]
        }
        return response

    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-model")
async def download_model():
    file_path = "values_classifier.pt"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл модели не найден")

    return FileResponse(path=file_path, filename="values_classifier.pt", media_type='application/octet-stream')

def extract_text_from_docx(contents: bytes) -> str:
    text = ""
    try:
        doc = docx.Document(BytesIO(contents))
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Ошибка чтения DOCX: {e}")
    return text

def extract_text_from_pdf(contents: bytes) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(contents))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logger.error(f"Ошибка чтения PDF: {e}")
    return text
