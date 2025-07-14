from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import load_model, predict_long_text
from config import MODEL_SAVE_PATH
import logging

app = FastAPI(title="Values Classifier Chatbot")

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка модели при старте
try:
    tokenizer, model, device = load_model(MODEL_SAVE_PATH)
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise Exception(f"Не удалось загрузить модель: {e}")

# Статика
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
