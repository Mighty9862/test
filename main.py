from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import load_model, predict_long_text, build_explanation
from config import MODEL_SAVE_PATH, VALUE_CATEGORIES
import logging
import torch

app = FastAPI(title="Values Classifier Chatbot")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели при старте
try:
    tokenizer, model, device = load_model(MODEL_SAVE_PATH)
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise Exception(f"Не удалось загрузить модель: {e}")

# Монтирование статических файлов
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

        # Предсказание
        results = predict_long_text(text, tokenizer, model, device)
        
        # Объяснение
        explanation, keywords = build_explanation(text[:1000], tokenizer, model, device)  # Ограничиваем для объяснения

        # Формирование ответа
        response = {
            "categories": [
                {"name": category, "probability": float(prob)} for category, prob in results
            ],
            "explanation": explanation,
            "keywords": [{"word": word, "score": float(score)} for word, score in keywords]
        }
        return response
    except Exception as e:
        logger.error(f"Ошибка анализа текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))