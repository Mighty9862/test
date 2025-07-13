import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from config import VALUE_CATEGORIES, MODEL_NAME, MAX_LENGTH
import re
import string
from captum.attr import IntegratedGradients

class ValuesDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=MAX_LENGTH):
        """Инициализация датасета с проверкой и очисткой данных."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден")
        
        self.df = pd.read_csv(filename)
        expected_columns = ['text'] + VALUE_CATEGORIES
        if not all(col in self.df.columns for col in expected_columns):
            raise ValueError(f"В файле {filename} отсутствуют необходимые столбцы: {expected_columns}")
        
        # Очистка текста
        self.df['text'] = self.df['text'].apply(self._clean_text)
        self.df = self.df.dropna(subset=['text'])
        self.df = self.df[self.df['text'].str.strip() != '']
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.df['text'].tolist()
        self.labels = self.df[VALUE_CATEGORIES].values.astype(np.float32)
        
        # Проверка меток на NaN и бесконечные значения
        if np.any(np.isnan(self.labels)) or np.any(np.isinf(self.labels)):
            raise ValueError("Обнаружены NaN или бесконечные значения в метках")

    def _clean_text(self, text):
        """Очистка текста."""
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
        text = re.sub(r'[^\w\s]', ' ', text)  # Удаление пунктуации
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def load_model(model_path=None):
    """Загрузка модели и токенизатора."""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(VALUE_CATEGORIES),
        problem_type="multi_label_classification"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    return tokenizer, model, device

def predict_long_text(text, tokenizer, model, device, threshold=None, chunk_size=MAX_LENGTH-2):
    """Предсказание для длинных текстов с разбиением на части."""
    model.eval()
    threshold = threshold or PREDICTION_THRESHOLD
    
    # Очистка текста
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    
    # Токенизация текста
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= chunk_size:
        return predict(text, tokenizer, model, device, threshold)
    
    # Разбиение на части
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    all_probs = []
    
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        encoding = tokenizer.encode_plus(
            chunk_text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
        all_probs.append(probs)
    
    # Усреднение вероятностей по всем частям
    avg_probs = np.mean(all_probs, axis=0) * 100
    results = [(VALUE_CATEGORIES[i], avg_probs[i]) for i in range(len(VALUE_CATEGORIES)) if avg_probs[i] >= threshold * 100]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def explain_keywords(text, tokenizer, model, device, top_n=5):
    """Объяснение предсказаний с использованием Integrated Gradients."""
    model.eval()
    ig = IntegratedGradients(model)

    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_attention_mask=True,
        return_token_type_ids=False
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    input_embeds = model.bert.embeddings(input_ids)
    input_embeds.requires_grad_()

    # Вычисляем атрибуции
    attributions = ig.attribute(
        inputs=input_embeds,
        target=None,
        additional_forward_args=(attention_mask,),
        n_steps=50
    )

    # Суммируем атрибуции по размерности эмбеддингов
    attributions = attributions.sum(dim=-1).squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    importance = attributions.cpu().detach().numpy()

    # Объединение подслов
    merged_tokens = []
    merged_scores = []
    current_token = ''
    current_score = 0
    current_count = 0

    for tok, score in zip(tokens, importance):
        if tok.startswith('##'):
            current_token += tok[2:]
            current_score += score
            current_count += 1
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(current_score / max(1, current_count))
            current_token = tok
            current_score = score
            current_count = 1

    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(current_score / max(1, current_count))

    # Фильтрация токенов
    def is_valid(token):
        if token in tokenizer.all_special_tokens:
            return False
        if token.lower() in string.punctuation:
            return False
        if len(token) <= 2 and not re.match(r'\w\w', token):
            return False
        if re.fullmatch(r'[\W_]+', token):
            return False
        return True

    filtered = [(tok, score) for tok, score in zip(merged_tokens, merged_scores) if is_valid(tok)]
    top_tokens = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]
    return top_tokens

def build_explanation(text, tokenizer, model, device, top_n=5):
    """Построение пояснения на основе ключевых слов."""
    keywords = explain_keywords(text, tokenizer, model, device, top_n)
    keyword_list = [word for word, _ in keywords]
    explanation = (
        f"Модель выделила ключевые слова: {', '.join(f'«{w}»' for w in keyword_list)}. "
        "Эти слова имеют наибольшее влияние на предсказание ценностей, так как они связаны с соответствующими категориями в обучающих данных."
    )
    return explanation, keywords