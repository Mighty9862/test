import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig
from config import VALUE_CATEGORIES, MODEL_NAME, MAX_LENGTH
from typing import List, Tuple, Dict
import os

# Применяем оптимизации для CPU
os.environ["OMP_NUM_THREADS"] = str(os.getenv("OMP_NUM_THREADS", 16))
os.environ["KMP_AFFINITY"] = os.getenv("KMP_AFFINITY", "granularity=fine,compact,1,0")

class ValuesDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=MAX_LENGTH):
        self.df = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.df['text'].tolist()
        self.labels = self.df[VALUE_CATEGORIES].values.astype(np.float32)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Конфигурация с явным указанием реализации внимания
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(VALUE_CATEGORIES),
        problem_type="multi_label_classification",
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attn_implementation="eager"  # Решает проблему с предупреждением
    )
    
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    )
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return tokenizer, model, device

def predict(text: str, tokenizer: BertTokenizer, model: BertForSequenceClassification, device: torch.device) -> List[Tuple[str, float]]:
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten() * 100
    
    # Create list of (value, probability) pairs
    results = [(VALUE_CATEGORIES[i], probs[i]) for i in range(len(VALUE_CATEGORIES))]
    
    # Sort by probability descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def explain_prediction(text: str, tokenizer: BertTokenizer, model: BertForSequenceClassification, device: torch.device) -> Dict[str, List[Tuple[str, float]]]:
    """Упрощенная функция объяснений с использованием attention весов"""
    model.eval()
    
    # Токенизация текста
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=MAX_LENGTH, 
        truncation=True,
        padding='max_length'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Получаем выходы модели и attention веса
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    
    # Берем attention из последнего слоя, усредняем по головам
    attentions = outputs.attentions[-1].mean(dim=1)[0]  # [batch, head, seq_len, seq_len] -> [seq_len, seq_len]
    
    # Усредняем внимание по всем токенам (получаем важность каждого токена)
    token_attentions = attentions.mean(dim=0).cpu().numpy()
    
    # Игнорируем специальные токены [CLS] и [SEP]
    token_attentions = token_attentions[1:-1]
    
    # Получаем токены
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[1:-1]
    
    # Формируем объяснения
    explanations = {}
    for i, value in enumerate(VALUE_CATEGORIES):
        # Для простоты используем общие веса внимания (можно адаптировать под класс)
        token_weights = [(token, weight) for token, weight in zip(tokens, token_attentions)]
        token_weights.sort(key=lambda x: x[1], reverse=True)
        explanations[value] = token_weights[:5]  # Топ-5 слов
    
    return explanations