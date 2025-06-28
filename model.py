import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from config import VALUE_CATEGORIES, MODEL_NAME, MAX_LENGTH
from typing import List, Tuple, Dict

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
    
    # Улучшенная конфигурация модели
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(VALUE_CATEGORIES),
        problem_type="multi_label_classification",
        output_attentions=True,  # Сохраняем attention для объяснений
        output_hidden_states=True
    )
    
    # Добавляем дополнительные слои для лучшей классификации
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, len(VALUE_CATEGORIES))
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
    
    # Создаем список пар (ценность, вероятность)
    results = [(VALUE_CATEGORIES[i], probs[i]) for i in range(len(VALUE_CATEGORIES))]
    
    # Сортировка по убыванию вероятности
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def explain_prediction(text: str, tokenizer: BertTokenizer, model: BertForSequenceClassification, device: torch.device) -> Dict[str, List[Tuple[str, float]]]:
    """
    Улучшенная функция объяснений с использованием Integrated Gradients
    """
    from captum.attr import LayerIntegratedGradients
    from captum.attr import visualization as viz
    
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
    
    # Функция для получения предсказаний модели
    def forward_func(input_ids, attention_mask):
        outputs = model(input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    # Инициализация метода объяснений
    lig = LayerIntegratedGradients(
        forward_func,
        model.bert.embeddings
    )
    
    # Базовые значения (нулевые эмбеддинги)
    baseline = torch.zeros_like(input_ids).to(device)
    
    # Вычисление атрибуций
    attributions, delta = lig.attribute(
        inputs=(input_ids, attention_mask),
        baselines=(baseline, attention_mask),
        return_convergence_delta=True
    )
    
    # Суммируем атрибуции по слоям
    attributions_sum = attributions.sum(dim=2).squeeze(0).cpu().detach().numpy()
    
    # Нормализация
    attributions_sum = np.abs(attributions_sum)
    attributions_sum /= attributions_sum.max()
    
    # Получаем токены
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Формируем объяснения для каждой ценности
    explanations = {}
    logits = model(input_ids, attention_mask=attention_mask).logits[0].cpu().detach().numpy()
    
    for i, value in enumerate(VALUE_CATEGORIES):
        # Вес класса
        class_weight = np.exp(logits[i]) / (1 + np.exp(logits[i]))
        
        # Умножаем атрибуции на вес класса
        weighted_attributions = attributions_sum * class_weight
        
        # Собираем пары (токен, вес)
        token_weights = [(token, weight) for token, weight in zip(tokens, weighted_attributions)]
        
        # Фильтруем специальные токены
        token_weights = [tw for tw in token_weights if tw[0] not in ['[CLS]', '[SEP]', '[PAD]']]
        
        # Сортируем по убыванию веса
        token_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Берем топ-5 самых важных слов
        explanations[value] = token_weights[:5]
    
    return explanations