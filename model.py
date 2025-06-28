import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig
from config import VALUE_CATEGORIES, MODEL_NAME, MAX_LENGTH, OMP_NUM_THREADS, KMP_AFFINITY
from typing import List, Tuple, Dict
import os

# Применяем оптимизации параллелизма
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["KMP_AFFINITY"] = KMP_AFFINITY
os.environ["KMP_BLOCKTIME"] = "1"
if not torch.cuda.is_available():
    torch.set_num_threads(OMP_NUM_THREADS)

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
    # Принудительно отключаем GPU
    torch.device('cpu')
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(VALUE_CATEGORIES),
        problem_type="multi_label_classification",
        torchscript=True  # Для оптимизации
    )
    
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    )
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Оптимизация модели для CPU
    model = torch.jit.script(model) if hasattr(torch.jit, 'script') else model
    model.eval()
    return tokenizer, model, torch.device('cpu')

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
    
    with torch.inference_mode():  # Ускоренный режим вывода
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten() * 100
    
    results = [(VALUE_CATEGORIES[i], probs[i]) for i in range(len(VALUE_CATEGORIES))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def explain_prediction(text: str, tokenizer: BertTokenizer, model: BertForSequenceClassification, device: torch.device) -> Dict[str, List[Tuple[str, float]]]:
    """Упрощенная функция объяснений"""
    model.eval()
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=MAX_LENGTH, 
        truncation=True,
        padding='max_length'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Упрощенные объяснения (без внимания)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_weights = np.ones(len(tokens))  # Заглушка
    
    explanations = {}
    for value in VALUE_CATEGORIES:
        token_weights = [(token, weight) for token, weight in zip(tokens, token_weights)]
        token_weights.sort(key=lambda x: x[1], reverse=True)
        explanations[value] = token_weights[:5]
    
    return explanations