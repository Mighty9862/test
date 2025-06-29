import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from config import VALUE_CATEGORIES, MODEL_NAME, MAX_LENGTH
import torch.nn.functional as F
import re
import string


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
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(VALUE_CATEGORIES),
        problem_type="multi_label_classification"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    return tokenizer, model, device


def predict(text, tokenizer, model, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
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

    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten() * 100

    results = [(VALUE_CATEGORIES[i], probs[i]) for i in range(len(VALUE_CATEGORIES))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results



def explain_keywords(text, tokenizer, model, device, top_n=5):
    model.eval()

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

    # Получение эмбеддингов и градиентов
    input_embeds = model.bert.embeddings(input_ids)
    input_embeds.requires_grad_()
    input_embeds.retain_grad()

    outputs = model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask
    )

    loss = outputs.logits.sum()
    loss.backward()

    grads = input_embeds.grad.abs().sum(dim=-1).squeeze()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    importance = grads.cpu().detach().numpy()

    # Объединение подслов в слова
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

    # Последний токен
    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(current_score / max(1, current_count))

    # Удаление специальных токенов, пунктуации и коротких слов
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
    keywords = explain_keywords(text, tokenizer, model, device, top_n)
    keyword_list = [word for word, _ in keywords]

    explanation = (
        f"Пояснение: модель выделила ключевые слова — {', '.join(f'«{w}»' for w in keyword_list)}. "
        "Эти слова, по мнению модели, чаще всего ассоциируются с определёнными ценностями в обучающей выборке, "
        "поэтому они повлияли на классификацию данного текста."
    )
    return explanation, keywords
