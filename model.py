import torch
import numpy as np
import re
from transformers import BertTokenizer, BertForSequenceClassification
from config import VALUE_CATEGORIES, MODEL_NAME, MAX_LENGTH, PREDICTION_THRESHOLD

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

def predict_long_text(text, tokenizer, model, device, threshold=None, chunk_size=MAX_LENGTH-2):
    model.eval()
    threshold = threshold or PREDICTION_THRESHOLD
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= chunk_size:
        return predict(text, tokenizer, model, device, threshold)

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

    avg_probs = np.mean(all_probs, axis=0) * 100
    top_indices = np.argsort(avg_probs)[::-1][:5]
    results = [(VALUE_CATEGORIES[i], avg_probs[i]) for i in top_indices]
    total_prob = sum(prob for _, prob in results)
    if total_prob > 0:
        results = [(cat, (prob / total_prob) * 100) for cat, prob in results]
    return results

def predict(text, tokenizer, model, device, threshold=None):
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
    threshold = threshold or PREDICTION_THRESHOLD
    top_indices = np.argsort(probs)[::-1][:5]
    results = [(VALUE_CATEGORIES[i], probs[i]) for i in top_indices]
    total_prob = sum(prob for _, prob in results)
    if total_prob > 0:
        results = [(cat, (prob / total_prob) * 100) for cat, prob in results]
    return results
