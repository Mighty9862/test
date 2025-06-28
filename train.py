import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from model import ValuesDataset, load_model
from config import MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, VALUE_CATEGORIES, MODEL_SAVE_PATH, MAX_LENGTH
import time
import numpy as np
from sklearn.metrics import f1_score
import os
import psutil

# Оптимизация использования памяти
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", 16)))

def calculate_f1(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Применяем порог для бинарной классификации
            preds = torch.sigmoid(logits) > 0.5
            true_labels.append(labels.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())
    
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    
    # Macro F1-score
    return f1_score(true_labels, pred_labels, average='macro')

def print_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 3)  # В гигабайтах
    print(f"Использование памяти: {mem:.2f} GB")

def main():
    print("Начало загрузки модели...")
    tokenizer, model, device = load_model()
    print(f"Модель загружена на {device}")
    print_memory_usage()
    
    # Загрузка и разделение данных
    try:
        print("Загрузка датасета...")
        full_dataset = ValuesDataset('values_dataset.csv', tokenizer)
        print(f"Загружено примеров: {len(full_dataset)}")
        
        # Разделение на обучение и валидацию (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Размер обучающей выборки: {len(train_dataset)}")
        print(f"Размер валидационной выборки: {len(val_dataset)}")
        
    except FileNotFoundError:
        print("Ошибка: Файл данных 'values_dataset.csv' не найден")
        return
    
    # Оптимизатор
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    best_f1 = 0.0
    print(f"Начало обучения на {device}...")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        model.train()
        batch_count = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Эпоха {epoch+1} | Батч {batch_count}/{len(train_loader)} | Loss: {loss.item():.4f}")
                print_memory_usage()
        
        # Валидация
        model.eval()
        val_f1 = calculate_f1(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        print(f"Эпоха {epoch+1}/{EPOCHS} завершена | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Время: {epoch_time:.2f}s")
        print_memory_usage()
        
        # Сохранение лучшей модели
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Лучшая модель сохранена с F1: {val_f1:.4f}")
    
    print(f"Обучение завершено. Лучший F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()