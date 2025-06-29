import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from model1 import ValuesDataset, load_model
from config import MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, PREDICTION_THRESHOLD
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os
import psutil

def calculate_metrics(model, dataloader, device):
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
            
            # Используем порог из конфига
            preds = torch.sigmoid(logits) > PREDICTION_THRESHOLD
            true_labels.append(labels.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())
    
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    
    # Вывод полного отчета
    print("\n" + classification_report(
        true_labels, 
        pred_labels, 
        target_names=VALUE_CATEGORIES,
        zero_division=0
    ))
    
    return f1_score(true_labels, pred_labels, average='macro', zero_division=0)

def main():
    print("Начало загрузки модели...")
    tokenizer, model, device = load_model()
    print(f"Модель загружена на {device}")
    
    # Загрузка и разделение данных
    try:
        print("Загрузка датасета...")
        full_dataset = ValuesDataset('values_dataset.csv', tokenizer)
        print(f"Загружено примеров: {len(full_dataset)}")
        
        # Разделение на обучение и валидацию (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Размер обучающей выборки: {len(train_dataset)}")
        print(f"Размер валидационной выборки: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return
    
    # Оптимизатор
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    best_loss = float('inf')
    print(f"Начало обучения на {device}...")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        model.train()
        
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
            optimizer.step()
        
        # Расчет метрик
        avg_loss = total_loss / len(train_loader)
        val_f1 = calculate_metrics(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nЭпоха {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Время: {epoch_time:.2f} сек")
        
        # Сохранение модели при улучшении loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Модель сохранена (лучший loss: {best_loss:.4f})")
    
    print(f"Обучение завершено. Финальный loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()