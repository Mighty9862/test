import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from model import ValuesDataset, load_model
from config import MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, PREDICTION_THRESHOLD, VALUE_CATEGORIES, OMP_NUM_THREADS
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os
import psutil

def calculate_metrics(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    pred_probs = []  # Добавляем сохранение вероятностей
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Получаем вероятности через сигмоиду
            probs = torch.sigmoid(logits)
            
            # Используем порог из конфига
            preds = probs > PREDICTION_THRESHOLD
            
            # Сохраняем метки, предсказания и вероятности
            true_labels.append(labels.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())
            pred_probs.append(probs.cpu().numpy())
    
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    pred_probs = np.concatenate(pred_probs)
    
    # Проверяем, есть ли хоть какие-то положительные предсказания
    positive_preds = np.sum(pred_labels)
    if positive_preds == 0:
        print(f"\nПредупреждение: нет положительных предсказаний. Возможно, порог {PREDICTION_THRESHOLD} слишком высокий.")
        # Выводим средние вероятности по каждой категории
        mean_probs = np.mean(pred_probs, axis=0)
        print("Средние вероятности по категориям:")
        for i, category in enumerate(VALUE_CATEGORIES):
            print(f"{category}: {mean_probs[i]:.4f}")
    
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
    best_f1 = 0.0
    print(f"Начало обучения на {device}...")
    
    # Создаем планировщик скорости обучения
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        model.train()
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Проверка на наличие NaN в метках
            if torch.isnan(labels).any():
                print("Предупреждение: обнаружены NaN в метках")
                continue
            
            # Проверка на наличие бесконечных значений в метках
            if torch.isinf(labels).any():
                print("Предупреждение: обнаружены бесконечные значения в метках")
                continue
                
            # Ручная реализация потери для многометочной классификации
            outputs = model(
                input_ids, 
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            # Применяем сигмоиду к логитам
            probs = torch.sigmoid(logits)
            
            # Вычисляем бинарную кросс-энтропию вручную
            bce_loss = torch.nn.BCELoss()
            loss = bce_loss(probs, labels)
            
            # Проверка на NaN в loss
            if torch.isnan(loss):
                print("Предупреждение: loss равен NaN, пропускаем батч")
                continue
                
            total_loss += loss.item()
            
            loss.backward()
            
            # Градиентный клиппинг для предотвращения взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Используем тот же подход, что и при обучении
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                
                # Вычисляем бинарную кросс-энтропию
                bce_loss = torch.nn.BCELoss()
                batch_loss = bce_loss(probs, labels)
                
                val_loss += batch_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Обновляем планировщик скорости обучения
        scheduler.step(avg_val_loss)
        
        # Расчет метрик
        val_f1 = calculate_metrics(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nЭпоха {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Время: {epoch_time:.2f} сек")
        
        # Сохранение модели при улучшении F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Модель сохранена с F1: {best_f1:.4f}")
    
    print(f"Обучение завершено. Лучший F1: {best_f1:.4f}")

def main():
    # Загрузка модели
    print("Начало загрузки модели...")
    tokenizer, model, device = load_model()
    print(f"Модель загружена на {device}")
    
    # Загрузка данных
    print("Загрузка датасета...")
    dataset = ValuesDataset('values_dataset.csv', tokenizer)
    print(f"Загружено примеров: {len(dataset)}")
    
    # Разделение на обучающую и валидационную выборки
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Размер обучающей выборки: {train_size}")
    print(f"Размер валидационной выборки: {val_size}")
    
    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Оптимизатор
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Создаем планировщик скорости обучения
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Обучение
    best_f1 = 0.0
    print(f"Начало обучения на {device}...")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Проверка на наличие NaN в метках
            if torch.isnan(labels).any():
                print("Предупреждение: обнаружены NaN в метках")
                continue
            
            # Проверка на наличие бесконечных значений в метках
            if torch.isinf(labels).any():
                print("Предупреждение: обнаружены бесконечные значения в метках")
                continue
                
            # Ручная реализация потери для многометочной классификации
            outputs = model(
                input_ids, 
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            # Применяем сигмоиду к логитам
            probs = torch.sigmoid(logits)
            
            # Вычисляем бинарную кросс-энтропию вручную
            bce_loss = torch.nn.BCELoss()
            loss = bce_loss(probs, labels)
            
            # Проверка на NaN в loss
            if torch.isnan(loss):
                print("Предупреждение: loss равен NaN, пропускаем батч")
                continue
                
            total_loss += loss.item()
            
            loss.backward()
            
            # Градиентный клиппинг для предотвращения взрыва градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Используем тот же подход, что и при обучении
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                
                # Вычисляем бинарную кросс-энтропию
                bce_loss = torch.nn.BCELoss()
                batch_loss = bce_loss(probs, labels)
                
                val_loss += batch_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Обновляем планировщик скорости обучения
        scheduler.step(avg_val_loss)
        
        # Расчет метрик
        val_f1 = calculate_metrics(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nЭпоха {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Время: {epoch_time:.2f} сек")
        
        # Сохранение модели при улучшении F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Модель сохранена с F1: {best_f1:.4f}")
    
    print(f"Обучение завершено. Лучший F1: {best_f1:.4f}")


if __name__ == "__main__":
    # Настройка параллелизма
    os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    os.environ["KMP_AFFINITY"] = KMP_AFFINITY
    
    main()