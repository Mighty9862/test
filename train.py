import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from model import ValuesDataset, load_model, predict_long_text
from config import MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, PREDICTION_THRESHOLD, VALUE_CATEGORIES, OMP_NUM_THREADS, KMP_AFFINITY, EARLY_STOPPING_PATIENCE, LOG_FILE
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os
import logging
import psutil
from torch.nn import BCEWithLogitsLoss

# Уменьшаем размер батча для CPU
BATCH_SIZE = 4

def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

def calculate_class_weights(dataset):
    """Вычисление весов для классов на основе их частоты."""
    labels = dataset.labels
    pos_weights = []
    for i in range(len(VALUE_CATEGORIES)):
        pos_count = np.sum(labels[:, i])
        neg_count = len(labels) - pos_count
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0  # Избегаем деления на ноль
        pos_weights.append(weight)
    return torch.tensor(pos_weights, dtype=torch.float)

def optimize_threshold(model, dataloader, device):
    """Оптимизация порога предсказания."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    best_f1 = 0
    best_threshold = PREDICTION_THRESHOLD
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = all_probs > threshold
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logging.info(f"Оптимальный порог: {best_threshold:.2f} с F1: {best_f1:.4f}")
    return best_threshold

def calculate_metrics(model, dataloader, device, threshold):
    """Расчёт метрик качества."""
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            preds = probs > threshold
            
            true_labels.append(labels.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())
    
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    
    positive_preds = np.sum(pred_labels)
    if positive_preds == 0:
        logging.warning(f"Нет положительных предсказаний при пороге {threshold}. Средние вероятности по категориям:")
        mean_probs = np.mean(probs.cpu().numpy(), axis=0)
        for i, category in enumerate(VALUE_CATEGORIES):
            logging.warning(f"{category}: {mean_probs[i]:.4f}")
    
    report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=VALUE_CATEGORIES,
        zero_division=0
    )
    logging.info("\n" + report)
    
    return f1_score(true_labels, pred_labels, average='macro', zero_division=0)

def main():
    """Основная функция обучения."""
    setup_logging()
    logging.info("Начало обучения...")

    # Загрузка модели
    logging.info("Загрузка модели...")
    tokenizer, model, device = load_model()
    logging.info(f"Модель загружена на {device}")

    # Загрузка данных
    try:
        logging.info("Загрузка датасета...")
        full_dataset = ValuesDataset('values_dataset.csv', tokenizer)
        logging.info(f"Загружено примеров: {len(full_dataset)}")
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Устанавливаем num_workers=0 для CPU
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        logging.info(f"Размер обучающей выборки: {len(train_dataset)}")
        logging.info(f"Размер валидационной выборки: {len(val_dataset)}")
        
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {e}")
        return

    # Вычисление весов классов
    pos_weights = calculate_class_weights(full_dataset).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Оптимизация порога
    logging.info("Оптимизация порога предсказания...")
    best_threshold = optimize_threshold(model, val_loader, device)

    # Обучение
    best_f1 = 0.0
    patience_counter = 0
    logging.info(f"Начало обучения на {device}...")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                logging.warning("Обнаружены NaN или бесконечные значения в метках")
                continue
                
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Взвешенная функция потерь
            loss_fn = BCEWithLogitsLoss(pos_weight=pos_weights)
            loss = loss_fn(logits, labels)
            
            if torch.isnan(loss):
                logging.warning("Loss равен NaN, пропускаем батч")
                continue
                
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Логирование каждые 10 батчей
            if i % 10 == 0:
                logging.info(f"Эпоха {epoch+1}/{EPOCHS}, батч {i}/{len(train_loader)}, loss: {loss.item():.4f}")
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Расчёт метрик
        val_f1 = calculate_metrics(model, val_loader, device, best_threshold)
        
        # Обновление планировщика
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        logging.info(f"Эпоха {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Время: {epoch_time:.2f} сек")
        
        # Сохранение модели после каждой эпохи
        torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}_epoch_{epoch+1}.pt")
        logging.info(f"Модель сохранена после эпохи {epoch+1}")

        # Сохранение лучшей модели
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Лучшая модель сохранена с F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Ранняя остановка на эпохе {epoch+1}")
                break
    
    # Сохранение финальной модели после завершения обучения
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Финальная модель сохранена как {MODEL_SAVE_PATH}")

    # Удаление промежуточных моделей
    for epoch in range(1, EPOCHS + 1):
        epoch_model_path = f"{MODEL_SAVE_PATH}_epoch_{epoch}.pt"
        if os.path.exists(epoch_model_path):
            os.remove(epoch_model_path)
            logging.info(f"Удалена промежуточная модель: {epoch_model_path}")

    logging.info(f"Обучение завершено. Лучший F1: {best_f1:.4f}")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    os.environ["KMP_AFFINITY"] = KMP_AFFINITY
    main()