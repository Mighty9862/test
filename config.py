import os
import psutil
import torch

VALUE_CATEGORIES = [
    "Жизнь", "Достоинство", "Права и свободы человека", 
    "Патриотизм", "Гражданственность", "Служение Отечеству и ответственность за его судьбу",
    "Высокие нравственные идеалы", "Крепкая семья", "Созидательный труд",
    "Приоритет духовного над материальным", "Гуманизм", "Милосердие",
    "Справедливость", "Коллективизм", "Взаимопомощь и взаимоуважение",
    "Историческая память и преемственность поколений", "Единство народов России"
]

MODEL_NAME = 'sberbank-ai/ruBert-base'
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 10
MODEL_SAVE_PATH = 'values_classifier.pt'
PREDICTION_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 3
LOG_FILE = 'training.log'

# Настройки параллелизма
OMP_NUM_THREADS = str(min(psutil.cpu_count(logical=False), 32))  # Динамическое определение потоков
KMP_AFFINITY = "granularity=fine,compact,1,0"
ATTENTION_IMPLEMENTATION = "sdpa" if torch.cuda.is_available() else "eager"  # Проверяем поддержку sdpa