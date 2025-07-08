import csv
import os

# Словарь соответствия имен файлов и индексов столбцов в test.csv
file_to_column = {
    'spiritual_priority_sentences.txt': 10,  # Приоритет духовного над материальным
    'unity_of_peoples_sentences.txt': 17,    # Единство народов России
    'mutual_aid_respect_sentences.txt': 15,  # Взаимопомощь и взаимоуважение
    'mercy_sentences.txt': 12,               # Милосердие
    'labor_sentences.txt': 9,                # Созидательный труд
    'justice_sentences.txt': 13,             # Справедливость
    'humanism_sentences.txt': 11,            # Гуманизм
    'historical_memory_sentences.txt': 16,   # Историческая память и преемственность поколений
    'family_sentences.txt': 8,               # Крепкая семья
    'collectivism_sentences.txt': 14,        # Коллективизм
}

# Путь к директории с файлами
base_path = '/Users/admin/Main/PythonProjects/Very_cool_parser_3000/'

# Чтение существующих данных из test.csv
with open(os.path.join(base_path, 'test.csv'), 'r', encoding='utf-8') as f:
    existing_data = list(csv.reader(f))

# Создание новых строк для добавления в test.csv
new_rows = []

# Обработка каждого файла с предложениями
for filename, column_index in file_to_column.items():
    file_path = os.path.join(base_path, filename)
    
    # Чтение предложений из файла
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    
    # Обработка каждого предложения
    for sentence in sentences:
        # Удаление номера предложения, если он есть
        if sentence.strip():
            # Проверка, начинается ли строка с цифры и точки (например, "1. ")
            if sentence.strip()[0].isdigit() and ". " in sentence[:5]:
                sentence = sentence.split(". ", 1)[1]
            
            # Удаление кавычек, если они есть
            sentence = sentence.strip().strip('"')
            
            # Создание новой строки для test.csv
            new_row = ['0'] * 18
            new_row[0] = f'"{sentence}"'  # Добавление предложения в первый столбец
            new_row[column_index] = '1'    # Установка 1 в соответствующем столбце ценности
            
            new_rows.append(new_row)

# Добавление новых строк в существующие данные
all_data = existing_data + new_rows

# Запись обновленных данных в test.csv
with open(os.path.join(base_path, 'test.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_data)

print(f'Добавлено {len(new_rows)} новых предложений в test.csv')