from model import load_model, predict, explain_prediction
from config import MODEL_SAVE_PATH, VALUE_CATEGORIES
import time

# Примеры текстов для тестирования с комментариями
test_texts = [
    ("Здоровье человека и его физическая неприкосновенность - фундаментальные основы общества", "Жизнь"),
    ("Любовь к Родине проявляется в готовности защищать её интересы в любой ситуации", "Патриотизм"),
    ("Семейные традиции и взаимная поддержка делают нас сильнее перед лицом трудностей", "Крепкая семья"),
    ("Творческий труд не только создаёт материальные блага, но и развивает духовный мир человека", "Созидательный труд, Приоритет духовного"),
    ("Равные права для всех граждан независимо от происхождения - основа справедливого общества", "Справедливость"),
    ("Защита национальных интересов требует от нас трудолюбия и сплочённости перед внешними вызовами", "Патриотизм, Коллективизм"),
    ("Милосердное отношение к слабым и уважение к старшим поколениям сохраняют нашу человечность", "Милосердие, Историческая память"),
    ("Духовное развитие личности важнее материальных накоплений, особенно в воспитании детей", "Приоритет духовного, Крепкая семья"),
    ("Врач, работающий в зоне военного конфликта, проявляет милосердие к раненым и служит Отечеству", "Милосердие, Служение Отечеству"),
    ("Труд учителя формирует нравственные идеалы молодёжи и способствует единству народов России", "Созидательный труд, Единство народов России")
]

def main():
    print("Загрузка обученной модели...")
    start_time = time.time()
    tokenizer, model, device = load_model(MODEL_SAVE_PATH)
    load_time = time.time() - start_time
    print(f"Модель загружена за {load_time:.2f} секунд")
    
    # Тестирование
    for i, (text, expected) in enumerate(test_texts):
        print(f"\nТекст #{i+1}: {text}")
        print(f"Ожидаемые ценности: {expected}")
        
        start_pred = time.time()
        predictions = predict(text, tokenizer, model, device)
        pred_time = time.time() - start_pred
        
        # Выводим результаты по убыванию вероятности
        print("\nРезультаты классификации:")
        top_predictions = []
        for value, prob in predictions:
            if prob > 5.0:  # Показываем только >5%
                print(f"- {value}: {prob:.1f}%")
                top_predictions.append((value, prob))
        
        print(f"Время предсказания: {pred_time:.4f} сек")
        
        # Генерация объяснений для первых 3 текстов
        if i < 3 and len(top_predictions) > 0:
            print("\nОбъяснение предсказания (топ-5 слов для каждой ценности):")
            try:
                start_explain = time.time()
                explanations = explain_prediction(text, tokenizer, model, device)
                
                for value, prob in top_predictions:
                    if value in explanations:
                        words = ", ".join([f"{word} ({weight:.3f})" for word, weight in explanations[value]])
                        print(f"- {value}: {words}")
                
                explain_time = time.time() - start_explain
                print(f"Время объяснения: {explain_time:.4f} сек")
            except Exception as e:
                print(f"Ошибка при генерации объяснений: {e}")

if __name__ == "__main__":
    main()