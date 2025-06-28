from model import load_model, predict
from config import MODEL_SAVE_PATH, VALUE_CATEGORIES
import time

test_texts = [
    ("Здоровье человека и его физическая неприкосновенность - фундаментальные основы общества", "Жизнь"),
    ("Любовь к Родине проявляется в готовности защищать её интересы в любой ситуации", "Патриотизм"),
    ("Семейные традиции и взаимная поддержка делают нас сильнее перед лицом трудностей", "Крепкая семья"),
    ("Творческий труд не только создаёт материальные блага, но и развивает духовный мир человека", "Созидательный труд, Приоритет духовного"),
    ("Равные права для всех граждан независимо от происхождения - основа справедливого общества", "Справедливость")
]

def main():
    print("Загрузка обученной модели...")
    start_time = time.time()
    tokenizer, model, device = load_model(MODEL_SAVE_PATH)
    load_time = time.time() - start_time
    print(f"Модель загружена за {load_time:.2f} сек")
    
    for i, (text, expected) in enumerate(test_texts):
        print(f"\nТекст #{i+1}: {text}")
        print(f"Ожидаемые ценности: {expected}")
        
        start_pred = time.time()
        predictions = predict(text, tokenizer, model, device)
        pred_time = time.time() - start_pred
        
        print("\nРезультаты классификации:")
        for value, prob in predictions:
            if prob > 5.0:
                print(f"- {value}: {prob:.1f}%")
        
        print(f"Время предсказания: {pred_time:.4f} сек")

if __name__ == "__main__":
    main()