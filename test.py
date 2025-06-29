from model import load_model, predict, build_explanation
from config import MODEL_SAVE_PATH
import time

test_texts = [
    ("Здоровье человека и его физическая неприкосновенность - фундаментальные основы общества", "Жизнь"),
    ("Любовь к Родине проявляется в готовности защищать её интересы в любой ситуации", "Патриотизм"),
    ("Семейные традиции и взаимная поддержка делают нас сильнее перед лицом трудностей", "Крепкая семья"),
    ("Творческий труд не только создаёт материальные блага, но и развивает духовный мир человека", "Созидательный труд, Приоритет духовного"),
    ("Равные права для всех граждан независимо от происхождения - основа справедливого общества", "Справедливость"),
    ("В жизни нет горячего и священнее чувства, чем любовь к родине, родной земле, отечеству", "Патриотизм"),
    ("Любовь к родным местам вдохновляет на заботу о будущем и уважение к прошлому.", "Историческая память и преемственность поколений"),
    ("С каждым рассветом приходит молчаливое напоминание: тепло, пульс и ритм мира всё ещё рядом.", "Жизнь")
]


def main():
    print("Загрузка обученной модели...")
    start_time = time.time()
    tokenizer, model, device = load_model(MODEL_SAVE_PATH)
    print(f"Модель загружена за {time.time() - start_time:.2f} сек")

    for i, (text, expected) in enumerate(test_texts):
        print(f"\nТекст #{i + 1}: {text}")
        print(f"Ожидаемые ценности: {expected}")

        start_pred = time.time()
        predictions = predict(text, tokenizer, model, device)
        explanation, keywords = build_explanation(text, tokenizer, model, device)
        print(f"\nРезультаты классификации:")
        for value, prob in predictions:
            if prob > 5.0:
                print(f"- {value}: {prob:.1f}%")

        print("\n" + explanation)
        print("Ключевые слова:", ", ".join(f"{w} ({round(s, 3)})" for w, s in keywords))
        print(f"Время предсказания: {time.time() - start_pred:.4f} сек")


if __name__ == "__main__":
    main()
