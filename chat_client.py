import argparse
from model import load_model, predict, build_explanation
from config import VALUE_CATEGORIES


def normalize_top_predictions(results, top_n=5):
    """Нормализует проценты только среди топ-N ценностей (в сумме 100%)."""
    top_results = results[:top_n]
    total = sum(score for _, score in top_results)
    if total == 0:
        return [(label, 0.0) for label, _ in top_results]
    return [(label, round((score / total) * 100, 2)) for label, score in top_results]


def print_response(text, tokenizer, model, device, top_n=5):
    print("\n📝 Ваш текст:", text)

    print("\n⏳ Обработка...")

    # Предсказание
    results = predict(text, tokenizer, model, device)
    top_normalized = normalize_top_predictions(results, top_n=top_n)

    # Объяснение
    explanation, keywords = build_explanation(text, tokenizer, model, device, top_n=top_n)

    # Вывод
    print("\n📊 Топ-ценности:")
    for label, percent in top_normalized:
        print(f"  - {label}: {percent}%")

    print("\n📌 Ключевые слова:")
    for word, score in keywords:
        print(f"  - {word}: важность {round(score, 2)}")

    print("\n🧠", explanation)
    print()


def main():
    parser = argparse.ArgumentParser(description="Модель определения ценностей в тексте")
    parser.add_argument('--text', type=str, help='Текст для анализа')
    parser.add_argument('--top_n', type=int, default=5, help='Количество топ-ценностей')
    parser.add_argument('--model_path', type=str, default=None, help='Путь к обученной модели (если не используется дефолтная)')

    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_path)

    if args.text:
        print_response(args.text, tokenizer, model, device, top_n=args.top_n)
    else:
        print("Введите текст (для завершения — Ctrl+C):")
        try:
            while True:
                user_text = input(">>> ")
                if user_text.strip():
                    print_response(user_text.strip(), tokenizer, model, device, top_n=args.top_n)
        except KeyboardInterrupt:
            print("\nВыход.")


if __name__ == "__main__":
    main()
