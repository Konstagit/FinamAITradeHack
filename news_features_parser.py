import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Загружает данные из файла формата JSONL (JSON Lines).
    Каждая строка файла — это отдельный JSON-объект.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def flatten_items(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Преобразует список JSON-объектов (новостей) в плоскую таблицу (DataFrame).

    Правила преобразования:
    - Создается одна строка на КАЖДЫЙ тикер из поля `scope.tickers`.
    - Если в новости нет тикеров в `scope.tickers`, создается одна общая строка,
      где `ticker` будет иметь значение None.
    """
    rows = []
    for obj in items:
        # --- 1. Извлечение вложенных данных ---
        # Безопасно получаем данные из разных уровней вложенности
        _id = obj.get("_id")
        title = obj.get("title")

        res = obj.get("result", {}) or {}
        meta = res.get("meta", {}) or {}
        scope = res.get("scope", {}) or {}
        sent = res.get("sentiment", {}) or {}
        target = res.get("target", {}) or {}

        # --- 2. Извлечение базовых полей новости ---
        published_at = meta.get("published_at")
        source = meta.get("source")
        source_reliability = meta.get("source_reliability")
        article_type = meta.get("article_type")

        sectors = scope.get("sectors")
        time_horizon = scope.get("time_horizon")

        overall_sentiment = sent.get("overall")
        certainty = sent.get("certainty")
        by_ticker_sentiment = sent.get("by_ticker") or []

        all_sector_impact = res.get("all_sector_impact")
        surprise = res.get("surprise")

        target_tickers = target.get("tickers") or []
        scope_tickers = scope.get("tickers") or []

        # --- 3. Подготовка словарей для быстрого доступа к данным по тикеру ---
        # Это позволяет избежать медленного поиска в списках внутри цикла
        sent_map = {
            d.get("ticker"): d.get("score")
            for d in by_ticker_sentiment if isinstance(d, dict)
        }
        
        target_map = {
            d.get("ticker"): {
                "price_growth": d.get("price_growth"),
                # Обработка возможной опечатки в ключе ("confidense" vs "confidence")
                "confidence": d.get("confidence") if "confidence" in d else d.get("confidense"),
                "price_speed": d.get("price_speed"),
            }
            for d in target_tickers if isinstance(d, dict)
        }

        # --- 4. Создание строк для DataFrame ---
        base_row_data = {
            "_id": _id,
            "title": title,
            "published_at": published_at,
            "source": source,
            "source_reliability": source_reliability,
            "article_type": article_type,
            "sectors": sectors,
            "time_horizon": time_horizon,
            "overall_sentiment": overall_sentiment,
            "certainty": certainty,
            "all_sector_impact": all_sector_impact,
            "surprise": surprise,
        }

        if scope_tickers:
            # Если есть тикеры в scope, создаем по одной строке на каждый
            for t in scope_tickers:
                ticker = t.get("ticker")
                
                # Собираем данные по конкретному тикеру
                ticker_specific_data = {
                    "ticker": ticker,
                    "relevance": t.get("relevance"),
                    "role": t.get("role"),
                    "sentiment_by_ticker": sent_map.get(ticker),
                    "target_price_growth": (target_map.get(ticker) or {}).get("price_growth"),
                    "target_confidence": (target_map.get(ticker) or {}).get("confidence"),
                    "target_price_speed": (target_map.get(ticker) or {}).get("price_speed"),
                }
                
                # Объединяем общие данные новости с данными по тикеру
                row = {**base_row_data, **ticker_specific_data}
                rows.append(row)
        else:
            # Если новость не привязана к тикерам, создаем одну строку с пустыми полями
            ticker_agnostic_data = {
                "ticker": None,
                "relevance": None,
                "role": None,
                "sentiment_by_ticker": None,
                "target_price_growth": None,
                "target_confidence": None,
                "target_price_speed": None,
            }
            row = {**base_row_data, **ticker_agnostic_data}
            rows.append(row)

    df = pd.DataFrame(rows)

    # --- 5. Пост-обработка данных в DataFrame ---
    # Преобразование типов и форматов для удобства анализа
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    if "sectors" in df.columns:
        # Преобразуем список секторов в единую строку через запятую
        df["sectors"] = df["sectors"].apply(
            lambda v: ",".join(v) if isinstance(v, list) else v
        )

    return df

def read_jsonl_to_dataset(path_jsonl: str) -> pd.DataFrame:
    """
    Основная функция-конвейер: читает JSONL, преобразует его в DataFrame
    и удаляет возможные дубликаты.
    """
    items = load_jsonl(path_jsonl)
    df = flatten_items(items)
    # Удаляем дубликаты по паре (_id, ticker) на случай, если в исходных данных есть аномалии
    df = df.drop_duplicates(subset=["_id", "ticker"]).reset_index(drop=True)
    return df

# ====== Точка входа в программу ======
if __name__ == "__main__":
    # Путь к файлу
    path_to_file = "llm_news_file.jsonl"
    
    # Проверка, существует ли файл
    if not Path(path_to_file).is_file():
        print(f"Ошибка: Файл не найден по пути '{path_to_file}'")
        # Для примера создадим пустой DataFrame
        df_news = pd.DataFrame()
    else:
        print(f"Загрузка и обработка данных из {path_to_file}...")
        df_news = read_jsonl_to_dataset(path_to_file)
        print("Обработка завершена.")
        print("Информация о полученном DataFrame:")
        df_news.info()
        
    df_news = df_news.drop('_id',axis=1)
    df_news = df_news.drop(['title','source'],axis=1)
    df_news.to_csv('df_news.csv', index=False)