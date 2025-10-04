from deduplication_news import deduplicate_news
from llm_news_call_main import run_llms_news_extraction_pipeline
from translate_tickers import MOEXTickerMapper, manual_mappings, unite_all_batches
from news_features_parser import read_jsonl_to_dataset
from forecaster_main import make_preds

from pathlib import Path
import os
import pandas as pd

# ====== Дедупликация новостей (+-1 день) ======
print("Начинаем процесс дедупликации из другого файла...")
os.makedirs(Path(r'data\processed'), exist_ok=True)
# Определяем пути к данным
INPUT_NEWS_1 = Path(r'data\raw\participants\news.csv')
INPUT_NEWS_2 = Path(r'data\raw\participants\news_2.csv')
OUTPUT_FILE = Path(r'data\processed\deduped_full.csv')

# Вызываем функцию
deduplicate_news(
    news_path1=INPUT_NEWS_1,
    news_path2=INPUT_NEWS_2,
    output_path=OUTPUT_FILE
)

print("Процесс дедупликации успешно завершен.")


# ====== Вычленяем признаки из новостей с помощью LLM ======
print("Начинаем обработку новостей для получения признаков...")

# Определяем пути к данным
input_csv_path = Path(r'data\processed\deduped_full.csv')
BATCH_SIZE        = 1000
MAX_CONCURRENCY   = 70         # сколько запросов одновременно внутри батча
ARTIFACTS_DIR     = Path("./artifacts")
MODEL_NAME        = "google/gemma-3-12b-it"

# Вызываем функцию
run_llms_news_extraction_pipeline(input_csv_path = input_csv_path,
                                      ARTIFACTS_DIR = ARTIFACTS_DIR,
                                      MODEL_NAME = MODEL_NAME,
                                      batch_size = BATCH_SIZE,
                                      max_concurrency = MAX_CONCURRENCY)

print("Процесс обработки новостей успешно завершен.")



# ====== Обьединяем все бачи и меняем переводим ticker на язык Биржи (Газпром -> GAZP)======
unite_all_batches(root_dir_name="artifacts", output_file_name="merged_parsed.jsonl")

# Создаем экземпляр маппера, передавая ему ручные маппинги
print("Создаю экземпляр маппера...")
mapper = MOEXTickerMapper(manual_mappings=manual_mappings)
# Обрабатываем файл и сохраняем результат
mapper.process_jsonl_file(
    input_file=Path(r'artifacts\merged_parsed.jsonl'),
    output_file=Path(r'data\processed\'llm_news_file.jsonl')
)

# ====== Парсим json в обычный датафрейм ======

path_to_llm_news_file = Path(r'data\processed\'llm_news_file.jsonl')
output_path =  Path(r'data\processed\df_news.csv')
# Проверка, существует ли файл
if not Path(path_to_llm_news_file).is_file():
    print(f"Ошибка: Файл не найден по пути '{path_to_llm_news_file}'")
    # Для примера создадим пустой DataFrame
    df_news = pd.DataFrame()
else:
    print(f"Загрузка и обработка данных из {path_to_llm_news_file}...")
    df_news = read_jsonl_to_dataset(path_to_llm_news_file)
    print("Обработка завершена.")
    print("Информация о полученном DataFrame:")
    df_news.info()

# Cохраняем для дебага    
df_news.to_csv(output_path, index=False)



# ====== Собственно прогноз ======

default_train_path = Path(r'data/raw/participants/candles.csv')
default_test_path = Path(r'data/raw/participants/candles_2.csv')
default_df_news_path = Path(r'data/processed/df_news.csv')
submission_file =('result_submission.csv')

# Вызываем основную функцию прогнозирования
make_preds(default_train_path, default_test_path, default_df_news_path, output_path=submission_file)