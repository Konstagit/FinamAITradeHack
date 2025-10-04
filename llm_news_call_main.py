# pip install -U langchain langchain-openai pydantic pandas tqdm tenacity python-dotenv
import os, json, re, asyncio, hashlib, csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json, re

# ====== настройки ======
BATCH_SIZE        = 1000         # сколько записей в одном поэтапном шаге
MAX_CONCURRENCY   = 25         # сколько запросов одновременно внутри батча
ARTIFACTS_DIR     = Path("./artifacts_new")
TIMEOUT_SECONDS   = 60          # таймаут одного запроса
MODEL_NAME        = "google/gemma-3-12b-it"
TEMPERATURE       = 0.0
REPROCESS_ERRORS = True 

def run_llms_news_extraction_pipeline(input_csv_path = "deduped_full.csv",
                                      ARTIFACTS_DIR = ARTIFACTS_DIR,
                                      MODEL_NAME = MODEL_NAME,
                                      batch_size = BATCH_SIZE,
                                      max_concurrency = MAX_CONCURRENCY):
    # ====== инициализация ======
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    os.environ["OPENAI_API_KEY"] = API_KEY

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,  # важное!
    )

    prompt = PromptTemplate.from_template(
    """Ты финансовый аналитик. Прочитай новость и верни СТРОГО валидный JSON по схеме.
    Требования:
    - Не выдумывай числа. Если в тексте нет значения — ставь null/пусто.
    - Направление события указывай с точки зрения будущей цены акций эмитента.
    - Укажи time_horizon из: intraday, 1d, 1w, 1m, 3m+.
    - в source укажи все источники, которые упоминаются в новости.
    - article_type указывай как "news", "report", "forecast", "other".
    - source_reliability в 0..1, где 0.0 — крайне низкая надёжность, 1.0 — высокая.
    - sectors указывай как список из "oil & gas", "banking", "transportation", "retail", "telecom", "tech", "metals", "chemicals", "real estate", "beverages", "agriculture", "insurance", "media & entertainment", "construction", "pharmaceuticals & healthcare", "other"
    - tickers определяй названию компании из текста новости. Например, если в новости говорится про Газпром и Лукойл, ticker будет "Газпром" и ticker будет "Лукойл".
    - Допустимые значения "role" - "main", "mentioned" 
    - sentiment.overall в -1..+1. certainty 0..1.
    - all_sector_impact - влияние новости на весь рыночный сектор True или False.Есть новости, которые отражают влияние на отдельные компании(e.g. Лукойл синзил добычу) или на весь сектор (e.g. Цена на нефть взлетела)
    - surprise это про то,является ли новость внезапной и внезапно влияющей на рынок(e.g. Газпром внезапно изобрел вечный источник энергии). Возможные значение от 0.0(очень ожидаемо) до 1.0(очень внезапно).
    - "target" отвечает за прогноз цены на акцию в price_growth от -1.0(очень сильное падение) до +1.0(сильный рост), в confidense уверенность в этом, в price_speed от 0.0(отсутсвие изменений) до +1.0(очень быстрые изменения)
    - target должен быть проставлен для каждой компании из tickers по новости
    - Оценивай показатели только по тексту новости, не используй внешние данные и свои знания об экономике компании.
    - ВЕРНИ ТОЛЬКО ОДИН JSON-ОБЪЕКТ. Без Markdown, без ```json, без комментариев.

    Схема JSON:
    {schema_json}

    Данные:
    publish_date: {publish_date}
    title: {title}

    Текст новости:
    <<<
    {publication}
    >>>
    """
    )


    schema_json = json.dumps({
    "meta": {
        "published_at": "",
        "source": "",
        "source_reliability": 0.5,
        "article_type": ""
    },
    "scope": {
        "tickers": [
        {"ticker":"", "relevance":0.0, "role":""}
        ],
        "sectors": [],
        "time_horizon": ""
    },
    "sentiment": {
        "overall": 0.0,
        "certainty": 0.0,
        "by_ticker": [{"ticker":"", "score":0.0}]
    },
    "all_sector_impact": True,
    "surprise": 0.0,
    "target": {
        "tickers": [
        {"ticker":"","price_growth": 0.5, "confidense": 0.5, "price_speed": 0.5}
        ],
        }
    }, ensure_ascii=False)

    # ====== Вспомогательные функции ======
    def trim_text(s: str, max_chars: int = 5000) -> str:
        if s is None:
            return ""
        s = re.sub(r"\s+", " ", str(s)).strip()
        return s[:max_chars]

    def _strip_md_fences(text: str) -> str:
        # Снимаем ```json ... ``` или просто ```
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        return m.group(1) if m else text

    def _first_balanced_json(text: str) -> str:
        # Берём с первого '{' до места, где баланс скобок вернулся в 0
        start = text.find("{")
        if start == -1:
            raise ValueError("В тексте нет '{' — не похоже на JSON.")
        level = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                level += 1
            elif ch == "}":
                level -= 1
                if level == 0:
                    return text[start:i+1]
        raise ValueError("Не нашли закрывающую '}' для первого объекта.")

    def postprocess_coerce_types(obj: dict) -> dict:
        # 1) булевы как строки → bool
        def _coerce_bools(x):
            if isinstance(x, str):
                if x.lower() == "true":  return True
                if x.lower() == "false": return False
            return x

        # 2) surprise: "unknown" → 0.0 (или None, если хочешь пусто)
        def _coerce_surprise(x):
            if isinstance(x, str) and x.lower() == "unknown":
                return 0.0
            return x

        # обходим нужные поля, если они есть
        if isinstance(obj, dict):
            # all_sector_impact
            if "all_sector_impact" in obj:
                obj["all_sector_impact"] = _coerce_bools(obj["all_sector_impact"])
            # surprise
            if "surprise" in obj:
                obj["surprise"] = _coerce_surprise(obj["surprise"])
            # внутри sentiment/certainty и т. п. трогать не будем
            # при желании можно добавить строгую проверку диапазонов

            # мягкая валидация target.tickers: предупреждаем, если нет поля ticker
            try:
                tickers = obj.get("target", {}).get("tickers", [])
                if isinstance(tickers, list):
                    for t in tickers:
                        if isinstance(t, dict) and "ticker" not in t:
                            t["_warning"] = "missing_ticker_field"
            except Exception:
                pass

        return obj

    # ===== JSON-ремонт и извлечение =====
    def _strip_trailing_commas(s: str) -> str:
        # ", }" -> "}", ", ]" -> "]"
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    def _cut_to_last_closer(s: str) -> str:
        # Обрезаем до последней '}' или ']'
        last_brace = s.rfind("}")
        last_bracket = s.rfind("]")
        last = max(last_brace, last_bracket)
        return s[:last+1] if last != -1 else s

    def _autoclose_brackets(s: str) -> str:
        # Достраиваем недостающие ] и } по стеку, учитывая строки и экранирование
        stack = []
        out = []
        in_str = False
        esc = False
        for ch in s:
            out.append(ch)
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack and ((stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]")):
                        stack.pop()
                    else:
                        # лишняя закрывающая — просто игнорируем логику, продолжаем
                        pass
        # теперь закрываем всё, что осталось открытым
        closers = {"{": "}", "[": "]"}
        while stack:
            opener = stack.pop()
            out.append(closers[opener])
        return "".join(out)

    def extract_json(text: str):
        s = _strip_md_fences(text)

        # 0) быстрый путь
        try:
            return postprocess_coerce_types(json.loads(s))
        except Exception:
            pass

        # 1) попытка вырезать первый сбалансированный объект
        try:
            candidate = _first_balanced_json(s)
            return postprocess_coerce_types(json.loads(candidate))
        except Exception:
            pass

        # 2) «ремонт»: режем хвост, прибираем запятые, автозакрываем
        base = s[s.find("{"):] if "{" in s else s
        repaired = _cut_to_last_closer(base)
        repaired = _strip_trailing_commas(repaired)
        try:
            return postprocess_coerce_types(json.loads(repaired))
        except Exception:
            # 3) автозакрытие скобок
            repaired2 = _autoclose_brackets(base)
            repaired2 = _strip_trailing_commas(repaired2)
            # ещё раз с обрезкой до последней закрывающей — иногда помогает
            repaired2 = _cut_to_last_closer(repaired2)
            try:
                return postprocess_coerce_types(json.loads(repaired2))
            except Exception as e:
                raise ValueError(f"Не удалось распарсить JSON после ремонта: {e}")


    class LLMError(Exception): ...

    def hash_row(row) -> str:
        # хеш для стабильной идентификации записи (чтобы можно было резюмиться)
        h = hashlib.sha256()
        h.update(str(row.publish_date).encode("utf-8"))
        h.update(str(row.title).encode("utf-8"))
        # опционально можно добавить первые 200 символов текста
        h.update(str(row.publication)[:200].encode("utf-8"))
        return h.hexdigest()

    def build_prompt_for_row(row):
        return prompt.format(
            schema_json=schema_json,
            publish_date=str(row.publish_date),
            title=trim_text(row.title, 300),
            publication=trim_text(row.publication, 5000),
        )

    class LLMError(Exception): ...

    # -------- низкоуровневая отправка одного запроса с ретраями --------
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=2),
        retry=retry_if_exception_type(LLMError),
    )
    async def process_one(row, sem: asyncio.Semaphore):
        ptext = build_prompt_for_row(row)
        async with sem:
            resp = await llm.ainvoke(ptext)
        if not hasattr(resp, "content") or not resp.content:
            raise LLMError("Пустой ответ от LLM")

        parsed = None
        err = None
        try:
            parsed = extract_json(resp.content)
            # вот здесь и вызываем «допил» типов
            parsed = postprocess_coerce_types(parsed)
        except Exception as e:
            err = f"json_parse_error: {e}"

        usage = {}
        if hasattr(resp, "response_metadata") and "token_usage" in resp.response_metadata:
            usage = resp.response_metadata["token_usage"]

        return {
            "raw": resp.content,
            "parsed": parsed,
            "error": err,
            "usage": usage,
        }

    # -------- запись артефактов по мере готовности --------
    def append_jsonl(path: Path, obj: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def append_csv_usage(path: Path, row: dict):
        header = ["id", "prompt_tokens", "completion_tokens", "total_tokens"]
        newfile = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if newfile:
                w.writeheader()
            w.writerow({
                "id": row.get("id"),
                "prompt_tokens": row.get("prompt_tokens", 0),
                "completion_tokens": row.get("completion_tokens", 0),
                "total_tokens": row.get("total_tokens", 0),
            })

    # -------- обработка одного батча --------
    async def process_batch(df_batch: pd.DataFrame, batch_idx: int, max_concurrency: int):
        batch_dir = ARTIFACTS_DIR / f"batch_{batch_idx:05d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        done_flag = batch_dir / ".done"
        if done_flag.exists() and not REPROCESS_ERRORS:
            print(f"[batch {batch_idx}] уже отмечен как .done — пропускаю")
            return

        # файлы артефактов
        raw_path    = batch_dir / "raw.jsonl"
        parsed_path = batch_dir / "parsed.jsonl"
        errors_path = batch_dir / "errors.jsonl"
        usage_path  = batch_dir / "usage.csv"

        # читаем уже выполненные id
        done_ids = set()
        if parsed_path.exists():
            with parsed_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "_id" in d:
                            done_ids.add(d["_id"])
                    except:
                        pass

        # готовим задания с метаданными
        sem = asyncio.Semaphore(max_concurrency)
        task_to_meta = {}
        
        for row in df_batch.itertuples(index=False):
            rid = hash_row(row)
            if rid in done_ids:
                continue
            
            meta = {
                "_id": rid, 
                "publish_date": str(row.publish_date),
                "title": str(row.title)
            }
            task = asyncio.create_task(process_one(row, sem))
            task_to_meta[task] = meta

        if not task_to_meta:
            print(f"[batch {batch_idx}] все элементы уже обработаны")
            done_flag.touch()
            return

        # ПРАВИЛЬНЫЙ СПОСОБ: await каждую корутину из as_completed
        completed_count = 0
        total = len(task_to_meta)
        
        with tqdm(total=total, desc=f"batch {batch_idx}", unit="item") as pbar:
            for coro in asyncio.as_completed(task_to_meta.keys()):
                try:
                    # Получаем ЗАВЕРШЁННУЮ задачу
                    completed_task = None
                    res = await coro  # ← Правильно: await корутину
                    
                    # Находим задачу, которая завершилась
                    for task in task_to_meta:
                        if task.done():
                            try:
                                if task.result() == res:
                                    completed_task = task
                                    break
                            except:
                                pass
                    
                    # Fallback: если не нашли, ищем по факту done()
                    if completed_task is None:
                        for task in task_to_meta:
                            if task.done() and task not in [None]:  # грубо, но работает
                                completed_task = task
                                break
                    
                    if completed_task is None:
                        print(f"[WARNING] Не смогли сопоставить задачу с результатом")
                        continue
                    
                    meta = task_to_meta[completed_task]
                    
                    # raw
                    append_jsonl(raw_path, {
                        "_id": meta["_id"], 
                        "title": meta["title"], 
                        "raw": res["raw"]
                    })
                    
                    # parsed / errors
                    if res["parsed"] is not None and res["error"] is None:
                        obj = {"_id": meta["_id"], "title": meta["title"], "result": res["parsed"]}
                        append_jsonl(parsed_path, obj)
                    else:
                        append_jsonl(errors_path, {
                            "_id": meta["_id"],
                            "title": meta["title"],
                            "error": res["error"],
                            "raw": res["raw"][:2000]
                        })
                    
                    # usage
                    u = res.get("usage") or {}
                    u["id"] = meta["_id"]
                    append_csv_usage(usage_path, u)

                except Exception as e:
                    # Обрабатываем ошибку - записываем для всех незавершённых задач
                    print(f"[ERROR] Task failed: {e}")
                    append_jsonl(errors_path, {
                        "_id": "unknown",
                        "title": "unknown", 
                        "error": f"task_error: {e}"
                    })
                
                finally:
                    completed_count += 1
                    pbar.update(1)

        done_flag.touch()
        print(f"[batch {batch_idx}] done")


    # -------- утилита: резка на батчи --------
    def iter_batches(df: pd.DataFrame, batch_size: int):
        n = len(df)
        for start in range(0, n, batch_size):
            yield (start // batch_size), df.iloc[start:start+batch_size]

    # -------- основной раннер --------
    def run_pipeline_in_batches(news_df: pd.DataFrame,
                                batch_size=BATCH_SIZE,
                                max_concurrency=MAX_CONCURRENCY,
                                start_from: int | None = None):
        ARTIFACTS_DIR.mkdir(exist_ok=True)

        # если start_from не задан — попробуем autodetect: первый батч без .done
        def _autodetect_start():
            i = 0
            while True:
                d = ARTIFACTS_DIR / f"batch_{i:05d}"
                if not d.exists() or not (d / ".done").exists():
                    return i
                i += 1

        begin = _autodetect_start() if start_from is None else start_from

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for bidx, dfb in iter_batches(news_df, batch_size):
            if bidx < begin:
                continue
            loop.run_until_complete(process_batch(dfb, bidx, max_concurrency))
        loop.close()
        print("Готово. Все батчи пройдены.")

    # ====== Запуск ======
    news = pd.read_csv(input_csv_path)
    run_pipeline_in_batches(news, batch_size=batch_size, max_concurrency=max_concurrency)
        

if __name__ == "__main__":
    print("Запуск пайплайна...")
    run_llms_news_extraction_pipeline(input_csv_path="deduped_full.csv")       