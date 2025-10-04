import json, re, ast, requests
from typing import Dict, Optional, List
from pathlib import Path

MOEX_BASE = "https://iss.moex.com/iss"

def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("ё", "е")
    # лат/кир миксы для "pao/pjsc/oao" и т.п.
    org_trash = [
        r"\b(pao|pjsc|oao|zao|ao)\b",
        r"\b(пао|оао|зао|ао|ооо)\b",
        r"\b(пao)\b",
    ]
    for pat in org_trash:
        s = re.sub(pat, " ", s)
    # убрать служебные символы
    s = re.sub(r"[\.\,\(\)\[\]\"\'/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_pref_from_shortname(shortname: str) -> bool:
    sn = shortname.lower()
    return any(x in sn for x in [" ап", " преф", " pref", " pr"])

class MOEXTickerMapper:
    def __init__(self, manual_mappings: Dict[str, str] = None, include_tqtf: bool = True,
                 log_not_found: bool = True, drop_records_without_tickers: bool = True):
        self.cache: Dict[str, str] = {}
        self.manual_mappings = manual_mappings or {}
        self.log_not_found = log_not_found                   # логировать «не найдено»
        self.drop_records_without_tickers = drop_records_without_tickers  # удалять записи без тикеров
        self._build_offline_directory(include_tqtf=include_tqtf)
        self._apply_manual_mappings()

    def _fetch_board(self, board: str):
        url = (
            f"{MOEX_BASE}/engines/stock/markets/shares/boards/{board}/securities.json"
            "?iss.meta=off&iss.only=securities&securities.columns=SECID,SHORTNAME,NAME,SECNAME,LATNAME,ISIN"
        )
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        blk = r.json().get("securities", {})
        columns = blk.get("columns", []) or []
        data = blk.get("data", []) or []
        # вернём и колонки, и строки, чтобы дальше маппить по имени
        return columns, data

    def _build_offline_directory(self, include_tqtf: bool):
        print("Гружу справочник TQBR…")
        cols, data = self._fetch_board("TQBR")
        if include_tqtf:
            print("Гружу справочник TQTF…")
            cols2, data2 = self._fetch_board("TQTF")
            # если набор колонок отличается, просто приведём строки ко второму формату, доставая по имени
            # но проще объединять как есть — будем читать по именам внутри цикла
            data += data2
            # columns нам важны только как «словарь имён» — возьмём из первого ответа,
            # а в цикле будем безопасно получать индексы через cols.index(...) с проверкой наличия

        # безопасно берём индексы (если колонки нет — вернём None)
        def idx(col):
            try:
                return cols.index(col)
            except ValueError:
                return None

        i_secid     = idx("SECID")
        i_shortname = idx("SHORTNAME")
        i_name      = idx("NAME")
        i_secname   = idx("SECNAME")
        i_latname   = idx("LATNAME")
        i_isin      = idx("ISIN")

        created = 0
        for row in data:
            # достаём по индексам, которые могут быть None
            def get(i):
                return (row[i] if i is not None and i < len(row) else None)

            secid     = get(i_secid)
            shortname = get(i_shortname)
            name      = get(i_name)
            secname   = get(i_secname)
            latname   = get(i_latname)
            isin      = get(i_isin)

            if not secid:
                continue
            ticker = str(secid).strip().upper()

            # 1) сам тикер как ключ
            self.cache[ticker.lower()] = ticker

            # 2) все доступные «имена» нормализуем и кладём
            for f in (shortname, name, secname, latname, isin):
                if f:
                    key = normalize_name(str(f))
                    if key:
                        self.cache.setdefault(key, ticker)

            # 3) базовое русское имя без меток ао/ап и его варианты
            base_ru = None
            if shortname:
                base_ru = normalize_name(
                    re.sub(r"\b(ап|ао|pref|преф|pr)\b", " ", str(shortname), flags=re.IGNORECASE)
                )
            if not base_ru and name:
                base_ru = normalize_name(str(name))

            def is_pref_from_shortname(sn: Optional[str]) -> bool:
                s = (sn or "").lower()
                return any(x in s for x in [" ап", " преф", " pref", " pr"])

            if base_ru:
                if is_pref_from_shortname(shortname):
                    for v in (f"{base_ru} ап", f"{base_ru} pref", f"{base_ru} преф"):
                        self.cache.setdefault(v, ticker)
                else:
                    for v in (f"{base_ru} ао", f"{base_ru} ao"):
                        self.cache.setdefault(v, ticker)

            created += 1

        print(f"Создано {created} бумаг, всего ключей в кэше: {len(self.cache)}")


    def _apply_manual_mappings(self):
        for k, v in (self.manual_mappings or {}).items():
            self.cache[normalize_name(k)] = v.upper()
            # и прямой тикер-ключ
            self.cache[k.lower()] = v.upper()

    def get_ticker(self, company_name: str) -> Optional[str]:
        if not company_name:
            return None
        raw = company_name.strip()
        # 1) если это уже тикер (в верхнем регистре и без пробелов) — вернем как есть,
        # но все равно проверим наличие в кэше
        direct_key = raw.lower()
        if direct_key in self.cache:
            return self.cache[direct_key]

        # 2) по нормализованному
        key = normalize_name(raw)
        if key in self.cache:
            return self.cache[key]

        # 3) простые эвристики для «-ао/-ап» через дефисы
        key_dash = normalize_name(raw.replace("-", " "))
        if key_dash in self.cache:
            return self.cache[key_dash]

        return None

    def replace_tickers_in_record(self, record: Dict, context: Optional[str] = None):
            """
            Заменяет названия на тикеры. Возвращает кортеж:
            (обновлённая_запись, stats_dict)

            stats_dict = {
                "not_found": set[str],           # какие имена не нашли
                "removed_in_scope": int,         # сколько объектов выкинули из scope.tickers
                "removed_in_sentiment": int,     # сколько из sentiment.by_ticker
                "removed_in_target": int,        # сколько из target.tickers
                "kept_total": int,               # сколько оставили суммарно
            }
            """
            work = record.get("result", record)
            stats = {
                "not_found": set(),
                "removed_in_scope": 0,
                "removed_in_sentiment": 0,
                "removed_in_target": 0,
                "kept_total": 0,
            }

            def _replace_in_list(lst, bucket_name):
                if not isinstance(lst, list):
                    return []
                out = []
                for obj in lst:
                    old = obj.get("ticker")
                    tk = self.get_ticker(old)
                    if tk:
                        new_obj = dict(obj)
                        new_obj["ticker"] = tk
                        out.append(new_obj)
                    else:
                        stats["not_found"].add(str(old))
                        stats[f"removed_in_{bucket_name}"] += 1
                return out

            # Сбор уникальных «кандидатов» для корректного отчёта
            # (чтобы «не найдено» выводилось один раз на имя)
            # — мы и так собираем в stats["not_found"].

            if "scope" in work and "tickers" in work["scope"]:
                work["scope"]["tickers"] = _replace_in_list(work["scope"]["tickers"], "scope")

            if "sentiment" in work and "by_ticker" in work["sentiment"]:
                work["sentiment"]["by_ticker"] = _replace_in_list(work["sentiment"]["by_ticker"], "sentiment")

            if "target" in work and "tickers" in work["target"]:
                work["target"]["tickers"] = _replace_in_list(work["target"]["tickers"], "target")

            # Подсчёт оставшегося
            kept = 0
            kept += len(work.get("scope", {}).get("tickers", []) or [])
            kept += len(work.get("sentiment", {}).get("by_ticker", []) or [])
            kept += len(work.get("target", {}).get("tickers", []) or [])
            stats["kept_total"] = kept

            # Логи по «не найдено» (разово на запись)
            if self.log_not_found and stats["not_found"]:
                where = f" [{context}]" if context else ""
                for name in sorted(stats["not_found"]):
                    print(f"⚠️ Не найден тикер для: '{name}'{where}")

            return record, stats

    def process_jsonl_file(self, input_file: str, output_file: str):
        """
        Обрабатывает JSONL. Если drop_records_without_tickers=True,
        записи без единого валидного тикера НЕ пишем (и логируем, почему).
        """
        print(f"\nОбработка файла: {input_file}")
        processed = 0
        written = 0
        dropped = 0

        import ast

        with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        rec = ast.literal_eval(line)

                    # Подготовим контекст для логов
                    rec_id = rec.get("_id") or rec.get("id") or ""
                    title = rec.get("title") or rec.get("result", {}).get("title") or ""
                    ctx = f"строка {line_num}" + (f", id={rec_id}" if rec_id else "") + (f", '{title[:60]}'" if title else "")

                    rec2, stats = self.replace_tickers_in_record(rec, context=ctx)

                    if self.drop_records_without_tickers and stats["kept_total"] == 0:
                        # ничего не осталось — дропаем и явно логируем
                        print(f"🗑  Запись удалена: нет валидных тикеров ({ctx}). "
                              f"Удалено: scope={stats['removed_in_scope']}, "
                              f"sentiment={stats['removed_in_sentiment']}, target={stats['removed_in_target']}")
                        dropped += 1
                    else:
                        fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                        written += 1

                    processed += 1
                    if processed % 500 == 0:
                        print(f"Прогресс: обработано {processed}, записано {written}, удалено {dropped}")

                except Exception as e:
                    print(f"Ошибка в строке {line_num}: {e}")
                    continue

        print(f"\n✅ Готово. Обработано: {processed}, записано: {written}, удалено: {dropped}")
        print(f"✅ Результат: {output_file}")

manual_mappings = {
    'банк Россия': 'RTKM',
    'ВТБ Капитал': 'VTBR',
    'РН-Юганскнефтегаз ': 'ROSN',
    'Новокуйбышевский завод катализаторов': 'ROSN',
    'Yandex N.V.': 'YDEX',
    'Газпром трансгаз Томск': 'GAZP',
    'Газпром газораспределение': 'GAZP',
    'Газпром добыча Ямбург': 'GAZP',
    'Газпром экспорт': 'GAZP',
    'Газпром трейдинг': 'GAZP',
    'Газпром добыча Оренбург': 'GAZP',
    'RUSAL Plc': 'RUAL',
    'Tele2': 'RTKM',
    'Теле2': 'RTKM ',
    'Т2 Мобайл': 'RTKM',
    'Газпром Межрегионгаз': 'GAZP',
    'Северсталь-метиз': 'CHMF',
    'АК АЛРОСА': 'ALRS',
    'Яндекс.Еда': 'YDEX',
    'Лента др': 'LNTA',
    'Х5': 'X5',
    'Московский Кредитный Банк': 'CBOM',
    'Delivery Club': 'SBER',
    'Российские сети': 'FEES',
    'Rosneft Trading': 'ROSN',
    'Башнефть': 'BANEP',
    'САФМАР ФИ': 'SFIN',
    'GAZP': 'GAZP',
    'AFLT': 'AFLT',
    'MVID': 'MVID',
    'SBER': 'SBER',
    'LKOH': 'LKOH',
    'DSKY': 'DSKY',
    'SFIN': 'SFIN',
    'МГНТ ': 'MGNT',
    'FIVE': 'X5',
    'NLMK': 'NLMK',
    'MOEX': 'MOEX',
    'MAGN': 'MAGN',
    'PLZL': 'PLZL',
    'ALRS': 'ALRS',
    'PHOR': 'PHOR',
    'IRAO': 'IRAO',
    'POLY': 'POLY',
    'CHMF': 'CHMF',
    'PLZL': 'PLZL',
    'GAZP': 'GAZP',
    'Мечел-ао': 'GAZP',
    'Сургутнефтегаз-ао': 'SNGS',
    'Сургутнефтегаз': 'SNGS',
    'Россети-ап': 'MSRS',
    'Банк ВТБ': 'VTBR',
    'Татнефть-ап': 'TATN',
    'Газпромнефть': 'GAZP',
    'Промсвязьбанк': 'PSBR',
    'Мосэнерго': 'MSNG',
    'X5': 'X5',
    'Россети-ао': 'MSRS',
    'Татнефть-ао': 'TATN',
    'Яндекс.Такси': 'YDEX',
    'Аптеки 36,6': 'APTK',
    'Аптеки 36и6': 'APTK',
    'ЮТэйр': 'UTAR',
    'Пятерочка': 'X5',
    'Уралкалий': 'URKA',
    'Газпромнефть-Аэро': 'GAZP',
    'Новатек': 'NVTK',
    'МКБ': 'CBOM',
    'Sberbank CIB': 'SBER',
    'Тинькофф Банк': 'Т',
    'Банк Санкт-Петербург': 'BSPB',
    'Кубаньэнерго': 'KBSB',
    'ЛСР': 'LSRG',
    'Mail.Ru Group': 'MAIL',
    'Mail.ru': 'MAIL',
    'Ситимобил': 'VKCO',
    'Nord Steam 2': 'GAZP',
    'Nord Stream 2': 'GAZP',
    'Nord Stream': 'GAZP',
    'МРСК ЦП': 'MRKP',
    'САФМАР': 'SFIN',
    'АФК Система': 'AFKS',
    'Норильский Никель': 'GMKN',
    'Транснефть': 'TRNFP',
    'Ленэнерго': 'LSNGP',
    'Мечел': 'MTLR',
    'газпром': 'GAZP',
    'лукойл': 'LKOH',
    'сбербанк': 'SBER',
    'яндекс': 'YDEX',
    'роснефть': 'ROSN',
    'новатэк': 'NVTK',
    'татнефть': 'TATN',
    'магнит': 'MGNT',
    'норникель': 'GMKN',
    'полюс': 'PLZL',
    'vtb': 'VTBR',
    'втб': 'VTBR',
    'мтс': 'MTSS',
    'алроса': 'ALRS',
    'сургутнефтегаз': 'SNGS',
    'северсталь': 'CHMF',
    'polymetal': 'POLY',
    'полиметалл': 'POLY',
    'аэрофлот': 'AFLT',
    'мосбиржа': 'MOEX',
    'тинькофф': 'TCSG',
    'ozon': 'OZON',
    'озон': 'OZON',
    'фикс прайс': 'FIXP',
    'фосагро': 'PHOR',
    'русал': 'RUAL',
    'нлмк': 'NLMK',
    'пик': 'PIKK',
    'интер рао': 'IRAO',
    'X5 Retail Group': 'X5',
    'ЧТПЗ': 'CHEP',
    'ГМКН': 'GMKN',
    'ТМК': 'TRMK',
    'НМТП': 'NMTP',
    'Газпром нефть': 'SIBN',
    'ММК': 'MAGN',
    'Магнитогорский металлургический комбинат': 'MAGN',
    'Детский Мир': 'DSKY',
    'SberCloud': 'SBER',
    'Московская биржа': 'MOEX',
    'Яндекс.Драйв': 'YDEX',
    'Ростелеком': 'RTKM',
    'ИнтерРао ': 'IRAO',
    'FIVE-гдр': 'X5',
    'Yandex': 'YDEX',
    'Газпромбанк': 'GAZP',
    'ТрансКонтейнер': 'TRCN ',
    'ТрансК': 'TRCN',
    'Уральские авиалинии': 'URAL',
    'Победа': 'AFLT',
    'Россия': 'AFLT',
    'Яндекс.Деньги': 'YDEX',
}

def unite_all_batches(root_dir_name: str = "artifacts", output_file_name: str = "merged_parsed.jsonl"):
    """
    Объединяет все файлы parsed.jsonl из папок batch_00000, batch_00001 и т.д.
    в один файл merged_parsed.jsonl.
    """
    root_dir = Path(root_dir_name)
    output_file = root_dir / output_file_name
    with open(output_file, "w", encoding="utf-8") as outfile:
        for batch_dir in sorted(root_dir.iterdir()):
            parsed_path = batch_dir / "parsed.jsonl"
            if parsed_path.exists():
                with open(parsed_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
    print(f"Готово! Все parsed.jsonl объединены в {output_file}")
    
# Пример использования
if __name__ == "__main__":
    # Объединяем батчи в один файл
    unite_all_batches(root_dir_name="artifacts", output_file_name="merged_parsed.jsonl")
    
    # Создаем маппер
    mapper = MOEXTickerMapper(manual_mappings=manual_mappings)
    
    # Обрабатываем файл
    mapper.process_jsonl_file(
        input_file=r'merged_parsed.jsonl',
        output_file='llm_news_file.jsonl'
    )
    
    # Примеры поиска отдельных тикеров:
    print("\n--- Примеры поиска ---")
    test_names = ['Газпром', 'ЛУКОЙЛ', 'Сбербанк', 'Яндекс']
    for name in test_names:
        ticker = mapper.get_ticker(name)
        print(f"{name} -> {ticker}")