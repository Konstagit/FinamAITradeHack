# !pip install transformers torch scikit-learn pandas tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from sklearn.neighbors import NearestNeighbors
import tiktoken
import re

# ====== настройки ======
DATE_COL  = "publish_date"
TITLE_COL = "title"
TEXT_COL  = "publication"

MODEL_NAME   = "sergeyzh/BERTA"  # можно "BAAI/bge-m3"
BATCH_SIZE   = 128
TITLE_WEIGHT = 1.0
NORMALIZE    = True

DAYS_WINDOW  = 1
SIM_THR      = 0.90
N_NEIGHBORS  = 20

# ==== DSU с контролем временного диапазона кластера ====
class DSUWithSpan:
    def __init__(self, dates_int):
        n = len(dates_int)
        self.p = list(range(n))
        self.sz = [1]*n
        # по каждому корню храним min/max день
        self.min_day = dates_int.copy()
        self.max_day = dates_int.copy()

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union_maybe(self, a, b, max_span_days):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # прогнозируемый диапазон после объединения
        new_min = min(self.min_day[ra], self.min_day[rb])
        new_max = max(self.max_day[ra], self.max_day[rb])
        if (new_max - new_min) > max_span_days:
            # объединение растянет кластер дальше окна — не склеиваем
            return False
        # нормальный union by size
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        self.min_day[ra] = new_min
        self.max_day[ra] = new_max
        return True
    

def deduplicate_news(news_path1, news_path2, output_path):
    """
    Выполняет полную дедупликацию новостей и сохраняет результат.
    :param news_path1: Путь к первому файлу с новостями.
    :param news_path2: Путь ко второму файлу с новостями.
    :param output_path: Путь для сохранения итогового файла.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # ====== Загружаем данные ======
    print("Загрузка данных...")
    news = pd.read_csv(news_path1)
    news2 = pd.read_csv(news_path2)
    news_full = pd.concat([news, news2], ignore_index=True)
    news = news_full.drop_duplicates(subset=[DATE_COL, TITLE_COL, TEXT_COL]).reset_index(drop=True)

    # ====== загрузка модели Эмбеддингов ======
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    @torch.no_grad()
    def mean_pool(last_hidden_state, attention_mask):
        # усреднение по токенам с маской
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.no_grad()
    def embed_texts(texts):
        embs_list = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="HF embeddings"):
            batch = texts[i:i+BATCH_SIZE]

            batch_prefixed = [f"passage: {t}" for t in batch]

            enc = tok(batch_prefixed, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            if NORMALIZE:
                emb = F.normalize(emb, p=2, dim=1)
            embs_list.append(emb.cpu())
        return torch.vstack(embs_list).numpy()



    
    # ====== Подготавливаем данные ======
    print("Подготовка данных...")
    # Считаем токены с помощью tiktoken
    def count_tokens(text):
        return len(enc.encode(str(text)))
    def make_text(row):
        # усиливаем заголовок повтором
        return ( (str(row[TITLE_COL]) + " ") * int(TITLE_WEIGHT) + str(row[TEXT_COL]) ).strip()

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # считаем по каждой публикации
    news["token_count"] = news["publication"].apply(count_tokens)
    # отсекаем короткие тексты
    news = news[news['token_count']>=80].reset_index(drop=True) 
    news[DATE_COL] = pd.to_datetime(news[DATE_COL],format="%Y-%m-%d %H:%M:%S",errors="coerce")
    news = news.sort_values(DATE_COL).reset_index(drop=True)
    news = news.dropna(subset=[DATE_COL, TITLE_COL, TEXT_COL]).reset_index(drop=True)


    # Сохраняем новости, с заголвком о выпуске однодневных облгиаций.Очень похожи, но не являются дупликатами
    ONE_DAY_PATTERN = r"(одноднев\w*)"
    mask_one_day = news[TITLE_COL].fillna("").str.contains(ONE_DAY_PATTERN, flags=re.I, regex=True)
    # Фильтруем записи с однодневный в заголовке
    news_one_day = news[mask_one_day].copy()
    # Остальное идет в дедуп
    news  = news[~mask_one_day].copy()        # это идёт в дедуп

    news = news.sort_values(DATE_COL).reset_index(drop=True)
    docs_raw = [make_text(r) for _, r in news.iterrows()]

    # ====== формируем бакеты по датам ======
    news["_date"] = news[DATE_COL].dt.date
    unique_days = sorted(news["_date"].unique())
    original_index_to_pos = {idx: pos for pos, idx in enumerate(news.index)}
    day2idx = {d: [original_index_to_pos[idx] for idx in news.index[news["_date"] == d]] for d in unique_days}
    # убедимся в типах дат
    assert np.issubdtype(news[DATE_COL].dtype, np.datetime64), news[DATE_COL].dtype
    # преобразуем в целые дни для быстрых проверок
    news["_day_int"] = (news[DATE_COL].dt.floor("D").astype("int64") // 86_400_000_000_000)
    # Сохраняем массив для быстрого доступа
    day_int_array = news["_day_int"].to_numpy()

    # ====== Вычисление эмбеддингов ======
    embs = embed_texts(docs_raw) 
    dsu = DSUWithSpan(day_int_array)

    # ==== поиск похожих внутри окна дат ====
    for d in tqdm(unique_days, desc="Dedup by time window"):
        d_ts = pd.to_datetime(d)
        left, right = d_ts - timedelta(days=DAYS_WINDOW), d_ts + timedelta(days=DAYS_WINDOW)
        days_in = [dd for dd in unique_days if (pd.to_datetime(dd) >= left) and (pd.to_datetime(dd) <= right)]
        if not days_in:
            continue
        cand_idx = np.concatenate([day2idx[dd] for dd in days_in])
        if len(cand_idx) <= 1:
            continue

        X = embs[cand_idx]
        nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, len(cand_idx)), metric="cosine")
        nn.fit(X)
        dist, nbrs = nn.kneighbors(X, return_distance=True)

        # корректный проход по соседям: синхронно dist_row и nbr_row
        for i_row, (dist_row, nbr_row) in enumerate(zip(dist, nbrs)):
            i_local = cand_idx[i_row]  # это уже позиция в news (0..len(news)-1)
            for d_ij, j_local_idx in zip(dist_row, nbr_row):
                if j_local_idx == i_row:
                    continue
                j_local = cand_idx[j_local_idx]  # это тоже позиция в news
                sim = 1.0 - float(d_ij)
                if sim < SIM_THR:
                    continue

                # строгая проверка пары по дням (без часов) - используем массив
                di = day_int_array[i_local]
                dj = day_int_array[j_local]
                if abs(int(di - dj)) > DAYS_WINDOW:
                    continue

                # объединяем ТОЛЬКО если итоговый кластер не растягивается дальше окна
                dsu.union_maybe(i_local, j_local, max_span_days=DAYS_WINDOW)

    # ==== сбор кусков и выбор репрезентативов ====
    root2members = {}
    for i in range(len(news)):
        r = dsu.find(i)
        root2members.setdefault(r, []).append(i)

    keep = np.ones(len(news), dtype=bool)
    rep_for = {}
    for r, members in root2members.items():
        if len(members) == 1:
            rep_for[members[0]] = members[0]
            continue
        # оставляем самую раннюю; при равенстве — самую длинную
        members_sorted = sorted(
            members,
            key=lambda i: (news.iloc[i][DATE_COL], -len(str(news.iloc[i][TEXT_COL])))
        )
        rep = members_sorted[0]
        rep_for[rep] = rep
        for i in members:
            if i != rep:
                keep[i] = False
                rep_for[i] = rep

    deduped = news[keep].copy().reset_index(drop=True)

    # Клеим к однодневным новостям (как у тебя)
    deduped_full = pd.concat([deduped, news_one_day], ignore_index=True)
    deduped_full = deduped_full.sort_values("publish_date").reset_index(drop=True)

    print(
        f"Итого: исходно {len(news_full)}; "
        f"однодневных вынесли {len(news_one_day)}; "
        f"дедуп по остальным: {len(news)} -> {len(deduped)}; "
        f"итоговый датафрейм: {len(deduped_full)}"
    )

    deduped_full.to_csv(output_path, index=False)
    print(f"Результат сохранен в {output_path}")
    
if __name__ == "__main__":
    # Указываем пути к файлам здесь
    path1 = r'data\raw\participants\news.csv'
    path2 = r'data\raw\participants\news_2.csv'
    output = 'deduped_full_direct_run.csv'
    
    deduplicate_news(news_path1=path1, news_path2=path2, output_path=output)
