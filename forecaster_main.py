import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import lightgbm as lgb
from xgboost import XGBRegressor
import xgboost as xgb

import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path


## Подготовка данных цен акций
train_path = Path(r'data\raw\participants\candles.csv')
test_path = Path(r'data\raw\participants\candles_2.csv')
df_news_path =  Path(r'data\processed\df_news.csv')

def make_preds(train_path: Path, test_path: Path, df_news_path: Path, output_path: str = 'result_submission.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = pd.concat([train, test])

    train['begin'] = pd.to_datetime(train['begin'])
    train = train.drop_duplicates(subset=['begin','ticker']).reset_index(drop=True)

    # Читаем датасет новостей и агрегируем по дням
    df_news = pd.read_csv(df_news_path)
    df_news = df_news.drop(['_id','title','source','sectors'],axis=1)

    # Приводим типы дат
    df_news["published_at"] = pd.to_datetime(df_news["published_at"])
    df_news["published_at"] = df_news["published_at"].dt.date
    df_news["published_at"] = pd.to_datetime(df_news["published_at"])
    df_news = df_news.sort_values(["published_at", "ticker"])

    cat_cols = ["article_type", "time_horizon", "role"]
    df_news = pd.get_dummies(df_news, columns=cat_cols, drop_first=True)
    df_news['all_sector_impact'] = df_news['all_sector_impact'].astype(bool)

    # исходные колонки в df_news после one-hot
    num_cols = [
        "source_reliability",
        "overall_sentiment",
        "certainty",
        "all_sector_impact",   
        "surprise",
        "relevance",
        "sentiment_by_ticker",
        "target_price_growth",
        "target_confidence",
        "target_price_speed",
    ]

    # one-hot колонки
    one_hot_cols = [
        "article_type_news",
        "article_type_other",
        "article_type_report",
        "time_horizon_1d",
        "time_horizon_1m",
        "time_horizon_1m+",
        "time_horizon_1w",
        "time_horizon_1y",
        "time_horizon_3m+",
        "time_horizon_intraday",
        "role_mentioned",
    ]

    df_news = df_news[["published_at", "ticker"] + num_cols].assign(**{col: False for col in one_hot_cols})

    # агрегация по дням
    df_news = (
        df_news
        .groupby(["ticker", pd.Grouper(key="published_at", freq="1D")])
        .agg({
            # числовые фичи — среднее
            "source_reliability": "mean",
            "overall_sentiment": "mean",
            "certainty": "mean",
            "all_sector_impact": "mean",   # доля положительных случаев
            "surprise": "mean",
            "relevance": "mean",
            "sentiment_by_ticker": "mean",
            "target_price_growth": "mean",
            "target_confidence": "mean",
            "target_price_speed": "mean",
            
            # one-hot — усредняем (фактически доля True)
            "article_type_news": "mean",
            "article_type_other": "mean",
            "article_type_report": "mean",
            "time_horizon_1d": "mean",
            "time_horizon_1m": "mean",
            "time_horizon_1m+": "mean",
            "time_horizon_1w": "mean",
            "time_horizon_1y": "mean",
            "time_horizon_3m+": "mean",
            "time_horizon_intraday": "mean",
            "role_mentioned": "mean",
        })
        .reset_index()
    )

    df_news = df_news.sort_values('published_at').reset_index(drop=True)

    #Обьединяем
    train["begin"] = pd.to_datetime(train["begin"])
    train = train.sort_values(["begin", "ticker"])
    df_news = df_news.sort_values(["published_at", "ticker"])

    # Соединяем
    merged = pd.merge_asof(
        train,
        df_news,
        by="ticker",
        left_on="begin",
        right_on="published_at",
        direction="backward",   # берём последнюю новость до даты
        tolerance=pd.Timedelta("7d")  #Ограничеваем окном, например 7 дней
    )

    ### Формирование фичей
    def add_forward_returns_trading(df: pd.DataFrame, horizons=range(1, 21)) -> pd.DataFrame:

        out = df.copy()
        out["begin"] = pd.to_datetime(out["begin"])
        out = out.sort_values(["ticker", "begin"]).reset_index(drop=True)

        for N in horizons:
            future_close = out.groupby("ticker")["close"].shift(-N)
            out[f"target_return_{N}d"] = future_close / out["close"] - 1

        return out

    def make_news_features(df_news: pd.DataFrame) -> pd.DataFrame:
        out = df_news.copy()
        out["published_at"] = pd.to_datetime(out["published_at"])
        out = out.sort_values(["ticker", "published_at"])

        # Числовые признаки, по которым будем считать лаги и окна
        num_cols = [
            "source_reliability",
            "overall_sentiment",
            "certainty",
            "surprise",
            "relevance",
            "sentiment_by_ticker",
            "target_price_growth",
            "target_confidence",
            "target_price_speed",
        ]

        # Лаги — значения прошлых новостей
        for L in [1, 3, 5, 7 ,10, 15, 20]:
            for col in num_cols:
                out[f"{col}_lag{L}"] = out.groupby("ticker")[col].shift(L)

        # Скользящие средние и стд
        for W in [3, 7, 14, 21]:
            for col in num_cols:
                out[f"{col}_mean{W}"] = (
                    out.groupby("ticker")[col]
                    .transform(lambda s: s.rolling(W, min_periods=1).mean())
                )
                out[f"{col}_std{W}"] = (
                    out.groupby("ticker")[col]
                    .transform(lambda s: s.rolling(W, min_periods=1).std())
                )

        # Всё, что осталось не числовым (тикер, дата и one-hot)
        out = out.fillna(0)
        return out


    train = make_news_features(merged)
    train = train.drop(['published_at'],axis=1)
    train = add_forward_returns_trading(train, horizons=range(1, 21))


    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """RSI по классике: EMA по gains/losses. Используются только прошлые значения."""
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD = EMA(fast) - EMA(slow); signal = EMA(MACD); hist = MACD - signal."""
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ---------- основной генератор фичей ----------

    def make_features(df: pd.DataFrame) -> pd.DataFrame:

        out = df.copy()

        # типы и порядок
        out["begin"] = pd.to_datetime(out["begin"])
        out = out.sort_values(["ticker", "begin"], kind="mergesort")  # стабильная сортировка

        # базовые доходности
        out["ret_1d"] = out.groupby("ticker")["close"].pct_change()

        # лаги (цена/доходность/объём)
        lag_list = [1, 2, 3, 5, 10, 20]
        for L in lag_list:
            out[f"close_lag{L}"]  = out.groupby("ticker")["close"].shift(L)
            out[f"ret_1d_lag{L}"] = out.groupby("ticker")["ret_1d"].shift(L)
            out[f"vol_lag{L}"]    = out.groupby("ticker")["volume"].shift(L)

        # многодневные доходности и ROC/momentum
        win_ret = [2, 3, 5, 10, 20]
        for W in win_ret:
            out[f"ret_{W}d"] = out.groupby("ticker")["close"].pct_change(W)
            out[f"roc_{W}d"] = out.groupby("ticker")["close"].transform(lambda s: (s / s.shift(W)) - 1)

        # скользящие средние/стд по close и volume
        roll_windows = [3, 5, 10, 20]
        for W in roll_windows:
            grp_close = out.groupby("ticker")["close"]
            grp_vol   = out.groupby("ticker")["volume"]
            grp_ret   = out.groupby("ticker")["ret_1d"]

            out[f"sma_{W}"]      = grp_close.transform(lambda s: s.rolling(W, min_periods=W).mean())
            out[f"sma_ratio_{W}"] = out["close"] / out[f"sma_{W}"]
            out[f"ema_{W}"]      = grp_close.transform(lambda s: s.ewm(span=W, adjust=False, min_periods=W).mean())
            out[f"ret_std_{W}"]  = grp_ret.transform(lambda s: s.rolling(W, min_periods=W).std())

            out[f"vol_mean_{W}"] = grp_vol.transform(lambda s: s.rolling(W, min_periods=W).mean())
            out[f"vol_std_{W}"]  = grp_vol.transform(lambda s: s.rolling(W, min_periods=W).std())
            out[f"vol_rel_{W}"]  = out["volume"] / (out[f"vol_mean_{W}"] + 1e-12)  # аномалии объёма
            out[f"vol_z_{W}"]    = (out["volume"] - out[f"vol_mean_{W}"]) / (out[f"vol_std_{W}"] + 1e-12)

        # ATR (true range) и варианты волатильности диапазона
        out["prev_close"] = out.groupby("ticker")["close"].shift(1)
        out["tr"] = true_range(out["high"], out["low"], out["prev_close"])
        for W in [5, 14, 20]:
            out[f"atr_{W}"] = out.groupby("ticker")["tr"].transform(lambda s: s.rolling(W, min_periods=W).mean())
        out.drop(columns=["prev_close"], inplace=True)

        # Bollinger Bands (по 20, можно добавить и другие окна)
        out["bb_mid_20"] = out["sma_20"]
        out["bb_std_20"] = out.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).std())
        out["bb_up_20"]  = out["bb_mid_20"] + 2 * out["bb_std_20"]
        out["bb_dn_20"]  = out["bb_mid_20"] - 2 * out["bb_std_20"]
        out["bb_width_20"] = (out["bb_up_20"] - out["bb_dn_20"]) / (out["bb_mid_20"] + 1e-12)
        out["bb_percB_20"] = (out["close"] - out["bb_dn_20"]) / ((out["bb_up_20"] - out["bb_dn_20"]) + 1e-12)

        # RSI и MACD
        out["rsi_14"] = out.groupby("ticker", group_keys=False)["close"].apply(lambda s: rsi(s, 14))
        macd_parts = out.groupby("ticker", group_keys=False)["close"].apply(lambda s: pd.DataFrame({
            "macd_line": macd(s)[0],
            "macd_signal": macd(s)[1],
            "macd_hist": macd(s)[2]
        }))
        out = out.join(macd_parts.reset_index(level=0, drop=True))

        # свечные фичи
        out["body"]        = out["close"] - out["open"]
        out["range"]       = (out["high"] - out["low"]).replace(0, np.nan)
        out["upper_shadow"]= (out["high"] - np.maximum(out["open"], out["close"])).clip(lower=0)
        out["lower_shadow"]= (np.minimum(out["open"], out["close"]) - out["low"]).clip(lower=0)
        out["body_to_range"]= out["body"] / out["range"]
        out["dir_up"]      = (out["close"] > out["open"]).astype(int)  # 1 если бычья свеча

        # объём/деньги
        out["log_volume"]      = np.log1p(out["volume"])
        out["dollar_turnover"] = out["close"] * out["volume"]
        out["dollar_turnover_rel20"] = out["dollar_turnover"] / (
            out.groupby("ticker")["dollar_turnover"].transform(lambda s: s.rolling(20, min_periods=20).mean()) + 1e-12
        )

        # календарь
        dt = out["begin"]
        out["weekday"]        = dt.dt.weekday
        out["month"]          = dt.dt.month
        out["is_month_end"]   = dt.dt.is_month_end.astype(int)
        out["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

        # кросс-фичи
        out["ret_vol_interact20"] = out["ret_1d"] * out["vol_z_20"]
        out["body_x_volrel20"]    = out["body_to_range"].fillna(0) * out["vol_rel_20"].fillna(0)

        return out


    # Выделение трейна и теста
    features_df = make_features(train)
    data_train = make_features(train).dropna(subset=['target_return_20d']).fillna(0)

    max_date = features_df['begin'].max()
    data_oot = features_df[features_df['begin']==max_date].fillna(0)


    print(f'Размер тренировочной выборки: {data_train.shape[0]}')
    print(f'Размер отложенной выборки: {data_oot.shape[0]}')



    # Выбор фичей
    drop_cols = ['begin', 'ticker']

    # Уберём все target_* из фичей
    drop_cols += [col for col in data_train.columns if col.startswith('target_')]

    x_train = data_train.drop(columns=drop_cols)
    x_oot   = data_oot.drop(columns=drop_cols)

    # Формируем все таргеты
    target_cols = [col for col in data_train.columns if col.startswith('target_return_')]

    y_train = data_train[target_cols]


    ### Фича селекшен
    base_lgb  = lgb.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=42)
    base_xgb  = XGBRegressor(n_jobs=-1, random_state=42)

    model_lgbm_full = MultiOutputRegressor(base_lgb)
    model_xgb_full  = MultiOutputRegressor(base_xgb)

    model_lgbm_full.fit(x_train, y_train)
    model_xgb_full.fit(x_train, y_train)

    def _xgb_importance_to_df(estimator, feature_names, importance_type="gain"):

        booster = estimator.get_booster()
        imp_dict = booster.get_score(importance_type=importance_type) or {}

        mapped = {}
        for k, v in imp_dict.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feature_names):
                    mapped[feature_names[idx]] = float(v)
            else:
                mapped[k] = float(v)

        s = pd.Series(0.0, index=pd.Index(feature_names, name="feature"))
        for f, val in mapped.items():
            if f in s.index:
                s.loc[f] = val
        return s

    def top_features_multi(model_lgbm_full, model_xgb_full, feature_names, topk=50,
                        lgbm_mode="gain", xgb_mode="gain", agg="mean"):

        feat_idx = pd.Index(feature_names, name="feature")
        n_feats = len(feat_idx)

        # LGBM: агрегируем важности по всем таргетам 
        lgbm_imps = []
        for est in model_lgbm_full.estimators_:
            imp = pd.Series(est.feature_importances_, index=est.feature_name_)
            imp = imp.reindex(feat_idx, fill_value=0.0).astype(float)
            lgbm_imps.append(imp.values)

        if agg == "mean":
            lgbm_vec = np.mean(lgbm_imps, axis=0)
        else:
            lgbm_vec = np.sum(lgbm_imps, axis=0)

        lgbm_vec = lgbm_vec / (lgbm_vec.sum() + 1e-12)
        features_lgbm = pd.DataFrame({"feature": feat_idx, "importance_lgbm": lgbm_vec}) \
                            .sort_values("importance_lgbm", ascending=False).head(topk)

        # XGB: агрегируем важности по всем таргетам 
        xgb_imps = []
        for est in model_xgb_full.estimators_:
            s = _xgb_importance_to_df(est, feature_names=feat_idx, importance_type=xgb_mode)
            xgb_imps.append(s.values)

        if agg == "mean":
            xgb_vec = np.mean(xgb_imps, axis=0)
        else:
            xgb_vec = np.sum(xgb_imps, axis=0)
        xgb_vec = xgb_vec / (xgb_vec.sum() + 1e-12)
        features_xgb = pd.DataFrame({"feature": feat_idx, "importance_xgb": xgb_vec}) \
                            .sort_values("importance_xgb", ascending=False).head(topk)

        # Объединяем топы и собираем финальный список
        features = features_lgbm.merge(features_xgb, on="feature", how="outer")

        features["rank_lgbm"] = features["importance_lgbm"].rank(ascending=False, method="min")
        features["rank_xgb"]  = features["importance_xgb"].rank(ascending=False, method="min")
        features["rank_sum"]  = features[["rank_lgbm", "rank_xgb"]].min(axis=1)  # лучшее из двух
        features = features.sort_values(["rank_sum"], ascending=True).reset_index(drop=True)
        return features

    feature_names = list(x_train.columns)
    features_full = top_features_multi(model_lgbm_full, model_xgb_full, feature_names, topk=50)


    ### Прогноз

    # Топ признаки
    feat_cols = list(features_full["feature"])
    
    xgb_base = XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",         # CPU
        predictor="cpu_predictor",  # CPU
        n_estimators=600,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        random_state=42,
    )

    # xgb_base = XGBRegressor(
    #     objective="reg:absoluteerror",
    #     tree_method="gpu_hist",
    #     predictor="gpu_predictor",
    #     n_estimators=600,
    #     learning_rate=0.05,
    #     max_depth=10,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     n_jobs=1,
    #     random_state=42,
    # )

    xgb_multi = MultiOutputRegressor(xgb_base, n_jobs=1)
    xgb_multi.fit(x_train[feat_cols], y_train)
    y_pred_xgb = pd.DataFrame(xgb_multi.predict(x_oot[feat_cols]), columns=target_cols, index=data_oot.index)

    result = pd.concat([data_oot['ticker'], y_pred_xgb], axis=1)
    result.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    # Этот код выполнится ТОЛЬКО если вы запустите файл напрямую
    # командой в терминале: python make_predictions_simple.py


    print("Скрипт запущен напрямую. Выполняется пайплайн с путями по умолчанию...")
    
    # Определяем пути по умолчанию для самостоятельного запуска
    default_train_path = Path('data/raw/participants/candles.csv')
    default_test_path = Path('data/raw/participants/candles_2.csv')
    default_df_news_path = Path('data/processed/df_news.csv')

    # Вызываем основную функцию
    make_preds(default_train_path, default_test_path, default_df_news_path)
    
    print("Работа скрипта в автономном режиме завершена.")