# -*- coding: utf-8 -*-
"""
화면이 쓰는 예측 진입점 — **최종 모델(LightGBM)** 을 쓴다.

여기서 하는 일은 하나뿐이다.
**사람이 화면에 입력한 것을 학습할 때와 똑같은 숫자로 바꾸는 것.**

이 변환을 화면 쪽에서 따로 짜면 안 된다. 미묘하게 달라지고,
에러 없이 조용히 틀린 예측이 나온다. (실제로 겪었다 — 언어별 기본값이
화면에만 따로 박혀 있어서 review_len_z 가 155 까지 튄 적이 있다)

모델 정보
    models/ml_model.joblib        LightGBM 튜닝본 · 전체 30개 언어 139,658행
    models/ml_threshold.json      0.3735  <- 0.5 가 아니다
    성능                          랜덤 0.818 · 게임5조각 0.748 · 봉인12게임 0.721

★ 이 모델은 파이프라인이 아니라 순수 LGBMClassifier 다.
  열 순서와 범주값을 맞춰줘야 한다. 그 처리는 팀원이 만든
  src/explain_ml.to_model_frame() 이 해준다. 직접 하지 말 것.
"""
import json
import re

import numpy as np
import pandas as pd

from src.config import DATA_PROC, MODELS, LANG_STATS, ENC_READ
from src.explain_ml import explain_one, load_final, to_model_frame
from src.preprocess import featurize

# 화면 전용 파일들. 팀 공유 config.py 를 늘리지 않는다 —
# 화면이 사라지면 같이 사라질 것들이라서. 폴더는 config 에서 받는다.
GAMES_CSV = DATA_PROC / "games.csv"
CARDS_JSON = DATA_PROC / "cards.json"
UNSEEN_CSV = DATA_PROC / "unseen.csv"
CATALOG_CSV = DATA_PROC / "catalog.csv"
ML_META = MODELS / "ml_meta.json"

STEAM_IMG = "https://cdn.akamai.steamstatic.com/steam/apps/{}/header.jpg"


def load_all():
    """화면이 필요로 하는 것 전부. Streamlit 쪽에서 캐시를 씌워 쓴다."""
    model, order, thr, meta = load_final()
    games = pd.read_csv(GAMES_CSV, encoding=ENC_READ)

    # games.csv 에는 설명이 없다. catalog.csv 를 만들 때 스팀에서 같이 받아뒀고
    # 우리 60개 중 58개가 거기 들어 있으니 appid 로 붙여 온다. 새로 받을 게 없다.
    if CATALOG_CSV.exists():
        cat = pd.read_csv(CATALOG_CSV, encoding=ENC_READ)
        if "설명" in cat.columns:
            games = games.merge(cat[["appid", "설명"]].drop_duplicates("appid"), on="appid", how="left")

    cards = json.load(open(CARDS_JSON, encoding=ENC_READ))
    unseen = pd.read_csv(UNSEEN_CSV, encoding=ENC_READ)
    lang = json.load(open(LANG_STATS, encoding=ENC_READ))
    meta = {**meta, "임계값": thr, "_order": order}
    
    return model, games, cards, unseen, meta, lang


def load_catalog():
    """추천용 게임 카탈로그. 없으면 None."""
    return pd.read_csv(CATALOG_CSV, encoding=ENC_READ) if CATALOG_CSV.exists() else None


def _game_dict(g):
    """games.csv 또는 catalog.csv 의 한 줄 -> featurize 가 받는 형태."""
    return {
        "genre_group": g["genre_group"],
        "era": g["era"],
        "grade": g["grade"],
        "release_year": int(g["release_year"]),
        # featurize 는 출시 시각(초)을 받아 game_age_days 를 만든다.
        # 표에는 이미 계산된 game_age_days 만 있으므로 거꾸로 되돌린다.
        "app_release_ts": _NOW - int(g["game_age_days"]) * 86400,
        "game": g.get("game", "UNKNOWN"),
    }


_NOW = 1787643023        # 수집 기준 시각 (03_수집/manifest.json 의 기준_unix)


def build_row(review, hours, owned, n_reviews, voted_up, game_row, lang_stats,
              language="english"):
    """화면 입력 한 줄 -> 모델이 먹는 표 한 줄."""
    return featurize({
        "review": review,
        "language": language,
        "playtime_at_review_min": int(float(hours) * 60),
        "num_games_owned": int(owned),
        "num_reviews": int(n_reviews),
        "created_ts": _NOW,
        "voted_up": bool(voted_up),
    }, _game_dict(game_row), lang_stats)


def predict(model, order, row):
    """확률 하나만. 근거가 필요하면 explain_one 을 쓴다."""
    return float(model.predict_proba(to_model_frame(row, order))[0, 1])


def predict_many(model, order, review, hours, owned, n_reviews, voted_up,
                 games_df, lang_stats):
    """게임 여러 개를 한 번에 채점한다 (추천·계산기 화면).

    한 줄씩 predict 를 부르면 1,500개에 몇 초가 걸린다.
    표를 한 번에 만들어 넘기면 훨씬 빠르다.
    """
    rows = [build_row(review, hours, owned, n_reviews, voted_up, g, lang_stats)
            for _, g in games_df.iterrows()]
    X = to_model_frame(pd.concat(rows, ignore_index=True), order)
    return model.predict_proba(X)[:, 1]


def explain(row, top_n=6):
    """이 예측에 각 변수가 얼마나 기여했는지 (SHAP).

    팀원이 만든 explain_one 을 그대로 쓴다.
    '기여' 는 확률이 아니라 로그오즈라, 화면에는 기여율(%)을 같이 보여준다.
    """
    r = explain_one(row, 상위=top_n)
    return pd.DataFrame([{"변수": x["한글"], "기여": x["기여"], "기여율": x["기여율"]}
                         for x in r["이유"]]), r


def gauge(p, thr):
    """확률을 사람이 읽는 말로. 임계값을 기준으로 나눈다."""
    if p >= thr + .15:
        return "🔴", "떠날 가능성이 높습니다"
    if p >= thr:
        return "🟡", "떠나는 쪽입니다"
    if p >= thr - .15:
        return "🟢", "남는 쪽입니다"
    return "🟢", "계속할 가능성이 높습니다"
