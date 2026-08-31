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
                                  튜닝이 고른 class_weight="balanced" 를 배포에서 뺐다.
                                  확률을 부풀리고 있었다 (봉인에서 실제 45.6% 인데
                                  60.4% 라고 말함 -> 52.4% 로 잡음)
    models/ml_threshold.json      0.2899  <- 0.5 가 아니다
                                  설정이 바뀌면 이 값도 바뀐다. 모델과 한 세트라
                                  하드코딩하지 말고 load_final() 로 읽을 것
    성능                          48게임OOF 0.754 · 봉인12게임 0.719
                                  F1 0.651 (전부-이탈 기준선 0.626 대비 +0.025)

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
PLAYTIME_STATS = DATA_PROC / "playtime_stats.json"
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


def load_playtime_stats():
    """리뷰 뒤 실제로 더 플레이한 시간의 중앙값 (살까말까 계산기용).

    src/build_playtime_stats.py 가 원본에서 만든다.
    게임별(60개) · 장르별 · 전체 세 겹으로 들어 있어서,
    카탈로그의 1,500개 중 우리가 안 모은 게임은 장르로 대신한다.
    """
    return json.load(open(PLAYTIME_STATS, encoding=ENC_READ))


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


# ── 딥러닝(글을 읽는 모델) — 화면 1 전용 ──────────────────────────
# 화면 1 의 의의는 "학습에 없던 리뷰라도 글을 읽고 확률을 낸다" 이다.
# 최종 모델(LightGBM)은 글의 겉모양만 본다 — 글자 수·단어 수·대문자 비율.
# 실제로 "인생 최고의 게임"과 "인생 최악의 게임"에 소수점 넷째 자리까지
# 같은 확률을 준다. 그래서 화면 1 만 딥러닝을 쓴다.
#
#   변동 폭 (같은 조건, 글만 바꿔 8문장)
#     LightGBM 0.018 · 딥러닝 0.486   <- 27배
#
#   models/dl_model.joblib      MLP(숫자 24개 + 글 임베딩 384개) · 영어 67,112행
#   models/dl_threshold.json    0.3346  <- LightGBM 의 0.2899 와 다르다. 한 세트다
#   성능                        랜덤 0.777 · 게임분할 0.7295
#
# ★ 성능은 최종 모델보다 낮다 (같은 영어 데이터·게임분할에서 0.730 vs 0.750).
#   화면에 "더 정확하다" 고 쓰면 안 된다. 쓰는 이유는 성능이 아니라 글을 읽어서다.
# ★ 영어 전용이다. 한국어 리뷰는 제대로 못 읽는다.
DL_META = MODELS / "dl_meta.json"


def load_dl():
    """딥러닝 모델을 준비하고 (임계값, 메타)를 돌려준다.

    첫 호출에 11초쯤 걸린다 — 임베딩 모델(all-MiniLM-L6-v2, 90MB)을
    올리는 시간이다. 이후는 한 건에 19ms. Streamlit 쪽에서 캐시를 씌워
    화면이 뜰 때 한 번만 치르게 한다.
    """
    from src.predict import _load
    c = _load()
    return float(c["thr"]), json.load(open(DL_META, encoding=ENC_READ))


def predict_dl(review, hours, owned, n_reviews, voted_up, game_row):
    """(딥러닝 확률, 글 없이 냈을 때의 확률) 을 돌려준다.

    글의 기여를 대리(surrogate) 모델의 SHAP 으로 추정하지 않는다.
    같은 입력에서 리뷰만 빈 문자열로 바꿔 한 번 더 돌리고 그 차이를 쓴다.
    실제 모델이 실제로 움직인 값이라 설명이 어긋날 일이 없다.
    (대리 모델은 "리뷰 글" 기여를 +0.06 으로 봤는데 진짜 모델은
     극찬↔혹평 사이에서 35%p 를 움직였다. 대리 모델을 믿으면 안 된다.)

    두 번 부르지만 합쳐서 39ms 다.
    """
    from src.predict import predict_one

    kw = dict(playtime_at_review_min=int(float(hours) * 60),
              num_games_owned=int(owned), num_reviews=int(n_reviews),
              voted_up=bool(voted_up), created_ts=_NOW,
              game=_game_dict(game_row))
    return (predict_one(review=review, **kw)["이탈확률"],
            predict_one(review="", **kw)["이탈확률"])


def 한글비율(s):
    """한국어 리뷰가 들어오면 알려주려고. 딥러닝은 영어로만 학습했다."""
    t = (s or "").strip()
    if not t:
        return 0.0
    return sum("가" <= c <= "힣" for c in t) / len(t)


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


def 판정말(p, thr):
    """계기판 옆에 넣을 짧은 판정말.

    gauge() 의 문장은 화면 0 처럼 넓은 칸에서는 괜찮지만,
    화면 1 의 좁은 칸에서는 "떠나는 쪽입..." 처럼 잘린다.
    잘린 말은 읽는 사람을 불안하게 하므로 여기서는 짧고 끝맺은 말을 쓴다.
    """
    if p >= thr + .15:
        return "🔴", "이탈 유력"
    if p >= thr:
        return "🟡", "이탈 쪽"
    if p >= thr - .15:
        return "🟢", "잔존 쪽"
    return "🟢", "잔존 유력"


def gauge(p, thr):
    """확률을 사람이 읽는 말로. 임계값을 기준으로 나눈다."""
    if p >= thr + .15:
        return "🔴", "떠날 가능성이 높습니다."
    if p >= thr:
        return "🟡", "떠나는 쪽입니다."
    if p >= thr - .15:
        return "🟢", "남는 쪽입니다."
    
    return "🟢", "계속할 가능성이 높습니다"
