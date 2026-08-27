# -*- coding: utf-8 -*-
"""
프로토타입 화면 전용 예측 — 임시 모델(HistGradientBoosting)을 쓴다.

★ 이건 최종이 아니다.
  최종 예측은 src/predict.py 의 predict_one() 이다(MLP + 임베딩).
  화면을 먼저 만들어 보여주려고 가벼운 임시 모델을 붙여둔 것이고,
  최종 모델로 바꿀 때는 app/_shared.py 의 import 한 줄만 고치면 된다.

여기서 하는 일은 하나뿐이다 —
**사람이 화면에 입력한 것을 학습할 때와 똑같은 숫자로 바꾸는 것.**

이 변환을 화면 쪽에서 따로 짜면 안 된다. 미묘하게 달라지고,
에러 없이 조용히 틀린 예측이 나온다. (실제로 겪었다: 언어별 기본값이
화면에만 따로 박혀 있어서 review_len_z 가 155 까지 튄 적이 있다)
"""
import json
import re

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROC, MODELS, LANG_STATS, ENC_READ
from src.preprocess import FEATURES_B

# 프로토타입 전용 파일들. 팀 규칙은 "주소는 config.py 에서만" 이지만,
# 이 파일들은 임시 모델과 함께 사라질 것이라 공유 config 를 늘리지 않는다.
# 폴더 위치는 config 의 DATA_PROC / MODELS 에서 받아오므로 하드코딩은 없다.
CLF_MODEL = MODELS / "clf.joblib"            # 이탈 확률 (임시)
REG_MODEL = MODELS / "reg.joblib"            # 예상 추가 플레이 시간 (임시)
MODEL_META = MODELS / "meta.json"
GAMES_CSV = DATA_PROC / "games.csv"          # 게임 60개 속성
CARDS_JSON = DATA_PROC / "cards.json"        # 사람 vs 모델 문제 카드
UNSEEN_CSV = DATA_PROC / "unseen.csv"        # 게임 단위 분할 결과
CATALOG_CSV = DATA_PROC / "catalog.csv"      # 추천용 게임 402개


def load_all():
    """화면이 필요로 하는 것 전부. Streamlit 쪽에서 캐시를 씌워 쓴다."""
    return (joblib.load(CLF_MODEL),
            joblib.load(REG_MODEL),
            pd.read_csv(GAMES_CSV, encoding=ENC_READ),
            json.load(open(CARDS_JSON, encoding=ENC_READ)),
            pd.read_csv(UNSEEN_CSV, encoding=ENC_READ),
            json.load(open(MODEL_META, encoding=ENC_READ)),
            json.load(open(LANG_STATS, encoding=ENC_READ)))


def load_catalog():
    """추천용 게임 카탈로그(우리 60개 + 처음 보는 게임). 없으면 None."""
    return pd.read_csv(CATALOG_CSV, encoding=ENC_READ) if CATALOG_CSV.exists() else None


STEAM_IMG = "https://cdn.akamai.steamstatic.com/steam/apps/{}/header.jpg"


def text_feats(t):
    """언어를 몰라도 뽑을 수 있는 신호. preprocess 의 _text_features 와 같은 정의."""
    t = t or ""
    n = len(t)
    letters = [c for c in t if c.isalpha()]
    return {
        "review_len": n,
        "review_words": len(t.split()),
        "has_text": int(n > 0),
        "excl_ratio": t.count("!") / n if n else 0.0,
        "caps_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "has_repeat": int(bool(re.search(r"(.)\1{3,}", t))),
    }


def build_row(review, hours, owned, n_reviews, voted_up, game_row, lang_stats,
              language="english", steam_purchase=1, free=0, ea=0, deck=0):
    """화면 입력 한 줄 -> 모델이 먹는 24개 변수."""
    tf = text_feats(review)

    # 언어마다 다른 것은 "기본 길이"(중앙값) 하나뿐이고, 퍼지는 정도는 전 언어 공통.
    # 목록에 없는 언어(히브리어 등)는 전체 중앙값을 쓴다.
    med = lang_stats["median"].get(language, lang_stats["median_기본"])
    std = lang_stats["std_공통"]
    clip = lang_stats["_z_clip"]

    row = {
        "hours_at_review": hours,
        "log_hours_at_review": np.log1p(hours),
        "log_num_games": np.log1p(owned) if owned > 0 else -1,
        "log_num_reviews": np.log1p(n_reviews),
        "game_age_days": float(game_row["game_age_days"]),
        "review_len_z": float(np.clip((tf["review_len"] - med) / std, -clip, clip)),
        "is_private": int(owned == 0),
        "is_spike": 0,                      # 화면에서는 알 수 없음 - 평소로 간주
        "language": language,
        "genre_group": game_row["genre_group"],
        "era": game_row["era"],
        "grade": game_row["grade"],
        "release_year": int(game_row["release_year"]),
        "voted_up": int(voted_up),
        "steam_purchase": int(steam_purchase),
        "received_for_free": int(free),
        "early_access": int(ea),
        "steam_deck": int(deck),
        **tf,
    }
    # FEATURES_B 순서로 맞춘다 - 컬럼 순서가 어긋나면 조용히 틀린 예측이 나온다
    return pd.DataFrame([row])[FEATURES_B]


def gauge(p):
    """확률을 사람이 읽는 말로."""
    if p >= .65:
        return "🔴", "떠날 가능성이 높습니다"
    if p >= .45:
        return "🟡", "애매합니다"
    return "🟢", "계속할 가능성이 높습니다"
