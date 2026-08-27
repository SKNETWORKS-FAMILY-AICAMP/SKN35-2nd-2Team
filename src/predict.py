# -*- coding: utf-8 -*-
"""
화면(app/) 이 쓰는 예측 진입점.

★ 화면에서 모델을 직접 불러오지 않는다.
  학습 때와 조금이라도 다르게 변환하면 조용히 틀린 예측이 나온다.
  진입점을 여기 하나로 막아둔다.

쓰는 법
    from src.predict import predict_one

    r = predict_one(
        review="Refunded after 30 minutes. Total waste.",
        playtime_at_review_min=25, num_games_owned=310, num_reviews=12,
        voted_up=False, created_ts=1743247376,
        game={"genre_group": "싱글 서사", "era": "S4 2023-25", "grade": "평가나쁨",
              "release_year": 2024, "app_release_ts": 1690000000},
    )
    r["이탈확률"]  -> 0.83
    r["판정"]     -> "이탈"
"""
import numpy as np
import pandas as pd

from src.config import EMB_MODEL, MODELS, load_json

_cache = {}


def _load():
    """모델·임계값·열순서·임베딩 모델을 한 번만 읽어 재사용한다."""
    if not _cache:
        import joblib
        _cache["model"] = joblib.load(MODELS / "dl_model.joblib")
        _cache["thr"] = load_json(MODELS / "dl_threshold.json")["threshold"]
        _cache["order"] = load_json(MODELS / "dl_feature_order.json")["열순서"]
        _cache["meta"] = load_json(MODELS / "dl_meta.json")
    return _cache


def _encoder():
    if "enc" not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache["enc"] = SentenceTransformer(EMB_MODEL)
    return _cache["enc"]


def predict_one(review, playtime_at_review_min, num_games_owned, num_reviews,
                voted_up, created_ts, game, **kw):
    """리뷰 한 건 -> 이탈 확률. 학습 때와 같은 변환을 거친다."""
    from src.preprocess import featurize

    c = _load()
    row = featurize(
        {"review": review, "language": "english",
         "playtime_at_review_min": playtime_at_review_min,
         "num_games_owned": num_games_owned, "num_reviews": num_reviews,
         "voted_up": voted_up, "created_ts": created_ts, **kw},
        game)

    text = (review or "").strip()
    emb = (_encoder().encode([text], normalize_embeddings=True)[0]
           if text else np.zeros(384, dtype=np.float32))
    # 384개를 한 번에 붙인다 (한 열씩 넣으면 pandas 가 매번 표를 다시 만든다)
    e = pd.DataFrame([emb], columns=[f"emb_{i}" for i in range(len(emb))],
                     index=row.index)
    row = pd.concat([row, e], axis=1)

    X = row.reindex(columns=c["order"])      # ★ 학습 때의 열 순서로 강제 정렬
    p = float(c["model"].predict_proba(X)[0, 1])
    return {"이탈확률": round(p, 4),
            "판정": "이탈" if p >= c["thr"] else "잔존",
            "임계값": c["thr"], "모델": c["meta"]["모델"]}


if __name__ == "__main__":
    cases = [
        ("Refunded after 30 minutes. Total waste of money.", 25, "평가나쁨"),
        ("300 hours in and still playing with friends every night. Best purchase ever.",
         18000, "압도적긍정"),
        ("fun", 90, "매우긍정"),
    ]
    for text, mins, grade in cases:
        r = predict_one(
            review=text, playtime_at_review_min=mins, num_games_owned=310,
            num_reviews=12, voted_up=(mins > 1000), created_ts=1743247376,
            game={"genre_group": "싱글 서사", "era": "S4 2023-25", "grade": grade,
                  "release_year": 2024, "app_release_ts": 1690000000, "game": "UNKNOWN"})
        print(f'  {r["판정"]}  {r["이탈확률"]:.3f}   "{text[:52]}"  ({mins}분)')
