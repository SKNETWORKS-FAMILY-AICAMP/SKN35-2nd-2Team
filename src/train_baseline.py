# -*- coding: utf-8 -*-
"""
화면용 모델 학습 — 분류(이탈 여부) + 회귀(몇 시간 더 할까).

이건 '임시' 모델이다. B가 LightGBM으로 더 좋은 걸 만들면 models/ 안의
파일만 교체하면 되고, app.py 는 건드리지 않는다.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (RAW_CSV, DATA_PROC, MODELS, MODEL_META,
                        GAMES_CSV, CARDS_JSON, UNSEEN_CSV, ENC_READ, ENC_WRITE)
from src.preprocess import FEATURES_B, TARGET, assert_no_leak  # noqa: E402

MODELS.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)
UNSEEN_GAMES = DATA_PROC / "unseen_games.csv"

print("데이터 로드…")
d = pd.read_csv(DATA_PROC / "dataset.csv", encoding=ENC_READ)
raw = pd.read_csv(RAW_CSV, encoding=ENC_READ, low_memory=True,
                  usecols=["recommendationid", "playtime_forever_min",
                           "playtime_at_review_min"])
d = d.merge(raw, on="recommendationid", how="left")
d["hours_after"] = (d.playtime_forever_min - d.playtime_at_review_min) / 60
d = d.drop(columns=["playtime_forever_min", "playtime_at_review_min"])
print(f"  {len(d):,}행")

assert_no_leak(FEATURES_B)
X, y = d[FEATURES_B], d[TARGET]
# pandas 3.0 부터 문자열이 object 가 아니라 str dtype 이다.
# dtype == object 로 판별하면 범주형을 놓친다. 숫자인지로 판별할 것.
NUM = [c for c in FEATURES_B if pd.api.types.is_numeric_dtype(d[c])]
CAT = [c for c in FEATURES_B if c not in NUM]


def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), CAT),
    ])


# ── 1. 분류 모델 ────────────────────────────────────────────────
print("\n[1] 분류 모델 (이탈 여부)")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
clf = Pipeline([("pre", make_pre()),
                ("m", HistGradientBoostingClassifier(max_iter=400, learning_rate=.08,
                                                     random_state=42))])
t0 = time.time()
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
auc = roc_auc_score(yte, p)
pr = average_precision_score(yte, p)
f1 = f1_score(yte, p > .5)
print(f"  AUC {auc:.3f} · PR-AUC {pr:.3f} · F1 {f1:.3f}  ({time.time()-t0:.0f}초)")

# ── 2. 회귀 모델 ────────────────────────────────────────────────
print("\n[2] 회귀 모델 (리뷰 후 몇 시간 더 할까)")
yh = np.log1p(d.hours_after.clip(lower=0))
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, yh, test_size=.2, random_state=42)
reg = Pipeline([("pre", make_pre()),
                ("m", HistGradientBoostingRegressor(max_iter=400, learning_rate=.08,
                                                    random_state=42))])
reg.fit(Xtr2, ytr2)
pred_h = np.expm1(reg.predict(Xte2))
true_h = np.expm1(yte2)
print(f"  중앙 절대오차 {np.median(np.abs(pred_h - true_h)):.1f}시간 "
      f"(실제 중앙 {np.median(true_h):.1f}시간)")

# ── 3. 처음 보는 게임 성능 (화면 4용) ───────────────────────────
print("\n[3] 게임 단위 분할 — 처음 보는 게임 성능")
rows, per_game = [], []
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=d.game), 1):
    m = Pipeline([("pre", make_pre()),
                  ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=.08,
                                                       random_state=42))])
    m.fit(X.iloc[tr], y.iloc[tr])
    pp = m.predict_proba(X.iloc[te])[:, 1]
    a = roc_auc_score(y.iloc[te], pp)
    games = sorted(d.game.iloc[te].unique())
    rows.append({"fold": k, "auc": round(a, 4), "n_games": len(games),
                 "n_rows": len(te), "games": " · ".join(games)})
    print(f"  조각 {k}: AUC {a:.3f}  게임 {len(games)}개")

    # 게임 하나씩 — 화면 ④에서 "이 게임을 가려놓고 시험했을 때" 를 보여주기 위해
    sub = pd.DataFrame({"game": d.game.iloc[te].values,
                        "y": y.iloc[te].values, "p": pp})
    for g, cell in sub.groupby("game"):
        # 한 게임 안에 이탈/잔존이 둘 다 있어야 AUC 를 낼 수 있다
        auc_g = (round(roc_auc_score(cell.y, cell.p), 4)
                 if cell.y.nunique() == 2 else np.nan)
        per_game.append({
            "game": g, "fold": k, "n_rows": len(cell),
            "실제_이탈률": round(float(cell.y.mean()), 4),
            "예측_평균": round(float(cell.p.mean()), 4),
            "auc": auc_g,
            "맞힌_비율": round(float(((cell.p >= .5).astype(int) == cell.y).mean()), 4),
        })

unseen = pd.DataFrame(rows)
unseen.to_csv(UNSEEN_CSV, index=False, encoding=ENC_WRITE)
pg = pd.DataFrame(per_game).sort_values("auc", ascending=False)
pg.to_csv(UNSEEN_GAMES, index=False, encoding=ENC_WRITE)
print(f"  게임별 결과 {len(pg)}개 저장 — {UNSEEN_GAMES.name}")
print(f"  평균 {unseen.auc.mean():.3f} ± {unseen.auc.std():.3f}")

# ── 4. 사람 vs 모델 카드 (화면 2용) ─────────────────────────────
print("\n[4] 사람 vs 모델 카드 12장")
d["_p"] = clf.predict_proba(X)[:, 1]
pool = d[(d.review.fillna("").str.len().between(20, 400)) &
         (d.language == "english")].copy()
# 난이도를 조작하지 않는다 — 정답 비율만 6:6으로 맞춘 무작위 표본.
# 일부러 '모델이 틀리는 카드'를 넣으면 모델을 나쁘게 보이게 조작하는 셈이다.
cards = pd.concat([
    pool[pool[TARGET] == 1].sample(6, random_state=7),
    pool[pool[TARGET] == 0].sample(6, random_state=7),
]).sample(frac=1, random_state=7)
cards_out = cards[["game", "genre_group", "grade", "review", "hours_at_review",
                   "is_private", "voted_up", "review_len", TARGET, "_p"]]
cards_out.to_json(CARDS_JSON, orient="records", force_ascii=False, indent=1)
print(f"  저장 {len(cards_out)}장")

# ── 5. 게임 목록 (화면 3용) ─────────────────────────────────────
games = d.groupby("game").agg(
    genre_group=("genre_group", "first"), era=("era", "first"),
    grade=("grade", "first"), release_year=("release_year", "first"),
    game_age_days=("game_age_days", "median"),
    churn_rate=(TARGET, "mean"), n=(TARGET, "size"),
).reset_index()
games.to_csv(GAMES_CSV, index=False, encoding=ENC_WRITE)
print(f"\n[5] 게임 {len(games)}개 목록 저장")

# ── 저장 ────────────────────────────────────────────────────────
joblib.dump(clf, MODELS / "clf.joblib")
joblib.dump(reg, MODELS / "reg.joblib")
json.dump({
    "학습일시": time.strftime("%Y-%m-%d %H:%M"),
    "모델": "HistGradientBoosting (임시 — B의 모델로 교체 예정)",
    "변수묶음": "B셋 (게임 이름 제외)",
    "행수": len(d),
    "이탈률": round(float(y.mean()), 4),
    "분류_AUC": round(auc, 4), "분류_PR_AUC": round(pr, 4), "분류_F1": round(f1, 4),
    "처음보는게임_AUC": round(float(unseen.auc.mean()), 4),
    "처음보는게임_편차": round(float(unseen.auc.std()), 4),
}, open(MODEL_META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n완료 — models/clf.joblib · reg.joblib · meta.json")
