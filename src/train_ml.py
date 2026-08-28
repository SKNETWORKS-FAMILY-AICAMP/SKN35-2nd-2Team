# -*- coding: utf-8 -*-
"""
머신러닝 4종 — 로지스틱 · 랜덤포레스트 · XGBoost · LightGBM

채점은 하지 않는다. 채점은 src/evaluate.py 가 전담한다.
여기서는 "모델을 어떻게 만드는가"만 정의해서 넘긴다.

두 벌로 돌린다
    전체  139,658행 · 30개 언어  → README 기준선(AUC 0.820) 재현 · 실제 배포용
    영어   67,112행             → 딥러닝(C)과 같은 행에서 공정 비교

    행 수가 다르면 AUC 를 나란히 놓을 수 없다. 그래서 두 번 잰다.
    results.csv 에서 구분되도록 모델명에 (전체) / (영어) 를 붙인다.

스케일링·원핫이 필요한 모델과 아닌 모델
    로지스틱 · 랜덤포레스트  →  evaluate.features() 가 만든 category 를 못 먹는다.
                               ValueError: could not convert string to float: 'english'
                               → prep() 으로 원핫해서 넣는다
    XGBoost · LightGBM      →  category 를 그대로 먹는다. prep() 불필요

쓰는 법
    uv run python -m src.train_ml              # 전체 + 영어, 4모델 x 4칸
    uv run python -m src.train_ml --영어만
    uv run python -m src.train_ml --빠르게     # 트리 개수를 줄여 리허설
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROC, SEED, load_dataset, load_json
from src.evaluate import evaluate_model, features, load_english, make_splits, summary

# 보유 게임 수가 -1 인 것은 "0개"가 아니라 프로필 비공개다.
# 트리는 -1 에서 갈라주면 그만이지만, 선형 모델은 -1 을 "게임이 매우 적은 사람"으로
# 읽는다 (실제 값 범위는 0~10.5). is_private 이 이미 비공개 여부를 들고 있으므로
# 선형 쪽에서는 -1 을 중앙값으로 바꾼다.
SENTINEL_COLS = ["log_num_games"]
SENTINEL = -1

# 원핫에서 이보다 드문 범주는 한 칸으로 합친다. 게임 분할에서 학습에 없던
# 게임 이름이 나와도 handle_unknown='ignore' 가 0 벡터로 처리한다.
MIN_CAT_FREQ = 20


# ── 데이터 ──────────────────────────────────────────────────────
def load_all():
    """
    전체 30개 언어. ML 담당 기본값.

    ★ evaluate.load_english() 와 같은 이유로 index 를 0..n-1 로 다시 매긴다.
      분할은 위치번호를 돌려주므로, index 가 어긋나면 에러 없이 조용히 틀린다.
    """
    d = load_dataset().reset_index(drop=True)
    return d, d["churn"].to_numpy(), d["game"].to_numpy()


def 변수묶음들():
    """meta 에서 A셋 / B셋 컬럼 목록을 읽어온다. preprocess.py 를 import 하지 않는다."""
    meta = load_json(DATA_PROC / "dataset_meta.json")
    return [("A셋", meta["변수_A셋"]["열"]), ("B셋", meta["변수_B셋"]["열"])]


# ── 전처리 (로지스틱 · 랜덤포레스트 전용) ────────────────────────
def prep(X, *, scale):
    """
    category 컬럼을 원핫으로 편다.

    scale=True  로지스틱용 — 숫자도 표준화하고 -1 을 중앙값으로 바꾼다
    scale=False 랜덤포레스트용 — 숫자는 건드리지 않는다 (트리는 크기에 무관)
    """
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    nums = [c for c in X.columns if c not in cats]
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_CAT_FREQ,
                           sparse_output=False)

    if not scale:
        return ColumnTransformer([("num", "passthrough", nums), ("cat", onehot, cats)])

    sentinel = [c for c in SENTINEL_COLS if c in nums]
    plain = [c for c in nums if c not in sentinel]
    return ColumnTransformer([
        ("num", StandardScaler(), plain),
        ("sent", Pipeline([
            ("비공개", SimpleImputer(missing_values=SENTINEL, strategy="median")),
            ("표준화", StandardScaler()),
        ]), sentinel),
        ("cat", onehot, cats),
    ])


# ── 모델 4종 ────────────────────────────────────────────────────
# 전부 evaluate_model 이 요구하는 fit_predict(X_tr, y_tr, X_te) -> 이탈 확률 형태.
def 로지스틱():
    def fit_predict(X_tr, y_tr, X_te):
        m = Pipeline([("전처리", prep(X_tr, scale=True)),
                      ("모델", LogisticRegression(max_iter=1000, random_state=SEED))])
        return m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return fit_predict


def 랜덤포레스트(n=300):
    def fit_predict(X_tr, y_tr, X_te):
        m = Pipeline([("전처리", prep(X_tr, scale=False)),
                      ("모델", RandomForestClassifier(
                          n_estimators=n, min_samples_leaf=5,
                          random_state=SEED, n_jobs=-1))])
        return m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return fit_predict


def xgboost(n=400):
    import xgboost as xgb

    def fit_predict(X_tr, y_tr, X_te):
        m = xgb.XGBClassifier(
            n_estimators=n, learning_rate=0.1, max_depth=6,
            tree_method="hist", enable_categorical=True,   # category 를 그대로 먹는다
            random_state=SEED, n_jobs=-1, eval_metric="logloss")
        return m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return fit_predict


def lightgbm(n=400):
    import lightgbm as lgb

    def fit_predict(X_tr, y_tr, X_te):
        m = lgb.LGBMClassifier(
            n_estimators=n, learning_rate=0.1, num_leaves=31,
            random_state=SEED, n_jobs=-1, verbose=-1)
        return m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return fit_predict


def 모델들(빠르게=False):
    n = 100 if 빠르게 else 300
    return {
        "로지스틱": 로지스틱(),
        "랜덤포레스트": 랜덤포레스트(n),
        "XGBoost": xgboost(n + 100),
        "LightGBM": lightgbm(n + 100),
    }


# ── 한 벌 돌리기 ────────────────────────────────────────────────
def 한벌(꼬리표, d, y, g, 빠르게=False, 기록=True):
    """
    A셋/B셋 x 랜덤/게임 = 4칸을 모델 4종으로 채운다.

    꼬리표 : "전체" 또는 "영어". results.csv 에서 두 벌을 구분한다.
             summary(최신만=True) 가 (모델명·변수묶음·분할방식) 으로 중복을 지우므로
             모델명에 꼬리표를 넣지 않으면 한쪽이 사라진다.
    """
    sp = make_splits(d, g)
    print(f"\n{'='*78}\n{꼬리표}  {len(d):,}행 · 게임 {len(set(g))}개 · 이탈률 {y.mean():.4f}")
    print(f"  랜덤 분할 학습 {len(sp['random'][0][0]):,} / 시험 {len(sp['random'][0][1]):,}"
          f"  |  게임 분할 {len(sp['group'])}조각\n{'='*78}")

    for 묶음, cols in 변수묶음들():
        X = features(d, cols)          # 누수 컬럼은 여기서 막힌다
        print(f"\n[{묶음}] {len(cols)}개 변수 "
              f"(범주형 {sum(1 for c in X.columns if str(X[c].dtype)=='category')}개)")
        for 이름, fp in 모델들(빠르게).items():
            try:
                evaluate_model(f"{이름}({꼬리표})", X, y, g, sp, fp,
                               변수묶음=묶음, 메모=f"ML 기본값 · {꼬리표}", 기록=기록)
            except ImportError as e:
                print(f"  {이름:<22s} 건너뜀 — 패키지 없음: {e}")
            except Exception as e:
                print(f"  {이름:<22s} 실패 — {type(e).__name__}: {str(e)[:80]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="머신러닝 4종을 4칸 채점표에 채운다")
    ap.add_argument("--전체만", action="store_true", help="30개 언어 139,658행만")
    ap.add_argument("--영어만", action="store_true", help="영어 67,112행만 (딥러닝 비교용)")
    ap.add_argument("--빠르게", action="store_true", help="트리 개수를 줄여 리허설")
    ap.add_argument("--기록안함", action="store_true", help="results.csv 에 쓰지 않는다")
    a = ap.parse_args(argv)

    기록 = not a.기록안함
    둘다 = not (a.전체만 or a.영어만)

    if a.전체만 or 둘다:
        한벌("전체", *load_all(), 빠르게=a.빠르게, 기록=기록)
    if a.영어만 or 둘다:
        한벌("영어", *load_english(), 빠르게=a.빠르게, 기록=기록)

    if 기록:
        print(f"\n{'='*78}\n누적 결과\n{'='*78}")
        summary()
    print("\n정확도(accuracy)는 보고하지 않는다 — 전부 '잔존'만 찍어도 58.9% 가 나온다.")


if __name__ == "__main__":
    # ★ 윈도우에서 n_jobs=-1 을 쓸 때 이 가드가 없으면 프로세스가 무한 복제된다.
    main(sys.argv[1:])
