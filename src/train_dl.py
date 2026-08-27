# -*- coding: utf-8 -*-
"""
딥러닝(C) 파트 — 영어 서브셋 67,112행.

답해야 할 질문 하나
    "리뷰에 쓴 말이 예측에 도움이 되는가?"

그래서 같은 행·같은 분할에서 세 가지를 잰다.
    참조   부스팅(숫자만)   — 표 데이터에서 흔히 가장 센 모델
    DL(1)  MLP(숫자만)      — 통제군. 신경망이라서 달라진 부분을 분리
    DL(2)  MLP(숫자+글)     — 여기서 오르는 만큼이 '글의 효과'

B(머신러닝)가 낸 전체 13.9만행 수치(AUC 0.820)는 행 수가 달라
비교 대상이 아니다. 반드시 여기서 다시 잰 숫자와 비교한다.

실행 (레포 루트에서)
    uv run python -m src.train_dl
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROC, SEED, load_json
from src.evaluate import evaluate_model, features, load_english, make_splits, summary


def feature_cols(변수묶음="B셋"):
    """dataset_meta.json 에서 가져온다. 손으로 타이핑하면 game 이 섞인다."""
    m = load_json(DATA_PROC / "dataset_meta.json")
    return m[f"변수_{변수묶음}"]["열"]


def _split_types(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return num, [c for c in X.columns if c not in num]


# ── 참조: 부스팅 (숫자만) ───────────────────────────────────────
def boosting(X_tr, y_tr, X_te):
    """트리 모델이라 스케일링·원핫이 필요 없다. 범주형을 그대로 먹는다."""
    m = HistGradientBoostingClassifier(
        max_iter=200, random_state=SEED, categorical_features="from_dtype")
    return m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


# ── DL(1): MLP (숫자만) ─────────────────────────────────────────
def mlp_numeric(X_tr, y_tr, X_te):
    """
    신경망은 값의 크기에 민감하다 → 반드시 스케일링.
    ★ 스케일러는 학습 데이터에서만 fit 한다. 전체로 fit 하면 누수.
      Pipeline 안에 넣으면 자동으로 지켜진다.
    """
    num, cat = _split_types(X_tr)
    pipe = make_pipeline(
        ColumnTransformer([
            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
        ]),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                      early_stopping=True, n_iter_no_change=10,
                      random_state=SEED),
    )
    return pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def main():
    d, y, g = load_english()
    sp = make_splits(d, g)
    cols = feature_cols("B셋")
    X = features(d, cols)

    print(f"영어 서브셋 {len(d):,}행 | 이탈률 {y.mean():.4f} | 변수 {len(cols)}개")
    print("─" * 78)
    evaluate_model("부스팅(숫자만)", X, y, g, sp, boosting,
                   변수묶음="B셋", 메모="참조 — 표 데이터 강자")
    evaluate_model("MLP(숫자만)", X, y, g, sp, mlp_numeric,
                   변수묶음="B셋", 메모="DL(1) 통제군")
    print("─" * 78)
    summary()


if __name__ == "__main__":
    main()
