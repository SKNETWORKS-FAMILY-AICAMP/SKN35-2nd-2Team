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
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROC, EMB_DIM, SEED, load_json
from src.embed import load as load_embeddings
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


# ── DL(2): MLP (숫자 + 글) ──────────────────────────────────────
EMB_PREFIX = "emb_"


def with_text(X, emb):
    """
    숫자표 옆에 임베딩 384개를 이어 붙인다.  24 + 384 = 408열.

    한 DataFrame 으로 합치는 이유
        분할 함수가 행을 잘라 넘길 때 숫자와 글이 같이 잘려야 한다.
        따로 들고 다니면 순서가 어긋나도 에러가 안 난다.
    """
    e = pd.DataFrame(emb, columns=[f"{EMB_PREFIX}{i}" for i in range(emb.shape[1])],
                     index=X.index)
    return pd.concat([X, e], axis=1)


def mlp_text(X_tr, y_tr, X_te):
    """
    숫자 블록과 글 블록을 각각 표준화한 뒤 신경망에 넣는다.

    글 블록도 스케일링하는 이유
        임베딩은 길이가 1로 맞춰져 있어 값이 대략 ±0.3 범위다.
        표준화된 숫자 변수(±1 이상)보다 작아서, 그대로 두면 신경망이
        글을 덜 보게 된다. 두 블록의 크기를 맞춰준다.
    """
    emb_cols = [c for c in X_tr.columns if c.startswith(EMB_PREFIX)]
    rest = [c for c in X_tr.columns if c not in emb_cols]
    num = [c for c in rest if pd.api.types.is_numeric_dtype(X_tr[c])]
    cat = [c for c in rest if c not in num]
    pipe = make_pipeline(
        ColumnTransformer([
            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
            ("e", StandardScaler(), emb_cols),
        ]),
        MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                      early_stopping=True, n_iter_no_change=10,
                      random_state=SEED),
    )
    return pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


# ── 진단용 ─────────────────────────────────────────────────────
def mlp_text_only(X_tr, y_tr, X_te):
    """글만. 숫자를 빼고 임베딩 384개만으로 맞혀본다."""
    emb = [c for c in X_tr.columns if c.startswith(EMB_PREFIX)]
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                      early_stopping=True, n_iter_no_change=10, random_state=SEED))
    return pipe.fit(X_tr[emb], y_tr).predict_proba(X_te[emb])[:, 1]


def make_mlp_text_pca(n_comp=64):
    """
    글을 384 -> n_comp 차원으로 줄여서 붙인다.

    왜 줄이는가
        숫자는 24개인데 글이 384개다. 입력의 94%가 글이라
        신경망이 숫자 변수를 거의 못 본다. 차원을 맞춰줘야
        '글을 더했을 때의 효과'를 제대로 잰다.
    """
    from sklearn.decomposition import PCA

    def f(X_tr, y_tr, X_te):
        emb = [c for c in X_tr.columns if c.startswith(EMB_PREFIX)]
        rest = [c for c in X_tr.columns if c not in emb]
        num = [c for c in rest if pd.api.types.is_numeric_dtype(X_tr[c])]
        cat = [c for c in rest if c not in num]
        pipe = make_pipeline(
            ColumnTransformer([
                ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
                ("e", make_pipeline(StandardScaler(), PCA(n_components=n_comp,
                                                          random_state=SEED)), emb),
            ]),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                          early_stopping=True, n_iter_no_change=10, random_state=SEED))
        return pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    return f


def make_boosting_text(n_comp=64):
    """부스팅 + 글. 가장 센 모델에도 글이 도움이 되는지 본다."""
    from sklearn.decomposition import PCA

    def f(X_tr, y_tr, X_te):
        emb = [c for c in X_tr.columns if c.startswith(EMB_PREFIX)]
        rest = [c for c in X_tr.columns if c not in emb]
        pca = PCA(n_components=n_comp, random_state=SEED).fit(X_tr[emb])
        def prep(X):
            z = pd.DataFrame(pca.transform(X[emb]),
                             columns=[f"pc{i}" for i in range(n_comp)], index=X.index)
            return pd.concat([X[rest], z], axis=1)
        m = HistGradientBoostingClassifier(max_iter=200, random_state=SEED,
                                           categorical_features="from_dtype")
        return m.fit(prep(X_tr), y_tr).predict_proba(prep(X_te))[:, 1]
    return f


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

    emb = load_embeddings(d)
    Xt = with_text(X, emb)
    print(f"  글 붙임: {X.shape[1]} + {emb.shape[1]} = {Xt.shape[1]}열")
    evaluate_model("MLP(숫자+글)", Xt, y, g, sp, mlp_text,
                   변수묶음="B셋+글", 메모="DL(2) 핵심 실험")
    print("─" * 78)
    summary()


if __name__ == "__main__":
    main()
