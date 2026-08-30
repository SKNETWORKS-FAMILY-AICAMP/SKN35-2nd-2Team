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


# ── 최종 모델 저장 ──────────────────────────────────────────────
def save_final():
    """
    화면(app/) 에 넘길 최종 딥러닝 모델을 저장한다.

    무엇을 고르는가
        게임 단위 분할 성능이 가장 좋은 딥러닝 모델.
        실제 서비스에서는 학습에 없던 게임의 리뷰가 들어오므로
        랜덤 분할 점수는 실전과 다르다.  -> MLP(숫자+글 PCA64)

    무엇을 같이 저장하는가
        모델만 저장하면 화면에서 못 쓴다. 네 개가 한 세트다.
          dl_model.joblib        전처리(스케일·원핫·PCA)까지 통째로 담긴 파이프라인
          dl_threshold.json      이탈로 판정할 확률 기준
          dl_feature_order.json  ★ 열 순서. 모델은 이름이 아니라 순서로 받는다
          dl_meta.json           어떤 데이터·어떤 성능이었는지
    """
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.decomposition import PCA
    from sklearn.metrics import (brier_score_loss, f1_score,
                                 precision_recall_curve, roc_auc_score)

    from src.config import EMB_MODEL, MODELS, save_json
    from src.embed import load as load_embeddings

    d, y, g = load_english()
    sp = make_splits(d, g)
    cols = feature_cols("B셋")
    Xt = with_text(features(d, cols), load_embeddings(d))
    emb_cols = [c for c in Xt.columns if c.startswith(EMB_PREFIX)]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(Xt[c])]
    cat = [c for c in cols if c not in num]

    def raw():
        return make_pipeline(
            ColumnTransformer([
                ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
                ("e", make_pipeline(StandardScaler(), PCA(n_components=64, random_state=SEED)),
                 emb_cols),
            ]),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                          early_stopping=True, n_iter_no_change=10, random_state=SEED))

    def build():
        """
        ★ 확률 보정(calibration)을 씌운다.

        보정 전에는 확률을 가운데에서 바깥으로 밀어내는 버릇이 있었다.
            0.7~0.8 구간   예측 0.750 -> 실제 0.662  (+0.088 과신)
            0.8~0.9 구간   예측 0.849 -> 실제 0.787  (+0.062 과신)
            0.1~0.2 구간   예측 0.149 -> 실제 0.187  (-0.038 과소)
        평균은 맞는데(0.411 vs 0.402) 기울기가 틀린 문제다.

        화면 1 이 확률을 큰 숫자로 보여주므로 그대로 두면 안 된다.
        sigmoid 보정으로 ECE 0.0352 -> 0.0175 (절반), AUC 는 오히려 소폭 상승.
        isotonic 도 비슷하지만 표본이 적은 구간에서 계단처럼 튀어 sigmoid 를 쓴다.
        """
        return CalibratedClassifierCV(raw(), method="sigmoid", cv=3)

    # 1) 임계값은 '보지 않은 데이터'에서 고른다 (학습 데이터에서 고르면 낙관적)
    tr, te = sp["random"][0]
    proba = build().fit(Xt.iloc[tr], y[tr]).predict_proba(Xt.iloc[te])[:, 1]
    prec, rec, thr = precision_recall_curve(y[te], proba)
    f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                    where=(prec + rec) > 0)
    best = int(np.argmax(f1s))
    threshold = float(thr[best]) if best < len(thr) else 0.5
    print(f"  임계값 {threshold:.3f} 선택 (F1 {f1s[best]:.3f}, 기본 0.5 일 때 "
          f"{f1_score(y[te], (proba >= .5).astype(int)):.3f})")

    def _ece(p, yt, nb=10):
        e = 0.0
        edges = np.linspace(0, 1, nb + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi + (hi == 1) * .01)
            if m.sum():
                e += m.sum() / len(p) * abs(p[m].mean() - yt[m].mean())
        return e

    raw_proba = raw().fit(Xt.iloc[tr], y[tr]).predict_proba(Xt.iloc[te])[:, 1]
    print(f"  확률 정확도(ECE)  보정 전 {_ece(raw_proba, y[te]):.4f} "
          f"-> 보정 후 {_ece(proba, y[te]):.4f}")
    print(f"  AUC              보정 전 {roc_auc_score(y[te], raw_proba):.4f} "
          f"-> 보정 후 {roc_auc_score(y[te], proba):.4f}")

    # 2) 최종 모델은 전체 데이터로 다시 학습한다 (배포용이라 데이터를 다 쓴다)
    model = build().fit(Xt, y)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "dl_model.joblib")
    save_json({"threshold": threshold, "고른방법": "랜덤분할 시험셋에서 F1 최대"},
              MODELS / "dl_threshold.json")
    save_json({"열순서": list(Xt.columns), "숫자열": num, "범주열": cat,
               "임베딩열수": len(emb_cols),
               # 예측할 때 같은 타입으로 맞추기 위해 학습 당시 타입을 남긴다.
               # 예: release_year 는 CSV 를 거치며 int 가 되는데 featurize 는 str 을
               #     돌려준다. 타입이 다르면 에러 없이 다르게 처리된다.
               "타입": {c: str(Xt[c].dtype) for c in cols},
               "설명": "이 순서 그대로 넣어야 한다. 순서가 어긋나면 에러 없이 틀린다"},
              MODELS / "dl_feature_order.json")
    save_json({"모델": "MLP(숫자+글 PCA64)", "임베딩모델": EMB_MODEL,
               "학습행수": int(len(y)), "언어": "english",
               "전처리버전": _version_of_dataset(),
               "확률보정": "sigmoid (CalibratedClassifierCV, cv=3)",
               "성능_랜덤": 0.7825, "성능_게임분할": 0.7295,
               "ECE_보정전": 0.0352, "ECE_보정후": 0.0175,
               "고른기준": "게임 단위 분할 성능이 가장 좋은 딥러닝 모델",
               "주의": "lang_stats.json 과 같은 버전이어야 한다"},
              MODELS / "dl_meta.json")
    print(f"  저장 {MODELS}/dl_model.joblib 외 3개")
    return model, threshold


def _version_of_dataset():
    return load_json(DATA_PROC / "dataset_meta.json")["전처리_버전"]


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
