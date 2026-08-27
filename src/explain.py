# -*- coding: utf-8 -*-
"""
SHAP — "모델이 왜 그렇게 판단했나" 를 숫자로 쪼갠다.

무엇인가
    모델은 이탈 확률 0.87 만 뱉는다. 왜 0.87 인지는 말해주지 않는다.
    SHAP 은 그 0.87 을 변수별 몫으로 나눈다.

        기본값 0.40  (아무것도 모를 때의 평균 이탈률)
          + 0.31   플레이 시간이 25분밖에 안 됨
          + 0.09   리뷰 글의 내용
          + 0.05   비추천을 누름
          - 0.02   보유 게임이 적음
        ─────────
          = 0.83   이 사람의 이탈 확률

    각 몫은 "그 변수를 몰랐다면 확률이 얼마나 달랐을까" 를 잰 값이다.

어디에 쓰나
    1) 화면 1번 — 리뷰를 붙여넣으면 "왜 그렇게 봤는지" 근거를 보여준다
    2) 결과서 — 어떤 변수가 전체적으로 중요한지 (reports/figures/04)

왜 LightGBM 으로 계산하나
    최종 모델은 신경망(MLP)인데, 신경망의 SHAP 은 근사값이라 느리다
    (건당 수 초). 트리 모델은 정확한 값을 0.3초에 2000건 계산한다.
    같은 입력을 쓰고 성능도 거의 같으므로(0.812 vs 0.810),
    "어떤 변수가 중요한가" 라는 질문의 답은 공유된다.
    ※ 화면의 확률은 최종 모델(MLP)이 내고, 근거만 이 모델이 만든다.

실행 (레포 루트에서)
    uv run python -m src.explain
"""
import numpy as np
import pandas as pd

from src.config import FIGURES, MODELS, SEED, save_json
from src.embed import load as load_embeddings
from src.evaluate import features, load_english, make_splits
from src.train_dl import EMB_PREFIX, feature_cols, with_text

N_PCA = 64
EMB_GROUP = "리뷰 글"
_MODEL_PATH = MODELS / "shap_model.joblib"

# 사람이 읽을 이름
LABEL = {
    "log_hours_at_review": "플레이 시간", "hours_at_review": "플레이 시간(원값)",
    "log_num_games": "보유 게임 수", "log_num_reviews": "작성한 리뷰 수",
    "game_age_days": "출시 후 며칠 만에 샀나", "review_len": "리뷰 길이",
    "review_words": "리뷰 단어 수", "review_len_z": "리뷰 길이(언어 보정)",
    "excl_ratio": "느낌표 비율", "caps_ratio": "대문자 비율",
    "voted_up": "추천 / 비추천", "steam_purchase": "스팀에서 직접 구매",
    "received_for_free": "무료로 받음", "early_access": "얼리액세스",
    "steam_deck": "스팀덱으로 플레이", "is_private": "프로필 비공개",
    "is_spike": "리뷰가 몰린 날", "has_text": "글이 있음",
    "has_repeat": "반복 문자 (ㅋㅋㅋ, !!!)", "language": "언어",
    "genre_group": "장르", "era": "출시 시기", "grade": "게임 평가 등급",
    "release_year": "출시 연도",
}
_pretty = lambda c: EMB_GROUP if c.startswith("글_") else LABEL.get(c, c)


# ── 모델 ────────────────────────────────────────────────────────
def _prep(X, cols, pca, emb_cols, ref=None):
    z = pd.DataFrame(pca.transform(X[emb_cols]),
                     columns=[f"글_{i}" for i in range(N_PCA)], index=X.index)
    out = pd.concat([X[cols], z], axis=1)
    for c in cols:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = (pd.Categorical(out[c].astype(str), categories=ref[c].cat.categories)
                      if ref is not None else out[c].astype(str).astype("category"))
    return out


def build(save=True):
    """근거 계산용 모델을 학습한다. 최종 모델과 같은 입력을 쓴다."""
    import joblib
    import lightgbm as lgb
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    d, y, g = load_english()
    sp = make_splits(d, g)
    cols = feature_cols("B셋")
    Xt = with_text(features(d, cols), load_embeddings(d))
    emb_cols = [c for c in Xt.columns if c.startswith(EMB_PREFIX)]
    tr, te = sp["random"][0]

    pca = PCA(n_components=N_PCA, random_state=SEED).fit(Xt[emb_cols].iloc[tr])
    Xtr = _prep(Xt.iloc[tr], cols, pca, emb_cols)
    Xte = _prep(Xt.iloc[te], cols, pca, emb_cols, ref=Xtr)

    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           random_state=SEED, verbose=-1).fit(Xtr, y[tr])
    auc = roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])
    print(f"  근거 모델 LightGBM(숫자+글{N_PCA}) 랜덤분할 AUC {auc:.4f}")

    bundle = {"model": m, "pca": pca, "cols": cols, "emb_cols": emb_cols,
              "template": Xtr.iloc[:1], "auc": float(auc)}
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, _MODEL_PATH)
        print(f"  저장 {_MODEL_PATH}")
    return bundle, Xte, y[te]


_cache = {}


def _load():
    if "b" not in _cache:
        import joblib
        if not _MODEL_PATH.exists():
            raise FileNotFoundError("먼저: uv run python -m src.explain")
        _cache["b"] = joblib.load(_MODEL_PATH)
    return _cache["b"]


# ── 1) 전체 변수 중요도 ─────────────────────────────────────────
def global_importance(X, top_n=12, save_fig=True):
    """변수별 평균 기여도. 글 64개는 하나로 묶는다."""
    import shap

    b = _load()
    sv = np.array(shap.TreeExplainer(b["model"]).shap_values(X))
    v = sv[..., 1] if sv.ndim == 3 else sv
    imp = pd.Series(np.abs(v).mean(0), index=X.columns)
    imp.index = [_pretty(c) for c in imp.index]
    imp = imp.groupby(level=0).sum().sort_values(ascending=False)

    if save_fig:
        import matplotlib.pyplot as plt
        from src.figures import C_GROUP, C_TEXT  # 폰트 설정도 함께 적용된다

        t = imp.head(top_n)[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.6))
        ax.barh(t.index, t.values,
                color=[C_TEXT if i == EMB_GROUP else C_GROUP for i in t.index])
        for i, val in enumerate(t.values):
            ax.text(val + t.max() * .012, i, f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel("평균 기여도 (SHAP 절댓값)")
        ax.set_title(f"무엇을 보고 판단하는가  —  '{EMB_GROUP}'은 {N_PCA}개 성분의 합",
                     fontsize=12, pad=12)
        ax.set_xlim(0, t.max() * 1.15)
        ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(FIGURES / "04_변수기여도.png"); plt.close(fig)
    return imp


# ── 2) 리뷰 한 건의 근거 (화면 1번) ─────────────────────────────
def explain_row(row_df, top_n=5):
    """
    변환된 한 줄 -> [(사람이 읽을 이름, 기여도), ...]

    기여도가 +면 "이탈 쪽으로 밀었다", -면 "잔존 쪽으로 밀었다".
    """
    import shap

    b = _load()
    X = row_df.reindex(columns=b["template"].columns)
    for c in b["template"].columns:                 # 범주 목록을 학습 기준으로
        if isinstance(b["template"][c].dtype, pd.CategoricalDtype):
            X[c] = pd.Categorical(X[c].astype(str),
                                  categories=b["template"][c].cat.categories)
    sv = np.array(shap.TreeExplainer(b["model"]).shap_values(X))
    v = (sv[..., 1] if sv.ndim == 3 else sv)[0]
    s = pd.Series(v, index=X.columns)
    s.index = [_pretty(c) for c in s.index]
    s = s.groupby(level=0).sum()
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)


def explain_review(**kw):
    """화면용 진입점. predict_one 과 같은 인자를 받는다."""
    from src.predict import _build_row, predict_one
    r = predict_one(**kw)
    r["근거"] = [(name, round(float(val), 4))
                for name, val in explain_row(_build_row(**kw)).items()]
    return r


if __name__ == "__main__":
    bundle, Xte, yte = build()
    imp = global_importance(Xte.iloc[:3000])
    print("\n변수 기여도 상위 10")
    for k, v in imp.head(10).items():
        print(f"  {v:.4f}  {k}")
    print(f"\n  그림: {FIGURES / '04_변수기여도.png'}")
    save_json({"근거모델": "LightGBM(숫자+글64)", "AUC": bundle["auc"],
               "기여도": {k: round(float(v), 4) for k, v in imp.items()}},
              MODELS / "shap_importance.json")
