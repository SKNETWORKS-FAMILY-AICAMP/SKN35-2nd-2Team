# -*- coding: utf-8 -*-
"""
하이퍼파라미터 튜닝 + 불균형 처리 실험.

절대 규칙 (코드로 강제한다)
    1. 학습 전에 assert_no_leak() 을 부른다 — preprocess.py 의 금지 목록으로 검사
    2. 라벨을 다시 만들지 않는다 — dataset.csv 의 churn 열을 그대로 쓴다
    3. 랜덤 80/20 과 게임 5조각, 두 가지로 평가한다
    4. 봉인 조각은 튜닝에 절대 쓰지 않는다 — 맨 마지막 한 번만 연다

분할 구조
    전체 게임 60개
      ├─ 튜닝용 48개 (111,759행)  ← 탐색·검증·임계값 선택이 전부 여기 안에서만
      │    └─ 안쪽 GroupKFold(4) — 검증도 '처음 보는 게임'
      └─ 봉인   12개 ( 27,899행)  ← 마지막 한 번. 여기로 설정을 고르지 않는다

    바깥 분할은 GroupKFold(5) 의 1조각으로 고정한다 (SEED 고정, 재현 가능).
    "이 조각으로 하니 별로네, 다른 조각으로" 를 하는 순간 봉인이 깨진다.

탐색 방법을 모델마다 다르게 쓰는 이유
    로지스틱   후보가 6개뿐 -> GridSearchCV (전수 탐색이 더 싸다)
    나머지 4종  연속값이 섞인 넓은 공간 -> RandomizedSearchCV
               (같은 예산이면 무작위 탐색이 격자보다 좋은 값을 잘 찾는다)

실행 (레포 루트에서)
    uv run python -m src.tune_ml                 # 전부
    uv run python -m src.tune_ml --모델 LightGBM
    uv run python -m src.tune_ml --불균형만
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, RandomizedSearchCV, cross_val_predict,
)
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import DATA_PROC, MODELS, RESULTS_CSV, SEED, load_json, save_json
from src.evaluate import features, make_splits
from src.preprocess import assert_no_leak          # ★ 규칙 1
from src.train_ml import load_all, prep, 변수묶음들

warnings.filterwarnings("ignore")

봉인조각 = 0          # GroupKFold(5) 의 몇 번째를 봉인할지. 절대 바꾸지 않는다
안쪽조각 = 4          # 튜닝용 안에서 몇 조각으로 검증할지
BEST_JSON = MODELS / "ml_best_params.json"

# 팀 공용 results.csv 는 열이 15개라 그대로 append 하면 표가 어긋난다.
# 튜닝 결과는 별도 파일에 요청받은 9열 형식으로 쌓는다.
TUNED_CSV = RESULTS_CSV.parent / "tuned_results.csv"
TUNE_DIR = RESULTS_CSV.parent / "tuning"          # 모델별 탐색 전 과정 (원시 기록)
IMB_CSV = RESULTS_CSV.parent / "imbalance.csv"


def _순수(v):
    """numpy 스칼라·튜플을 JSON 이 먹는 형태로."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_순수(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


# ── 데이터 · 분할 ───────────────────────────────────────────────
def 준비(묶음="B셋"):
    """
    표를 읽고 봉인 분할을 만든다.

    ★ 규칙 2 — 라벨을 만들지 않는다. load_all() 이 dataset.csv 의 churn 을 그대로 준다.
    ★ 규칙 1 — 열 목록을 meta 에서 가져와 assert_no_leak() 으로 검사한 뒤에만 쓴다.
    """
    d, y, g = load_all()
    cols = dict(변수묶음들())[묶음]

    assert_no_leak(cols)                  # 금지 컬럼이 하나라도 있으면 여기서 멈춘다
    assert "churn" not in cols, "정답 열이 입력에 섞였습니다"

    X = features(d, cols)                 # features() 도 FORBIDDEN 을 한 번 더 본다
    assert_no_leak(list(X.columns))

    idx = np.arange(len(d))
    바깥 = list(GroupKFold(n_splits=5).split(idx, y, groups=g))
    튜닝, 봉인 = 바깥[봉인조각]
    return d, y, g, X, 튜닝, 봉인


def 안쪽CV(y, g, 튜닝):
    """튜닝용 안에서만 도는 게임 단위 CV. 검증도 '처음 보는 게임'."""
    return list(GroupKFold(n_splits=안쪽조각).split(튜닝, y[튜닝], groups=g[튜닝]))


# ── 탐색 공간 ───────────────────────────────────────────────────
def 공간(이름, X):
    """
    (추정기, 파라미터공간, 탐색방법, 시도횟수) 를 돌려준다.

    로지스틱·랜덤포레스트·MLP 는 category 를 못 먹으므로 prep() 으로 원핫을 씌운다.
    XGBoost·LightGBM 은 category 를 그대로 먹으므로 맨 모델을 쓴다.
    """
    if 이름 == "로지스틱":
        est = Pipeline([("전처리", prep(X, scale=True)),
                        ("모델", LogisticRegression(max_iter=2000, random_state=SEED))])
        공간_ = {"모델__C": [0.01, 0.1, 1.0],
                "모델__class_weight": [None, "balanced"]}
        return est, 공간_, "grid", None          # 후보 6개 -> 전수 탐색

    if 이름 == "랜덤포레스트":
        est = Pipeline([("전처리", prep(X, scale=False)),
                        ("모델", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
        공간_ = {"모델__n_estimators": randint(200, 600),
                "모델__max_depth": [None, 12, 20, 30],
                "모델__min_samples_leaf": randint(2, 40),
                "모델__max_features": ["sqrt", "log2", 0.3, 0.5],
                "모델__class_weight": [None, "balanced", "balanced_subsample"]}
        return est, 공간_, "random", 12          # 1회가 비싸므로 적게

    if 이름 == "XGBoost":
        import xgboost as xgb
        est = xgb.XGBClassifier(tree_method="hist", enable_categorical=True,
                                random_state=SEED, n_jobs=-1, eval_metric="logloss")
        비율 = None                              # 아래 튜닝() 에서 채운다
        공간_ = {"n_estimators": randint(200, 900),
                "learning_rate": loguniform(0.02, 0.3),
                "max_depth": randint(3, 10),
                "min_child_weight": randint(1, 30),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_lambda": loguniform(0.5, 20),
                "scale_pos_weight": [1.0, "AUTO"]}   # AUTO = 불균형 보정
        return est, 공간_, "random", 30

    if 이름 == "LightGBM":
        import lightgbm as lgb
        est = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1)
        공간_ = {"n_estimators": randint(200, 900),
                "learning_rate": loguniform(0.02, 0.3),
                "num_leaves": randint(15, 128),
                "min_child_samples": randint(10, 200),
                "subsample": uniform(0.6, 0.4), "subsample_freq": [1],
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_lambda": loguniform(0.5, 20),
                "class_weight": [None, "balanced"]}
        return est, 공간_, "random", 30

    if 이름 == "MLP":
        est = Pipeline([("전처리", prep(X, scale=True)),
                        ("모델", MLPClassifier(max_iter=80, early_stopping=True,
                                             n_iter_no_change=6, random_state=SEED))])
        공간_ = {"모델__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                "모델__alpha": loguniform(1e-6, 1e-2),
                "모델__learning_rate_init": loguniform(5e-4, 1e-2),
                "모델__batch_size": [256, 512]}
        return est, 공간_, "random", 10          # class_weight 를 지원하지 않는다
    raise ValueError(이름)


# ── 튜닝 ────────────────────────────────────────────────────────
def 튜닝(이름, X, y, g, 튜닝idx, 묶음="B셋"):
    """
    ★ 규칙 4 — 여기서 봉인 조각은 한 번도 등장하지 않는다.
    선택 기준은 '안쪽 CV 의 평균 AUC' 하나로 미리 못 박는다.
    """
    est, 공간_, 방법, n_iter = 공간(이름, X)
    cv = 안쪽CV(y, g, 튜닝idx)
    Xt, yt, gt = X.iloc[튜닝idx], y[튜닝idx], g[튜닝idx]

    if 이름 == "XGBoost":                        # AUTO 를 실제 비율로 바꾼다
        비율 = float((yt == 0).sum() / (yt == 1).sum())
        공간_["scale_pos_weight"] = [1.0, 비율]

    공통 = dict(scoring="roc_auc", cv=GroupKFold(n_splits=안쪽조각),
               refit=True, n_jobs=1, verbose=0, return_train_score=False)
    if 방법 == "grid":
        s = GridSearchCV(est, 공간_, **공통)
        후보수 = int(np.prod([len(v) for v in 공간_.values()]))
    else:
        s = RandomizedSearchCV(est, 공간_, n_iter=n_iter, random_state=SEED, **공통)
        후보수 = n_iter

    print(f"\n  [{이름}] {방법} · 후보 {후보수}개 x {안쪽조각}조각 "
          f"= {후보수 * 안쪽조각}회 학습")
    t0 = time.time()
    s.fit(Xt, yt, groups=gt)
    걸린 = time.time() - t0

    # 탐색 전 과정을 그대로 남긴다. 모델마다 파라미터 열이 달라 파일을 나눈다.
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(s.cv_results_).sort_values("rank_test_score")
    r.insert(0, "모델명", 이름)
    r.insert(1, "변수묶음", 묶음)
    r.to_csv(TUNE_DIR / f"{이름}.csv", index=False, encoding="utf-8-sig")

    print(f"    {걸린/60:.1f}분 | 안쪽 CV AUC {s.best_score_:.4f} "
          f"± {r.loc[s.best_index_, 'std_test_score']:.4f}")
    print(f"    최고 설정: {s.best_params_}")
    print(f"    (참고) 후보들의 CV AUC 분포 "
          f"최저 {r.mean_test_score.min():.4f} / 중앙 {r.mean_test_score.median():.4f} "
          f"/ 최고 {r.mean_test_score.max():.4f}")
    return s.best_estimator_, s.best_params_, float(s.best_score_), 걸린


# ── 임계값 (봉인 밖에서만) ──────────────────────────────────────
def 임계값찾기(est, X, y, g, 튜닝idx):
    """
    ★ 봉인으로 임계값을 고르지 않는다.
    튜닝용 안에서 CV out-of-fold 확률을 모아 F1 이 최대인 지점을 찾는다.
    """
    Xt, yt, gt = X.iloc[튜닝idx], y[튜닝idx], g[튜닝idx]
    oof = cross_val_predict(est, Xt, yt, groups=gt,
                            cv=GroupKFold(n_splits=안쪽조각),
                            method="predict_proba", n_jobs=1)[:, 1]
    prec, rec, thr = precision_recall_curve(yt, oof)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    i = int(np.argmax(f1))
    t = float(thr[i]) if i < len(thr) else 0.5
    return t, float(f1[i]), float(f1_score(yt, (oof >= 0.5).astype(int)))


# ── 불균형 처리 비교 ────────────────────────────────────────────
def 불균형비교(X, y, g, 튜닝idx, 묶음="B셋"):
    """
    기본 / class_weight=balanced / SMOTE 를 같은 CV 에서 비교한다.

    ★ SMOTE 는 imblearn Pipeline 안에 둔다. 그래야 각 조각의 '학습 쪽' 에만
      적용되고 검증 쪽에는 합성 데이터가 새지 않는다.
      나누기 전에 SMOTE 를 걸면 합성된 이웃이 검증에 들어가 점수가 부풀려진다.

    세 방식 모두 원핫을 거친다 (SMOTE 가 숫자만 먹으므로 조건을 맞춘다).
    ※ 원핫 열을 보간하면 0/1 사이 값이 생긴다. SMOTE 의 알려진 한계이고,
      그래서 이 비교는 '트리에 SMOTE 가 필요한가' 를 보는 용도다.
    """
    import lightgbm as lgb
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    Xt, yt, gt = X.iloc[튜닝idx], y[튜닝idx], g[튜닝idx]
    cv = GroupKFold(n_splits=안쪽조각)

    def 기본모델(**kw):
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.1,
                                  random_state=SEED, n_jobs=-1, verbose=-1, **kw)

    안들 = {
        "기본": Pipeline([("전처리", prep(X, scale=False)), ("모델", 기본모델())]),
        "class_weight=balanced": Pipeline([("전처리", prep(X, scale=False)),
                                           ("모델", 기본모델(class_weight="balanced"))]),
        "SMOTE": ImbPipeline([("전처리", prep(X, scale=False)),
                              ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
                              ("모델", 기본모델())]),
    }

    print(f"\n{'='*72}\n불균형 처리 비교 (LightGBM 고정 · 튜닝용 {len(튜닝idx):,}행 · "
          f"안쪽 {안쪽조각}조각)\n{'='*72}")
    print(f"  {'방식':<24s}{'AUC':>8s}{'PR-AUC':>9s}{'F1@0.5':>9s}"
          f"{'best_F1':>9s}{'임계값':>8s}{'시간':>8s}")
    표 = []
    for 이름, est in 안들.items():
        t0 = time.time()
        oof = cross_val_predict(est, Xt, yt, groups=gt, cv=cv,
                                method="predict_proba", n_jobs=1)[:, 1]
        걸린 = time.time() - t0
        prec, rec, thr = precision_recall_curve(yt, oof)
        f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                        where=(prec + rec) > 0)
        i = int(np.argmax(f1s))
        행 = {"방식": 이름, "AUC": roc_auc_score(yt, oof),
             "PR-AUC": average_precision_score(yt, oof),
             "F1@0.5": f1_score(yt, (oof >= 0.5).astype(int)),
             "best_F1": float(f1s[i]),
             "임계값": float(thr[i]) if i < len(thr) else 0.5,
             "초": 걸린}
        표.append(행)
        print(f"  {이름:<24s}{행['AUC']:>8.4f}{행['PR-AUC']:>9.4f}"
              f"{행['F1@0.5']:>9.4f}{행['best_F1']:>9.4f}"
              f"{행['임계값']:>8.3f}{걸린:>7.0f}s")
    return pd.DataFrame(표)


# ── 최종 평가 ───────────────────────────────────────────────────
def 최종평가(이름, est, X, y, g, 튜닝idx, 봉인idx, 임계값, 묶음="B셋"):
    """
    ★ 규칙 3 — 랜덤 80/20 과 게임 5조각 둘 다 보고한다.
    ★ 규칙 4 — 봉인은 여기서 처음이자 마지막으로 열린다.
    """
    from sklearn.base import clone

    def 점수(y_true, p, t):
        pred = (p >= t).astype(int)
        return {"AUC": roc_auc_score(y_true, p),
                "PR-AUC": average_precision_score(y_true, p),
                "Recall": recall_score(y_true, pred, zero_division=0),
                "F1": f1_score(y_true, pred, zero_division=0)}

    기록 = []

    # (1) 봉인 12게임 — 오염 0. 튜닝용 전체로 학습하고 딱 한 번 채점
    t0 = time.time()
    m = clone(est).fit(X.iloc[튜닝idx], y[튜닝idx])
    p = m.predict_proba(X.iloc[봉인idx])[:, 1]
    기록.append({"모델명": f"{이름}(튜닝)", "변수묶음": 묶음, "분할방식": "봉인(게임12)",
               **점수(y[봉인idx], p, 임계값), "학습시간": f"{time.time()-t0:.1f}s"})

    # (2) 랜덤 80/20 · (3) 게임 5조각 — evaluate.py 의 채점표를 그대로 쓴다
    d_all = pd.DataFrame({"churn": y, "game": g})
    sp = make_splits(d_all, g)

    def fit_predict(X_tr, y_tr, X_te):
        return clone(est).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

    for kind in ("random", "group"):
        t0 = time.time()
        점수들 = []
        for tr, te in sp[kind]:
            pp = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
            점수들.append(점수(y[te], pp, 임계값))
        agg = {k: float(np.mean([s[k] for s in 점수들])) for k in 점수들[0]}
        기록.append({"모델명": f"{이름}(튜닝)", "변수묶음": 묶음,
                   "분할방식": "랜덤" if kind == "random" else f"게임({len(sp[kind])})",
                   **agg, "학습시간": f"{time.time()-t0:.1f}s"})
    return 기록


def _버전():
    try:
        return "v" + load_json(DATA_PROC / "dataset_meta.json")["전처리_버전"]
    except Exception:
        return "?"


# ── 실행 ────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="하이퍼파라미터 튜닝 + 불균형 실험")
    ap.add_argument("--모델", nargs="*", default=None)
    ap.add_argument("--묶음", default="B셋", choices=["A셋", "B셋"])
    ap.add_argument("--불균형만", action="store_true")
    ap.add_argument("--최종표", action="store_true",
                    help="저장된 최고 설정으로 다시 재서 results.csv 에 합친다")
    a = ap.parse_args(argv)

    d, y, g, X, 튜닝idx, 봉인idx = 준비(a.묶음)
    print(f"{'='*72}")
    print(f"봉인 분할 (GroupKFold(5) 의 {봉인조각+1}조각 고정)")
    print(f"{'='*72}")
    print(f"  튜닝용  게임 {len(set(g[튜닝idx])):2}개 · {len(튜닝idx):,}행 · "
          f"이탈률 {y[튜닝idx].mean():.4f}")
    print(f"  봉인    게임 {len(set(g[봉인idx])):2}개 · {len(봉인idx):,}행 · "
          f"이탈률 {y[봉인idx].mean():.4f}")
    print(f"  겹치는 게임 {len(set(g[튜닝idx]) & set(g[봉인idx]))}개 · "
          f"변수 {X.shape[1]}개 ({a.묶음}) · 누수검사 통과")

    if a.최종표:
        최종표(a.묶음)
        return

    if a.불균형만:
        불균형비교(X, y, g, 튜닝idx, a.묶음)
        return

    이름들 = a.모델 or ["로지스틱", "랜덤포레스트", "XGBoost", "LightGBM", "MLP"]
    저장 = load_json(BEST_JSON) if BEST_JSON.exists() else {}
    기록 = []

    print(f"\n{'='*72}\n튜닝 — 선택 기준은 '안쪽 CV 평균 AUC' 하나로 고정\n{'='*72}")
    for 이름 in 이름들:
        est, params, cv점수, 걸린 = 튜닝(이름, X, y, g, 튜닝idx, a.묶음)
        t, bf1, f1_05 = 임계값찾기(est, X, y, g, 튜닝idx)
        print(f"    임계값(CV out-of-fold): {t:.4f}  "
              f"F1 {f1_05:.4f}(0.5) -> {bf1:.4f}(최적)")
        저장[이름] = {"변수묶음": a.묶음,
                   "최고설정": {k: _순수(v) for k, v in params.items()},
                   "안쪽CV_AUC": round(cv점수, 4), "임계값": round(t, 4),
                   "탐색시간_초": round(걸린, 1), "전처리버전": _버전(),
                   "봉인조각": 봉인조각}
        save_json(저장, BEST_JSON)
        기록 += 최종평가(이름, est, X, y, g, 튜닝idx, 봉인idx, t, a.묶음)

    표 = pd.DataFrame(기록)
    표["전처리버전"] = _버전()
    열 = ["모델명", "변수묶음", "분할방식", "AUC", "PR-AUC", "Recall", "F1",
         "학습시간", "전처리버전"]
    표 = 표[열].round(4)
    표.to_csv(TUNED_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*72}\n원시 결과\n{'='*72}")
    print(표.to_string(index=False))

    imb = 불균형비교(X, y, g, 튜닝idx, a.묶음)
    imb.to_csv(IMB_CSV, index=False, encoding="utf-8-sig")

    print("\n기록")
    for f in (BEST_JSON, TUNED_CSV, IMB_CSV, TUNE_DIR):
        print(f"  {f}")




# ── 팀 공용 results.csv 로 합치기 ────────────────────────────────
def 재구성(이름, X, params):
    """ml_best_params.json 의 최고 설정으로 추정기를 다시 만든다."""
    est, _, _, _ = 공간(이름, X)
    return est.set_params(**params)


def 최종표(묶음="B셋", 기록=True):
    """
    저장된 최고 설정으로 봉인·랜덤·게임(5) 를 다시 재서
    팀 공용 results.csv (15열) 에 append 한다.

    ★ 임계값은 evaluate.THRESHOLD(0.5) 가 아니라 모델별 튜닝 임계값을 쓴다.
      배포되는 모델이 0.4015 로 판정하는데 표에 0.5 기준 F1 을 적으면
      실제 나가는 설정과 다른 숫자를 보고하는 셈이다.
      임계값은 메모 열에 명시하고, best_F1 · best_임계값 은 진단용으로 함께 남긴다.
      AUC · PR-AUC 는 임계값과 무관하므로 다른 줄과 그대로 비교된다.

    ※ evaluate.py 는 딥러닝 담당(C)의 공용 파일이라 고치지 않는다.
      채점 로직만 여기서 따로 쓴다.
    """
    from datetime import datetime
    from sklearn.base import clone

    d, y, g, X, 튜닝idx, 봉인idx = 준비(묶음)
    저장 = load_json(BEST_JSON)
    d_all = pd.DataFrame({"churn": y, "game": g})
    sp = make_splits(d_all, g)
    sp["봉인"] = [(튜닝idx, 봉인idx)]          # 튜닝용으로 학습 -> 봉인으로 시험

    def 점수(y_true, proba, t):
        pred = (proba >= t).astype(int)
        prec, rec, thr = precision_recall_curve(y_true, proba)
        f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                        where=(prec + rec) > 0)
        i = int(np.argmax(f1s))
        return {"AUC": roc_auc_score(y_true, proba),
                "PR-AUC": average_precision_score(y_true, proba),
                "Recall": recall_score(y_true, pred, zero_division=0),
                "F1": f1_score(y_true, pred, zero_division=0),
                "best_F1": float(f1s[i]),
                "best_임계값": float(thr[i]) if i < len(thr) else 1.0}

    이름표 = {"봉인": "봉인(게임12)", "random": "랜덤", "group": f"게임({len(sp['group'])})"}
    행들 = []
    print("=" * 78)
    print("최종표 -> results.csv (임계값은 모델별 튜닝값)")
    print("=" * 78)
    for 이름, 정보 in 저장.items():
        est = 재구성(이름, X, 정보["최고설정"])
        t = 정보["임계값"]
        for kind in ("봉인", "random", "group"):
            t0, 조각들 = time.time(), []
            for tr, te in sp[kind]:
                p = clone(est).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
                조각들.append(점수(y[te], p, t))
            초 = time.time() - t0
            agg = {k: float(np.mean([r[k] for r in 조각들])) for k in 조각들[0]}
            행 = {"모델명": f"{이름}(튜닝)", "변수묶음": 묶음, "분할방식": 이름표[kind],
                 **{k: round(agg[k], 4) for k in
                    ["AUC", "PR-AUC", "Recall", "F1"]},
                 "편차": round(float(np.std([r["AUC"] for r in 조각들])), 4),
                 "best_F1": round(agg["best_F1"], 4),
                 "best_임계값": round(agg["best_임계값"], 3),
                 "학습시간": f"{초:.1f}s", "전처리버전": _버전(), "행수": len(y),
                 "시각": datetime.now().strftime("%m-%d %H:%M"),
                 "메모": (f"튜닝 · 임계값 {t} (OOF 선택) · 봉인조각{정보['봉인조각']} · "
                        f"안쪽CV {정보['안쪽CV_AUC']}")}
            행들.append(행)
            pm = f" ± {행['편차']:.3f}" if kind == "group" else ""
            print(f"  {이름:<12s} {이름표[kind]:<11s} 임계값 {t:.4f} | "
                  f"AUC {행['AUC']:.4f}{pm} | PR {행['PR-AUC']:.3f} | "
                  f"Recall {행['Recall']:.3f} | F1 {행['F1']:.3f} | {행['학습시간']}")

    표 = pd.DataFrame(행들)
    기존 = pd.read_csv(RESULTS_CSV, encoding="utf-8")
    표 = 표[list(기존.columns)]                 # ★ 열 순서를 기존 파일에 맞춘다
    if 기록:
        표.to_csv(RESULTS_CSV, mode="a", index=False, header=False,
                  encoding="utf-8-sig")
        print(f"\nresults.csv 에 {len(표)}줄 추가 -> {RESULTS_CSV}")
    return 표


if __name__ == "__main__":
    main()
