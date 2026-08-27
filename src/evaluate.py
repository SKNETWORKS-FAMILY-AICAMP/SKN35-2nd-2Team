# -*- coding: utf-8 -*-
"""
채점표 — 모든 모델은 이 함수로만 점수를 매긴다.

왜 여기 모아두는가
    모델마다 다른 방식으로 재면 나중에 표를 못 그린다.
    채점 방식을 먼저 못 박고, 모델을 그 틀에 넣는다.

딥러닝(C) 담당은 영어 서브셋만 쓴다. 머신러닝(B)이 낸 전체 데이터
숫자(AUC 0.820)는 행 수가 달라 비교 대상이 아니다.
같은 67,112행 · 같은 분할에서 다시 재야 한다.

쓰는 법
    from src.evaluate import load_english, make_splits, evaluate_model, summary

    d, y, groups = load_english()
    sp = make_splits(d, groups)

    def fit_predict(X_tr, y_tr, X_te):
        m = LogisticRegression().fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1]

    evaluate_model("로지스틱", d[cols], y, groups, sp, fit_predict, 변수묶음="B셋")
    summary()
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

from src.config import (
    DATA_PROC, ENC_CSV, ENC_READ, FORBIDDEN, RESULTS_CSV, SEED, load_dataset, load_json,
)

N_FOLDS = 5
THRESHOLD = 0.5          # A 가 낸 기준선과 맞추기 위해 기본 0.5 로 보고한다


# ── 데이터 ──────────────────────────────────────────────────────
def load_english():
    """
    영어 서브셋을 돌려준다.

    ★ index 를 0..n-1 로 다시 매긴다.
      전체 13.9만 행 기준 index 를 그대로 쓰면 분할 번호가 어긋나
      에러 없이 조용히 틀린다.
    """
    d = load_dataset()
    d = d[d.language == "english"].reset_index(drop=True)
    return d, d["churn"].to_numpy(), d["game"].to_numpy()


def features(d, cols):
    """모델 입력용 표. 금지 컬럼이 섞였는지 여기서 막는다."""
    bad = [c for c in cols if c in FORBIDDEN]
    if bad:
        raise ValueError(f"누수 컬럼이 입력에 섞였습니다: {bad}")
    X = d[list(cols)].copy()
    for c in X.columns:                       # 글자 컬럼은 범주형으로
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.Categorical(X[c].astype(str))
    return X


def make_splits(d, groups, n_folds=N_FOLDS, seed=SEED):
    """
    분할 두 벌. 영어 서브셋 기준 위치번호(0..n-1)로 돌려준다.

      random : 층화 80/20      — 아는 게임에서의 실력
      group  : 게임 단위 n조각  — 처음 보는 게임에서의 실력
    """
    idx = np.arange(len(d))
    y = d["churn"].to_numpy()
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    folds = list(GroupKFold(n_splits=n_folds).split(idx, y, groups=groups))
    return {"random": [(tr, te)], "group": folds}


# ── 채점 ────────────────────────────────────────────────────────
def _scores(y_true, proba):
    pred = (proba >= THRESHOLD).astype(int)
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = np.divide(2 * prec * rec, prec + rec,
                    out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1s))
    return {
        "AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "best_F1": float(f1s[best]),
        "best_임계값": float(thr[best]) if best < len(thr) else 1.0,
    }


def evaluate_model(모델명, X, y, groups, splits, fit_predict,
                   변수묶음="B셋", 분할방식=("random", "group"), 메모="", 기록=True):
    """
    fit_predict(X_tr, y_tr, X_te) -> 이탈 확률 배열

    표/넘파이/임베딩 무엇이든 이 형태만 지키면 채점된다.
    """
    take = (lambda i: X.iloc[i]) if hasattr(X, "iloc") else (lambda i: X[i])
    out = []
    for kind in ([분할방식] if isinstance(분할방식, str) else 분할방식):
        rows, t0 = [], time.time()
        for tr, te in splits[kind]:
            rows.append(_scores(y[te], np.asarray(fit_predict(take(tr), y[tr], take(te)))))
        sec = time.time() - t0
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        agg["편차"] = float(np.std([r["AUC"] for r in rows]))
        rec = {
            "모델명": 모델명, "변수묶음": 변수묶음,
            "분할방식": "랜덤" if kind == "random" else f"게임({len(splits[kind])})",
            **{k: round(agg[k], 4) for k in ["AUC", "PR-AUC", "Recall", "F1"]},
            "편차": round(agg["편차"], 4),
            "best_F1": round(agg["best_F1"], 4),
            "best_임계값": round(agg["best_임계값"], 3),
            "학습시간": f"{sec:.1f}s",
            "전처리버전": _version(), "행수": len(y),
            "시각": datetime.now().strftime("%m-%d %H:%M"), "메모": 메모,
        }
        out.append(rec)
        if 기록:
            _append(rec)
        pm = f" ± {rec['편차']:.3f}" if kind == "group" else ""
        print(f"  {모델명:<22s} {변수묶음:<5s} {rec['분할방식']:<8s} "
              f"AUC {rec['AUC']:.4f}{pm} | PR {rec['PR-AUC']:.3f} | "
              f"F1 {rec['F1']:.3f} | {rec['학습시간']}")
    return out


# ── 기록 ────────────────────────────────────────────────────────
def _version():
    try:
        return "v" + load_json(DATA_PROC / "dataset_meta.json")["전처리_버전"]
    except Exception:
        return "?"


def _append(rec):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([rec])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False, encoding=ENC_CSV)


def summary(정렬="AUC"):
    """results.csv 를 표로 출력한다. 학습 결과서에 그대로 붙인다."""
    if not RESULTS_CSV.exists():
        print("아직 기록된 결과가 없습니다."); return None
    df = pd.read_csv(RESULTS_CSV, encoding=ENC_READ)
    cols = ["모델명", "변수묶음", "분할방식", "AUC", "편차", "PR-AUC", "Recall",
            "F1", "학습시간", "행수", "전처리버전"]
    print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    d, y, g = load_english()
    sp = make_splits(d, g)
    print(f"영어 서브셋 {len(d):,}행 | 이탈률 {y.mean():.4f} | 게임 {len(set(g))}개")
    print(f"  랜덤 분할  학습 {len(sp['random'][0][0]):,} / 시험 {len(sp['random'][0][1]):,}")
    print(f"  게임 분할  {len(sp['group'])}조각")
    for i, (tr, te) in enumerate(sp["group"], 1):
        te_games = len(set(g[te]))
        print(f"    {i}조각: 학습 {len(tr):,} / 시험 {len(te):,} (게임 {te_games}개)")
    print(f"\n  ※ 전부 '잔존'만 찍어도 정확도 {1 - y.mean():.1%} → 정확도는 쓰지 않는다")
