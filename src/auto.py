# -*- coding: utf-8 -*-
"""
AutoGluon — 자동화 도구와 우리가 손으로 만든 모델을 비교한다.

무엇을 재나
    우리 최고 (부스팅 + 글)   게임분할 0.757
    AutoGluon 자동            ???

    이기면  "자동화 도구가 이만큼 한다. 우리가 놓친 건 이것"
    비기면  "우리 피처 엔지니어링이 값어치가 있었다"
    어느 쪽이든 결과서에 들어간다.

두 가지 방식
    tabular      숫자 24개 + 미리 뽑은 임베딩(PCA 64). 우리와 같은 재료. 빠르다
    multimodal   리뷰 원문을 그대로 준다. 언어모델을 우리 문제에 맞게 재학습.
                 "글을 최선을 다해 써도 안 오르는가" 를 확인하는 실험. 느리다

밤새 돌릴 때 지켜지는 것
    - 단계마다 results.csv 에 즉시 기록한다. 3단계에서 죽어도 1·2단계는 남는다
    - 로그를 results/auto_log.txt 에 남긴다. 아침에 무슨 일이 있었는지 읽을 수 있다
    - 한 단계가 실패해도 다음 단계로 넘어간다 (전체가 멈추지 않는다)

실행 (레포 루트에서)
    uv run python -m src.auto --mode smoke                     # 1~2분, 먼저 이것부터
    uv run python -m src.auto --mode tabular
    uv run python -m src.auto --mode multimodal --sample 20000 --time 1800
"""
import argparse
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import RESULTS, SEED, load_json, DATA_PROC
from src.evaluate import evaluate_model, features, load_english, make_splits, summary

LOG = RESULTS / "auto_log.txt"


def log(msg=""):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def device_note():
    import torch
    if torch.backends.mps.is_available():
        log("  장치: MPS(맥 GPU) 사용 가능 — 다만 AutoGluon 이 CPU 로 떨어질 수 있음")
    else:
        log("  장치: CPU")


# ── 재료 준비 ───────────────────────────────────────────────────
def feature_cols():
    return load_json(DATA_PROC / "dataset_meta.json")["변수_B셋"]["열"]


def build_tabular(d, emb):
    """숫자 24개 + 임베딩 PCA 64. 우리 부스팅과 똑같은 재료."""
    from sklearn.decomposition import PCA

    cols = feature_cols()
    z = PCA(n_components=64, random_state=SEED).fit_transform(emb)
    X = features(d, cols).copy()
    for i in range(z.shape[1]):
        X[f"글_{i}"] = z[:, i]
    return X


def build_multimodal(d):
    """숫자 24개 + 리뷰 원문. AutoGluon 이 글을 직접 읽는다."""
    X = features(d, feature_cols()).copy()
    X["review"] = d["review"].fillna("").astype(str).values
    return X


# ── AutoGluon 을 채점표에 끼워 넣기 ─────────────────────────────
def make_fit_predict(kind, time_limit, presets, label="churn", num_cpus=4):
    """
    evaluate_model 이 요구하는 fit_predict(X_tr, y_tr, X_te) 형태로 감싼다.
    이렇게 해야 우리 모델들과 '같은 분할·같은 채점'으로 비교된다.
    """
    def f(X_tr, y_tr, X_te):
        import tempfile
        tr = X_tr.copy(); tr[label] = y_tr
        with tempfile.TemporaryDirectory() as tmp:
            if kind == "tabular":
                from autogluon.tabular import TabularPredictor
                p = TabularPredictor(label=label, eval_metric="roc_auc", path=tmp,
                                     verbosity=1)
                # ★ num_cpus 를 명시하지 않으면 맥에서 자원 탐지 단계가
                #   멀티프로세싱으로 교착돼 CPU 0% 로 영원히 멈춘다.
                #   (오류도 안 나므로 밤새 돌리면 아침에 아무 결과가 없다)
                p.fit(tr, time_limit=time_limit, presets=presets,
                      num_cpus=num_cpus, ag_args_fit={"num_cpus": num_cpus})
            else:
                from autogluon.multimodal import MultiModalPredictor
                p = MultiModalPredictor(label=label, eval_metric="roc_auc", path=tmp,
                                        problem_type="binary", verbosity=1)
                p.fit(tr, time_limit=time_limit, presets=presets)
            proba = p.predict_proba(X_te)
        return proba[1].values if hasattr(proba, "columns") else np.asarray(proba)[:, 1]
    return f


def run(name, kind, X, y, g, sp, splits, time_limit, presets, 메모, num_cpus=4):
    """한 단계. 실패해도 예외를 삼키고 다음으로 넘어간다."""
    log(f"▶ {name}  (분할 {splits}, 제한 {time_limit}s, presets={presets})")
    t0 = time.time()
    try:
        evaluate_model(name, X, y, g, sp,
                       make_fit_predict(kind, time_limit, presets, num_cpus=num_cpus),
                       변수묶음="B셋+글", 분할방식=splits, 메모=메모)
        log(f"  ✅ 완료 {time.time()-t0:.0f}초")
        return True
    except Exception:
        log(f"  ❌ 실패 {time.time()-t0:.0f}초")
        log(traceback.format_exc())
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke",
                    choices=["smoke", "tabular", "multimodal", "all"])
    ap.add_argument("--sample", type=int, default=0, help="행 수 제한 (0=전체)")
    ap.add_argument("--time", type=int, default=600, help="AutoGluon 제한시간(초)")
    ap.add_argument("--presets", default="medium_quality")
    ap.add_argument("--group", action="store_true", help="게임 단위 분할도 (5배 오래 걸림)")
    ap.add_argument("--cpus", type=int, default=4, help="AutoGluon 이 쓸 CPU 수")
    a = ap.parse_args()

    log("=" * 66)
    log(f"AutoGluon 시작 — mode={a.mode} sample={a.sample or '전체'} "
        f"time={a.time}s presets={a.presets} group={a.group}")
    device_note()

    d, y, g = load_english()

    # ★ 임베딩은 표본을 줄이기 전에 읽는다.
    #   embed.load() 가 전체 데이터 기준으로 행 정렬을 검사하기 때문.
    #   (줄인 뒤에 읽으면 "행이 어긋납니다" 로 멈춘다 — 안전장치가 제대로 동작한 것)
    emb = None
    if a.mode in ("smoke", "tabular", "all"):
        from src.embed import load as load_emb
        emb = load_emb(d)

    if a.mode == "smoke":
        a.sample, a.time = a.sample or 3000, min(a.time, 120)
    if a.sample and a.sample < len(d):
        rs = np.random.RandomState(SEED)
        keep = np.sort(rs.choice(len(d), a.sample, replace=False))
        d = d.iloc[keep].reset_index(drop=True)
        y, g = y[keep], g[keep]
        if emb is not None:
            emb = emb[keep]                      # 임베딩도 같은 행만 남긴다
        log(f"  표본 {len(d):,}행으로 축소 (이탈률 {y.mean():.4f})")
    else:
        log(f"  전체 {len(d):,}행 (이탈률 {y.mean():.4f})")

    sp = make_splits(d, g)
    splits = ("random", "group") if a.group else ("random",)
    ok = []

    if a.mode in ("smoke", "tabular", "all"):
        X = build_tabular(d, emb)
        log(f"  tabular 재료 {X.shape[1]}열 (숫자 24 + 글 64)")
        ok.append(run("AutoGluon(tabular)", "tabular", X, y, g, sp, splits,
                      a.time, a.presets, f"자동화 비교 · {len(d)}행", a.cpus))

    if a.mode in ("smoke", "multimodal", "all"):
        X = build_multimodal(d)
        log(f"  multimodal 재료 {X.shape[1]}열 (숫자 24 + 리뷰 원문)")
        ok.append(run("AutoGluon(multimodal)", "multimodal", X, y, g, sp, splits,
                      a.time, a.presets, f"글 원문 직접 · {len(d)}행", a.cpus))

    log(f"성공 {sum(ok)}/{len(ok)} 단계")
    log("현재까지의 전체 결과:")
    summary()
    log("=" * 66)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
