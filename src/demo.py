# -*- coding: utf-8 -*-
"""
모델링이 실제로 무슨 일인지 한 단계씩 보여준다.

results.csv 에 아무것도 쓰지 않는다. 모델도 저장하지 않는다.
"우리가 뭘 한 건지" 를 눈으로 확인하는 용도다 — 발표 준비에도 쓴다.

실행 (레포 루트에서)
    uv run python -m src.demo                 # 전체 데이터 · B셋
    uv run python -m src.demo --영어           # 영어 6.7만행 (빠름)
    uv run python -m src.demo --묶음 A셋       # 게임 이름을 넣고 돌려보기
    uv run python -m src.demo --분할 group    # 처음 보는 게임으로 시험

노트북으로 한 셀씩 보고 싶으면
    notebooks/ 안의 *_demo.ipynb
"""
import argparse
import time
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluate import features, load_english, make_splits
from src.train_ml import load_all, 모델들, 변수묶음들

warnings.filterwarnings("ignore")


def 줄(t):
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


def run(영어=False, 묶음="B셋", 분할="random", 모델목록=None):
    # ── 1 ───────────────────────────────────────────────────────
    줄("1단계 · 표를 읽는다")
    d, y, g = load_english() if 영어 else load_all()
    print(f"  dataset.csv  ->  {d.shape[0]:,}행 x {d.shape[1]}열"
          f"  ({'영어만' if 영어 else '30개 언어'})")
    print(f"  정답 y       ->  {y[:12]} ...   (1=이탈, 0=잔존)")
    print(f"  게임 g       ->  {g[:3]} ...   ({len(set(g))}개)")
    print(f"  이탈률       ->  {y.mean():.1%}")

    # ── 2 ───────────────────────────────────────────────────────
    cols = dict(변수묶음들())[묶음]
    줄(f"2단계 · 모델에게 줄 열만 고른다 ({묶음} {len(cols)}개)")
    X = features(d, cols)          # 누수 컬럼이 섞이면 여기서 멈춘다
    print(f"  X  ->  {X.shape[0]:,}행 x {X.shape[1]}열")
    print("\n  맨 앞 3줄 · 앞 5열만:")
    print(X.iloc[:3, :5].to_string())
    print(f"\n  정답(churn) 이 섞여 있나? -> {'churn' in X.columns}"
          f"   (True 면 부정행위 — AUC 0.99 가 나온다)")
    print(f"  글자 열은 범주형으로 바뀌었나? -> "
          f"{[c for c in X.columns if str(X[c].dtype) == 'category']}")

    # ── 3 ───────────────────────────────────────────────────────
    줄("3단계 · 문제집과 시험지로 나눈다")
    sp = make_splits(d, g)
    조각 = sp[분할]
    if 분할 == "random":
        tr, te = 조각[0]
        print(f"  랜덤 분할 — 그냥 섞어서 자른다 (이미 아는 게임에서의 실력)")
    else:
        tr, te = 조각[0]
        print(f"  게임 분할 — 게임 통째로 자른다 (처음 보는 게임에서의 실력)")
        print(f"  {len(조각)}조각 중 1조각만 써서 보여준다")
        print(f"  시험지에 나오는 게임 {len(set(g[te]))}개는 "
              f"학습에 한 번도 안 나온다: {sorted(set(g[te]))[:3]} ...")
    print(f"\n  train  {len(tr):>7,}줄   <- 답까지 보여주고 공부시킴")
    print(f"  test   {len(te):>7,}줄   <- 답을 숨기고 시험")
    print(f"  겹치는 줄 : {len(set(tr) & set(te))}줄   (0 이어야 정상)")

    # ── 4 ───────────────────────────────────────────────────────
    줄("4단계 · 모델을 같은 문제집에 번갈아 넣는다")
    print("  train_ml.py 의 루프가 하는 일이 이것뿐이다:\n")
    print("      for 이름, fit_predict in 모델들().items():")
    print("          확률 = fit_predict(X[train], y[train], X[test])")
    print("          AUC  = roc_auc_score(y[test], 확률)\n")
    print("  모델 4개는 안이 전혀 다른데 쓰는 법이 같다.")
    print("  전부 fit() 으로 배우고 predict_proba() 로 확률을 뱉는다.\n")

    전부 = 모델들()
    쓸것 = {k: v for k, v in 전부.items() if 모델목록 is None or k in 모델목록}
    결과 = {}
    for 이름, fit_predict in 쓸것.items():
        print(f"  -- {이름} --")
        t0 = time.time()
        확률 = fit_predict(X.iloc[tr], y[tr], X.iloc[te])   # ★ 학습+예측이 이 한 줄
        걸린 = time.time() - t0
        auc = roc_auc_score(y[te], 확률)
        결과[이름] = auc
        print(f"     학습+예측 {걸린:6.1f}초")
        print(f"     뱉은 확률 {np.round(확률[:6], 3)} ...  ({len(확률):,}개)")
        print(f"     실제 정답 {y[te][:6]} ...")
        print(f"     AUC  {auc:.4f}\n")

    # ── 5 ───────────────────────────────────────────────────────
    줄(f"5단계 · 표 한 칸이 채워진다 ({묶음} · {'랜덤' if 분할=='random' else '게임'}분할)")
    for 이름, auc in sorted(결과.items(), key=lambda x: -x[1]):
        print(f"  {이름:<14s} {auc:.4f} {'#' * max(0, int((auc - 0.70) * 200))}")
    if 분할 == "group":
        print("\n  ★ 여기는 5조각 중 1조각만 쓴 값이다. results.csv 는 5조각 평균이라")
        print("    순위가 다를 수 있다. 조각마다 AUC 가 ±0.04 씩 흔들린다")
        print("    (results.csv 의 '편차' 열이 그 흔들림이다).")
    print(f"\n  전부 '잔존'만 찍어도 정확도 {1 - y.mean():.1%} 가 나온다.")
    print("  그래서 정확도가 아니라 AUC 로 잰다.")
    print(f"\n  이 {len(결과)}줄이 results.csv 32줄 중 {len(결과)}줄이다.")
    print("  나머지는 A셋으로 한 번 더 x 게임분할로 한 번 더 x 영어로 한 번 더.")
    return 결과


def main(argv=None):
    ap = argparse.ArgumentParser(description="모델링 과정을 한 단계씩 본다")
    ap.add_argument("--영어", action="store_true", help="영어 6.7만행 (빠름)")
    ap.add_argument("--묶음", default="B셋", choices=["A셋", "B셋"])
    ap.add_argument("--분할", default="random", choices=["random", "group"])
    ap.add_argument("--모델", nargs="*", default=None,
                    help="예: --모델 LightGBM 로지스틱")
    a = ap.parse_args(argv)
    run(영어=a.영어, 묶음=a.묶음, 분할=a.분할, 모델목록=a.모델)


if __name__ == "__main__":
    # 윈도우에서 n_jobs=-1 을 쓰므로 이 가드가 필요하다
    main()
