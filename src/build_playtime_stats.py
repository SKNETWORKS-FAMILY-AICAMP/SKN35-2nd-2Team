# -*- coding: utf-8 -*-
"""
'살까 말까 계산기' 가 쓸 **실제 플레이 시간 통계**를 만든다.

왜 필요한가
  전에는 예상 플레이 시간을 이렇게 지어냈다.

      예상시간 = 리뷰까지시간 x (1 + 6 x 완주확률)

  이 '6' 에 근거가 없다. 우리 모델은 "리뷰 뒤 1시간을 넘기나" 라는
  예/아니오만 예측하지, 몇 시간 더 할지는 예측하지 않는다.
  시간당 비용은 그 시간을 분모로 쓰므로, 분모가 임의값이면 결과도 임의값이다.

무엇으로 바꾸나
  원본에 답이 그대로 있다.

      리뷰 뒤 실제로 더 한 시간 = playtime_forever_min - playtime_at_review_min

  이걸 게임별로, 그리고 **이탈한 사람 / 남은 사람** 을 나눠 중앙값을 낸다.
  화면에서는 모델이 낸 이탈 확률로 두 값을 섞는다.

      예상 추가시간 = p x (이탈자 중앙값) + (1-p) x (잔존자 중앙값)

  이러면 "모델이 낸 확률" 과 "실제 사람들이 한 시간" 만으로 계산된다.

평균이 아니라 중앙값을 쓰는 이유
  플레이 시간은 한쪽으로 심하게 쏠려 있다. 5,000시간 한 사람 몇 명이
  평균을 통째로 끌어올린다. 중앙값이 "보통 사람" 에 가깝다.

세 겹으로 준비한다
  게임별  — 우리가 리뷰를 모은 60개. 가장 정확하다.
  장르별  — 카탈로그 1,500개 중 나머지는 게임별 값이 없다. 장르로 대신한다.
  전체    — 장르도 모를 때.

실행
    uv run python -m src.build_playtime_stats
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.config import DATA_PROC, ENC_READ

RAW = DATA_PROC.parent.parent / "_보관" / "원래폴더" / "03_수집" / "steam_raw.csv"
OUT = DATA_PROC / "playtime_stats.json"

CHURN_H = 1.0        # 리뷰 뒤 1시간 미만이면 이탈 (preprocess.py 와 같은 기준)
최소표본 = 30        # 이보다 적으면 게임별 값을 믿지 않는다


def 중앙값들(g):
    """이탈자 / 잔존자 각각의 '리뷰 뒤 추가 시간' 중앙값."""
    이탈 = g.loc[g.churn == 1, "after"]
    잔존 = g.loc[g.churn == 0, "after"]
    return {
        "이탈": round(float(이탈.median()), 2) if len(이탈) else None,
        "잔존": round(float(잔존.median()), 2) if len(잔존) else None,
        "n": int(len(g)),
    }


def main():
    if not RAW.exists():
        sys.exit(f"원본을 못 찾았습니다: {RAW}")

    d = pd.read_csv(RAW, encoding=ENC_READ, low_memory=False,
                    usecols=["game", "genre_group", "playtime_forever_min",
                             "playtime_at_review_min"])
    print(f"원본 {len(d):,}행")

    for c in ["playtime_forever_min", "playtime_at_review_min"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["playtime_forever_min", "playtime_at_review_min"])
    d["after"] = (d.playtime_forever_min - d.playtime_at_review_min) / 60

    이상 = (d.after < 0).sum()
    d = d[d.after >= 0]
    print(f"음수(수집 오류) {이상}건 제외 -> {len(d):,}행")

    d["churn"] = (d.after < CHURN_H).astype(int)
    print(f"이탈률 {d.churn.mean():.4f}\n")

    전체 = 중앙값들(d)
    장르 = {k: 중앙값들(v) for k, v in d.groupby("genre_group")}
    게임 = {k: 중앙값들(v) for k, v in d.groupby("game")
           if len(v) >= 최소표본}

    print(f"{'장르':12} {'표본':>7} {'이탈자':>8} {'잔존자':>10}")
    for k, v in sorted(장르.items(), key=lambda x: -x[1]["n"]):
        print(f"{k:12} {v['n']:>7,} {v['이탈']:>7.2f}h {v['잔존']:>9.1f}h")
    print(f"{'전체':12} {전체['n']:>7,} {전체['이탈']:>7.2f}h {전체['잔존']:>9.1f}h")

    print(f"\n게임별 통계 {len(게임)}개 (표본 {최소표본}건 이상)")
    표 = pd.DataFrame(게임).T.sort_values("잔존")
    print("  잔존자가 가장 짧게 한 게임 3개")
    for n, r in 표.head(3).iterrows():
        print(f"    {n[:34]:36} 잔존 {r['잔존']:>7.1f}h · 이탈 {r['이탈']:.2f}h")
    print("  잔존자가 가장 길게 한 게임 3개")
    for n, r in 표.tail(3).iterrows():
        print(f"    {n[:34]:36} 잔존 {r['잔존']:>7.1f}h · 이탈 {r['이탈']:.2f}h")

    json.dump({
        "설명": "리뷰를 쓴 뒤 실제로 더 플레이한 시간의 중앙값. "
              "이탈자(1시간 미만)와 잔존자를 나눠서 담았다.",
        "쓰는법": "예상 추가시간 = p x 이탈 + (1-p) x 잔존  (p = 모델의 이탈 확률)",
        "기준": {"이탈_시간": CHURN_H, "게임별_최소표본": 최소표본},
        "출처": "03_수집/steam_raw.csv 의 playtime_forever - playtime_at_review",
        "전체": 전체, "장르": 장르, "게임": 게임,
    }, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n저장 {OUT}  ({OUT.stat().st_size/1024:.0f}KB)")


if __name__ == "__main__":
    main()
