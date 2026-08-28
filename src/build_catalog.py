# -*- coding: utf-8 -*-
"""
추천 화면(나에게 맞는 게임)이 쓸 게임 카탈로그를 만든다.

모델은 게임 이름을 보지 않는다. 게임에 대해 아는 것은 이 5개뿐이다.
    genre_group · era · grade · release_year · game_age_days
그래서 이 5개만 채우면 우리가 리뷰를 모으지 않은 게임도 점수를 매길 수 있다.
정확도는 떨어진다 — 화면 ④가 그 낙폭(0.818 -> 0.749)을 측정한 화면이다.

이미지는 받지 않는다. 스팀 CDN 주소가 appid 로 정해져 있어 화면에서
바로 부르면 되고, 그러면 우리 용량은 0 이다.
    https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg

실행 (레포 맨 위에서)
    uv run python -m src.build_catalog          # 기본 400개
    uv run python -m src.build_catalog 800      # 개수 지정
"""
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from src.config import DATA_PROC, DATA_RAW, ENC_READ, ENC_WRITE, SEED

POOL = DATA_RAW.parent.parent / "_보관" / "원래폴더" / "02_게임선정" / "pool.csv"
OUT = DATA_PROC / "catalog.csv"
# 장르·카테고리는 영어로 받아야 한다. select_games.py 의 분류 규칙이 영어 기준이라
# 한국어로 받으면 같은 게임이 다른 장르로 분류된다 — 모델이 본 적 없는 조합이 된다.
API_EN = "https://store.steampowered.com/api/appdetails?appids={}"
API_KR = "https://store.steampowered.com/api/appdetails?appids={}&l=korean"
UA = {"User-Agent": "Mozilla/5.0"}
# 스팀은 5분에 200요청쯤(분당 40회)에서 막는다. 게임당 2번 부르므로 여유를 둔다.
# 0.7 로 돌렸다가 200개 넘어가면서 전부 실패한 적이 있다.
SLEEP = 2.0
MIN_REVIEWS = 5000     # 이보다 리뷰가 적은 게임은 추천 목록에 넣지 않는다.
                       # 아무도 모르는 게임이 추천에 뜨면 오히려 신뢰가 떨어진다.
NOW = pd.Timestamp("2026-08-25")


def classify(genres, categories):
    """select_games.py 의 classify() 와 똑같은 규칙. 다르면 모델이 본 적 없는 값이 된다."""
    g, c = set(genres), set(categories)
    if "Massively Multiplayer" in g or "Online PvP" in c or "PvP" in c:
        return "멀티/경쟁"
    if any(x.startswith("Co-op") or x == "Co-op" for x in c):
        return "협동"
    if "Strategy" in g or "Simulation" in g:
        return "전략/시뮬"
    if "RPG" in g or "Adventure" in g:
        return "싱글 서사"
    return "액션/캐주얼"


# 학습 데이터가 본 출시연도는 2001~2024 뿐이다. 카탈로그에는 1999·2025·2026 도
# 섞이는데, release_year 는 범주형이라 OneHotEncoder(handle_unknown="ignore") 가
# 전부 0 으로 넘긴다. 에러는 안 나지만 그 게임은 출시연도 신호를 잃는다.
# era(4단계)로 대략의 시기 정보는 남으므로 걸러내지 않고 그대로 둔다.
# 화면에서는 "처음 보는 게임은 정확도가 낮다" 로 이미 안내하고 있다.


def to_era(y):
    return ("S1 ~2016" if y <= 2016 else "S2 2017-18" if y <= 2018
            else "S3 2019-22" if y <= 2022 else "S4 2023-25")


def to_grade(pos, neg):
    """스팀 평가 등급 — games.csv 와 같은 4단계로."""
    n = pos + neg
    if n == 0:
        return None
    r = pos / n
    if r >= 0.95:
        return "압도적긍정"
    if r >= 0.80:
        return "매우긍정"
    if r >= 0.70:
        return "대체로긍정"
    return "평가나쁨"


def _get(url, appid):
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url.format(appid), headers=UA), timeout=15) as r:
            d = json.load(r)[str(appid)]
        return d["data"] if d.get("success") else None
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None


def fetch(appid):
    """영어로 받아 장르를 분류하고, 한국어 설명만 따로 덧붙인다."""
    en = _get(API_EN, appid)
    if not en:
        return None
    kr = _get(API_KR, appid)
    if kr and kr.get("short_description"):
        en["short_description"] = kr["short_description"]
    return en


def trim(total=1500):
    """카탈로그를 total 개로 줄인다.

    우리가 리뷰를 모은 60개는 무조건 남긴다 — 모델이 실제로 학습한 게임이라
    예측이 가장 정확하고, 리뷰 수가 적어도 화면의 기준점이 되기 때문이다.
    (그냥 리뷰순으로 자르면 TheoTown 등 4개가 잘려 나간다)

    나머지는 리뷰 많은 순으로 채운다.
    """
    cat = pd.read_csv(OUT, encoding=ENC_READ)
    ours = cat[cat.학습함 == 1]
    rest = (cat[cat.학습함 == 0].sort_values("리뷰수", ascending=False)
            .head(max(0, total - len(ours))))
    out = pd.concat([ours, rest], ignore_index=True)
    out.to_csv(OUT, index=False, encoding=ENC_WRITE)
    print(f"{len(cat)}개 -> {len(out)}개 "
          f"(우리 {len(ours)}개 + 처음 보는 게임 {len(rest)}개)")
    print(f"  잘라낸 기준 — 처음 보는 게임 중 리뷰 {int(rest.리뷰수.min()):,}건 이상")
    return out


def finalize():
    """카탈로그를 games.csv 기준으로 정리한다.

    우리 60개는 games.csv 의 이름과 속성을 쓴다. 스팀 API 이름에는 상표 기호가
    붙어 있어(Yakuza Kiwami vs Yakuza Kiwami (Legacy)) 이름으로 맞추면 어긋난다.
    맞추는 열쇠는 appid 하나뿐이다.
    """
    cat = pd.read_csv(OUT, encoding=ENC_READ)
    ours = pd.read_csv(DATA_PROC / "games.csv", encoding=ENC_READ)

    extra = cat[cat.학습함 == 0].copy()                    # 처음 보는 게임은 그대로
    desc = cat.set_index("appid")[["설명", "개발사", "리뷰수"]]

    mine = ours[["game", "appid", "genre_group", "era", "grade",
                 "release_year", "game_age_days"]].copy()
    mine["학습함"] = 1
    mine = mine.join(desc, on="appid")
    mine["설명"] = mine["설명"].fillna("")
    mine["개발사"] = mine["개발사"].fillna("")
    mine["리뷰수"] = mine["리뷰수"].fillna(0).astype(int)
    mine["무료"] = 0

    out = pd.concat([mine, extra], ignore_index=True).drop_duplicates("appid")

    # 가격 붙이기 — SteamSpy 는 센트 단위로 준다 (2999 = $29.99)
    pool = pd.read_csv(POOL, encoding=ENC_READ)
    out = out.merge(pool[["appid", "price", "initialprice"]], on="appid", how="left")
    out["정가"] = (out.initialprice.fillna(out.price) / 100).round(2)
    out["현재가"] = (out.price / 100).round(2)
    out = out.drop(columns=["price", "initialprice"])

    out.to_csv(OUT, index=False, encoding=ENC_WRITE)
    print(f"정리 완료 — {len(out)}개 "
          f"(우리 {int(out.학습함.sum())}개 + 처음 보는 게임 {int((out.학습함 == 0).sum())}개)")
    return out


def main(n_target=400, append=False):
    pool = pd.read_csv(POOL, encoding=ENC_READ)
    pool["total"] = pool.positive + pool.negative
    pool = pool[pool.total >= MIN_REVIEWS].copy()
    print(f"후보 {len(pool):,}개 (누적 리뷰 {MIN_REVIEWS:,}건 이상)")

    # 이미 리뷰를 모은 60개는 '학습함' 으로 표시해 구분한다
    ours = set(pd.read_csv(DATA_RAW / "selected_60.csv", encoding=ENC_READ).appid)

    have = set()
    if append and OUT.exists():
        have = set(pd.read_csv(OUT, encoding=ENC_READ).appid)
        print(f"이미 받아둔 {len(have)}개는 건너뜁니다")

    # 무작위가 아니라 **리뷰 많은 순**으로 가져온다.
    # 추천 화면에서 사용자가 검색할 게임은 결국 아는 게임이다.
    # 무작위 표본이면 유명한 게임이 절반쯤 빠져서 검색이 안 된다.
    need = ours - have
    cand = (pool[~pool.appid.isin(ours | have)]
            .sort_values("total", ascending=False).appid.values)
    pick = list(need) + list(cand[:max(0, n_target - len(need))])
    print(f"조회 대상 {len(pick)}개 (리뷰 많은 순)")
    print(f"예상 소요 {len(pick) * SLEEP / 60:.0f}분\n")

    rows, t0 = [], time.time()
    for i, appid in enumerate(pick, 1):
        d = fetch(int(appid))
        time.sleep(SLEEP)
        if i % 25 == 0 or i == len(pick):
            print(f"  {i}/{len(pick)}  성공 {len(rows)}  "
                  f"{(time.time() - t0) / 60:.1f}분")
        if not d or d.get("type") != "game":
            continue
        rel = d.get("release_date", {})
        if rel.get("coming_soon") or not rel.get("date"):
            continue
        try:
            date = pd.to_datetime(rel["date"], errors="coerce")
        except Exception:
            continue
        if pd.isna(date) or date > NOW:
            continue
        rec = d.get("recommendations", {}).get("total", 0)
        p = pool[pool.appid == appid]
        pos = int(p.positive.iloc[0]) if len(p) else rec
        neg = int(p.negative.iloc[0]) if len(p) else 0
        grade = to_grade(pos, neg)
        if grade is None:
            continue
        rows.append({
            "appid": int(appid),
            "game": d["name"],
            "genre_group": classify([x["description"] for x in d.get("genres", [])],
                                    [x["description"] for x in d.get("categories", [])]),
            "era": to_era(date.year),
            "grade": grade,
            "release_year": int(date.year),
            "game_age_days": int((NOW - date).days),
            "학습함": int(appid in ours),
            "설명": (d.get("short_description") or "")[:180],
            "개발사": ", ".join(d.get("developers", []) or [])[:60],
            "무료": int(bool(d.get("is_free"))),
            "리뷰수": pos + neg,
        })

    cat = pd.DataFrame(rows)
    if append and OUT.exists():
        cat = pd.concat([pd.read_csv(OUT, encoding=ENC_READ), cat], ignore_index=True)
    cat = cat.drop_duplicates("appid")
    cat.to_csv(OUT, index=False, encoding=ENC_WRITE)

    print(f"\n완료 — {len(cat)}개")
    print(f"  학습한 게임 {cat.학습함.sum()}개 · 처음 보는 게임 {(~cat.학습함.astype(bool)).sum()}개")
    print(f"  장르 {dict(cat.genre_group.value_counts())}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "400"
    if arg == "finalize":
        finalize()
    elif arg == "trim":
        trim(int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    else:
        main(int(arg), append="--append" in sys.argv)
        finalize()
