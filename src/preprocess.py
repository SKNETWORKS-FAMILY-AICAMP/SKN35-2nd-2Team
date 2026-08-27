# -*- coding: utf-8 -*-
"""
전처리 — 원본을 모델이 먹을 수 있는 표로 바꾼다.

여기가 라벨과 파생 변수를 만드는 **유일한 곳**이다.
다른 스크립트에서 라벨을 다시 만들지 않는다. 정의가 갈라지면 조용히 어긋난다.

쓰는 법
    from src.preprocess import clean, featurize, FEATURES_A, FEATURES_B, assert_no_leak

실행 (레포 루트에서)
    uv run python -m src.preprocess

경로는 전부 src/config.py 가 정한다. 여기에 직접 적지 않는다.
"""
import json
import re

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROC, DATASET, ENC_READ, ENC_WRITE, LANG_STATS, RAW_CSV,
    load_json, save_json,
)

CHURN_HOURS = 1.0          # 리뷰 후 이 시간 미만 플레이 = 이탈
SPIKE_RATIO = 3.0          # 그날 리뷰가 평소(중앙값)의 몇 배면 '급증'인가
MIN_LANG_N = 30            # 이보다 표본이 적은 언어는 자기 통계를 못 믿는다

# 마지막 안전장치. 여기 걸리는 건 학습 데이터 기준 3,000자 이상 리뷰뿐이고,
# "엄청 길다" 는 뜻은 잘려도 남는다. 트리 모델은 영향 없고 로지스틱·MLP 가 보호된다.
Z_CLIP = 10.0

# 전처리 규칙이 바뀔 때마다 올린다. 모델·화면이 어느 버전으로 만든
# 데이터를 썼는지 추적하기 위한 것. 바꿀 때 CHANGELOG 도 같이 적는다.
VERSION = "1.2"
CHANGELOG = {
    "1.0": "최초 — 라벨·파생변수 15개, 누수 8열 제외",
    "1.1": "표본 30건 미만 언어의 리뷰 길이 기준을 전체 통계로 대체 "
           "(아랍어 1건 때문에 review_len_z 가 197 까지 튀던 문제)",
    "1.2": "review_len_z 의 표준편차를 언어별이 아닌 전체 공통으로 바꾸고 "
           "±10 으로 잘랐다. 표본이 적은 언어는 표준편차가 과소추정돼 같은 "
           "글자수가 언어마다 다른 z 를 받았다(노르웨이어 2000자 26.5 vs "
           "영어 2000자 3.6). 언어별 표준편차는 성능 이득이 없었다",
}

# ── 절대 모델에 넣으면 안 되는 컬럼 ──────────────────────────────
# 정답을 만드는 데 쓴 것 + 리뷰 작성 시점엔 존재하지 않던 것
LEAK_COLS = [
    "playtime_forever_min",   # 정답 그 자체
    "playtime_2weeks_min",    # 미래 정보
    "last_played_ts",         # 미래 정보
    "updated_ts",             # 나중에 수정된 시각
    "votes_up", "votes_funny", "comment_count", "weighted_vote_score",  # 남들 반응
    "hours_total", "hours_after",   # 라벨 계산 중간값
]

ID_COLS = ["recommendationid", "appid", "steamid", "game"]

NUMERIC = [
    "log_hours_at_review", "hours_at_review",
    "log_num_games", "log_num_reviews",
    "game_age_days", "review_len", "review_words", "review_len_z",
    "excl_ratio", "caps_ratio",
]
BINARY = [
    "voted_up", "steam_purchase", "received_for_free", "early_access",
    "steam_deck", "is_private", "is_spike", "has_text", "has_repeat",
]
CATEGORICAL_COMMON = ["language", "genre_group", "era", "grade", "release_year"]

# A셋: 랜덤 분할용 — 게임 이름을 알려줘도 된다 (test에도 같은 게임이 있으므로)
FEATURES_A = NUMERIC + BINARY + CATEGORICAL_COMMON + ["game"]
# B셋: 게임 단위 분할용 — 처음 보는 게임이 나오므로 이름 대신 '속성'만
FEATURES_B = NUMERIC + BINARY + CATEGORICAL_COMMON

TARGET = "churn"

_STATS_PATH = LANG_STATS      # data/processed/lang_stats.json (config 가 정함)


# ── 도우미 ──────────────────────────────────────────────────────
def _text_features(text: str) -> dict:
    t = text or ""
    n = len(t)
    letters = [c for c in t if c.isalpha()]
    return {
        "review_len": n,
        "review_words": len(t.split()),
        "has_text": int(n > 0),
        "excl_ratio": t.count("!") / n if n else 0.0,
        "caps_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        # 같은 글자 4번 이상 반복 — "ㅋㅋㅋㅋ", "!!!!", "wwww"
        "has_repeat": int(bool(re.search(r"(.)\1{3,}", t))),
    }


def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── 본체 ────────────────────────────────────────────────────────
def clean(raw: pd.DataFrame, save_stats: bool = True) -> pd.DataFrame:
    """원본 → 학습용 표. 라벨과 파생 변수를 만든다."""
    d = raw.copy()

    # 1) 타입 정리 — 스팀이 같은 컬럼을 숫자로도 글자로도 준다
    d = _to_num(d, ["playtime_forever_min", "playtime_at_review_min",
                    "playtime_2weeks_min", "last_played_ts", "num_games_owned",
                    "num_reviews", "created_ts", "updated_ts", "app_release_ts",
                    "votes_up", "votes_funny", "comment_count",
                    "weighted_vote_score", "release_year"])
    for c in ["voted_up", "steam_purchase", "received_for_free",
              "early_access", "steam_deck"]:
        d[c] = d[c].astype(str).str.lower().isin(["true", "1"]).astype(int)

    # 2) 정답 만들기
    d["hours_at_review"] = d.playtime_at_review_min / 60
    d["hours_total"] = d.playtime_forever_min / 60
    d["hours_after"] = d.hours_total - d.hours_at_review
    d[TARGET] = (d.hours_after < CHURN_HOURS).astype(int)

    # 음수는 스팀 쪽 기록 오류 — 소수라 제거
    bad = d.hours_after < 0
    if bad.any():
        print(f"  이상치 제거: 리뷰 후 플레이가 음수인 {bad.sum()}건")
        d = d[~bad].copy()

    # 3) '0인데 0이 아닌 것' — 보유 게임 수 0은 프로필 비공개
    d["is_private"] = (d.num_games_owned == 0).astype(int)
    d["num_games_owned"] = d.num_games_owned.replace(0, np.nan)

    # 4) 큰 숫자 누르기
    d["log_hours_at_review"] = np.log1p(d.hours_at_review)
    d["log_num_games"] = np.log1p(d.num_games_owned).fillna(-1)   # 비공개는 -1
    d["log_num_reviews"] = np.log1p(d.num_reviews)

    # 5) 없는 정보 만들기
    d["game_age_days"] = (d.created_ts - d.app_release_ts) / 86400

    tf = pd.DataFrame([_text_features(t) for t in d.review.fillna("")], index=d.index)
    d = pd.concat([d, tf], axis=1)

    # 리뷰 길이는 언어마다 뜻이 다르다 (한국어 15자 = 영어 57자)
    #
    # 언어마다 다르게 쓰는 것은 "기본 길이"(중앙값) 하나뿐이다.
    # 퍼지는 정도(표준편차)는 전체 공통값을 쓴다.
    #
    # 왜  언어별 표준편차는 표본이 적은 언어에서 과소추정된다.
    #     노르웨이어는 197건이 모두 짧아 표준편차가 75 로 잡혔고,
    #     2000자 리뷰의 z 가 26.5 가 됐다. 같은 2000자가 영어에서는 3.6 이다
    #     — 언어끼리 비교가 안 되는 값이다. 아랍어(1건)는 더 심해서 197 이었다.
    #
    #     실제로 재보니 언어별 표준편차를 써도 성능 이득이 없었다.
    #       로지스틱 0.7647 (동일) · 부스팅 0.818 (차이 0.0005 = 노이즈)
    #     이득이 없는데 버그가 생기는 쪽을 고를 이유가 없다.
    stats = d.groupby("language").review_len.agg(["median", "std", "count"])
    all_med = float(d.review_len.median())
    all_std = float(d.review_len.std())

    weak = stats["count"] < MIN_LANG_N          # 중앙값조차 못 믿는 언어
    if weak.any():
        print(f"  표본 부족 언어 {int(weak.sum())}개 → 중앙값을 전체값으로 대체: "
              f"{', '.join(stats.index[weak])}")
    stats.loc[weak, "median"] = all_med
    stats["std"] = all_std                      # 전 언어 공통

    d["review_len_z"] = (((d.review_len - d.language.map(stats["median"]))
                          / d.language.map(stats["std"]))
                         .fillna(0).clip(-Z_CLIP, Z_CLIP))
    if save_stats:
        save_json({
            # 화면 담당이 자기 파일이 최신인지 눈으로 확인할 수 있게 버전을 심는다
            "_version": VERSION,
            # 언어마다 다른 것은 "기본 길이" 하나뿐이다
            "median": stats["median"].to_dict(),
            # 퍼지는 정도는 전 언어 공통 — 값 하나만 둔다
            "std_공통": all_std,
            # 목록에 없는 언어(히브리어 등)가 들어올 때 쓸 기본 길이
            "median_기본": all_med,
            "_min_n": MIN_LANG_N,
            "_z_clip": Z_CLIP,
        }, _STATS_PATH)

    # 급증 구간 — 세일·업데이트·화제성을 한꺼번에 잡는다
    d["date"] = pd.to_datetime(d.created_ts, unit="s").dt.date
    cnt = d.groupby(["game", "date"]).size().rename("n").reset_index()
    med = cnt.groupby("game").n.median().rename("med")
    cnt = cnt.merge(med, on="game")
    cnt["is_spike"] = (cnt.n / cnt.med >= SPIKE_RATIO).astype(int)
    d = d.merge(cnt[["game", "date", "is_spike"]], on=["game", "date"], how="left")
    d["is_spike"] = d.is_spike.fillna(0).astype(int)
    d = d.drop(columns=["date"])

    d["release_year"] = d.release_year.astype("Int64").astype(str)   # 범주로 취급

    return d


def assert_no_leak(cols) -> None:
    """모델에 넣기 직전 호출. 금지 컬럼이 하나라도 섞이면 멈춘다."""
    bad = [c for c in cols if c in LEAK_COLS]
    if bad:
        raise ValueError(f"데이터 누수! 입력에 들어가면 안 되는 컬럼: {bad}")


def featurize(review: dict, game: dict, lang_stats: dict = None) -> pd.DataFrame:
    """
    화면용 — 리뷰 하나를 모델 입력 한 줄로.

    review : {'review','language','voted_up','playtime_at_review_min',
              'num_games_owned','num_reviews','created_ts',
              'steam_purchase','received_for_free','early_access','steam_deck'}
    game   : {'genre_group','era','grade','release_year','app_release_ts','game'}
    """
    if lang_stats is None:
        lang_stats = load_json(_STATS_PATH)

    hours = (review.get("playtime_at_review_min") or 0) / 60
    owned = review.get("num_games_owned") or 0
    tf = _text_features(review.get("review", ""))
    lang = review.get("language", "english")
    # 목록에 없는 언어(히브리어 등)는 전체 중앙값을 쓴다
    med = lang_stats["median"].get(lang, lang_stats["median_기본"])
    std = lang_stats["std_공통"]

    row = {
        "hours_at_review": hours,
        "log_hours_at_review": np.log1p(hours),
        "log_num_games": np.log1p(owned) if owned > 0 else -1,
        "log_num_reviews": np.log1p(review.get("num_reviews") or 0),
        "game_age_days": ((review.get("created_ts") or 0) - game["app_release_ts"]) / 86400,
        "review_len_z": float(np.clip((tf["review_len"] - med) / std,
                                      -Z_CLIP, Z_CLIP)),
        "is_private": int(owned == 0),
        "is_spike": 0,                      # 화면에서는 알 수 없음 — 평소로 간주
        "language": lang,
        "genre_group": game["genre_group"],
        "era": game["era"],
        "grade": game["grade"],
        "release_year": str(game["release_year"]),
        "game": game.get("game", "UNKNOWN"),
        **tf,
        **{k: int(bool(review.get(k, False))) for k in
           ["voted_up", "steam_purchase", "received_for_free", "early_access", "steam_deck"]},
    }
    return pd.DataFrame([row])


def make_splits(d: pd.DataFrame, n_folds: int = 5, seed: int = 42):
    """
    시험 문제 나누기.
      random : 그냥 섞어서 80/20  → 아는 게임에서의 실력
      group  : 게임 단위 5조각    → 처음 보는 게임에서의 실력
    """
    from sklearn.model_selection import GroupKFold, train_test_split

    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                              stratify=d[TARGET])
    group_folds = list(GroupKFold(n_splits=n_folds).split(idx, d[TARGET], groups=d.game))
    return {"random": (tr, te), "group": group_folds}


if __name__ == "__main__":
    print("원본 읽는 중…")
    raw = pd.read_csv(RAW_CSV, encoding=ENC_READ, low_memory=False)
    print(f"  {len(raw):,}행 × {len(raw.columns)}열")

    print("전처리…")
    d = clean(raw)

    assert_no_leak(FEATURES_A)
    assert_no_leak(FEATURES_B)

    # 중복 없이 순서 유지 (game 은 식별용이자 A셋 변수라 두 번 나온다)
    keep, seen = [], set()
    for c in ID_COLS + sorted(set(FEATURES_A)) + [TARGET, "review"]:
        if c in d.columns and c not in seen:
            keep.append(c)
            seen.add(c)
    out = d[keep]
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATASET, index=False, encoding=ENC_WRITE)

    import time
    meta = {
        "전처리_버전": VERSION,
        "이번_변경": CHANGELOG[VERSION],
        "생성일시": time.strftime("%Y-%m-%d %H:%M"),
        "원본": str(RAW_CSV.relative_to(RAW_CSV.parent.parent.parent)),
        "행수": int(len(out)),
        "열수": int(len(out.columns)),
        "이탈률": round(float(out[TARGET].mean()), 4),
        "이탈판정_시간": CHURN_HOURS,
        "급증_배수": SPIKE_RATIO,
        "언어_표본하한": MIN_LANG_N,
        "길이z_자르기": Z_CLIP,
        # 변수 목록을 여기 적어둔다 — 모델 담당자가 preprocess.py 를
        # import 하지 않고도 dataset.csv 만으로 작업할 수 있게.
        "변수_A셋": {"설명": "game 포함 — 아는 게임 실력", "n": len(FEATURES_A),
                   "열": FEATURES_A},
        "변수_B셋": {"설명": "game 제외 — 처음 보는 게임 실력", "n": len(FEATURES_B),
                   "열": FEATURES_B},
        "정답": TARGET,
        "모델입력_아님": ID_COLS + ["review"],
        "누수로_제외한_원본열": LEAK_COLS,
    }
    save_json(meta, DATA_PROC / "dataset_meta.json")

    print(f"\n완료 — {len(out):,}행 × {len(out.columns)}열  [전처리 v{VERSION}]")
    print(f"  이탈률 {out[TARGET].mean():.1%}")
    print(f"  A셋(랜덤 분할용) {len(FEATURES_A)}개 · B셋(게임 분할용) {len(FEATURES_B)}개")
    print(f"  저장: {DATASET}")
    print(f"        {_STATS_PATH}")
    print(f"        {DATA_PROC / 'dataset_meta.json'}")
