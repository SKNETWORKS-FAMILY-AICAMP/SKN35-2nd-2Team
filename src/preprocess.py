# -*- coding: utf-8 -*-
"""
전처리 — 원본을 모델이 먹을 수 있는 표로 바꾼다.

여기가 라벨과 파생 변수를 만드는 **유일한 곳**이다.
다른 스크립트에서 라벨을 다시 만들지 않는다. 정의가 갈라지면 조용히 어긋난다.

쓰는 법 — 레포 루트에서
    from src.preprocess import clean, featurize, FEATURES_A, FEATURES_B
    df = clean(pd.read_csv(RAW_CSV, encoding=ENC_READ))

원본 → dataset.csv 통째로 다시 만들기
    uv run python -m src.preprocess

경로는 직접 쓰지 않는다. 전부 src/config.py 에서 가져온다.
"""
import json
import re

import numpy as np
import pandas as pd

from src.config import (
    DATASET,
    ENC_READ,
    FORBIDDEN,
    LANG_STATS,
    RAW_CSV,
    SEED,
    check_leakage,
    save_csv,
)

CHURN_HOURS = 1.0          # 리뷰 후 이 시간 미만 플레이 = 이탈
SPIKE_RATIO = 3.0          # 그날 리뷰가 평소(중앙값)의 몇 배면 '급증'인가
MIN_LANG_N  = 30           # 이보다 표본이 적은 언어는 전체 기준을 쓴다
LANG_STATS_VERSION = "1.1"  # lang_stats.json 형식 — _overall 이 들어간 판

# ── 절대 모델에 넣으면 안 되는 컬럼 ──────────────────────────────
# 목록의 원본은 src/config.py 의 FORBIDDEN 하나뿐이다.
# 여기에 사본을 두면 두 목록이 조용히 갈라져서, 한쪽으로 검사할 때 누수가 통과한다.
LEAK_COLS = FORBIDDEN

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

    # 리뷰 길이는 언어마다 뜻이 다르다 (중국어 75자 = 영어 283자)
    stats = d.groupby("language").review_len.agg(["median", "std", "count"])
    overall_med = float(d.review_len.median())
    overall_std = float(d.review_len.std())

    # 표본이 너무 적은 언어는 자기 기준을 못 만든다.
    # 예전처럼 std 를 1.0 으로 채우면 그 언어만 z 가 수백까지 튀어서,
    # 모델이 학습 때 본 적 없는 크기가 들어간다 (아랍어 1건 → z 197).
    thin = (stats["count"] < MIN_LANG_N) | stats["std"].isna() | (stats["std"] == 0)
    stats.loc[thin, "median"] = overall_med
    stats.loc[thin, "std"] = overall_std

    d["review_len_z"] = ((d.review_len - d.language.map(stats["median"]))
                         / d.language.map(stats["std"])).fillna(0)
    if save_stats:
        LANG_STATS.parent.mkdir(parents=True, exist_ok=True)
        # 윈도우 기본값(CRLF)으로 쓰면 .gitattributes 의 eol=lf 와 어긋나서,
        # 내용이 같은데도 매번 "수정된 파일"로 잡힌다
        with open(LANG_STATS, "w", encoding="utf-8", newline="\n") as f:
            json.dump({
                "_version": LANG_STATS_VERSION,
                "median": stats["median"].to_dict(),
                "std": stats["std"].to_dict(),
                # 목록에 없는 언어가 화면으로 들어올 때 쓸 기준.
                # 이걸 같이 저장하지 않으면 재실행할 때마다 사라진다.
                "_overall": {"median": overall_med, "std": overall_std},
                "_min_n": MIN_LANG_N,
            }, f, ensure_ascii=False, indent=1)

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
    """모델에 넣기 직전 호출. 금지 컬럼이 하나라도 섞이면 멈춘다.

    config.check_leakage 와 같은 목록을 본다. DataFrame 도 컬럼 리스트도 받는다.
    """
    check_leakage(cols)


def featurize(review: dict, game: dict, lang_stats: dict = None) -> pd.DataFrame:
    """
    화면용 — 리뷰 하나를 모델 입력 한 줄로.

    review : {'review','language','voted_up','playtime_at_review_min',
              'num_games_owned','num_reviews','created_ts',
              'steam_purchase','received_for_free','early_access','steam_deck'}
    game   : {'genre_group','era','grade','release_year','app_release_ts','game'}
    """
    if lang_stats is None:
        with open(LANG_STATS, encoding="utf-8") as f:
            lang_stats = json.load(f)

    hours = (review.get("playtime_at_review_min") or 0) / 60
    owned = review.get("num_games_owned") or 0
    tf = _text_features(review.get("review", ""))
    lang = review.get("language", "english")
    # 학습 때 못 본 언어(히브리어 등)가 화면으로 들어올 수 있다.
    # _overall 이 없는 옛 파일도 읽히도록 기본값을 남겨둔다.
    overall = lang_stats.get("_overall") or {}
    med_fb = overall.get("median", 45.0)
    std_fb = overall.get("std") or 1.0

    med = lang_stats["median"].get(lang, med_fb)
    std = lang_stats["std"].get(lang) or std_fb

    row = {
        "hours_at_review": hours,
        "log_hours_at_review": np.log1p(hours),
        "log_num_games": np.log1p(owned) if owned > 0 else -1,
        "log_num_reviews": np.log1p(review.get("num_reviews") or 0),
        "game_age_days": ((review.get("created_ts") or 0) - game["app_release_ts"]) / 86400,
        "review_len_z": (tf["review_len"] - med) / std,
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


def make_splits(d: pd.DataFrame, n_folds: int = 5, seed: int = SEED):
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
    save_csv(out, DATASET)

    print(f"\n완료 — {len(out):,}행 × {len(out.columns)}열")
    print(f"  이탈률 {out[TARGET].mean():.1%}")
    print(f"  A셋(랜덤 분할용) {len(FEATURES_A)}개 · B셋(게임 분할용) {len(FEATURES_B)}개")
    print(f"  저장: {DATASET}")
    print(f"        {LANG_STATS}")
