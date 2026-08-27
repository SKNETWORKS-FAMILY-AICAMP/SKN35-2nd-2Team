# -*- coding: utf-8 -*-
"""
프로토타입 화면 전용 SHAP — 임시 모델의 기여도를 쪼갠다.

★ 이건 최종이 아니다. 최종은 src/explain.py 다.
  임시 모델(HistGradientBoosting)에 맞춰 놓은 것이라 따로 둔다.

변수 이름을 한국어로 바꿔주는 것까지 여기서 한다 - 발표에서
log_hours_at_review 같은 이름이 그대로 보이면 안 되기 때문이다.
"""
import numpy as np
import pandas as pd

LABEL = {
    "log_hours_at_review": "플레이 시간", "hours_at_review": "플레이 시간",
    "log_num_games": "보유 게임 수", "log_num_reviews": "작성 리뷰 수",
    "game_age_days": "게임 출시 후 경과", "review_len": "리뷰 길이",
    "review_words": "리뷰 단어 수", "review_len_z": "리뷰 길이(언어보정)",
    "excl_ratio": "느낌표 비율", "caps_ratio": "대문자 비율",
    "voted_up": "추천 여부", "is_private": "프로필 비공개",
    "has_repeat": "반복 문자", "has_text": "본문 유무",
    "steam_purchase": "스팀 구매", "received_for_free": "무료 수령",
    "early_access": "얼리액세스", "steam_deck": "스팀덱",
    "release_year": "게임 출시 연도", "is_spike": "리뷰 급증 시기",
}
CAT_LABEL = {"language": "언어", "genre_group": "장르",
             "era": "출시 시기", "grade": "게임 평가"}


def explain(clf, X, top_n=6):
    """기여도 상위 top_n 개를 DataFrame 으로. 양수 = 이탈 쪽."""
    import shap

    pre, model = clf.named_steps["pre"], clf.named_steps["m"]
    Xt = pre.transform(X)
    names = pre.get_feature_names_out()

    sv = np.array(shap.TreeExplainer(model).shap_values(Xt))
    v = sv[0] if sv.ndim == 2 else sv[0, :, 1]
    order = np.argsort(-np.abs(v))[:top_n]

    out = []
    for i in order:
        raw = names[i]
        key = raw.split("__", 1)[-1]
        if raw.startswith("cat__"):
            base, _, val = key.rpartition("_")
            nm = f"{CAT_LABEL.get(base, base)} = {val}"
        else:
            nm = LABEL.get(key, key)
        out.append({"변수": nm, "기여": float(v[i])})
    return pd.DataFrame(out)
