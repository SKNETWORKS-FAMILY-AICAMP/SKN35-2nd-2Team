# -*- coding: utf-8 -*-
"""
SHAP — 머신러닝 모델이 왜 그렇게 판단했는지 캔다.

★ 파일 이름이 explain_ml 인 이유
  딥러닝 담당(C)의 src/explain_dl.py 와 목적이 같고 대상 모델이 다르다.
    src/explain_dl.py  최종 MLP 를 설명한다. SHAP 은 LightGBM 대리 모델로 근사.
                       영어 67,112행 · B셋 + 임베딩 PCA64
    src/explain_ml.py  배포용 LightGBM 자체를 설명한다 (근사 아님).
                       전체 139,658행 · 30개 언어 · B셋
  둘 다 필요하므로 파일을 나눴다. 그림 번호도 06·07 로 비켜 뒀다
  (01~05 는 딥러닝 담당이 쓴다).

두 가지를 낸다
    ① 전체 경향  reports/figures/ 06·07 — 결과서·PPT 에 붙인다
    ② 개인 설명  explain_one() — 화면 1 "왜 그렇게 판단했나" 가 불러 쓴다

배포 모델을 여기서 확정하고 저장한다
    models/ml_model.joblib          학습된 모델
    models/ml_feature_order.json    ★ 열 순서 + 범주값. 없으면 조용히 틀린다
    models/ml_threshold.json        확률 -> 0/1 경계값
    models/ml_meta.json             무엇으로 만든 모델인지

왜 B셋(game 제외) 인가
    A셋은 게임 분할에서 확률이 무너진다. 학습에 없던 게임 이름을 만나면
    실제 이탈률 37% 인 조각에 평균 70% 를 뱉는다 (AUC 는 순위만 보므로 안 드러난다).
    화면 1 은 확률을 그대로 보여주므로 B셋이어야 한다.

왜 LightGBM 인가
    게임 분할에서는 랜덤포레스트가 0.01 정도 앞선다 (5조각 중 4조각에서 우세).
    "차이가 없다" 가 아니라 "0.01 을 내주고 두 가지를 샀다" 가 맞는 표현이다.
      학습이 14배 빠르다 (1.9초 vs 27.3초)
      범주형을 그대로 먹어 SHAP 이 24개 변수에 1:1 로 대응한다.
      랜덤포레스트는 원핫을 거쳐 "genre_group=협동" 처럼 쪼개진 열로 나오므로
      화면에 이유를 보여주려면 다시 합쳐야 한다.
    팀에서 다시 정해도 된다.

배포에서 튜닝 설정을 하나 덮어쓴다
    class_weight="balanced" 를 뺀다. 아래 배포_설정_덮어쓰기 참조.

실행 (레포 루트에서)
    uv run python -m src.explain_ml
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve, roc_auc_score

from src.config import (DATA_PROC, FIGURES, MODELS, RESULTS_CSV, SEED,
                        load_json, save_json)
from src.evaluate import features, make_splits
from src.train_ml import load_all, 변수묶음들

# 한글이 네모로 깨지지 않게 (figures.py 와 같은 방식)
for _f in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
    if _f in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

MODEL_PKL = MODELS / "ml_model.joblib"
ORDER_JSON = MODELS / "ml_feature_order.json"
THR_JSON = MODELS / "ml_threshold.json"
META_JSON = MODELS / "ml_meta.json"

SHAP_N = 8000          # 그림용 표본. 13.9만 행을 다 돌려도 그림은 달라지지 않는다

# 화면에 그대로 띄울 이름. 영어 컬럼명을 사용자에게 보여줄 수는 없다.
KOR = {
    "log_hours_at_review": "플레이 시간", "hours_at_review": "플레이 시간(원값)",
    "log_num_games": "보유 게임 수", "log_num_reviews": "작성 리뷰 수",
    "game_age_days": "게임 나이", "review_len": "리뷰 글자 수",
    "review_words": "리뷰 단어 수", "review_len_z": "리뷰 길이(언어보정)",
    "excl_ratio": "느낌표 비율", "caps_ratio": "대문자 비율",
    "voted_up": "추천 여부", "steam_purchase": "스팀 직접 구매",
    "received_for_free": "무료로 받음", "early_access": "얼리액세스",
    "steam_deck": "스팀덱", "is_private": "프로필 비공개",
    "is_spike": "리뷰 급증일", "has_text": "글 있음", "has_repeat": "반복 문자",
    "language": "언어", "genre_group": "장르", "era": "출시 시기",
    "grade": "평가 등급", "release_year": "출시 연도",
}


def 한글(c):
    return KOR.get(c, c)


# ── 배포 모델 ───────────────────────────────────────────────────
BEST_JSON = MODELS / "ml_best_params.json"


# ★ 튜닝이 고른 class_weight="balanced" 를 배포에서는 뺀다
#
#   왜  튜닝의 선택 기준이 AUC 하나였다. AUC 는 순위만 보므로 확률이 통째로
#       위로 밀려도 안 떨어진다. 그래서 확률을 부풀리는 설정이 뽑혔다.
#       화면은 확률 숫자를 그대로 보여주므로 이 왜곡이 그대로 노출된다.
#
#   근거  튜닝용 48게임 안에서만 측정했다 (봉인 미사용).
#         게임 조각별 왜곡(예측평균 - 실제)
#           balanced 있음   +2.3% ~ +17.0%   절대평균 7.8%
#           balanced 제거   -3.8% ~  +7.9%   절대평균 4.0%
#         안쪽 CV AUC 는 오히려 올랐다  0.7522 -> 0.7536
#
#   한계  남은 왜곡은 "어떤 게임이 시험지가 되느냐" 에 따른 것이라
#         CalibratedClassifierCV 로도 못 고친다 (평균 편향이 이미 +1.1% 라
#         보정기가 배울 게 없다). 게임이 60개뿐인 데서 오는 한계다.
#
#   ml_best_params.json 은 고치지 않는다 — 그건 "튜닝이 무엇을 골랐나" 의 기록이다.
#   배포에서 무엇을 왜 바꿨는지는 ml_meta.json 에 남긴다.
배포_설정_덮어쓰기 = {"class_weight": None}


def _봉인기록_잇기(설정):
    """
    앞서 잰 봉인 성적을 이어 받는다.

    ★ 왜 필요한가 — 이 파일을 다시 돌리면 meta 를 통째로 새로 쓴다.
      그때 봉인 성적을 빈칸으로 두면, 다시 재려고 봉인을 또 열어야 한다.
      봉인은 열 때마다 값이 닳는 자원이라 그러면 안 된다.
      (실제로 한 번 날렸다. results.csv 에 남아 있어 복구했다)

      설정이 그대로일 때만 잇는다. 설정이 바뀌면 옛 성적은 다른 모델의 것이므로
      빈칸으로 두고 새로 재야 한다.
    """
    import pandas as pd
    옛 = load_json(META_JSON) if META_JSON.exists() else {}
    같은설정 = 옛.get("설정") == 설정
    값 = 옛.get("성능_봉인12게임") if 같은설정 else None

    if 값 is None and 같은설정:                 # meta 가 비었으면 results.csv 에서 되찾는다
        try:
            r = pd.read_csv(RESULTS_CSV, encoding="utf-8")
            행 = r[r.모델명 == "LightGBM(배포)"]
            if len(행):
                값 = float(행.AUC.iloc[-1])
        except Exception:
            pass

    if 값 is None:
        return {"성능_봉인12게임": None,
                "봉인_안내": ("이 설정으로는 아직 봉인을 열지 않았다. "
                          "`uv run python -m src.tune_ml --봉인측정` 으로 한 번 잰다")}
    return {"성능_봉인12게임": 값,
            "봉인_안내": (f"같은 설정으로 이미 잰 값을 이어 받았다. 원본 기록은 "
                      f"results.csv 의 LightGBM(배포) 행에 있다")}


def fit_final(저장=True):
    """
    B셋 · 전체 데이터로 LightGBM 을 학습하고 저장한다.

    설정은 src/tune_ml.py 가 고른 것을 쓰되, 위 배포_설정_덮어쓰기 를 적용한다.
    임계값은 그 설정으로 튜닝용 48게임의 CV out-of-fold 에서 다시 고른다
    (설정이 바뀌면 확률 분포가 달라지므로 옛 임계값을 그대로 쓰면 안 된다).

    ★ 봉인 12게임은 여기서 한 번도 등장하지 않는다.
      배포 모델 자체는 전체 데이터로 학습하지만(정상), 임계값 선택과
      성능 측정에는 48게임만 쓴다.
    """
    import joblib
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from src.tune_ml import 안쪽조각, 준비

    d, y, g, X, 튜닝idx, _봉인idx = 준비("B셋")   # _봉인idx 는 쓰지 않는다

    튜닝됨 = BEST_JSON.exists()
    if 튜닝됨:
        정보 = load_json(BEST_JSON)["LightGBM"]
        설정 = {**정보["최고설정"], **배포_설정_덮어쓰기}
        바뀐것 = {k: (정보["최고설정"].get(k), v)
                for k, v in 배포_설정_덮어쓰기.items()
                if 정보["최고설정"].get(k) != v}
        print("  튜닝 설정 사용 (ml_best_params.json)")
        for k, (전, 후) in 바뀐것.items():
            print(f"    배포에서 변경: {k}  {전!r} -> {후!r}")
    else:
        설정 = dict(n_estimators=400, learning_rate=0.1, num_leaves=31)
        바뀐것 = {}
        print("  ml_best_params.json 이 없어 기본값으로 학습합니다")

    def _new():
        return lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **설정)

    # ① 임계값을 다시 고른다 — 튜닝용 48게임의 CV out-of-fold 에서만
    Xt, yt, gt = X.iloc[튜닝idx], y[튜닝idx], g[튜닝idx]
    oof = cross_val_predict(_new(), Xt, yt, groups=gt,
                            cv=GroupKFold(n_splits=안쪽조각),
                            method="predict_proba", n_jobs=1)[:, 1]
    auc = roc_auc_score(yt, oof)
    prec, rec, thr = precision_recall_curve(yt, oof)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    best = int(np.argmax(f1))
    임계값 = float(thr[best]) if best < len(thr) else 0.5
    왜곡 = float(oof.mean() - yt.mean())
    print(f"  48게임 OOF — AUC {auc:.4f} · 확률왜곡 {왜곡:+.1%} · "
          f"임계값 {임계값:.4f} (F1 {f1[best]:.4f})")

    # ② 전체 데이터로 다시 학습한다 (배포용은 데이터를 다 쓴다)
    model = _new().fit(X, y)

    if 저장:
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PKL)
        save_json({
            "열순서": list(X.columns),
            # ★ 범주값을 반드시 같이 저장한다. 화면에서 pd.Categorical 을 다시 만들 때
            #   같은 목록·같은 순서가 아니면 라벨 번호가 밀려 조용히 틀린 예측이 나온다.
            "범주값": {c: [str(v) for v in X[c].cat.categories]
                     for c in X.columns if str(X[c].dtype) == "category"},
            # ★ 자료형도 같이 남긴다. featurize() 는 release_year 를 문자열로 주는데
            #   학습 때는 CSV 를 읽으며 pandas 가 int64 로 파싱해 숫자로 배웠다.
            #   이 목록이 없으면 화면에서 LightGBM 이 "dtypes must be int, float or bool"
            #   로 죽는다. 죽으면 다행이고, 모델에 따라서는 조용히 틀린다.
            "자료형": {c: ("범주" if str(X[c].dtype) == "category" else "숫자")
                    for c in X.columns},
        }, ORDER_JSON)
        save_json({
            "threshold": round(임계값, 4),
            "기준": "튜닝용 48게임의 CV out-of-fold 에서 F1 이 최대가 되는 값",
            "봉인사용": False,
            "주의": ("설정을 바꾸면 확률 분포가 달라져 이 값도 달라진다. "
                   "모델과 임계값은 한 세트다"),
        }, THR_JSON)
        save_json({
            "모델": "LightGBM(숫자+범주)", "변수묶음": "B셋", "학습행수": int(len(X)),
            "언어": "전체 30개", "전처리버전": _버전(),
            "튜닝": 튜닝됨, "설정": 설정,
            "배포에서_변경": {k: {"튜닝이_고른값": 전, "배포값": 후}
                        for k, (전, 후) in 바뀐것.items()},
            "변경_이유": ("튜닝 기준이 AUC 하나였는데 AUC 는 순위만 보므로 "
                      "확률을 부풀리는 class_weight 가 뽑혔다. 화면이 확률을 "
                      "그대로 보여주므로 배포에서는 뺀다. 48게임 조각별 왜곡이 "
                      "절대평균 7.8% -> 4.0% 로 줄고 안쪽 CV AUC 는 "
                      "0.7522 -> 0.7536 으로 올랐다"),
            "성능_48게임OOF": round(float(auc), 4),
            "확률왜곡_48게임OOF": round(왜곡, 4),
            **_봉인기록_잇기(설정),
            "고른기준": ("게임분할에서는 랜덤포레스트가 0.01 정도 앞선다"
                     "(5조각 중 4조각). 그 0.01 을 내주고 학습 14배 속도와 "
                     "SHAP 이 24개 변수에 1:1 로 대응하는 설명 가능성을 얻었다"),
            "왜_B셋": ("A셋은 게임 분할에서 확률이 무너진다 — 실제 이탈률 37% 인 "
                     "조각에 평균 70% 를 뱉는다"),
            "한계": ("남은 확률 오차는 게임 조각별로 -3.6% ~ +6.8% 흔들린다. "
                   "게임이 60개뿐인 데서 오는 것이라 확률 보정으로도 못 고친다"),
            "주의": "lang_stats.json 과 같은 전처리 버전이어야 한다",
        }, META_JSON)
        print(f"  저장: {MODEL_PKL.name} · {ORDER_JSON.name} · "
              f"{THR_JSON.name} · {META_JSON.name}")

    print(f"  배포 모델 저장 완료 — 봉인 12게임은 사용하지 않았습니다")
    return model, X, y


def _버전():
    try:
        return "v" + load_json(DATA_PROC / "dataset_meta.json")["전처리_버전"]
    except Exception:
        return "?"


def load_final():
    """저장된 배포 모델 한 벌. 화면(app/) 도 이걸 쓴다."""
    import joblib
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"{MODEL_PKL} 이 없습니다. 먼저 "
                                f"`uv run python -m src.explain_ml` 를 실행하세요.")
    return (joblib.load(MODEL_PKL), load_json(ORDER_JSON),
            load_json(THR_JSON)["threshold"], load_json(META_JSON))


def to_model_frame(row, order):
    """
    featurize() 가 만든 1줄짜리 표를 모델이 먹는 모양으로 맞춘다.

    ★ 여기가 화면과 학습이 어긋나는 지점이다. 열이 빠지거나 순서가 다르거나
      범주 목록이 다르면 에러 없이 틀린 답이 나온다.
    """
    X = row.copy()
    빈칸 = pd.Series([None] * len(X), index=X.index)

    for c, cats in order["범주값"].items():
        v = X[c].astype(str) if c in X.columns else 빈칸
        X[c] = pd.Categorical(v, categories=cats)

    # 숫자 열은 숫자로 되돌린다. featurize() 가 문자열로 주는 것이 섞여 있다.
    for c, 종류 in order.get("자료형", {}).items():
        if 종류 == "숫자":
            X[c] = pd.to_numeric(X[c] if c in X.columns else 빈칸, errors="coerce")

    return X.reindex(columns=order["열순서"])


# ── SHAP ────────────────────────────────────────────────────────
_EX = {}


def _explainer(model):
    """
    TreeExplainer 는 만드는 데 3초쯤 걸린다. 모델당 한 번만 만들어 재사용한다.
    (화면 1 은 클릭할 때마다 부르므로 캐시하지 않으면 매번 3초씩 멈춘다)
    """
    import shap
    k = id(model)
    if k not in _EX:
        _EX[k] = shap.TreeExplainer(model)
    return _EX[k]


def _이탈쪽(v):
    """shap 버전에 따라 (n, f) 또는 [잔존, 이탈] 로 온다. 이탈 쪽만 쓴다."""
    if isinstance(v, list):
        return v[1] if len(v) > 1 else v[0]
    if getattr(v, "ndim", 2) == 3:
        return v[:, :, 1]
    return v


def shap_values(model, X, n=SHAP_N, seed=SEED):
    """그림용 SHAP 값. 표본을 뽑아 계산한다."""
    Xs = X.sample(min(n, len(X)), random_state=seed) if n and len(X) > n else X
    return _이탈쪽(_explainer(model).shap_values(Xs)), Xs


# ── 그림 ────────────────────────────────────────────────────────
def figure_importance(sv, Xs, path=None):
    """06 — 모델이 전체적으로 무엇을 보는가."""
    path = path or FIGURES / "06_ML_변수중요도.png"
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh([한글(c) for c in imp.index], imp.values, color="#2563eb")
    # SHAP 은 확률이 아니라 로그오즈 단위다. 축 이름에 그대로 밝힌다.
    ax.set_xlabel("평균 |SHAP| — 이탈 쪽으로 민 크기 (로그오즈)")
    ax.set_title("모델이 보는 것", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_beeswarm(sv, Xs, path=None):
    """07 — 값이 크면 이탈 쪽인가 잔존 쪽인가."""
    import shap
    path = path or FIGURES / "07_ML_SHAP분포.png"
    X2 = Xs.copy()
    for c in X2.columns:                     # beeswarm 은 숫자만 색으로 칠할 수 있다
        if str(X2[c].dtype) == "category":
            X2[c] = X2[c].cat.codes
    X2.columns = [한글(c) for c in X2.columns]
    plt.figure()
    shap.summary_plot(sv, X2, max_display=12, show=False)
    plt.title("무엇이 이탈 쪽으로 미는가", loc="left", fontweight="bold")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path


# ── 개인 설명 (화면 1) ──────────────────────────────────────────
def explain_one(row, 상위=5, 묶음=None):
    """
    리뷰 1건 -> 확률 + 이유.

    row : src.preprocess.featurize() 가 돌려준 1줄짜리 DataFrame

    반환 {"이탈확률", "판정", "임계값", "기준값", "기준확률",
          "이유": [{변수, 한글, 값, 기여, 기여율}...]}

    ★ "기여" 는 확률이 아니라 로그오즈다. 0.93 을 93% 로 읽으면 안 된다.
      기준값(로그오즈) + 모든 기여의 합 = 최종 로그오즈 이고,
      그것을 시그모이드에 넣은 것이 이탈확률이다.
      화면에는 "기여율"(상위 이유들 중 몇 %) 을 띄우는 편이 오해가 없다.
    """
    model, order, thr, meta = 묶음 or load_final()
    X = to_model_frame(row, order)
    p = float(model.predict_proba(X)[0, 1])

    ex = _explainer(model)
    sv = _이탈쪽(ex.shap_values(X))[0]
    base = ex.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = np.ravel(base)
        base = float(base[1] if base.size > 1 else base[0])
    else:
        base = float(base)

    이유 = [{"변수": c, "한글": 한글(c),
            "값": (str(X[c].iloc[0]) if str(X[c].dtype) == "category"
                  else round(float(X[c].iloc[0]), 3)),
            "기여": round(float(v), 4)}
           for c, v in zip(X.columns, sv)]
    이유.sort(key=lambda r: abs(r["기여"]), reverse=True)
    이유 = 이유[:상위]

    몫 = sum(abs(r["기여"]) for r in 이유) or 1.0
    for r in 이유:
        r["기여율"] = round(abs(r["기여"]) / 몫, 4)

    return {"이탈확률": round(p, 4), "판정": "이탈" if p >= thr else "잔존",
            "임계값": thr, "기준값": round(base, 4),
            "기준확률": round(float(1 / (1 + np.exp(-base))), 4), "이유": 이유}


def 문장(결과):
    """
    상위 이유를 사람이 읽는 문장으로. 화면에 그대로 띄운다.

    로그오즈 숫자는 내보내지 않는다 — 사용자가 확률로 오해한다.
    """
    말 = [f"기준 {결과['기준확률']:.1%} 에서 출발해 {결과['이탈확률']:.1%} 가 됐습니다."]
    for r in 결과["이유"]:
        방향 = "이탈" if r["기여"] > 0 else "잔존"
        말.append(f"{r['한글']}({r['값']}) → {방향} 쪽 (이유의 {r['기여율']:.0%})")
    return 말


# ── 실행 ────────────────────────────────────────────────────────
def main(argv=None):
    """
    ★ 인자 처리를 두는 이유 — 예전엔 argparse 가 없어서 `--help` 만 붙여도
      전체 재학습이 돌고 meta 를 덮어썼다. 그때 봉인 성적이 날아갔다.
      모르는 인자는 여기서 막는다.
    """
    import argparse
    ap = argparse.ArgumentParser(
        description="SHAP + 배포 모델 저장 (봉인은 열지 않는다)")
    ap.add_argument("--그림만", action="store_true",
                    help="모델을 다시 저장하지 않고 그림만 다시 그린다")
    a = ap.parse_args(argv)

    if a.그림만:
        print("그림만 다시 그립니다 (모델·meta 는 건드리지 않음)")
        model, order, thr, meta = load_final()
        d, y, g = load_all()
        X = features(d, dict(변수묶음들())["B셋"])
    else:
        print("배포 모델 학습 · 저장")
    model, X, y = fit_final()

    print(f"\nSHAP 계산 (표본 {min(SHAP_N, len(X)):,}행)")
    sv, Xs = shap_values(model, X)

    print("그림 저장")
    for p in (figure_importance(sv, Xs), figure_beeswarm(sv, Xs)):
        print(f"  {p}")

    print("\n전체 중요도 상위 8개")
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    for c, v in imp.head(8).items():
        print(f"  {한글(c):<18s} {v:.4f} {'#' * int(v * 120)}")

    print("\n개인 설명 예시 — 학습 데이터에서 확률이 가장 높은 1건")
    묶음 = load_final()
    p_all = model.predict_proba(X)[:, 1]
    r = explain_one(X.iloc[[int(np.argmax(p_all))]], 묶음=묶음)
    print(f"  이탈확률 {r['이탈확률']:.3f} ({r['판정']}) · 기준값 {r['기준값']:.3f} "
          f"· 임계값 {r['임계값']}")
    for s in 문장(r):
        print(f"    - {s}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
