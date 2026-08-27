"""
프로젝트 공통 설정.

경로 / 인코딩 / 랜덤시드 / 누수 금지 목록을 여기 한 곳에만 적는다.
팀원 전원이 이 파일 하나만 import 해서 쓴다.
"""

import json
from pathlib import Path

import pandas as pd

# ── 경로 ────────────────────────────────────────────────
# __file__ = .../SKN35-2nd-2Team/src/config.py
# parent   = .../SKN35-2nd-2Team/src
# parent.parent = .../SKN35-2nd-2Team   ← 레포 루트
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
EMBEDDINGS = ROOT / "data" / "embeddings"
MODELS     = ROOT / "models"
RESULTS    = ROOT / "results"
FIGURES    = ROOT / "reports" / "figures"

RAW_CSV     = DATA_RAW / "steam_raw.csv"      # 스팀 원본 (깃에 없음 — 수집 담당에게 받는다)
DATASET     = DATA_PROC / "dataset.csv"       # 전처리 담당이 준 학습용 표
LANG_STATS  = DATA_PROC / "lang_stats.json"   # 언어별 리뷰 길이 기준 — preprocess 가 만든다
                                              #   화면에서 리뷰 1건 변환할 때도 필요하므로 커밋한다
RESULTS_CSV = RESULTS / "results.csv"         # 실험 기록이 쌓이는 곳
EMB_NPY     = EMBEDDINGS / "review_emb_en.npy"  # 영어 리뷰 임베딩 (한 번 뽑아 재사용)

# 딥러닝 임베딩 모델 — 영어 전용, 가볍다 (384차원)
#   다국어를 섞으면 모델이 글은 안 읽고 "이건 러시아어 → 그 게임" 지름길을 탄다.
#   영어만 써도 이탈률 차이가 0.9%p 뿐이라 영어로 간다. (A 가이드 결정)
EMB_MODEL = "all-MiniLM-L6-v2"
EMB_DIM   = 384

# ── 인코딩 ──────────────────────────────────────────────
# 한국어 윈도우 파이썬은 기본이 cp949 라서 명시하지 않으면 한글이 깨진다.
# ★★ 규칙: CSV 는 utf-8-sig, JSON 은 utf-8. 섞으면 반드시 터진다. ★★
#
#   CSV  → utf-8-sig  맨 앞에 BOM 표식이 있어야 엑셀이 한글을 알아본다.
#                     안 붙이면 팀원이 엑셀로 열 때 전부 깨져 보인다.
#
#   JSON → utf-8      BOM 이 붙으면 json.load 가 그 자리에서 죽는다.
#                       JSONDecodeError: Unexpected UTF-8 BOM
#                     이 프로젝트에서 실제로 두 번 당했다
#                     (preprocess 의 lang_stats.json, embed 의 meta.json).
#
#   → 손으로 encoding= 을 적지 말고 아래 save_json / save_csv 를 쓸 것.
ENC_READ  = "utf-8"
ENC_CSV   = "utf-8-sig"   # CSV 쓰기 전용
ENC_JSON  = "utf-8"       # JSON 쓰기 전용 — 절대 sig 를 붙이지 않는다
ENC_WRITE = ENC_CSV       # 옛 이름 (CSV 를 가리킨다)

# ── 랜덤시드 ────────────────────────────────────────────
SEED = 42

# ── 정답을 미리 보는 컬럼 (모델 입력 금지) ──────────────
FORBIDDEN = [
    "playtime_forever_min",   # 라벨을 만든 원천
    "playtime_2weeks_min",    # 리뷰 이후 2주
    "last_played_ts",         # 리뷰 이후 마지막 접속 — 이거 하나로 AUC 0.82
    "votes_up",               # 리뷰 쓴 순간엔 아직 아무도 안 봤음
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "updated_ts",             # 리뷰를 나중에 수정한 시각
    "extra",                  # 라벨 그 자체
    "hours_total",            # 라벨 계산 중간값 — 넣으면 AUC 0.99
    "hours_after",            # 라벨 그 자체를 시간으로 바꾼 것
]


def check_leakage(X):
    """모델에 넣기 직전 호출. 금지 컬럼이 하나라도 있으면 멈춘다.

    DataFrame 도 컬럼 이름 리스트도 받는다.
    """
    cols = X.columns if hasattr(X, "columns") else X
    hit = [c for c in cols if c in FORBIDDEN]
    if hit:
        raise ValueError(f"누수 컬럼이 입력에 섞였습니다: {hit}")
    return X


def load_dataset(path=None):
    """학습용 표를 읽는다. 팀원 둘 다 이 함수만 쓴다."""
    path = Path(path) if path else DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다.\n"
            f"전처리 담당에게 받은 dataset.csv 를 {DATA_PROC} 안에 넣으세요."
        )
    return pd.read_csv(path, encoding=ENC_READ, low_memory=False)


def save_csv(df, path):
    """표를 저장한다. 엑셀에서 한글이 안 깨지는 형식(utf-8-sig)으로."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=ENC_CSV)
    return path


def save_json(obj, path):
    """JSON 을 저장한다. BOM 을 붙이지 않는다 (붙이면 json.load 가 깨진다)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=ENC_JSON) as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


def load_json(path):
    """JSON 을 읽는다. 옛 파일에 BOM 이 남아 있어도 utf-8-sig 로 걷어낸다."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


if __name__ == "__main__":
    # python src/config.py 로 실행하면 설정이 제대로 잡혔는지 확인된다
    print("ROOT      :", ROOT)
    print("DATASET   :", DATASET)
    print("있는가?   :", DATASET.exists())
    print("SEED      :", SEED)
    print("금지 컬럼 :", len(FORBIDDEN), "개")
