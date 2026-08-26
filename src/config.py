"""
프로젝트 공통 설정.

경로 / 인코딩 / 랜덤시드 / 누수 금지 목록을 여기 한 곳에만 적는다.
팀원 전원이 이 파일 하나만 import 해서 쓴다.
"""

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

DATASET     = DATA_PROC / "dataset.csv"       # 전처리 담당이 준 학습용 표
RESULTS_CSV = RESULTS / "results.csv"         # 실험 기록이 쌓이는 곳
EMB_NPY     = EMBEDDINGS / "review_emb.npy"   # 다국어 임베딩 (한 번 뽑아 재사용)

# 딥러닝 임베딩 모델 — 30개 언어를 한 모델로 처리
EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── 인코딩 ──────────────────────────────────────────────
# 한국어 윈도우 파이썬은 기본이 cp949 라서 명시하지 않으면 한글이 깨진다.
ENC_READ  = "utf-8"
ENC_WRITE = "utf-8-sig"   # sig 를 붙여야 엑셀에서 한글이 안 깨진다

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
]


def check_leakage(X):
    """모델에 넣기 직전 호출. 금지 컬럼이 하나라도 있으면 멈춘다."""
    hit = [c for c in X.columns if c in FORBIDDEN]
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
    """표를 저장한다. 엑셀에서 한글이 안 깨지는 형식으로."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=ENC_WRITE)
    return path


if __name__ == "__main__":
    # python src/config.py 로 실행하면 설정이 제대로 잡혔는지 확인된다
    print("ROOT      :", ROOT)
    print("DATASET   :", DATASET)
    print("있는가?   :", DATASET.exists())
    print("SEED      :", SEED)
    print("금지 컬럼 :", len(FORBIDDEN), "개")
