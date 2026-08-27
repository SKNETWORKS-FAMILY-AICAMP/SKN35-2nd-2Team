# -*- coding: utf-8 -*-
"""
리뷰 글 → 숫자벡터(임베딩).

컴퓨터는 글자를 못 읽으므로 문장을 384개 숫자로 바꾼다.
뜻이 비슷한 문장이 비슷한 위치에 놓이도록 학습된 모델을 쓴다.

    "this game is fun"        -> [ 0.03, -0.12, 0.44, ... ]
    "really enjoyed it"       -> [ 0.05, -0.10, 0.41, ... ]   가까움
    "refunded, waste of time" -> [-0.31,  0.22, -0.08, ... ]  멂

★ 한 번만 뽑아서 .npy 로 저장한다.
  모델 실험할 때마다 다시 뽑으면 하루가 날아간다. 저장해두면 이후엔 1초.

★ 누수가 아니다.
  임베딩은 라벨(churn)을 전혀 보지 않는다. 글자만 보고 위치를 정한다.
  그래서 전체 데이터에서 한 번에 뽑아도 된다.
  (반대로 StandardScaler 는 학습 데이터에서만 fit 해야 한다 — 그건 진짜 누수)

실행 (레포 루트에서)
    uv run python -m src.embed
"""
import json

import numpy as np

from src.config import EMB_DIM, EMB_MODEL, EMB_NPY
from src.evaluate import load_english

BATCH = 256
_META = EMB_NPY.with_suffix(".meta.json")


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"          # 맥 GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build():
    """영어 리뷰를 임베딩해 .npy 로 저장한다."""
    from sentence_transformers import SentenceTransformer

    d, _, _ = load_english()
    texts = d["review"].fillna("").astype(str).tolist()
    blank = [i for i, t in enumerate(texts) if not t.strip()]

    dev = _device()
    print(f"모델 {EMB_MODEL} | 장치 {dev} | 문장 {len(texts):,}개")
    print(f"  빈 텍스트 {len(blank)}건 → 0벡터로 채움 (has_text 컬럼이 이미 알려줌)")

    model = SentenceTransformer(EMB_MODEL, device=dev)
    emb = model.encode(
        texts, batch_size=BATCH, convert_to_numpy=True,
        normalize_embeddings=True,      # 길이를 1로 맞춤 — 스케일이 고르다
        show_progress_bar=True,
    ).astype(np.float32)

    emb[blank] = 0.0                     # 빈 글은 0벡터

    EMB_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_NPY, emb)

    # 행 순서가 어긋나면 조용히 틀린다 → 대조용 지문을 같이 남긴다
    meta = {
        "모델": EMB_MODEL, "차원": int(emb.shape[1]), "행수": int(emb.shape[0]),
        "빈텍스트": len(blank),
        "첫_recommendationid": int(d.recommendationid.iloc[0]),
        "끝_recommendationid": int(d.recommendationid.iloc[-1]),
    }
    json.dump(meta, open(_META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)  # JSON 은 BOM 금지

    print(f"\n저장 {EMB_NPY}")
    print(f"  {emb.shape} · {emb.nbytes / 1024**2:.0f}MB")
    return emb


def load(d=None):
    """
    저장된 임베딩을 읽는다.

    ★ 행 순서가 지금 데이터와 같은지 확인한다.
      어긋난 채로 학습하면 에러 없이 엉뚱한 글이 붙는다.
    """
    if not EMB_NPY.exists():
        raise FileNotFoundError(
            f"{EMB_NPY} 가 없습니다.  먼저: uv run python -m src.embed")
    emb = np.load(EMB_NPY)
    if d is None:
        d, _, _ = load_english()
    m = json.load(open(_META, encoding="utf-8"))
    if (len(d) != m["행수"]
            or int(d.recommendationid.iloc[0]) != m["첫_recommendationid"]
            or int(d.recommendationid.iloc[-1]) != m["끝_recommendationid"]):
        raise ValueError(
            "임베딩과 데이터의 행이 어긋납니다. dataset.csv 가 바뀌었으면 "
            "다시 뽑으세요: uv run python -m src.embed")
    assert emb.shape[1] == EMB_DIM
    return emb


if __name__ == "__main__":
    build()
