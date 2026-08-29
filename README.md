# 스팀 리뷰 이탈 예측

**주제** 게임 리뷰를 쓴 그 순간, 이 사람이 게임을 계속할지 그만둘지 맞히기
**발표** 2026-08-31

---

## 환경 세팅

패키지 관리는 [uv](https://docs.astral.sh/uv/)를 씁니다. **맥·윈도우 명령이 같습니다.**

```bash
uv sync
```

이 한 줄이면 파이썬 3.12와 모든 패키지가 `uv.lock`에 박힌 **정확히 같은 버전**으로 깔립니다.
가상환경을 activate 할 필요 없이 `uv run`을 앞에 붙여서 실행합니다.

```bash
uv run python src/config.py     # 설정 확인
uv run jupyter lab              # 노트북
```

| 상황 | 명령 |
|---|---|
| uv가 없다 (맥) | `brew install uv` |
| uv가 없다 (윈도우) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| 패키지 추가 | `uv add 패키지명` — **`pip install` 금지** (lock에 안 남아 팀원이 못 받음) |
| 맥에서 xgboost 오류 | `brew install libomp` (맥 전용, 윈도우는 불필요) |

```bash
uv run streamlit run app/main.py     # 화면 띄우기
```

> **Streamlit Cloud 배포 시:** 클라우드는 `uv.lock`을 못 읽습니다. 배포 직전에
> `uv export --no-hashes --format requirements-txt > requirements.txt` 로 뽑아서 올리세요.
> 평소에는 만들어두지 않습니다 — `pyproject.toml`과 어긋나기 때문입니다.

---

## 폴더 구조

```
SKN35-2nd-2Team/
│
├── pyproject.toml            # 필요한 패키지 목록                    → 커밋
├── uv.lock                   # 패키지 버전 고정 (팀 전원 동일)       → 커밋 (반드시)
├── .python-version           # 파이썬 3.12 고정                      → 커밋
├── .gitattributes            # 줄바꿈 LF 통일 (맥/윈도우 충돌 방지)
├── .gitignore
│
├── src/                      # 코드. 영어·소문자 — import 해야 하므로
│   ├── __init__.py           #   빈 파일. 있어야 from src.* 가 됨
│   ├── config.py             #   ★ 경로 · 인코딩 · 시드 · 누수금지목록 · 모델이름
│   │                         #     파일 위치는 전부 여기서만 정한다
│   │
│   ├── preprocess.py         #   ★ 원본 → 학습용 표. 라벨을 만드는 유일한 곳
│   │                         #     uv run python -m src.preprocess
│   │                         #     dataset.csv · lang_stats.json · dataset_meta.json 생성
│   │
│   ├── evaluate.py           #   채점표. 모든 모델이 이 함수로만 채점된다
│   │                         #     A셋/B셋 x 랜덤/게임분할 = 4칸
│   │                         #     결과를 results/results.csv 에 자동 기록
│   │
│   ├── train_ml.py           #   머신러닝 — 로지스틱 · RF · XGBoost · LightGBM
│   ├── embed.py              #   ★ 리뷰 글 → 숫자벡터. 오래 걸리므로 한 번만
│   │                         #     결과를 data/embeddings/ 에 .npy 로 저장
│   ├── train_dl.py           #   딥러닝 — (1) MLP 통제군 (숫자만)
│   │                         #            (2) 임베딩+MLP (숫자+글)
│   │                         #     (1)은 '글의 효과'를 분리하려는 대조군이다
│   ├── explain_dl.py         #   SHAP (딥러닝 담당) — 화면 1의 "왜 그렇게 판단했나"
│   ├── explain_ml.py         #   SHAP (머신러닝 담당)
│   │                         #     ※ 같은 파일을 둘이 고치면 머지에서 덮인다
│   ├── predict.py            #   저장된 모델로 리뷰 1건 예측 ← app/ 이 불러 씀
│   └── db.py                 #   DB 적재 (DB 담당)
│
├── app/                      # Streamlit — 배포용
│   ├── main.py               #   첫 화면 · 프로젝트 소개
│   ├── pages/                #   파일명이 곧 메뉴 이름. 숫자로 순서 지정
│   │   ├── 1_작별인사_판별기.py      #  리뷰 붙여넣으면 이탈 확률 + 근거
│   │   ├── 2_사람_vs_모델.py         #  10장 찍어보고 모델과 점수 비교
│   │   ├── 3_게임_붙잡기_랭킹.py     #  게임 60개 이탈률 · 초반형/후반형
│   │   └── 4_처음보는_게임_테스트.py  #  ★ 게임 이름을 외운 게 아님을 증명
│   └── .streamlit/
│       └── config.toml       #   테마 · 업로드 용량 설정
│
├── data/
│   ├── raw/                  # 스팀 원본 139,667행         → 깃에 안 올림 (60MB)
│   │                         #   단, manifest.json(수집 영수증)과
│   │                         #   selected_60.csv(게임 선정 근거)는 올린다
│   ├── processed/            # 전처리 결과
│   │                         #   dataset.csv (64MB)  → 깃에 안 올림
│   │                         #   lang_stats.json     → 커밋 (언어별 리뷰길이 기준)
│   │                         #   dataset_meta.json   → 커밋 (행수·열목록·버전)
│   │                         #   ※ csv 는 코드로 다시 만든다:
│   │                         #     uv run python -m src.preprocess
│   └── embeddings/           # 리뷰 임베딩 .npy            → 깃에 안 올림
│                             #   139,658 x 384 = 수백 MB
│
├── models/                   # 학습된 모델 .pkl / .pt      → 깃에 안 올림
│                             #   최종 제출본 1개만 git add -f 로 강제 추가
│                             #   함께 저장할 것:
│                             #     scaler.pkl        스케일링 기준
│                             #     threshold.json    이탈 판정 임계값
│                             #     feature_order.json ★ 컬럼 순서
│                             #       (없으면 화면에서 조용히 틀린 예측이 나옴)
│
├── results/                  # results.csv 실험 기록       → 커밋
│                             #   "그 0.83 나왔던 설정이 뭐였지" 방지용
│
├── reports/figures/          # 그래프 png (ROC · SHAP 등)  → 커밋
│                             #   문서와 PPT에 그대로 붙인다
│
├── notebooks/                # 탐색용. 개인 소유
│                             #   파일명에 본인 이름 (chaeyeong_eda.ipynb)
│                             #   ※ 노트북은 충돌이 심해 공유 금지. 공유는 .py 로
│
├── db/
│   └── schema.sql            # 테이블 정의 (DB 담당)
│
└── docs/                     # 필수 산출물 4종
                              #   00_제안서.pdf
                              #   01_전처리결과서.md
                              #   02_학습결과서.md      ← 모델 담당
                              #   03_발표자료.pptx
```

### 담당별로 건드리는 곳

| 담당 | 폴더 | 산출물 |
|---|---|---|
| 수집·전처리 | `data/` · `src/preprocess.py` | 01_전처리결과서 |
| **머신러닝·딥러닝** | `src/evaluate.py` `train_ml.py` `embed.py` `train_dl.py` `explain_dl.py` · `models/` · `results/` | **02_학습결과서 + 학습된 모델** |
| Streamlit | `app/` · `src/predict.py` | 배포 주소 |
| DB | `db/` · `src/db.py` | 스키마 |

서로 다른 폴더만 건드리므로 **충돌이 거의 안 납니다.** 공통으로 건드리는 건 `src/config.py` 하나뿐이라, 여기만 조심하면 됩니다.

### 이름 규칙

| 종류 | 규칙 | 이유 |
|---|---|---|
| **코드** | 영어 소문자 (`src/config.py`) | `import 04_전처리` 는 문법 오류 — 파이썬이 못 읽음 |
| **문서·데이터** | 한글·숫자 자유 (`docs/01_전처리결과서.md`) | import 하지 않으므로 상관없음 |

### 경로는 `config.py`에서만

```python
from src.config import load_dataset, check_leakage, SEED

df = load_dataset()        # 인코딩 자동 처리
check_leakage(X)           # 금지 컬럼 섞이면 에러
```

상대경로(`../data/...`)나 절대경로(`C:/Users/...`)를 코드에 직접 쓰면
실행 위치나 사람이 바뀔 때 깨집니다. `config.py`가 레포 루트를 자동으로 찾습니다.

> **⚠️ 레포 루트에서 실행하세요.** `src` 안으로 들어가면 `from src.config` 가 안 됩니다.
> 노트북은 첫 셀에 `%cd <레포 경로>`.

### 인코딩 — 윈도우 필수

한국어 윈도우 파이썬은 기본이 `cp949`라 명시하지 않으면 **한글이 깨집니다.**

```python
pd.read_csv(path, encoding='utf-8')                    # 읽기
df.to_csv(path, index=False, encoding='utf-8-sig')     # 쓰기 (sig = 엑셀 대응)
```

`config.py`의 `load_dataset()` / `save_csv()`를 쓰면 신경 안 써도 됩니다.

---

## 진행 상황

| 단계 | 상태 | 결과 |
|---|---|---|
| 제안서 | 완료 | — |
| 사전조사 (게임 12개 파일럿) | 완료 | 주제 성립 확인 |
| 게임 선정 | 완료 | 5,992개 → **60개** |
| 본 수집 | 완료 | **139,667행** · 30개 언어 · 2026-08-25 07:30 UTC |
| 전처리 | 완료 | **139,658행 × 31열** · 이탈률 41.1% |
| 머신러닝 · 딥러닝 | **진행 중** | — |
| Streamlit 화면 | 대기 | — |

### 전처리 결과 요약

```
원본 139,667행 → 학습용 139,658행 × 31열
이탈률 41.1%
변수묶음 2개 — A셋(게임 이름 포함) / B셋(장르·연도·평가등급만)
분할 2가지 — 랜덤 / 게임 단위
```

기준 모델(참고용)

| 변수묶음 | 랜덤 분할 | 게임 분할 |
|---|---|---|
| A셋 | AUC 0.820 | 0.750 |
| B셋 | 0.818 | 0.751 |

누수 컬럼을 일부러 넣으면 **0.998**이 나옵니다. **0.95를 넘으면 먼저 누수를 의심합니다.**

---

## 다음 할 일

- [ ] `dataset.csv` 받아서 `data/processed/`에 넣기
- [ ] `evaluate.py` — 4칸 채점표 + `results.csv` 자동 기록
- [ ] 기준선(로지스틱)으로 팀원 수치 0.820 재현 → 데이터 검증
- [ ] ML 3종(RF · XGBoost · LightGBM) 기본값으로 표 채우기
- [ ] 튜닝 (**게임 분할 기준으로**) · SHAP
- [ ] 딥러닝 ① MLP · ② 다국어 텍스트 임베딩
- [ ] 최종 모델 저장 → 화면 담당에게 전달
- [ ] 수집 담당에게 `collect.py` · `selected_60.csv` 받아 `src/` · `data/raw/`에 넣기

---

## 팀 규칙

> **수집은 한 사람만 합니다.**
> 스팀은 요청할 때마다 그 사람의 *현재* 플레이 시간을 돌려줍니다.
> 각자 받으면 **정답이 서로 달라져서** 성능 비교가 무의미해집니다.
> 한 명이 받아 파일을 공유하고, 수집 일시를 함께 적습니다.

> **원본 CSV는 아무도 수정하지 않습니다.**
> 전처리 결과는 항상 원본에서 다시 만듭니다. 그래야 코드를 고쳐도 재수집이 필요 없습니다.

> **전처리는 노트북이 아니라 `preprocess.py` 파일로.**
> Streamlit 화면에서 사용자가 붙여넣은 리뷰를 학습 때와 똑같이 변환해야 하는데,
> 노트북 셀에 흩어져 있으면 불러 쓸 수 없습니다.

> **`main` 에 직접 push 하지 않습니다.**
> 역할별 브랜치 — `feat/preprocess` `feat/model` `feat/app` `feat/db`
> 4명이 main에 직접 밀면 하루에 몇 번씩 충돌합니다.

> **정확도(accuracy)는 보고하지 않습니다.**
> 이탈률이 41.1%라 전부 "잔존"이라고만 찍어도 58.9%가 나옵니다.
> 주지표는 **AUC**, 보조로 PR-AUC · F1.

> **실행하는 스크립트는 `if __name__ == '__main__':` 로 감쌉니다.**
> 윈도우에서 `n_jobs=-1` 을 쓰면 이 가드가 없을 때 프로세스가 무한 복제됩니다.
