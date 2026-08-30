# -*- coding: utf-8 -*-
"""
화면 2 「사람 vs 모델」 이 쓰는 DB 어댑터 (TiDB · MySQL 호환).

★ 이 파일의 제1원칙 — **DB 때문에 화면이 죽으면 안 된다.**
  발표장에서 와이파이가 끊기거나 TiDB 가 응답하지 않아도
  퀴즈는 끝까지 돌아가야 한다. 그래서 모든 함수가
  실패하면 조용히 None / False 를 돌려주고, 예외를 위로 올리지 않는다.
  화면은 "저장됨 / 저장 안 됨" 만 보고 판단한다.

접속정보
  .streamlit/secrets.toml 에서 읽는다. 코드에 절대 적지 않는다.
  이 파일은 .gitignore 에 걸려 있다. 공유는 secrets.toml.example 로 한다.

      [tidb]
      host     = "gateway01....tidbcloud.com"
      port     = 4000
      user     = "xxxxx.root"
      password = "..."
      database = "steam_churn"

TiDB Serverless 는 TLS 가 필수라 ssl 설정을 켜서 붙는다.

테이블은 db/schema.sql 참고. 컬럼을 바꾸면 여기 SQL 도 같이 바꿔야 한다.
"""
from datetime import datetime

import streamlit as st


# ── 연결 ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _engine():
    """접속 엔진. 못 붙으면 None 을 돌려주고 화면은 DB 없이 간다.

    cache_resource 라 프로세스당 한 번만 붙는다.
    실패도 캐시되므로, 접속정보를 고친 뒤에는 서버를 다시 띄워야 한다.
    """
    try:
        cfg = st.secrets["tidb"]
    except Exception:
        return None                      # secrets.toml 이 없다 — 정상 상황

    try:
        import certifi
        from sqlalchemy import create_engine

        # ★ SSL 은 반드시 ca 를 줘야 한다.
        #   {"ssl_mode": "VERIFY_IDENTITY"} 는 mysqlclient 용 키라 PyMySQL 이 무시한다.
        #   무시되면 check_hostname=False · verify_mode=CERT_NONE 이 되어
        #   암호화는 되지만 상대가 진짜 TiDB 인지 검증하지 않는다.
        #   ca 를 주면 check_hostname=True · CERT_REQUIRED 로 올라간다.
        #   (윈도우는 시스템 CA 경로가 제각각이라 certifi 번들을 쓴다)
        # secrets 에 ca 를 적어두면 그걸 쓰고, 없으면 certifi 번들을 쓴다.
        # (TiDB 콘솔이 안내하는 CA 파일을 받아 쓰고 싶을 때를 위한 통로)
        eng = create_engine(
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg.get('port', 4000)}/{cfg['database']}",
            connect_args={"ssl": {"ca": cfg.get("ca") or certifi.where()},
                          "connect_timeout": 5},   # 발표 중 5초 넘게 매달리지 않는다
            pool_pre_ping=True,          # 끊긴 연결을 미리 걸러낸다
            pool_recycle=280,            # TiDB 가 유휴 연결을 끊기 전에 갈아끼운다
        )
        with eng.connect():              # 진짜 붙는지 여기서 확인
            pass
        return eng
    except Exception:
        return None


def 연결됨() -> bool:
    return _engine() is not None


# ── 쓰기 ────────────────────────────────────────────────────────
def 저장(참가자: dict, 사람점수: int, 모델점수: int, 로그: list) -> bool:
    """퀴즈 1회를 저장한다. 성공하면 True.

    세션 한 줄과 문제별 여러 줄을 **한 트랜잭션**으로 넣는다.
    중간에 끊기면 반쪽짜리 기록이 남아 통계가 틀어지기 때문이다.
    """
    eng = _engine()
    if eng is None:
        return False

    from sqlalchemy import text
    try:
        with eng.begin() as conn:        # begin() 이라 예외 시 자동 롤백
            r = conn.execute(text("""
                INSERT INTO quiz_session
                    (played_at, nickname, age_group, play_hours, steam_years,
                     n_questions, human_score, model_score)
                VALUES
                    (:played_at, :nickname, :age_group, :play_hours, :steam_years,
                     :n, :human, :model)
            """), {
                "played_at": datetime.now(),
                "nickname":    참가자.get("닉네임") or None,
                "age_group":   참가자.get("연령대") or None,
                "play_hours":  참가자.get("게임시간") or None,
                "steam_years": 참가자.get("스팀경력") or None,
                "n": len(로그), "human": 사람점수, "model": 모델점수,
            })
            sid = r.lastrowid

            if 로그:
                conn.execute(text("""
                    INSERT INTO quiz_answer
                        (session_id, q_no, game, human_pick, model_pick, truth, model_prob)
                    VALUES
                        (:sid, :q, :game, :human, :model, :truth, :prob)
                """), [{
                    "sid": sid, "q": i + 1, "game": r_["게임"],
                    "human": int(r_["당신"] == "그만뒀다"),
                    "model": int(r_["모델"] == "그만뒀다"),
                    "truth": int(r_["정답"] == "그만뒀다"),
                    "prob":  r_.get("_확률"),
                } for i, r_ in enumerate(로그)])
        return True
    except Exception:
        return False


# ── 읽기 ────────────────────────────────────────────────────────
def 통계(_새로고침=0):
    """지금까지 몇 명이 풀었고 평균이 몇 점인지.

    _새로고침 은 캐시를 깨기 위한 인자다. 방금 내 기록을 저장했는데
    화면에 안 잡히면 이상하므로, 저장 후에는 값을 바꿔서 다시 부른다.
    """
    return _통계(_새로고침)


@st.cache_data(ttl=30, show_spinner=False)
def _통계(_새로고침):
    eng = _engine()
    if eng is None:
        return None
    from sqlalchemy import text
    try:
        with eng.connect() as conn:
            row = conn.execute(text("""
                SELECT COUNT(*)         AS n,
                       AVG(human_score) AS human_avg,
                       AVG(model_score) AS model_avg
                FROM quiz_session
            """)).mappings().first()
        if not row or not row["n"]:
            return None
        return {"참가자수": int(row["n"]),
                "사람평균": float(row["human_avg"]),
                "모델평균": float(row["model_avg"])}
    except Exception:
        return None
