# -*- coding: utf-8 -*-
"""
TiDB 접속 확인 + 테이블 만들기 + 동작 시험을 한 번에.

.streamlit/secrets.toml 을 읽어서
  1. 붙는지 확인 (SSL 검증 포함)
  2. db/schema.sql 을 실행해 테이블 생성
  3. 시험 데이터를 넣었다 지워서 화면과 같은 SQL 이 도는지 확인

실행
    uv run python db/setup_tidb.py            # 확인 + 생성
    uv run python db/setup_tidb.py --확인만    # 붙는지만 보고 아무것도 안 만든다
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

여기 = Path(__file__).resolve().parent
SECRETS = 여기.parent / ".streamlit" / "secrets.toml"
SCHEMA = 여기 / "schema.sql"
확인만 = "--확인만" in sys.argv


def 설정읽기():
    if not SECRETS.exists():
        sys.exit(
            f"접속정보가 없습니다 — {SECRETS}\n\n"
            f"  1. .streamlit/secrets.toml.example 을 secrets.toml 로 복사\n"
            f"  2. TiDB Cloud 콘솔 > Connect 에서 값을 복사해 채우기\n"
            f"  (secrets.toml 은 .gitignore 에 걸려 있어 커밋되지 않습니다)")
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    cfg = tomllib.loads(SECRETS.read_text(encoding="utf-8")).get("tidb")
    if not cfg:
        sys.exit("secrets.toml 에 [tidb] 항목이 없습니다.")
    빈칸 = [k for k in ("host", "user", "password", "database") if not cfg.get(k)]
    if 빈칸:
        sys.exit(f"secrets.toml 의 값이 비어 있습니다: {', '.join(빈칸)}")
    return cfg


def main():
    cfg = 설정읽기()
    import certifi
    from sqlalchemy import create_engine, text

    print(f"접속 시도 — {cfg['user']}@{cfg['host']}:{cfg.get('port', 4000)}"
          f"/{cfg['database']}")

    연결설정 = {"ssl": {"ca": cfg.get("ca") or certifi.where()},
              "connect_timeout": 10}
    주소 = (f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
          f"@{cfg['host']}:{cfg.get('port', 4000)}")

    # 데이터베이스가 아직 없으면 만들어 준다.
    # (새 클러스터를 만들면 sys/test 밖에 없어서, 콘솔에서 CREATE DATABASE 를
    #  따로 하지 않으면 여기서 'Unknown database' 로 죽는다)
    try:
        임시 = create_engine(주소, connect_args=연결설정)
        with 임시.connect() as c:
            있음 = c.execute(text("SHOW DATABASES LIKE :d"),
                            {"d": cfg["database"]}).first()
        if not 있음:
            with 임시.connect() as c:
                c.execute(text(f"CREATE DATABASE `{cfg['database']}`"))
            print(f"  ✅ 데이터베이스 생성 — {cfg['database']}")
        임시.dispose()
    except Exception:
        pass                             # 못 만들면 아래에서 제대로 된 오류가 난다

    eng = create_engine(f"{주소}/{cfg['database']}", connect_args=연결설정)
    try:
        with eng.connect() as c:
            버전 = c.execute(text("SELECT VERSION()")).scalar()
            암호 = c.execute(text("SHOW STATUS LIKE 'Ssl_cipher'")).mappings().first()
    except Exception as e:
        sys.exit(f"\n접속 실패 — {type(e).__name__}\n  {e}\n\n"
                 f"자주 나오는 원인\n"
                 f"  · 비밀번호 오타 (TiDB 는 콘솔에서 새로 만들어야 다시 볼 수 있습니다)\n"
                 f"  · database 이름이 아직 없음 → 콘솔 SQL Editor 에서\n"
                 f"      CREATE DATABASE {cfg['database']};\n"
                 f"  · 회사/학원 방화벽이 4000 포트를 막음\n"
                 f"  · 'certificate verify failed' 라고 나오면 인증서 검증 실패다.\n"
                 f"      TiDB 콘솔 Connect 창의 'Download the CA cert' 로 파일을 받아\n"
                 f"      secrets.toml 에 한 줄 추가하세요 —  ca = \"C:/받은경로/ca.pem\"")

    print(f"  ✅ 연결됨 · {버전}")
    print(f"  ✅ 암호화 · {(암호 or {}).get('Value') or '(확인 불가)'}")

    if 확인만:
        print("\n--확인만 이라 아무것도 만들지 않았습니다.")
        return

    # ── 스키마 ────────────────────────────────────────────────
    print(f"\n스키마 실행 — {SCHEMA.name}")
    문장들 = [s.strip() for s in SCHEMA.read_text(encoding="utf-8").split(";")
            if s.strip() and not all(l.strip().startswith("--") or not l.strip()
                                     for l in s.strip().splitlines())]
    with eng.begin() as c:
        for s in 문장들:
            c.execute(text(s))
            이름 = next((w for w in s.split() if w.startswith("quiz_")), "?")
            print(f"  ✅ {이름}")

    with eng.connect() as c:
        표 = [r[0] for r in c.execute(text("SHOW TABLES"))]
    print(f"  현재 테이블 — {', '.join(표) or '(없음)'}")

    # ── 동작 시험 ─────────────────────────────────────────────
    # 화면(app/_db.py)이 쓰는 것과 같은 SQL 을 넣었다 지운다.
    print("\n동작 시험 (넣고 → 읽고 → 지운다)")
    from datetime import datetime
    with eng.begin() as c:
        r = c.execute(text("""
            INSERT INTO quiz_session
                (played_at, nickname, age_group, play_hours, steam_years,
                 n_questions, human_score, model_score)
            VALUES (:t, '__setup_test__', '20대', '5~20시간', '1~5년', 2, 1, 2)
        """), {"t": datetime.now()})
        sid = r.lastrowid
        c.execute(text("""
            INSERT INTO quiz_answer
                (session_id, q_no, game, human_pick, model_pick, truth, model_prob)
            VALUES (:sid, 1, '__setup_test__', 0, 1, 1, 0.7800)
        """), {"sid": sid})
    with eng.connect() as c:
        n = c.execute(text("SELECT COUNT(*) FROM quiz_answer WHERE session_id=:s"),
                      {"s": sid}).scalar()
    print(f"  ✅ 세션 1건 + 문제 {n}건 저장됨 (session_id={sid})")

    with eng.begin() as c:
        c.execute(text("DELETE FROM quiz_session WHERE id=:s"), {"s": sid})
    with eng.connect() as c:
        남음 = c.execute(text("SELECT COUNT(*) FROM quiz_answer WHERE session_id=:s"),
                       {"s": sid}).scalar()
    print(f"  ✅ 시험 데이터 삭제 · 딸린 답도 함께 지워짐(ON DELETE CASCADE) → 남은 행 {남음}")

    with eng.connect() as c:
        s = c.execute(text("""SELECT COUNT(*) n, AVG(human_score) h, AVG(model_score) m
                              FROM quiz_session""")).mappings().first()
    print(f"\n누적 통계 조회 — 참가자 {s['n']}명")
    print("\n끝났습니다. 이제 화면을 다시 띄우면 저장이 붙습니다.")
    print("    uv run streamlit run app/main.py")


if __name__ == "__main__":
    main()
