-- 테이블 정의 (DB 담당)
--
-- ★ 화면 담당이 만든 초안입니다. DB 담당이 검토하고 확정해 주세요.
--   화면 쪽(app/_db.py)은 이 이름 그대로 읽고 씁니다.
--   컬럼을 바꾸시면 app/_db.py 의 INSERT/SELECT 도 같이 바꿔야 합니다.
--
-- 대상 : TiDB (MySQL 8 호환)
-- 용도 : 화면 2 「사람 vs 모델」 퀴즈 참여 기록
--
-- 왜 DB 인가
--   이 화면은 우리 앱에서 유일하게 **데이터를 만들어내는** 곳입니다.
--   나머지 화면은 CSV 를 읽기만 하므로 DB 가 오히려 느려집니다.
--   퀴즈는 여러 사람의 기록을 모아야 의미가 생기므로(누적 평균),
--   파일이나 세션 메모리로는 안 됩니다.
--
-- 개인정보 원칙
--   실명·연락처·이메일은 받지 않습니다. 발표장에서 받은 개인정보를
--   보관할 이유가 없고, 분석에도 쓸모가 없습니다.
--   닉네임은 화면 표시용이며 없어도 됩니다(익명 참여 허용).
--
-- 실행
--   mysql -h <host> -P 4000 -u <user> -p <db> < db/schema.sql

-- ── 참가 1회 = 1행 ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quiz_session (
    id            BIGINT       AUTO_INCREMENT PRIMARY KEY,
    played_at     DATETIME     NOT NULL,

    -- 인적사항 (전부 선택 입력 · 식별정보 아님)
    nickname      VARCHAR(20)  NULL COMMENT '화면 표시용. 비워도 됨',
    age_group     VARCHAR(10)  NULL COMMENT '10대/20대/30대/40대 이상',
    play_hours    VARCHAR(20)  NULL COMMENT '주당 게임 시간대',
    steam_years   VARCHAR(20)  NULL COMMENT '스팀 이용 기간',

    -- 결과
    n_questions   TINYINT      NOT NULL COMMENT '문제 수 (현재 12)',
    human_score   TINYINT      NOT NULL,
    model_score   TINYINT      NOT NULL,

    INDEX idx_played_at (played_at)
) COMMENT '화면 2 퀴즈 참여 1회';

-- ── 문제별 기록 ──────────────────────────────────────────────
--   세션만 저장하면 "사람이 어떤 문제에서 틀리나" 를 못 본다.
--   문제 단위로 남겨야 "짧은 리뷰에서 사람이 특히 못 맞힌다" 같은
--   분석이 가능하다. 발표 이후에 쓸 수 있는 진짜 데이터가 된다.
CREATE TABLE IF NOT EXISTS quiz_answer (
    id            BIGINT       AUTO_INCREMENT PRIMARY KEY,
    session_id    BIGINT       NOT NULL,
    q_no          TINYINT      NOT NULL COMMENT '1..N',

    game          VARCHAR(200) NULL,
    human_pick    TINYINT      NOT NULL COMMENT '0=계속했다 1=그만뒀다',
    model_pick    TINYINT      NOT NULL,
    truth         TINYINT      NOT NULL,
    model_prob    DECIMAL(6,4) NULL COMMENT '모델이 낸 이탈 확률',

    INDEX idx_session (session_id),
    CONSTRAINT fk_answer_session
        FOREIGN KEY (session_id) REFERENCES quiz_session(id)
        ON DELETE CASCADE
) COMMENT '화면 2 퀴즈 문제별 응답';

-- ── 화면이 쓰는 조회 (참고) ──────────────────────────────────
-- 누적 통계 — 화면 2 상단에 띄운다
--   SELECT COUNT(*)          AS n,
--          AVG(human_score)  AS human_avg,
--          AVG(model_score)  AS model_avg
--   FROM quiz_session;
--
-- 사람이 특히 못 맞힌 문제 (발표 이후 분석용)
--   SELECT game,
--          COUNT(*)                                   AS n,
--          AVG(human_pick = truth)                    AS human_acc,
--          AVG(model_pick = truth)                    AS model_acc
--   FROM quiz_answer
--   GROUP BY game
--   ORDER BY human_acc ASC;
