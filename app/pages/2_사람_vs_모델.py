# -*- coding: utf-8 -*-
"""
사람 vs 모델 — 리뷰를 보고 직접 맞혀본다.

발표에서 관객이 직접 풀어볼 화면이라, 한 문제가 한 화면에 크게 들어오고
정답 공개가 한눈에 보이는 것을 우선했다.
"""
import html

import pandas as pd
import streamlit as st

from app._shared import get_all, page
from app import _db

_, _, CARDS, _, META, _ = get_all()
THR = float(META["임계값"])  # 0.5 가 아니다 — 데이터로 고른 판정 기준
N = len(CARDS)

page("🆚 사람 vs 모델",
     f"리뷰 {N}장을 보고 이 사람이 게임을 계속했을지 맞혀보세요. 모델과 점수를 비교합니다.")

ss = st.session_state
for k, v in [("i", 0), ("me", 0), ("ai", 0), ("log", []), ("shown", False),
             ("참가자", None), ("저장됨", None)]:
    ss.setdefault(k, v)


def 누적통계():
    """지금까지 몇 명이 풀었는지. DB 가 없으면 아무것도 안 그린다.

    이 화면의 주장은 "사람은 이 문제를 잘 못 맞힌다" 인데,
    한 사람의 점수만으로는 그게 우연인지 알 수 없다.
    누적 평균이 쌓여야 주장이 근거를 갖는다. DB 를 쓰는 이유가 이것이다.
    """
    s = _db.통계(ss.get("저장카운터", 0))
    if not s:
        return
    st.markdown(
        f'<div class="dbstat">지금까지 <b>{s["참가자수"]}명</b>이 풀었습니다 &nbsp;·&nbsp; '
        f'사람 평균 <b>{s["사람평균"]:.1f}</b> &nbsp;vs&nbsp; '
        f'모델 평균 <b>{s["모델평균"]:.1f}</b> <span>(각 {N}문제 중)</span></div>',
        unsafe_allow_html=True)


# ── 시작 전 : 참가자 입력 ───────────────────────────────────────
# 실명·연락처는 받지 않는다. 발표장에서 받은 개인정보를 보관할 이유가 없고
# 분석에도 쓸모가 없다. 여기서 받는 것은 전부 "어떤 사람이 잘 맞히나" 를
# 나눠 볼 축이다. 전부 선택이라 그냥 시작해도 된다.
if ss.참가자 is None and not ss.log:
    누적통계()
    # ★ 폼 key 를 "참가자" 로 두면 안 된다.
    #   ss.참가자 를 우리가 직접 대입하는데, 같은 이름의 위젯 key 가 있으면
    #   스트림릿이 "위젯 값은 session_state 로 못 바꾼다" 며 막는다.
    with st.form("참가자입력폼"):
        st.markdown("##### 참여자 정보 *(선택 — 안 쓰고 바로 시작해도 됩니다)*")
        c1, c2 = st.columns(2)
        닉 = c1.text_input("닉네임", max_chars=20, placeholder="비워두면 익명")
        연령 = c2.selectbox("연령대", ["선택 안 함", "10대", "20대", "30대", "40대 이상"])
        c3, c4 = st.columns(2)
        시간 = c3.selectbox("주당 게임 시간",
                          ["선택 안 함", "거의 안 함", "5시간 미만", "5~20시간", "20시간 이상"])
        경력 = c4.selectbox("스팀 이용 기간",
                          ["선택 안 함", "안 씀", "1년 미만", "1~5년", "5년 이상"])

        b1, b2 = st.columns(2)
        시작 = b1.form_submit_button("입력하고 시작", width="stretch", type="primary")
        건너 = b2.form_submit_button("그냥 시작", width="stretch")

        if 시작 or 건너:
            빈값 = lambda v: None if (not v or v == "선택 안 함") else v
            ss.참가자 = {} if 건너 else {
                "닉네임": 빈값(닉), "연령대": 빈값(연령),
                "게임시간": 빈값(시간), "스팀경력": 빈값(경력)}
            st.rerun()

    if _db.연결됨():
        st.caption("입력한 내용은 퀴즈가 끝나면 팀 DB(TiDB)에 저장됩니다. "
                   "실명·연락처는 받지 않습니다.")
    else:
        st.caption("⚠️ 지금은 DB에 연결되어 있지 않습니다. 퀴즈는 그대로 풀 수 있고, "
                   "기록만 저장되지 않습니다.")
    st.stop()

# 진행 상태를 세는 기준은 "답한 문제 수" 하나뿐이다.
# ss.i(현재 위치)와 섞어 쓰면 정답 공개 중에 하나씩 어긋난다.
answered = len(ss.log)


def score_board():
    """점수판.

    분모(/12)를 반드시 같이 보여준다. 숫자만 나란히 두면
    두 점수를 더해서 "총 몇 문제 풀었나" 로 읽는 사람이 생긴다.
    각자 12문제 중 몇 개를 맞혔는지이지, 합치는 숫자가 아니다.
    """
    st.markdown(
        f'''<div class="score-board">
          <div class="side me"><div class="who">당신이 맞힌 개수</div>
            <div class="pts">{ss.me}<u>/ {N}</u></div></div>
          <div class="vs">VS</div>
          <div class="side ai"><div class="who">모델이 맞힌 개수</div>
            <div class="pts">{ss.ai}<u>/ {N}</u></div></div>
        </div>''', unsafe_allow_html=True)


def grid(answered):
    """문제별 채점표 — 같은 문제를 각자 풀었다는 걸 눈으로 보여준다.

    점수 두 개를 나란히 두면 자꾸 더해서 읽힌다 (5 + 9 = 14).
    문제를 가로축에 두고 사람/모델 두 줄을 겹쳐 놓으면
    "같은 12문제"라는 게 한눈에 보인다.
    """
    def cell(k, who):
        if k < answered:
            hit = ss.log[k][who] == ss.log[k]["정답"]
            return f'<span class="mk {"o" if hit else "x"}">{"O" if hit else "X"}</span>'
        if k == answered and not ss.shown:
            return '<span class="mk q">?</span>'
        return '<span class="mk n">·</span>'

    head = "".join(f'<th class="qn">{k + 1}</th>' for k in range(N))
    rows = ""
    for who, label, cls in [("당신", "당신", "me"), ("모델", "모델", "ai")]:
        tds = "".join(
            f'<td class="{"cur" if k == answered and not ss.shown else ""}">{cell(k, who)}</td>'
            for k in range(N))
        rows += f'<tr class="{cls}"><th class="hdr">{label}</th>{tds}</tr>'
    st.markdown(
        f'<div class="grid"><table><tr><th class="hdr">문제</th>{head}</tr>'
        f'{rows}</table></div>', unsafe_allow_html=True)


# ── 다 풀었을 때 ────────────────────────────────────────────────
if ss.i >= N:
    score_board()
    if ss.ai > ss.me:
        st.error(f"### 모델이 이겼습니다 — 모델 {ss.ai}개 · 당신 {ss.me}개 (각 {N}문제 중)")
    elif ss.ai < ss.me:
        st.success(f"### 사람이 이겼습니다! — 당신 {ss.me}개 · 모델 {ss.ai}개 (각 {N}문제 중)")
    else:
        st.info(f"### 무승부 — 각 {ss.me}개 (각 {N}문제 중)")

    st.caption(f"같은 {N}문제를 각자 푼 결과입니다. 두 점수를 더하는 것이 아닙니다. "
               f"동전 던지기로 찍으면 평균 {N / 2:.0f}개를 맞힙니다.")

    # ── DB 저장 ────────────────────────────────────────────────
    # 한 번만 시도한다. ss.저장됨 이 None 일 때만 들어온다.
    # 실패해도 화면은 그대로 간다 — 결과는 이미 위에 다 나와 있다.
    if ss.저장됨 is None:
        ss.저장됨 = _db.저장(ss.참가자 or {}, ss.me, ss.ai, ss.log)
        if ss.저장됨:
            # 방금 넣은 내 기록이 누적 통계에 잡히도록 캐시를 깬다
            ss.저장카운터 = ss.get("저장카운터", 0) + 1

    if ss.저장됨:
        st.success("기록이 저장되었습니다. 아래 누적 통계에 이번 판이 포함되어 있습니다.",
                   icon="💾")
    elif _db.연결됨():
        st.warning("저장에 실패했습니다. 결과는 화면에 그대로 있습니다.", icon="⚠️")

    누적통계()

    st.dataframe(pd.DataFrame(ss.log).drop(columns=["_확률"], errors="ignore"),
                 width="stretch", hide_index=True)

    c = st.columns([2, 1, 2])[1]
    if c.button("다시 풀기", width="stretch"):
        # 참가자 정보는 남긴다 — 같은 사람이 한 번 더 풀 때 또 입력시키지 않는다.
        for k in ["i", "me", "ai", "log", "shown", "저장됨"]:
            ss.pop(k, None)
        st.rerun()
    if st.button("다른 사람이 풀기 (정보 다시 입력)", width="stretch"):
        for k in ["i", "me", "ai", "log", "shown", "저장됨", "참가자"]:
            ss.pop(k, None)
        st.rerun()
    st.stop()

# ── 문제 ────────────────────────────────────────────────────────
card = CARDS[ss.i]
answered = len(ss.log)
score_board()
grid(answered)
st.caption(f"{N}문제 중 {answered}개 답함"
           + ("  ·  결과를 확인하고 다음으로 넘어가세요" if ss.shown else ""))

vote = ('<span class="pill up">👍 추천</span>' if card["voted_up"]
        else '<span class="pill down">👎 비추천</span>')
priv = "비공개" if card["is_private"] else "공개"
st.markdown(
    f'''<div class="quiz-card">
      <div class="gtitle">{html.escape(str(card["game"]))}</div>
      <div class="gmeta">{vote}<span class="pill">{html.escape(str(card["genre_group"]))}</span>
        <span class="pill">{html.escape(str(card["grade"]))}</span></div>
      <div class="rtext">{html.escape(str(card["review"]))}</div>
      <div class="facts">
        <div>리뷰 쓸 때까지 플레이<b>{card["hours_at_review"]:.1f}시간</b></div>
        <div>프로필<b>{priv}</b></div>
        <div>리뷰 길이<b>{card["review_len"]}자</b></div>
      </div>
    </div>''', unsafe_allow_html=True)

# ── 답하기 / 정답 공개 ──────────────────────────────────────────
if not ss.shown:
    st.markdown("##### 이 사람은 이 리뷰를 쓴 뒤 게임을 계속했을까요?")
    c1, c2 = st.columns(2)
    pick = None
    if c1.button("🟢  계속했다", width="stretch"):
        pick = 0
    if c2.button("🔴  그만뒀다", width="stretch"):
        pick = 1
    if pick is not None:
        truth, ai = card["churn"], int(card["_p"] >= THR)
        ss.me += int(pick == truth)
        ss.ai += int(ai == truth)
        ss.log.append({
            "게임": card["game"],
            "당신": "그만뒀다" if pick else "계속했다",
            "모델": "그만뒀다" if ai else "계속했다",
            "정답": "그만뒀다" if truth else "계속했다",
            "모델 확률": f"{card['_p']:.0%}",
            "결과": ("둘 다 정답" if pick == truth == ai else
                    "나만 정답" if pick == truth else
                    "모델만 정답" if ai == truth else "둘 다 오답"),
            # DB 에 넣을 원값. 화면 표에서는 빼고 보여준다("모델 확률"이 이미 있다).
            "_확률": round(float(card["_p"]), 4),
        })
        ss.shown = True
        st.rerun()
else:
    last = ss.log[-1]
    me_hit = last["당신"] == last["정답"]
    ai_hit = last["모델"] == last["정답"]
    st.markdown(
        f'''<div class="answers">
          <div class="box {"hit" if me_hit else "miss"}">
            <div class="lbl">당신</div><div class="val">{last["당신"]}</div>
            <div class="tag">{"정답" if me_hit else "오답"}</div></div>
          <div class="box {"hit" if ai_hit else "miss"}">
            <div class="lbl">모델 ({last["모델 확률"]})</div><div class="val">{last["모델"]}</div>
            <div class="tag">{"정답" if ai_hit else "오답"}</div></div>
          <div class="box truth">
            <div class="lbl">정답</div><div class="val">{last["정답"]}</div>
            <div class="tag">&nbsp;</div></div>
        </div>''', unsafe_allow_html=True)

    label = "결과 보기" if ss.i + 1 >= N else "다음 문제  →"
    if st.button(label, width="stretch"):
        ss.i += 1
        ss.shown = False
        st.rerun()

# st.caption(f"문제 {N}장은 정답 비율 {N//2}:{N//2} 무작위 표본입니다. 난이도를 조작하지 않았습니다 · 모델 판정 기준 {THR:.0%}")
