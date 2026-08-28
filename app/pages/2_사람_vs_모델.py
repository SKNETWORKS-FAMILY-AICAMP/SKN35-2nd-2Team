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

_, _, CARDS, _, META, _ = get_all()
THR = float(META["임계값"])  # 0.5 가 아니다 — 데이터로 고른 판정 기준
N = len(CARDS)

page("🆚 사람 vs 모델",
     f"리뷰 {N}장을 보고 이 사람이 게임을 계속했을지 맞혀보세요. 모델과 점수를 비교합니다.")

ss = st.session_state
for k, v in [("i", 0), ("me", 0), ("ai", 0), ("log", []), ("shown", False)]:
    ss.setdefault(k, v)

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
    st.dataframe(pd.DataFrame(ss.log), width="stretch", hide_index=True)

    c = st.columns([2, 1, 2])[1]
    if c.button("다시 풀기", width="stretch"):
        for k in ["i", "me", "ai", "log", "shown"]:
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

st.caption(f"문제 {N}장은 정답 비율 {N//2}:{N//2} 무작위 표본입니다. 난이도를 조작하지 않았습니다 · 모델 판정 기준 {THR:.0%}")
