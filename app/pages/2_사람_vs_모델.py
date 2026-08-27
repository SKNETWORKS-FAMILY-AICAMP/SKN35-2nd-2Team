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

_, _, _, CARDS, _, _, _ = get_all()
N = len(CARDS)

page("🆚 사람 vs 모델",
     f"리뷰 {N}장을 보고 이 사람이 게임을 계속했을지 맞혀보세요. 모델과 점수를 비교합니다.")

ss = st.session_state
for k, v in [("i", 0), ("me", 0), ("ai", 0), ("log", []), ("shown", False)]:
    ss.setdefault(k, v)

done = ss.i if not ss.shown else ss.i + 1


def score_board():
    st.markdown(
        f'''<div class="score-board">
          <div class="side me"><div class="who">당신</div><div class="pts">{ss.me}</div></div>
          <div class="vs">VS</div>
          <div class="side ai"><div class="who">모델</div><div class="pts">{ss.ai}</div></div>
        </div>''', unsafe_allow_html=True)


def dots():
    marks = "".join(
        f'<i class="{"done" if k < ss.i else "now" if k == ss.i else ""}"></i>'
        for k in range(N))
    st.markdown(f'<div class="dots">{marks}</div>', unsafe_allow_html=True)


# ── 다 풀었을 때 ────────────────────────────────────────────────
if ss.i >= N:
    score_board()
    if ss.ai > ss.me:
        st.error(f"### 모델이 이겼습니다  {ss.ai} : {ss.me}")
    elif ss.ai < ss.me:
        st.success(f"### 사람이 이겼습니다!  {ss.me} : {ss.ai}")
    else:
        st.info(f"### 무승부  {ss.me} : {ss.ai}")

    st.caption(f"동전 던지기로 찍으면 평균 {N / 2:.0f}점입니다.")
    st.dataframe(pd.DataFrame(ss.log), width="stretch", hide_index=True)

    c = st.columns([2, 1, 2])[1]
    if c.button("다시 풀기", width="stretch"):
        for k in ["i", "me", "ai", "log", "shown"]:
            ss.pop(k, None)
        st.rerun()
    st.stop()

# ── 문제 ────────────────────────────────────────────────────────
card = CARDS[ss.i]
score_board()
dots()
st.caption(f"문제 {ss.i + 1} / {N}")

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
        truth, ai = card["churn"], int(card["_p"] >= .5)
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
        f'''<div class="verdict">
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

st.caption(f"문제 {N}장은 정답 비율 6:6 무작위 표본입니다. 난이도를 조작하지 않았습니다.")
