# -*- coding: utf-8 -*-
"""시작 화면 — 리뷰 하나 바로 넣어보기."""
import numpy as np
import streamlit as st

from app._shared import get_all, build_row, gauge
from app.theme import apply_theme

apply_theme()
clf, reg, GAMES, CARDS, _, META, LANG = get_all()

st.markdown(
    "<h1 style='text-align:center;margin:8px 0 4px;font-size:30px'>"
    "스팀 게임 리뷰 기반 유저 이탈 예측</h1>"
    "<p style='text-align:center;color:#8f98a0;margin:0 0 24px'>"
    "게임 리뷰를 입력하여 리뷰를 작성한 유저의 이탈 가능성을 예측해보세요</p>",
    unsafe_allow_html=True)

# 지표 4개 — st.metric 기본 스타일이 밋밋해서 직접 그린다
churn = float(META["이탈률"])
auc = float(META["분류_AUC"])
st.markdown(f"""
<div class="kpis">
  <div class="kpi">
    <div class="top"><div class="ico">📊</div><div class="lab">학습 데이터</div></div>
    <div class="val">{META['행수']:,}<u>행</u></div>
    <div class="sub">스팀 리뷰 · 30개 언어</div>
  </div>
  <div class="kpi">
    <div class="top"><div class="ico">🎮</div><div class="lab">분석한 게임</div></div>
    <div class="val">{len(GAMES)}<u>개</u></div>
    <div class="sub">2001~2024년 · 5개 장르</div>
  </div>
  <div class="kpi">
    <div class="top"><div class="ico">🚪</div><div class="lab">이탈률</div></div>
    <div class="val">{churn:.1%}</div>
    <div class="gauge"><i style="width:{churn*100:.0f}%"></i></div>
    <div class="sub">리뷰 후 1시간도 안 한 비율</div>
  </div>
  <div class="kpi accent">
    <div class="top"><div class="ico">🎯</div><div class="lab">모델 성능</div></div>
    <div class="val">{auc:.3f}<u>AUC</u></div>
    <div class="sub">동전 던지기는 0.500</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")
left, center, right = st.columns([1, 2, 1])
with center:
    with st.form("quick"):
        text = st.text_area(
            "리뷰 (영어)", height=110,
            placeholder="스팀에서 아무 리뷰나 복사해 붙여넣어 보세요",
            value="Refunded after 2 hours. The tutorial is confusing.")
        c1, c2 = st.columns(2)
        gname = c1.selectbox("게임", GAMES.game.tolist())
        hours = c2.number_input("리뷰까지 플레이 (시간)", 0.0, 5000.0, 2.0, step=0.5)
        submitted = st.form_submit_button("이탈 확률 예측하기", width="stretch")

    if submitted:
        if not text.strip():
            st.warning("리뷰 내용을 입력해주세요.")
        else:
            with st.spinner("리뷰 분석 중…"):
                grow = GAMES[GAMES.game == gname].iloc[0]
                X = build_row(text, hours, 250, 12, False, grow, LANG)
                p = float(clf.predict_proba(X)[0, 1])
                more = float(np.expm1(reg.predict(X)[0]))
            icon, msg = gauge(p)
            (st.error if p >= .65 else st.warning if p >= .45 else st.success)(
                f"{icon}  이탈 확률 {p:.0%} — {msg}")
            st.caption(f"예상 추가 플레이 {more:.1f}시간 · "
                       f"근거가 궁금하면 **작별 인사 판별기**로 가세요")

st.write("")
left, center, right = st.columns([5, 2, 5])
with center:
    if st.button("▶  PLAY", width="stretch"):
        st.switch_page("pages/1_작별인사_판별기.py")
