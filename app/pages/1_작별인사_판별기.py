# -*- coding: utf-8 -*-
"""리뷰를 붙여넣으면 이탈 확률 + 왜 그렇게 판단했는지."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app._shared import (get_all, build_row, gauge, page,
                         STEAM_BLUE, STEAM_GREEN, STEAM_RED,
                         STEAM_NAVY_LIGHT, STEAM_TEXT_MUTED)
from app._explain import explain

clf, reg, GAMES, _, _, META, LANG = get_all()

page("🔍 작별 인사 판별기",
     "리뷰를 붙여넣으면 이 사람이 이 리뷰를 끝으로 게임을 접었을 확률을 알려줍니다.")

c1, c2 = st.columns([3, 2])
with c1:
    review = st.text_area(
        "리뷰 본문 (영어)", height=150,
        value="Refunded after 2 hours. The tutorial is confusing and the controls feel awful.",
        help="스팀에서 아무 리뷰나 복사해 붙여넣어 보세요")
    voted = st.radio("추천 / 비추천", ["👍 추천", "👎 비추천"], index=1, horizontal=True)
with c2:
    gname = st.selectbox("게임", GAMES.game.tolist(), index=0)
    hours = st.number_input("리뷰 쓸 때까지 플레이 시간 (시간)", 0.0, 5000.0, 2.0, step=0.5)
    owned = st.number_input("보유 게임 수 (0 = 비공개)", 0, 20000, 250)
    nrev = st.number_input("이 사람이 쓴 리뷰 수", 1, 5000, 12)

if st.button("판별하기", width="stretch"):
    grow = GAMES[GAMES.game == gname].iloc[0]
    X = build_row(review, hours, owned, nrev, voted.startswith("👍"), grow, LANG)
    p = float(clf.predict_proba(X)[0, 1])
    hours_more = float(np.expm1(reg.predict(X)[0]))
    icon, msg = gauge(p)

    g_col, m_col = st.columns([2, 1])
    with g_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 44, "color": STEAM_BLUE}},
            title={"text": "이탈 확률", "font": {"size": 14, "color": STEAM_TEXT_MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": STEAM_TEXT_MUTED},
                "bar": {"color": STEAM_RED if p >= .65 else
                        (STEAM_BLUE if p >= .45 else STEAM_GREEN), "thickness": .72},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1, "bordercolor": STEAM_NAVY_LIGHT,
                "steps": [
                    {"range": [0, 45], "color": "rgba(164,208,7,.12)"},
                    {"range": [45, 65], "color": "rgba(102,192,244,.12)"},
                    {"range": [65, 100], "color": "rgba(224,92,92,.14)"},
                ],
            }))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=44, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#c7d5e0")
        st.plotly_chart(fig, width="stretch")
    with m_col:
        st.write("")
        st.metric("판정", f"{icon} {msg}")
        st.metric("예상 추가 플레이", f"{hours_more:.1f}시간")
        st.caption("이탈 = 리뷰 후 1시간도 더 안 함")

    st.markdown("##### 왜 이렇게 판단했나")
    e = explain(clf, X)
    e["방향"] = np.where(e.기여 > 0, "이탈 쪽으로", "잔존 쪽으로")
    fig2 = px.bar(e.sort_values("기여"), x="기여", y="변수", orientation="h",
                  color="방향",
                  color_discrete_map={"이탈 쪽으로": STEAM_RED, "잔존 쪽으로": STEAM_GREEN})
    fig2.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#c7d5e0", legend_title_text="",
                       xaxis=dict(title="기여도 (양수 = 이탈 쪽)", gridcolor=STEAM_NAVY_LIGHT),
                       yaxis=dict(title=""))
    st.plotly_chart(fig2, width="stretch")
    st.caption("빨간 막대는 이탈 확률을 올린 요인, 초록 막대는 낮춘 요인입니다.")
