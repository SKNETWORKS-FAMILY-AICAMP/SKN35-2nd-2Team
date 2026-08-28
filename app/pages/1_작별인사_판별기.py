# -*- coding: utf-8 -*-
"""
작별 인사 판별기 — 리뷰를 넣으면 이탈 확률 + 왜 그렇게 봤는지.

입력은 왼쪽, 결과는 오른쪽에 둔다.
아래로 길게 늘어놓으면 발표할 때 스크롤을 해야 해서
입력과 결과를 한 화면에 같이 볼 수 없다.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app._shared import (get_all, build_row, predict, explain, gauge, page,
                         STEAM_BLUE, STEAM_GREEN, STEAM_RED,
                         STEAM_NAVY_LIGHT, STEAM_TEXT_MUTED)

MODEL, GAMES, _, _, META, LANG = get_all()
THR = float(META["임계값"])

page("🔍 작별 인사 판별기",
     "리뷰를 붙여넣으면 이 사람이 이 리뷰를 끝으로 게임을 접었을 확률을 알려줍니다.")

ss = st.session_state
left, right = st.columns(2, gap="large")

# ── 왼쪽 : 입력 ─────────────────────────────────────────────────
with left:
    review = st.text_area(
        "리뷰 본문 (영어)", height=130,
        value="Refunded after 2 hours. The tutorial is confusing and the controls feel awful.",
        help="스팀에서 아무 리뷰나 복사해 붙여넣어 보세요")
    voted = st.radio("추천 / 비추천", ["👍 추천", "👎 비추천"], index=1, horizontal=True)

    c1, c2 = st.columns(2)
    gname = c1.selectbox("게임", GAMES.game.tolist(), index=0)
    hours = c2.number_input("리뷰까지 플레이 (시간)", 0.0, 5000.0, 2.0, step=0.5)
    c3, c4 = st.columns(2)
    owned = c3.number_input("보유 게임 수 (0 = 비공개)", 0, 5000, 120, step=10)
    nrev = c4.number_input("이 사람이 쓴 리뷰 수", 1, 2000, 12)

    if st.button("판별하기", width="stretch"):
        grow = GAMES[GAMES.game == gname].iloc[0]
        row = build_row(review, hours, owned, nrev, voted.startswith("👍"), grow, LANG)
        df, raw = explain(row)
        ss.res = {"p": predict(MODEL, META["_order"], row), "df": df, "game": gname}

# ── 오른쪽 : 결과 ───────────────────────────────────────────────
with right:
    if "res" not in ss:
        st.markdown(
            '<div class="resultbox"><div class="empty">'
            '<div class="big">🎮</div>'
            '왼쪽에 리뷰를 넣고<br><b>판별하기</b>를 눌러보세요'
            '</div></div>', unsafe_allow_html=True)
    else:
        r = ss.res
        p = r["p"]
        icon, msg = gauge(p, THR)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "%", "font": {"size": 40, "color": STEAM_BLUE}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": STEAM_TEXT_MUTED,
                         "tickfont": {"size": 10}},
                "bar": {"color": STEAM_RED if p >= THR + .15 else
                        (STEAM_BLUE if p >= THR else STEAM_GREEN), "thickness": .74},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1, "bordercolor": STEAM_NAVY_LIGHT,
                # 판정선을 눈에 보이게 그린다 — 0.5 가 아니라 0.3735 다
                "threshold": {"line": {"color": "#e8b64c", "width": 3},
                              "thickness": .85, "value": THR * 100},
                "steps": [
                    {"range": [0, THR * 100], "color": "rgba(164,208,7,.12)"},
                    {"range": [THR * 100, 100], "color": "rgba(224,92,92,.12)"},
                ]}))
        fig.update_layout(height=150, margin=dict(l=30, r=30, t=6, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#c7d5e0")
        st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        a.metric("판정", f"{icon} {msg}")
        b.metric("판정 기준선", f"{THR:.1%}",
                 help="이 값을 넘으면 이탈로 봅니다. 0.5가 아니라 데이터로 고른 값입니다")

        st.markdown("##### 왜 이렇게 판단했나")
        e = r["df"].copy()
        e["방향"] = np.where(e.기여 > 0, "이탈 쪽으로", "잔존 쪽으로")
        fig2 = px.bar(e.sort_values("기여"), x="기여", y="변수", orientation="h",
                      color="방향",
                      color_discrete_map={"이탈 쪽으로": STEAM_RED,
                                          "잔존 쪽으로": STEAM_GREEN},
                      custom_data=["기여율"])
        fig2.update_traces(hovertemplate="%{y}<br>기여율 %{customdata[0]:.0%}<extra></extra>")
        fig2.update_layout(
            height=200, margin=dict(l=4, r=4, t=4, b=4),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c7d5e0", legend_title_text="",
            legend=dict(orientation="h", y=-0.24, font=dict(size=11)),
            xaxis=dict(title="", gridcolor=STEAM_NAVY_LIGHT, tickfont=dict(size=11)),
            yaxis=dict(title="", tickfont=dict(size=12)))
        st.plotly_chart(fig2, width="stretch")
        st.caption("빨간 막대는 이탈 확률을 올린 요인, 초록 막대는 낮춘 요인입니다. "
                   "막대에 마우스를 올리면 기여율이 나옵니다.")
