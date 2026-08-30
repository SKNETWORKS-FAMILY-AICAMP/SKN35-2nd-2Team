# -*- coding: utf-8 -*-
"""
작별 인사 판별기 — 리뷰를 넣으면 이탈 확률.

★ 이 화면만 딥러닝(MLP + 글 임베딩)을 쓴다. 다른 화면은 최종 모델(LightGBM)이다.

왜 나눴나
  이 화면의 의의는 "학습에 없던 리뷰라도 글을 읽고 확률을 낸다" 이다.
  그런데 최종 모델은 글의 겉모양만 본다 — 글자 수·단어 수·대문자 비율.
  실제로 재보면 이렇다 (같은 조건, 글만 바꿔 8문장):

      변동 폭   최종 모델 0.018   ·   딥러닝 0.486      <- 27배
      극찬↔혹평  최종 모델 0.000   ·   딥러닝 0.355

  최종 모델은 "인생 최고의 게임"과 "인생 최악의 게임"에 **소수점 넷째 자리까지
  같은 확률**을 준다. 그러면 리뷰 입력칸이 장식이 된다.

정직하게 밝힐 것 — 화면에도 적어 둔다
  · 딥러닝이 더 정확한 게 아니다. 같은 영어 데이터·게임 분할에서
    부스팅 0.750 > 딥러닝 0.730 이다. 쓰는 이유는 성능이 아니라 글을 읽어서다.
  · 영어 전용이다 (영어 67,112행 학습). 한국어는 제대로 못 읽는다.
  · 임계값이 다르다. 딥러닝 0.3346 · 최종 모델 0.2899. 모델과 임계값은 한 세트다.

입력은 왼쪽, 결과는 오른쪽. 아래로 늘어놓으면 발표할 때 스크롤해야 해서
입력과 결과를 한 화면에 같이 볼 수 없다.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app._shared import (
    get_all,
    get_dl,
    build_row,
    predict,
    predict_dl,
    explain,
    판정말,
    page,
    한글비율,
    STEAM_BLUE,
    STEAM_GREEN,
    STEAM_RED,
    STEAM_NAVY_LIGHT,
    STEAM_TEXT_MUTED,
)

MODEL, GAMES, _, _, META, LANG = get_all()
ML_THR = float(META["임계값"])          # 최종 모델 기준선 (비교용)

page("🔍 작별 인사 판별기",
     "리뷰를 붙여넣으면 이 사람이 이 리뷰를 끝으로 게임을 접었을 확률을 알려줍니다.",
     모델="딥러닝")

st.markdown(
    '<div class="usenote">이 화면은 <b>글을 읽는 모델(딥러닝)</b>을 씁니다. '
    '다른 화면은 최종 모델(LightGBM)을 씁니다 — '
    '최종 모델은 리뷰의 <b>길이·대문자 비율</b>만 보고 내용은 읽지 않기 때문입니다.</div>',
    unsafe_allow_html=True)

ss = st.session_state
left, right = st.columns(2, gap="large")

# ── 왼쪽 : 입력 ─────────────────────────────────────────────────
with left:
    review = st.text_area(
        "리뷰 본문 (영어)", height=120,
        value="Refunded after 2 hours. The tutorial is confusing and the controls feel awful.",
        help="스팀에 없는 리뷰를 직접 지어내도 됩니다. 모델이 글을 읽습니다.",
    )
    if 한글비율(review) > 0.2:
        st.warning("이 모델은 **영어 리뷰**로 학습했습니다. 한국어는 제대로 읽지 못합니다.",
                   icon="⚠️")

    voted = st.radio("추천 / 비추천", ["👍 추천", "👎 비추천"], index=0, horizontal=True)

    c1, c2 = st.columns(2)
    gname = c1.selectbox("게임", GAMES.game.tolist(), index=0)
    hours = c2.number_input("리뷰까지 플레이 (시간)", 0.0, 5000.0, 2.0, step=0.5)
    c3, c4 = st.columns(2)
    owned = c3.number_input("보유 게임 수 (0 = 비공개)", 0, 5000, 120, step=10)
    nrev = c4.number_input("이 사람이 쓴 리뷰 수", 1, 2000, 12)

    if st.button("판별하기", width="stretch"):
        grow = GAMES[GAMES.game == gname].iloc[0]
        up = voted.startswith("👍")
        dl_thr, dl_meta = get_dl()
        with st.spinner("글을 읽는 중…"):
            p_dl, p_notext = predict_dl(review, hours, owned, nrev, up, grow)
        # 비교용 — 최종 모델은 같은 입력을 글 없이 본 셈이다
        row = build_row(review, hours, owned, nrev, up, grow, LANG)
        df, _ = explain(row)
        ss.res = {"p": p_dl, "p_notext": p_notext, "thr": dl_thr, "meta": dl_meta,
                  "p_ml": predict(MODEL, META["_order"], row), "df": df, "game": gname}

# ── 오른쪽 : 결과 ───────────────────────────────────────────────
with right:
    # 왼쪽은 "리뷰 본문 (영어)" 라벨이 한 줄 차지한다.
    # 그만큼 오른쪽도 내려야 계기판이 위쪽에서 잘리지 않는다.
    st.markdown('<div class="rightpad"></div>', unsafe_allow_html=True)

    if "res" not in ss:
        st.markdown(
            '<div class="resultbox">'
            '<div class="empty">'
            '<div class="big">🎮</div>'
            '왼쪽에 리뷰를 넣고<br><b>판별하기</b>를 눌러보세요'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        r = ss.res
        p, thr = r["p"], r["thr"]
        icon, msg = 판정말(p, thr)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "%", "font": {"size": 40, "color": STEAM_BLUE}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": STEAM_TEXT_MUTED,
                         "tickfont": {"size": 10}},
                "bar": {"color": STEAM_RED if p >= thr + .15 else
                        (STEAM_BLUE if p >= thr else STEAM_GREEN), "thickness": .74},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1, "bordercolor": STEAM_NAVY_LIGHT,
                # 판정선을 눈에 보이게 그린다 — 0.5 가 아니라 데이터로 고른 값이다
                "threshold": {"line": {"color": "#e8b64c", "width": 3},
                              "thickness": .85, "value": thr * 100},
                "steps": [
                    {"range": [0, thr * 100], "color": "rgba(164,208,7,.12)"},
                    {"range": [thr * 100, 100], "color": "rgba(224,92,92,.12)"},
                ]}))
        # t=6 이면 눈금 40·60·80 이 위에서 잘린다. 여백과 높이를 함께 키운다.
        fig.update_layout(height=190, margin=dict(l=34, r=34, t=34, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#c7d5e0")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # st.metric 은 칸이 좁으면 "떠나는 쪽입..." 처럼 말을 자른다.
        # 직접 그려서 끝맺은 말이 온전히 보이게 한다.
        st.markdown(
            f'<div class="verdicts">'
            f'  <div class="vbox"><div class="vlab">판정</div>'
            f'    <div class="vval">{icon} {msg}</div></div>'
            f'  <div class="vbox"><div class="vlab">판정 기준선</div>'
            f'    <div class="vval mono">{thr:.1%}</div>'
            f'    <div class="vsub">0.5 가 아니라 데이터로 고른 값</div></div>'
            f'</div>', unsafe_allow_html=True)

        # ── 글이 실제로 얼마나 움직였나 ─────────────────────────
        # 대리 모델의 SHAP 이 아니라, 같은 입력에서 리뷰만 지우고 한 번 더 돌려
        # 그 차이를 쓴다. 진짜 모델이 진짜로 움직인 값이다.
        d = p - r["p_notext"]
        방향 = "이탈 쪽으로" if d > 0 else "잔존 쪽으로"
        색 = STEAM_RED if d > 0 else STEAM_GREEN
        st.markdown(
            f'<div class="textgap">이 리뷰의 <b>글</b>이 확률을 '
            f'<b style="color:{색}">{abs(d):.1%}p {방향}</b> 밀었습니다'
            f'<span>글을 지우고 같은 조건으로 다시 돌리면 {r["p_notext"]:.1%} · '
            f'{r["meta"]["모델"]} · 영어 전용</span></div>',
            unsafe_allow_html=True)

        # ── 비교: 글을 안 읽는 최종 모델 ────────────────────────
        with st.expander("📊 글을 안 읽는 최종 모델은 이 리뷰를 어떻게 볼까"):
            st.markdown(
                f'<div class="mnote">최종 모델(LightGBM)은 같은 조건에 '
                f'<b>{r["p_ml"]:.1%}</b> 를 줍니다 (기준선 {ML_THR:.1%}).<br>'
                f'리뷰 글을 바꿔도 이 숫자는 거의 안 움직입니다 — '
                f'글에서 <b>길이·단어 수·대문자 비율</b>만 뽑기 때문입니다.<br>'
                f'<i>성능은 최종 모델이 더 높습니다. 같은 영어 데이터·게임 분할에서 '
                f'0.750 vs 0.730 입니다.</i></div>',
                unsafe_allow_html=True)

            st.markdown("###### 최종 모델이 무엇을 보고 판단했나")
            st.caption("빨간 막대는 이탈 확률을 올린 요인, 초록 막대는 낮춘 요인입니다. "
                       "막대에 마우스를 올리면 기여율이 나옵니다.")
            e = r["df"].copy()
            e["방향"] = np.where(e.기여 > 0, "이탈 요인", "잔존 요인")
            fig2 = px.bar(e.sort_values("기여"), x="기여", y="변수", orientation="h",
                          color="방향",
                          color_discrete_map={"이탈 요인": STEAM_RED,
                                              "잔존 요인": STEAM_GREEN},
                          custom_data=["기여율"])
            fig2.update_traces(
                hovertemplate="%{y}<br>기여율 : <b>%{customdata[0]:.0%}</b><extra></extra>")
            fig2.update_layout(
                height=200, margin=dict(l=4, r=4, t=4, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
                legend=dict(orientation="h", y=-0.24, x=0.3, font=dict(size=11),
                            font_color="#c7d5e0"),
                xaxis=dict(title="", tickfont=dict(size=12)),
                yaxis=dict(title="", tickfont=dict(size=12)),
                dragmode=False)
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
