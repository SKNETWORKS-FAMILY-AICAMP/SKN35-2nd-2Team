# -*- coding: utf-8 -*-
"""
나에게 맞는 게임 — 내 성향을 넣으면 게임별 완주 확률을 매긴다.

질문은 "사람이 실제로 답할 수 있는 것" 으로 만들었다.
'보통 몇 시간 하고 리뷰를 쓰나요' 같은 건 아무도 모른다.

모델이 게임에 대해 아는 것은 장르·출시시기·평가·연도뿐이라,
우리가 리뷰를 모으지 않은 게임도 점수를 매길 수 있다.
다만 정확도가 떨어지므로(0.818 -> 0.749) 화면에 표시한다.
"""
import html

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app._shared import get_all, predict_many, page
from app._predict import load_catalog, STEAM_IMG

MODEL, GAMES, _, _, META, LANG = get_all()
CAT = load_catalog()

page("🎯 나에게 맞는 게임",
     "내 성향을 넣으면 게임별로 **끝까지 할 확률**을 매깁니다.")

# ── 질문 ────────────────────────────────────────────────────────
GENRES = sorted(GAMES.genre_group.unique())
PLAY = {"한두 시간 해보고 판단해요": 2.5,
        "재밌으면 20~30시간은 해요": 25.0,
        "붙잡으면 100시간도 넘겨요": 120.0}
WRITE = {"거의 안 써요": 2, "가끔 써요": 12, "자주 쓰는 편이에요": 60}
ERA = {"상관없어요": None, "최근 게임 (2019~)": ["S3 2019-22", "S4 2023-25"],
       "예전 게임 (~2018)": ["S1 ~2016", "S2 2017-18"]}

c1, c2 = st.columns([2, 1])
with c1:
    genres = st.multiselect("어떤 장르를 찾으세요?", GENRES, default=GENRES,
                            help="비워두면 전체에서 찾습니다")
with c2:
    era_k = st.selectbox("출시 시기", list(ERA))

c3, c4 = st.columns(2)
play_k = c3.radio("한 게임을 보통 얼마나 하세요?", list(PLAY))
write_k = c4.radio("스팀에 리뷰를 자주 쓰세요?", list(WRITE))

c5, c6 = st.columns(2)
owned = c5.slider("스팀에 게임이 몇 개쯤 있나요?", 0, 500, 120, step=10,
                      help="스팀 유저 중앙값이 114개입니다")
generous = c6.radio("평가는 후한 편인가요?",
                    ["👍 웬만하면 추천해요", "👎 깐깐한 편이에요"], horizontal=True)

# 이미 산 게임은 추천에서 뺀다 — 갖고 있는 걸 추천하면 쓸모가 없다
n_cat = len(CAT) if CAT is not None else len(GAMES)
have = st.multiselect(
    "이미 갖고 있는 게임 (추천에서 제외됩니다)",
    (CAT.game.tolist() if CAT is not None else GAMES.game.tolist()),
    placeholder="게임 이름을 입력해 고르세요",
    help=f"스팀 전체가 아니라 **리뷰가 많은 상위 {n_cat}개**만 담고 있습니다")
st.caption(f"※ 스팀 전체 게임이 아니라 **누적 리뷰가 많은 순으로 {n_cat}개**만 다룹니다. "
           f"목록에 없는 게임은 아직 수집하지 않은 게임입니다.")

scope = "우리가 리뷰를 모은 60개"
if CAT is not None and (~CAT.학습함.astype(bool)).any():
    n_new = int((~CAT.학습함.astype(bool)).sum())
    scope = st.radio(
        "어디서 찾을까요?",
        [f"우리가 리뷰를 모은 {len(GAMES)}개 (정확도 높음)",
         f"처음 보는 게임 {n_new}개까지 포함 (정확도 낮음)"],
        horizontal=True)

hours, n_rev = PLAY[play_k], WRITE[write_k]
voted = generous.startswith("👍")
sample = "Pretty good game, worth the price."   # 리뷰 글은 성향과 무관하게 고정

# ── 대상 게임 고르기 ────────────────────────────────────────────
use_catalog = CAT is not None and scope.startswith("처음 보는")
if use_catalog:
    pool = CAT.copy()
else:
    pool = GAMES.copy()
    pool["학습함"] = 1
    # 설명은 _predict.load_all() 이 catalog.csv 에서 appid 로 붙여준다.
    # 예전에는 여기서 무조건 "" 로 덮어쓰고, appid 가 없을 때만 다시 붙였다.
    # games.csv 에는 appid 가 있어서 그 분기가 한 번도 안 돌았고 설명이 늘 비었다.
    if "설명" not in pool.columns:
        pool["설명"] = ""
    pool["설명"] = pool["설명"].fillna("")

if genres:
    pool = pool[pool.genre_group.isin(genres)]
if ERA[era_k]:
    pool = pool[pool.era.isin(ERA[era_k])]
if have:
    pool = pool[~pool.game.isin(have)]

st.caption(f"대상 게임 **{len(pool)}개**"
           + (f" · 보유 {len(have)}개 제외" if have else ""))

if st.button("추천 받기", width="stretch"):
    if pool.empty:
        st.warning("조건에 맞는 게임이 없습니다. 장르나 시기를 넓혀보세요.")
        st.stop()

    with st.spinner(f"게임 {len(pool):,}개 점수 매기는 중…"):
        p_churn = predict_many(MODEL, META["_order"], sample, hours, owned,
                               n_rev, voted, pool, LANG)
        res = pool.reset_index(drop=True).copy()
        res["완주확률"] = 1 - p_churn

    def card(r, risk=False):
        appid = r.get("appid")
        thumb = (f'background-image:url({STEAM_IMG.format(int(appid))})'
                 if pd.notna(appid) else "")
        new_game = ('<span class="new">처음 보는 게임</span>'
             if not int(r.get("학습함", 1)) else "")
        desc = html.escape(str(r.get("설명") or ""))[:110]
        return (
            f'<div class="rec{" risk" if risk else ""}">'
            f'  <div class="thumb" style="{thumb}"></div>'
            f'  <div class="body">'
            f'    <div class="name">{html.escape(str(r.game))}</div>'
            f'    <div class="desc">{desc}</div>'
            f'    <div class="tags">{new_game}'
            f'      <span>{html.escape(str(r.genre_group))}</span>'
            f'      <span>{html.escape(str(r.grade))}</span>'
            f'      <span>{int(r.release_year)}</span></div>'
            f'  </div>'
            f'  <div class="pct"><b>{r.완주확률:.0%}</b>'
            f'    <span>끝까지 할 확률</span></div>'
            f'</div>'
        )

    st.markdown("##### 🟢 끝까지 할 것 같은 게임")
    for _, r in res.nlargest(8, "완주확률").iterrows():
        st.markdown(card(r), unsafe_allow_html=True)

    with st.expander("🔴 금방 접을 것 같은 게임도 보기"):
        for _, r in res.nsmallest(8, "완주확률").iterrows():
            st.markdown(card(r, risk=True), unsafe_allow_html=True)

    if use_catalog:
        st.info(f"**처음 보는 게임**은 리뷰를 모으지 않은 게임입니다. "
                f"장르·출시시기·평가만 보고 예측하므로 정확도가 낮습니다 "
                f"(AUC {META.get('성능_랜덤분할', 0.818):.3f} → "
                f"{META.get('성능_봉인12게임', 0.7209):.3f}). "
                f"학습에 한 번도 쓰지 않은 12개 게임으로 잰 값입니다.", icon="⚠️")

    with st.expander(f"전체 {len(res)}개 표로 보기"):
        cols = ["game", "genre_group", "grade", "release_year", "완주확률"]
        st.dataframe(
            res.sort_values("완주확률", ascending=False)[cols]
            .rename(columns={"game": "게임", "genre_group": "장르",
                             "grade": "평가", "release_year": "출시"})
            .style.format({"완주확률": "{:.1%}", "출시": "{:.0f}"}),
            width="stretch", hide_index=True, height=360)

# st.caption("이 예측은 **\"당신 같은 사람이 이 게임에 리뷰를 쓴다면\"** 을 전제로 합니다. 우리 데이터가 리뷰를 쓴 사람만 담고 있기 때문입니다.")
