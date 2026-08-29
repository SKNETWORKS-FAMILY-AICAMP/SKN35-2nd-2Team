# -*- coding: utf-8 -*-
"""
살까 말까 계산기 — 위시리스트에 담아둔 게임의 구매 우선순위를 매긴다.

스팀 게이머가 실제로 하는 계산은 "재밌을까" 가 아니라 **"시간당 얼마냐"** 다.
모델이 예상 플레이 시간을 내주므로, 가격을 나누면 그 계산이 된다.

    시간당 비용 = 게임 가격 / 내가 할 것으로 예상되는 시간

같은 4만원이라도 100시간 할 게임은 시간당 400원이고,
3시간 하고 접을 게임은 시간당 13,000원이다.
"""
import html

import pandas as pd
import streamlit as st

from app._shared import get_all, predict_many, page, load_playtime_stats
from app._predict import load_catalog, STEAM_IMG

MODEL, GAMES, _, _, META, LANG = get_all()
CAT = load_catalog()

page("💰 살까 말까 계산기",
     "위시리스트에 담아둔 게임을 골라보세요. **시간당 비용**으로 우선순위를 매깁니다.")

if CAT is None:
    st.warning("게임 카탈로그가 없습니다. `uv run python -m src.build_catalog` 를 실행하세요.")
    st.stop()

# 원화로 보여준다 (스팀 가격은 달러 기준이라 대략 환산)
RATE = 1380

PLAY = {"한두 시간 해보고 판단해요": 2.5,
        "재밌으면 20~30시간은 해요": 25.0,
        "붙잡으면 100시간도 넘겨요": 120.0}
WRITE = {"거의 안 써요": 2, "가끔 써요": 12, "자주 쓰는 편이에요": 60}

# ── 내 성향 ─────────────────────────────────────────────────────
with st.expander("내 게임 성향 (한 번만 설정하면 됩니다)", expanded=True):
    c1, c2 = st.columns(2)
    play_k = c1.radio("한 게임을 보통 얼마나 하세요?", list(PLAY))
    write_k = c2.radio("스팀에 리뷰를 자주 쓰세요?", list(WRITE))
    c3, c4 = st.columns(2)
    owned = c3.slider("스팀에 게임이 몇 개쯤 있나요?", 0, 500, 120, step=10,
                      help="스팀 유저 중앙값이 114개입니다")
    generous = c4.radio("평가는 후한 편인가요?",
                        ["👍 웬만하면 추천해요", "👎 깐깐한 편이에요"], horizontal=True)

hours, n_rev = PLAY[play_k], WRITE[write_k]
voted = generous.startswith("👍")

# ── 게임 고르기 ─────────────────────────────────────────────────
# st.markdown("##### 사려는 게임을 고르세요")
paid = CAT[CAT.현재가 > 0].sort_values("리뷰수", ascending=False)

# 이미 산 게임은 목록에서 뺀다 — 또 살 일이 없다
have = st.multiselect("이미 갖고 있는 게임 (목록에서 제외됩니다)",
                      sorted(paid.game.tolist()), placeholder="게임 이름을 입력해 고르세요",
                      help=f"스팀 전체가 아니라 **리뷰가 많은 상위 {len(CAT)}개** 중 "
                           f"유료 게임만 담고 있습니다")
st.caption(f"※ 스팀 전체 게임이 아니라 **누적 리뷰가 많은 순으로 {len(CAT)}개**를 다루며, "
           f"그중 **유료 게임 {len(paid)}개**가 목록에 있습니다 (무료는 계산할 게 없어 제외).")
if have:
    paid = paid[~paid.game.isin(have)]
    
default = [g for g in ["Assassin's Creed Mirage", "Planet Coaster 2", "Tales of ARISE"] if g in set(paid.game)][:3]
picks = st.multiselect("사려고 하는 게임 (여러 개 고를 수 있습니다)", sorted(paid.game.tolist()), default=default, max_selections=12)

budget = st.number_input("예산 (원) — 0이면 제한 없음", 0, 1_000_000, 0, step=10_000)

if not picks:
    st.info("게임을 하나 이상 고르세요. 무료 게임은 목록에 없습니다 (계산할 게 없으니까요).")
    st.stop()

if st.button("계산하기", width="stretch"):
    sel = paid[paid.game.isin(picks)].reset_index(drop=True)
    p_churn = predict_many(MODEL, META["_order"], "Pretty good game, worth the price.",
                           hours, owned, n_rev, voted, sel, LANG)
    sel["완주확률"] = 1 - p_churn
    # ── 예상 플레이 시간 ─────────────────────────────────────
    # 전에는 `hours * (1 + 6 * 완주확률)` 이었다. 그 '6' 에 근거가 없었다.
    # 우리 모델은 "리뷰 뒤 1시간을 넘기나" 만 예측하지 몇 시간 더 할지는
    # 예측하지 않는데, 시간당 비용이 그 시간을 분모로 쓰니 결과도 임의값이 됐다.
    #
    # 지금은 원본에 있는 실제 값을 쓴다.
    #   리뷰 뒤 실제로 더 한 시간 = playtime_forever - playtime_at_review
    # 이걸 게임별로, 이탈자/잔존자를 나눠 중앙값을 내둔 것이 playtime_stats.json 이고
    # (src/build_playtime_stats.py 가 만든다), 모델의 확률로 두 값을 섞는다.
    #
    #   추가시간 = p x (이탈자 중앙값) + (1-p) x (잔존자 중앙값)
    #
    # 이탈자 중앙값은 어느 게임에서든 0.00 시간이다 — 떠난 사람은 리뷰를 쓰고
    # 정말로 한 시간도 더 하지 않는다. 그래서 사실상 (완주확률 x 잔존자 시간) 이다.
    #
    # 평균이 아니라 중앙값을 쓴다. 5,000시간 한 사람 몇 명이 평균을 통째로
    # 끌어올려서, 중앙값이 '보통 사람' 에 가깝다.
    _PT = load_playtime_stats()

    def 추가시간(g, p):
        s = _PT["게임"].get(g) or _PT["장르"].get(g) or _PT["전체"]
        이탈 = s["이탈"] if s["이탈"] is not None else 0.0
        잔존 = s["잔존"] if s["잔존"] is not None else _PT["전체"]["잔존"]
        return p * 이탈 + (1 - p) * 잔존

    sel["예상시간"] = [
        추가시간(r.game if r.game in _PT["게임"] else r.genre_group, pc)
        for r, pc in zip(sel.itertuples(), p_churn)
    ]
    # 근거가 게임별인지 장르별인지 화면에 밝힌다
    sel["시간근거"] = [("이 게임" if r.game in _PT["게임"] else "같은 장르")
                    for r in sel.itertuples()]
    sel["가격원"] = (sel.현재가 * RATE).round(-2)
    # 예상 플레이 시간은 '리뷰 이후 추가' 시간이라, 리뷰까지의 시간을 더해야 총 시간이다
    sel["총시간"] = sel.예상시간 + hours
    sel["시간당"] = (sel.가격원 / sel.총시간).round(-1)
    sel["할인중"] = sel.현재가 < sel.정가

    def verdict(r):
        # 스팀 게이머 통념 — 시간당 1,000원 아래면 잘 산 것
        if r.시간당 <= 700:
            return "good", "지금 구매", "시간당 700원 아래"
        if r.시간당 <= 1500:
            return "good", "구매 무난", "시간당 1,500원 아래"
        if r.시간당 <= 3000:
            return "wait", "세일 대기", "지금은 조금 비쌉니다"
        return "no", "구매 보류", "시간당 3,000원 넘습니다"

    sel[["cls", "말", "이유"]] = sel.apply(
        lambda r: pd.Series(verdict(r)), axis=1)
    sel = sel.sort_values("시간당")

    # ── 요약 ────────────────────────────────────────────────────
    st.markdown("##### 결과 — 시간당 비용이 싼 순서")
    k1, k2, k3 = st.columns(3)
    k1.metric("고른 게임", f"{len(sel)}개")
    k2.metric("전부 사면", f"{int(sel.가격원.sum()):,}원")
    k3.metric("예상 총 플레이", f"{sel.총시간.sum():.0f}시간")

    if budget:
        # 예산 안에서 시간당 싼 것부터 담는다
        cum = sel.가격원.cumsum()
        fit = sel[cum <= budget]
        st.success(f"**예산 {budget:,}원으로는 {len(fit)}개** — "
                   f"{' · '.join(fit.game.head(6))}"
                   f"{' …' if len(fit) > 6 else ''}  "
                   f"(합계 {int(fit.가격원.sum()):,}원 · "
                   f"예상 {fit.총시간.sum():.0f}시간)"
                   if len(fit) else
                   f"**예산 {budget:,}원으로는 아무것도 못 삽니다.** "
                   f"가장 싼 게임이 {int(sel.가격원.min()):,}원입니다.")

    for _, r in sel.iterrows():
        thumb = (f'background-image:url({STEAM_IMG.format(int(r.appid))})'
                 if pd.notna(r.appid) else "")
        new = ('<span>처음 보는 게임</span>' if not int(r.학습함) else "")
        sale = ('<span style="border-color:#a4d007;color:#a4d007">할인중</span>'
                if r.할인중 else "")
        st.markdown(
            f'<div class="buy {r.cls}">'
            f'  <div class="thumb" style="{thumb}"></div>'
            f'  <div class="body">'
            f'    <div class="name">{html.escape(str(r.game))}</div>'
            f'    <div class="tags">{sale}{new}'
            f'      <span>{html.escape(str(r.genre_group))}</span>'
            f'      <span>{html.escape(str(r.grade))}</span></div>'
            f'    <div class="nums">'
            f'      <div>가격<b>{int(r.가격원):,}원</b></div>'
            f'      <div>예상 플레이<b>{r.총시간:.0f}시간</b></div>'
            f'      <div>완주 확률<b>{r.완주확률:.0%}</b></div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="verdict"><div class="v">{r.말}</div>'
            f'    <div class="hour">{int(r.시간당):,}원</div>'
            f'    <small>시간당 · {r.이유}</small></div>'
            f'</div>', unsafe_allow_html=True)
