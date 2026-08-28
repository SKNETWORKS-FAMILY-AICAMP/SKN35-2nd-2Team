# -*- coding: utf-8 -*-
"""
화면 전체가 공통으로 쓰는 것.

디자인은 팀원이 만든 theme.py + style.css 를 그대로 쓴다.
색을 새로 만들지 않는다 — 색의 유일한 출처는 theme.py 의 STEAM_* 이다.

메뉴에 대해
    이 Streamlit 빌드는 사이드바가 렌더링되지 않는다(DOM 노드 자체가 안 생김).
    그래서 st.navigation(position="top") 으로 메뉴를 위에 둔다.
    메뉴 이름이 길면 "N more" 로 접히므로 짧게 유지할 것.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st                                          # noqa: E402

from app.theme import (apply_theme, STEAM_BLUE, STEAM_GREEN,     # noqa: E402
                       STEAM_RED, STEAM_NAVY_LIGHT, STEAM_TEXT_MUTED)
from app._predict import (load_all, load_catalog, build_row, predict,   # noqa: E402
                          predict_many, explain, gauge, STEAM_IMG)

PAGES = [
    ("pages/0_시작.py",                 "시작",                    "▶"),
    ("pages/1_작별인사_판별기.py",       "작별 인사 판별기",         "🔍"),
    ("pages/2_사람_vs_모델.py",          "사람 vs 모델",            "🆚"),
    ("pages/3_나에게_맞는_게임.py",      "나에게 맞는 게임",         "🎯"),
    ("pages/4_살까말까_계산기.py",      "살까 말까 계산기",      "💰"),
]


@st.cache_resource
def get_all():
    """모델과 데이터를 한 번만 읽는다."""
    return load_all()


def steam_header(tagline="리뷰를 쓴 그 순간, 이 사람이 게임을 계속할지 맞힌다"):
    """팀원이 만든 스팀 워드마크 헤더."""
    st.markdown(
        f"""
        <div class="steam-header">
            <div class="badge-icon">▶</div>
            <div>
                <div class="word">STEAM <span>ChurnLens</span></div>
                <div class="tagline">{tagline}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nav(current=None):
    """본문 맨 위 메뉴."""
    cols = st.columns(len(PAGES))
    for col, (path, label, icon) in zip(cols, PAGES):
        col.page_link(path, label=label, icon=icon, width="stretch")


def kpi_bar():
    """페이지 상단 지표 — 시작 화면 카드와 같은 스타일.

    어느 화면을 보고 있든 "무슨 데이터로 만든 건지" 가 눈에 있어야
    발표에서 설명을 반복하지 않아도 된다.
    시작 화면에는 넣지 않는다 — 결과를 미리 보여주는 셈이 되기 때문이다.
    """
    _, games, _, _, meta, _ = get_all()
    churn = 0.411                       # dataset_meta.json 의 이탈률
    auc = float(meta.get("성능_봉인12게임", 0.7209))
    name = meta["모델"]
    st.markdown(f"""
<div class="kpis compact">
  <div class="kpi">
    <div class="top"><div class="ico">📊</div><div class="lab">학습 데이터</div></div>
    <div class="val">{meta.get("학습행수", 139658):,}<u>행</u></div>
    <div class="sub">스팀 리뷰 · 30개 언어</div>
  </div>
  <div class="kpi">
    <div class="top"><div class="ico">🎮</div><div class="lab">분석한 게임</div></div>
    <div class="val">{len(games)}<u>개</u></div>
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
    <div class="sub">봉인 12게임 · {name}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def page(title, caption=""):
    """모든 페이지가 첫 줄에서 부른다 — 테마 + 미니지표 + 제목."""
    apply_theme()
    kpi_bar()
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)
    st.write("")
