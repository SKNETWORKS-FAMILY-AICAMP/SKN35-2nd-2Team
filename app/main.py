# -*- coding: utf-8 -*-
"""
화면 진입점 — 페이지를 묶고 라우팅한다.

실행 (이 폴더 맨 위에서):
    uv run streamlit run app/main.py
"""
import sys
from pathlib import Path
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="스팀 게임 리뷰 기반 유저 이탈 예측", page_icon="🎮", layout="wide")

st.navigation([
    st.Page("pages/0_시작.py", title="시작", icon="▶", default=True),
    st.Page("pages/1_작별인사_판별기.py", title="작별 인사 판별기", icon="🔍"),
    st.Page("pages/2_사람_vs_모델.py", title="사람 vs 모델", icon="🆚"),
    st.Page("pages/3_나에게_맞는_게임.py",title="나에게 맞는 게임", icon="🎯"),
    st.Page("pages/4_살까말까_계산기.py", title="살까 말까 계산기", icon="💰")
], position="top").run()
