"""
스팀 테마 공용 모듈.

- 색상 상수(STEAM_*): 이 앱의 유일한 색상 소스. app.py의 Plotly 차트 색상과
  style.css의 CSS 변수가 모두 여기서 나온 값을 씁니다.
- apply_theme(): style.css를 읽어와 화면에 주입하는 함수.

멀티페이지 구조 사용 시 주의:
    Streamlit은 페이지(스크립트)마다 독립적으로 실행되므로, 이 CSS는
    "어디선가 한 번" 부르는 게 아니라 CSS가 필요한 "모든 페이지 파일 각각"의
    맨 위에서 apply_theme()을 호출해야 합니다.

    예) pages/1_review_predict.py
        from theme import apply_theme
        apply_theme()
        ... (이후 해당 페이지 내용)
"""

from pathlib import Path

import streamlit as st

# 스팀 시그니처 컬러 팔레트 (공식 로고 대신 톤만 차용) - 색상의 유일한 소스
STEAM_NAVY = "#1b2838"
STEAM_NAVY_LIGHT = "#2a475e"
STEAM_BLUE = "#66c0f4"
STEAM_BLUE_DARK = "#1999ff"
STEAM_GREEN = "#a4d007"   # 긍정/추천
STEAM_RED = "#e05c5c"     # 이탈 위험/비추천
STEAM_TEXT_MUTED = "#8f98a0"

_CSS_PATH = Path(__file__).parent / "style.css"

# style.css가 var(--steam-*)로 참조하는 값을 여기서 주입
_ROOT_VARS = f"""
<style>
:root {{
    --steam-navy: {STEAM_NAVY};
    --steam-navy-light: {STEAM_NAVY_LIGHT};
    --steam-blue: {STEAM_BLUE};
    --steam-blue-dark: {STEAM_BLUE_DARK};
    --steam-green: {STEAM_GREEN};
    --steam-red: {STEAM_RED};
    --steam-text-muted: {STEAM_TEXT_MUTED};
}}
</style>
"""


@st.cache_data
def _load_css_text() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


def apply_theme():
    """색상 변수 + style.css를 현재 페이지에 주입한다.
    CSS가 필요한 모든 페이지 파일의 최상단에서 호출할 것."""
    st.markdown(_ROOT_VARS, unsafe_allow_html=True)
    st.markdown(f"<style>{_load_css_text()}</style>", unsafe_allow_html=True)