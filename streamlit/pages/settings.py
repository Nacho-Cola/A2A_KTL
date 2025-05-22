# pages/Settings.py
import streamlit as st
from state.state import AppState, SettingsState

# 1) 세션 상태 초기화
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
if "settings_state" not in st.session_state:
    st.session_state.settings_state = SettingsState()

state    = st.session_state.app_state
settings = st.session_state.settings_state

# 페이지 설정
st.set_page_config(page_title="⚙️ Settings", layout="centered")
st.header("⚙️ Settings")

# ─────────────────────────────────────────────────────────────
# 2) 출력 MIME 타입 설정
# ─────────────────────────────────────────────────────────────
st.subheader("출력 MIME Types")
all_mime = [
    "text/plain",
    "application/json",
    "image/png",
    "image/jpeg",
    "application/pdf",
    "text/markdown",
]
selected = st.multiselect(
    "허용할 MIME 타입을 선택하세요",
    options=all_mime,
    default=settings.output_mime_types,
)
# 변경 반영
settings.output_mime_types = selected
st.caption(f"현재: {settings.output_mime_types}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 3) 폴링 간격 설정
# ─────────────────────────────────────────────────────────────
st.subheader("폴링 간격 (초)")
interval = st.number_input(
    "백그라운드 폴링 주기를 초 단위로 입력하세요",
    min_value=1,
    value=state.polling_interval,
)
state.polling_interval = interval
st.caption(f"현재 폴링 간격: {state.polling_interval}초")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 4) API 키 관리
# ─────────────────────────────────────────────────────────────
st.subheader("API Key 설정")
api_input = st.text_input(
    "API Key",
    value=state.api_key,
    type="password",
    help="Google API Key 또는 Vertex AI Key를 입력하세요",
)
state.api_key = api_input

use_vertex = st.checkbox(
    "Vertex AI 사용",
    value=state.uses_vertex_ai,
    help="체크 시 Vertex AI를 사용하고, 해제 시 API Key 기반으로 호출합니다",
)
state.uses_vertex_ai = use_vertex

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 5) 테마 모드 (AppState에 추가로 관리하기 원할 경우)
# ─────────────────────────────────────────────────────────────
st.subheader("테마 모드")
theme = st.selectbox(
    "시스템/라이트/다크 모드를 선택하세요",
    options=["system", "light", "dark"],
    index=["system", "light", "dark"].index(state.theme_mode),
)
state.theme_mode = theme
st.caption(f"현재 테마 모드: {state.theme_mode}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 6) 저장 완료 알림
# ─────────────────────────────────────────────────────────────
if st.button("💾 설정 저장"):
    st.success("설정이 저장되었습니다!")
