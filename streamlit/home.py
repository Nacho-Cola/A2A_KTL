import os
import nest_asyncio # 애플리케이션 시작 시 매우 초기에 적용
nest_asyncio.apply()

import asyncio # asyncio.run 등을 위해 유지 (다른 비동기 작업이 있을 수 있음)
import streamlit as st
import logging

# Streamlit 프로젝트 구조에 따른 임포트 경로
from state.state import AppState, SettingsState # AppState, SettingsState는 유지
# host_agent_service 임포트는 home.py에서 직접 사용하지 않으면 제거 가능
# from streamlit.state.host_agent_service import (
#     fetch_app_state_service,
#     create_conversation_service,
#     server_url as _default_backend_url,
# )
# import streamlit.state.host_agent_service as host_agent_service

logger = logging.getLogger(__name__)

# --- 설정 (백엔드 URL 등) ---
# 백엔드 URL 설정은 다른 페이지나 설정 모듈에서 관리될 수 있음
# 이 페이지에서 직접적인 백엔드 호출이 없다면, 이 설정은 불필요할 수 있음
# backend_url = os.getenv("A2A_STREAMLIT_BACKEND_URL") or os.getenv("A2A_BACKEND_URL", "http://localhost:12000")
# if host_agent_service.server_url != backend_url: # host_agent_service를 사용하지 않으면 이 부분도 수정/제거
#     logger.info(f"Updating host_agent_service.server_url to: {backend_url}")
#     host_agent_service.server_url = backend_url


# --- 세션 상태 초기화 ---
# AppState와 SettingsState는 다른 페이지에서도 사용될 수 있으므로 초기화 로직 유지
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
    logger.info("Initialized AppState in st.session_state (from home.py).")
if "settings_state" not in st.session_state:
    st.session_state.settings_state = SettingsState()
    logger.info("Initialized SettingsState in st.session_state (from home.py).")


# --- 메인 페이지 UI ---
def home_page_simplified():
    st.set_page_config(
        page_title="🏠 A2A Home",
        layout="wide",
        initial_sidebar_state="auto"
    )

    app_state: AppState = st.session_state.app_state # AppState는 계속 사용 가능

    st.header("🏠 A2A Application Home")
    st.markdown("---")

    st.subheader("Welcome!")
    st.write(
        "A2A (Agent-to-Agent) 애플리케이션에 오신 것을 환영합니다."
    )
    st.write(
        "사이드바 메뉴를 사용하여 다른 기능들 (예: 에이전트 관리, 설정)을 이용할 수 있습니다."
    )
    st.write(
        "대화 기능은 'Conversations' 페이지에서 사용 가능합니다."
    )
    
    # 현재 AppState의 일부 정보 표시 (선택 사항)
    st.markdown("---")
    st.subheader("⚙️ Current Application Info (Example)")
    st.write(f"**Theme Mode:** {app_state.theme_mode}")
    st.write(f"**API Key Set:** {'Yes' if app_state.api_key else 'No'}")
    st.write(f"**Using Vertex AI:** {app_state.uses_vertex_ai}")

    # 대화방 생성 및 목록 확인 부분은 이 페이지에서 제거됨.
    # 해당 기능은 pages/Conversations.py 또는 다른 지정된 페이지에서 담당합니다.

    # --- 기타 링크 또는 정보 ---
    # 예: st.page_link("pages/Conversations.py", label="Go to Conversations", icon="💬")
    # 예: st.page_link("pages/agent.py", label="Manage Agents", icon="🤖")
    # 예: st.page_link("pages/settings.py", label="Settings", icon="⚙️")

    st.markdown("---")
    st.caption("A2A Application - Home Page")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )
    logger.info("Streamlit Home Page (Simplified) starting.")
    home_page_simplified()