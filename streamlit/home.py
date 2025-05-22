# streamlit/home.py
import os
import nest_asyncio
nest_asyncio.apply()

import asyncio
import streamlit as st
from state.state import AppState, SettingsState
from state.host_agent_service import (
    FetchAppState,
    ListRemoteAgents,
    AddRemoteAgent,
    CreateConversation,
    server_url as _default_url,
)

# 백엔드 URL 설정
backend = os.getenv("A2A_BACKEND_URL", _default_url)
import state.host_agent_service as hasvc
hasvc.server_url = backend

# 세션 상태 초기화
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
if "settings_state" not in st.session_state:
    st.session_state.settings_state = SettingsState()

def update_app_state(conversation_id: str = ""):
    loop = asyncio.get_event_loop()
    new_state: AppState = loop.run_until_complete(FetchAppState(conversation_id))
    st.session_state.app_state = new_state

def main():
    st.set_page_config(page_title="🏠 Home", layout="wide")
    app_state: AppState = st.session_state.app_state

    # ── 1) 최초 로드: 대화 목록 가져오기 ──
    if not app_state.conversations:
        update_app_state()

    st.header("🏠 Home Dashboard")

    # ── 2) 대화 선택 UI ──
    st.subheader("💬 Conversations")
    if app_state.conversations:
        # 표시용 이름 리스트와 ID 맵핑 생성
        display_names = [
            c.conversation_name or c.conversation_id
            for c in app_state.conversations
        ]
        conv_map = {
            display: c.conversation_id
            for display, c in zip(display_names, app_state.conversations)
        }
        chosen = st.selectbox("Select a conversation", display_names, key="conv_select")
        chosen_id = conv_map[chosen]

        # “Go” 버튼을 눌러 URL 쿼리파라 세팅
        if st.button("▶️ Open Conversation"):
            st.session_state.selected_conversation = conv_map[chosen]
            conv_id = conv_map[chosen]
            st.query_params["id"] = conv_id  # URL 파라미터 설정
            st.switch_page("pages/conversation.py")
            st.rerun()
    else:
        st.write("_No conversations yet. Click below to create one._")

    # ── 3) 새 대화 생성 ──
    if st.button("➕ New Conversation", type="primary"):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(CreateConversation())
        update_app_state()
        st.rerun()

    # ── 4) 대화 요약 ──
    st.markdown("---")
    st.subheader("📝 Summary of Current Conversation")
    st.write("Current Conversation ID:", app_state.current_conversation_id)
    st.write("Messages count:", len(app_state.messages))

    # ── 5) Remote Agents / Settings 등 기존 코드...
    #    (생략)

if __name__ == "__main__":
    main()
