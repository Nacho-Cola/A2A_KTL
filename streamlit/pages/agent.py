# pages/agents.py
import streamlit as st
import asyncio 
from utils.agent_card import get_agent_card
import httpx

from state.host_agent_service import AddRemoteAgent, RemoveRemoteAgent


st.set_page_config(page_title="📡 Remote Agents", page_icon="🤖", layout="wide")

def normalize_url(addr: str) -> str:
    """
    - Repairs “http:/” → “http://” (and same for https)
    - Prepends “http://” if no scheme is present
    - Strips any trailing slash
    """
    a = addr.strip()
    # fix single-slash schemes
    if a.startswith("http:/") and not a.startswith("http://"):
        a = "http://" + a[len("http:/"):]
    if a.startswith("https:/") and not a.startswith("https://"):
        a = "https://" + a[len("https:/"):]
    # add missing scheme
    if not a.startswith(("http://", "https://")):
        a = "http://" + a
    # strip trailing slash
    return a.rstrip("/")
    
def main():
    st.title("📡 Remote Agents")

    if "agent_servers" not in st.session_state:
        st.session_state.agent_servers = []

    # 1) Add Agent 폼
    with st.form("add_agent_server", clear_on_submit=True):
        raw_addr = st.text_input("Add A2A Server (host:port)", placeholder="localhost:12000")
        if st.form_submit_button("➕ Add Server"):
            addr = normalize_url(raw_addr)
            if not raw_addr.strip():
                st.error("❗ 주소를 입력하세요.")
            elif addr in st.session_state.agent_servers:
                st.warning(f"❗ 이미 목록에 있습니다: `{addr}`")
            else:
                # ① 백엔드에 등록 호출
                try:
                    asyncio.run(AddRemoteAgent(addr))
                    st.success(f"✅ Registered on backend: `{addr}`")
                    st.session_state.agent_servers.append(addr)
                except Exception as e:
                    st.error(f"❌ Backend registration failed: {e}")

    st.markdown("---")
    st.subheader("🔍 Registered A2A Servers")

    # 2) 리스트 출력 + 삭제 버튼
    for idx, addr in enumerate(st.session_state.agent_servers):
        cols = st.columns([9, 1])
        with cols[0]:
            try:
                httpx.get(f"{addr}/.well-known/agent.json", timeout=2.0)
                card = get_agent_card(addr)
                name      = card.name or addr
                desc      = card.description or "No description"
                framework = card.provider.organization if card.provider else "N/A"
                inputs    = ", ".join(card.defaultInputModes)
                outputs   = ", ".join(card.defaultOutputModes)
                st.markdown(
                    f"- **{name}**  \n"
                    f"  • 주소: `{addr}`  \n"
                    f"  • 설명: {desc}  \n"
                    f"  • 프레임워크: {framework}  \n"
                    f"  • 입력 모드: {inputs}  \n"
                    f"  • 출력 모드: {outputs}"
                )
            except Exception as e:
                st.error(f"⚠️ 연결 실패: `{addr}`  (Error: {e!r})")

        with cols[1]:
            if st.button("🗑️", key=f"del_{idx}", use_container_width=True):
                # ② 백엔드에서 unregister 호출
                try:
                    asyncio.run(RemoveRemoteAgent(addr))
                    st.success(f"✅ Unregistered on backend: `{addr}`")
                except Exception as e:
                    st.error(f"❌ Backend unregister failed: {e}")
                # ③ 로컬 목록에서 제거
                st.session_state.agent_servers.pop(idx)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
