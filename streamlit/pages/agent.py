import streamlit as st
import asyncio
import httpx # 직접 사용되지는 않지만, get_agent_card가 내부적으로 사용할 수 있음 (또는 이전 버전)
import logging # 로깅 추가

# streamlit 폴더 구조에 따른 임포트 경로 수정
from utils.agent_card import get_agent_card
from state.host_agent_service import add_remote_agent_service, remove_remote_agent_service # 수정된 함수 이름 사용

logger = logging.getLogger(__name__)

st.set_page_config(page_title="📡 Remote Agents", page_icon="🤖", layout="wide")

def normalize_url(addr: str) -> str:
    """
    - "http:/" 형식을 "http://"로 (https도 동일하게) 수정합니다.
    - 스킴(scheme)이 없으면 "http://"를 앞에 추가합니다.
    - 주소 끝의 슬래시를 제거합니다.
    """
    a = addr.strip()
    # 단일 슬래시 스킴 수정
    if a.startswith("http:/") and not a.startswith("http://"):
        a = "http://" + a[len("http:/"):]
    if a.startswith("https:/") and not a.startswith("https://"):
        a = "https://" + a[len("https:/"):]
    # 누락된 스킴 추가
    if not a.startswith(("http://", "https://")):
        a = "http://" + a
    # 끝 슬래시 제거
    return a.rstrip("/")

async def fetch_and_display_agent_card(address: str):
    """에이전트 카드를 비동기적으로 가져와 UI에 표시합니다."""
    try:
        # get_agent_card가 비동기 함수로 리팩토링되었다고 가정
        card = await get_agent_card(address) # 비동기 호출
        if card:
            name = card.name or address
            desc = card.description or "No description"
            framework = card.provider.organization if card.provider else "N/A"
            # common.types.AgentCard의 defaultInputModes/defaultOutputModes 필드명 확인 필요
            # 예시로 defaultInputModes, defaultOutputModes 사용
            inputs = ", ".join(card.defaultInputModes or []) # None일 경우 빈 리스트 처리
            outputs = ", ".join(card.defaultOutputModes or []) # None일 경우 빈 리스트 처리
            st.markdown(
                f"- **{name}** \n"
                f"  • 주소: `{address}`  \n"
                f"  • 설명: {desc}  \n"
                f"  • 프레임워크: {framework}  \n"
                f"  • 기본 입력 모드: {inputs or 'N/A'}  \n" # None 또는 빈 문자열일 경우 N/A
                f"  • 기본 출력 모드: {outputs or 'N/A'}"
            )
        else:
            st.error(f"⚠️ 에이전트 카드 정보를 가져오는 데 실패했습니다: `{address}`")
    except Exception as e: # get_agent_card 또는 httpx 호출 시 발생할 수 있는 모든 예외 처리
        logger.error(f"Error fetching or displaying agent card for {address}: {e!r}", exc_info=True)
        st.error(f"⚠️ 연결 또는 정보 조회 실패: `{address}`  (Error: {e!r})")


def main():
    st.title("📡 Remote Agents Management")

    # 세션 상태에 에이전트 서버 목록 초기화
    if "agent_servers" not in st.session_state:
        st.session_state.agent_servers = [] # 로컬 UI에서 관리하는 목록

    # 1) 에이전트 추가 폼
    with st.form("add_agent_server_form", clear_on_submit=True): # 폼 키 변경
        raw_addr = st.text_input("A2A 에이전트 서버 주소 (예: host:port 또는 http(s)://host:port)", placeholder="localhost:8080")
        submitted = st.form_submit_button("➕ 에이전트 추가")

        if submitted:
            normalized_addr = normalize_url(raw_addr)
            if not raw_addr.strip():
                st.error("❗ 주소를 입력해주세요.")
            elif normalized_addr in st.session_state.agent_servers:
                st.warning(f"❗ 이미 목록에 추가된 주소입니다: `{normalized_addr}`")
            else:
                # 백엔드에 에이전트 등록 요청 (비동기 함수 호출)
                try:
                    # add_remote_agent_service가 bool을 반환한다고 가정 (리팩토링된 버전)
                    success = asyncio.run(add_remote_agent_service(normalized_addr))
                    if success:
                        st.session_state.agent_servers.append(normalized_addr) # 성공 시 로컬 목록에도 추가
                        st.success(f"✅ 백엔드에 에이전트 등록 요청 성공: `{normalized_addr}`")
                        # st.experimental_rerun() # 또는 st.rerun() # 목록 즉시 반영
                    else:
                        st.error(f"❌ 백엔드 에이전트 등록 요청 실패: `{normalized_addr}`. 서버 로그를 확인하세요.")
                except Exception as e: # asyncio.run 또는 서비스 함수 내부의 예외
                    logger.error(f"Error during AddRemoteAgent for {normalized_addr}: {e}", exc_info=True)
                    st.error(f"❌ 백엔드 등록 중 예기치 않은 오류 발생: {e}")
            st.rerun() # 폼 제출 후 목록 새로고침

    st.markdown("---")
    st.subheader("📝 등록된 A2A 에이전트 서버 목록")

    # 2) 등록된 에이전트 목록 표시 및 삭제 버튼
    if not st.session_state.agent_servers:
        st.info("현재 등록된 에이전트 서버가 없습니다. 위에서 추가해주세요.")
    
    # 목록 순회를 안전하게 하기 위해 복사본 사용 또는 인덱스 기반 삭제 시 주의
    # 여기서는 st.session_state.agent_servers를 직접 수정하므로, 삭제 시 인덱스 문제 발생 가능성 있음
    # 더 안전한 방법은 삭제 후 st.rerun()을 즉시 호출하는 것
    
    for idx, addr_to_display in enumerate(list(st.session_state.agent_servers)): # 순회 중 변경 방지를 위해 복사본 사용
        st.markdown(f"### 에이전트 {idx + 1}")
        cols = st.columns([0.9, 0.1]) # 비율 조정
        with cols[0]:
            # 에이전트 카드 정보 비동기 로드 및 표시
            # Streamlit은 위에서 아래로 스크립트를 실행하므로,
            # 각 에이전트 카드 표시는 개별적으로 이루어짐.
            # 많은 에이전트가 있을 경우 로딩 시간이 길어질 수 있음.
            # 이 부분은 asyncio.gather 등으로 한 번에 가져와서 표시하는 방식으로 최적화 가능
            asyncio.run(fetch_and_display_agent_card(addr_to_display))

        with cols[1]:
            if st.button("🗑️ 삭제", key=f"delete_agent_{addr_to_display}_{idx}", help=f"`{addr_to_display}` 등록 해제", use_container_width=True):
                try:
                    # 백엔드에 에이전트 등록 해제 요청 (비동기 함수 호출)
                    # remove_remote_agent_service가 bool을 반환한다고 가정
                    unregister_success = asyncio.run(remove_remote_agent_service(addr_to_display))
                    if unregister_success:
                        # 로컬 목록에서 제거 (성공 시에만)
                        if addr_to_display in st.session_state.agent_servers:
                             st.session_state.agent_servers.remove(addr_to_display)
                        st.success(f"✅ 백엔드에 에이전트 등록 해제 요청 성공: `{addr_to_display}`")
                    else:
                        st.error(f"❌ 백엔드 에이전트 등록 해제 요청 실패: `{addr_to_display}`. 서버 로그를 확인하세요.")
                except Exception as e:
                    logger.error(f"Error during RemoveRemoteAgent for {addr_to_display}: {e}", exc_info=True)
                    st.error(f"❌ 백엔드 등록 해제 중 예기치 않은 오류 발생: {e}")
                st.rerun() # 목록 변경을 즉시 UI에 반영

if __name__ == "__main__":
    # 기본 로깅 설정 (애플리케이션 전체에 한 번만 수행하는 것이 좋음)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )
    main()