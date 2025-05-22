import httpx # requests 대신 httpx 사용
import json # JSONDecodeError 처리를 위해
import logging # 로깅 모듈 추가
from urllib.parse import urlparse, urlunparse # URL 파싱 및 재구성

from pydantic import ValidationError # Pydantic 유효성 검사 오류 처리를 위해

# streamlit 폴더 구조에 따라 common.types 경로 수정
from common.types import AgentCard
from typing import Optional

logger = logging.getLogger(__name__) # 로거 인스턴스 생성

# 사용자 정의 예외 (선택 사항이지만, 호출하는 쪽에서 오류를 구분하기 용이)
class AgentCardError(Exception):
    """AgentCard를 가져오는 과정에서 발생하는 기본 에러"""
    pass

class AgentCardFetchError(AgentCardError):
    """네트워크 요청 관련 에러 (연결, 타임아웃 등)"""
    pass

class AgentCardHTTPError(AgentCardError):
    """HTTP 상태 코드 에러 (4xx, 5xx)"""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP Error {status_code}: {detail}")

class AgentCardJSONError(AgentCardError):
    """JSON 파싱 에러"""
    pass

class AgentCardValidationError(AgentCardError):
    """Pydantic 모델 유효성 검사 에러"""
    pass


async def get_agent_card(remote_agent_address: str, timeout: float = 5.0) -> Optional[AgentCard]:
    """
    원격 에이전트 주소에서 /.well-known/agent.json 경로의 메타데이터를 비동기적으로 가져옵니다.
    - 스킴(scheme)이 없으면 http://를 기본값으로 사용합니다.
    - 주소 끝의 슬래시를 제거한 후 표준 경로를 추가합니다.
    - 오류 발생 시 None을 반환하거나, 특정 AgentCardError를 발생시킬 수 있습니다. (여기서는 None 반환)
    """
    addr = remote_agent_address.strip()

    if not addr:
        logger.error("Remote agent address cannot be empty.")
        # 또는 raise ValueError("Remote agent address cannot be empty.")
        return None

    # 스킴이 없으면 http://를 기본값으로 추가
    parsed_url = urlparse(addr)
    if not parsed_url.scheme:
        logger.debug(f"No scheme found in address '{addr}', defaulting to 'http'.")
        # urlunparse를 사용하여 안전하게 URL 재구성 시도
        addr_with_scheme = "http://" + addr
        parsed_url = urlparse(addr_with_scheme) # 다시 파싱하여 netloc 등이 올바르게 설정되도록 함
    
    # path 재구성: netloc까지만 사용하고, 표준 경로 추가
    # 예: "http://localhost:8000/some/path" -> "http://localhost:8000/.well-known/agent.json"
    # 원본 로직: addr.rstrip("/") + "/.well-known/agent.json"
    # urljoin을 사용하거나, netloc까지만 추출하여 경로를 만드는 것이 더 안전할 수 있음
    # 여기서는 원본 로직을 최대한 따르되, parsed_url을 활용하여 명확성 증대
    
    base_addr_for_wellknown = urlunparse((parsed_url.scheme, parsed_url.netloc, '', '', '', '')).rstrip("/")
    target_url = f"{base_addr_for_wellknown}/.well-known/agent.json"

    logger.info(f"Attempting to fetch agent card from: {target_url}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(target_url)
            response.raise_for_status()  # 4xx 또는 5xx 응답 코드일 경우 HTTPStatusError 발생
            
            json_data = response.json()
            agent_card = AgentCard(**json_data) # Pydantic 모델로 파싱 및 유효성 검사
            logger.info(f"Successfully fetched and parsed agent card from {target_url}")
            return agent_card

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text[:200] + "..." if e.response.text and len(e.response.text) > 200 else e.response.text
            logger.error(f"HTTP error {e.response.status_code} while fetching agent card from {target_url}. Detail: {error_detail}", exc_info=False)
            # raise AgentCardHTTPError(e.response.status_code, error_detail) from e # 호출부에서 처리하도록 예외 발생
            return None # 또는 None 반환
        except httpx.TimeoutException as e:
            logger.error(f"Timeout while fetching agent card from {target_url}.", exc_info=False)
            # raise AgentCardFetchError(f"Timeout: {str(e)}") from e
            return None
        except httpx.RequestError as e: # ConnectError, ReadTimeout 등 httpx의 다른 요청 관련 오류
            logger.error(f"Request error while fetching agent card from {target_url}: {type(e).__name__} - {str(e)}", exc_info=False)
            # raise AgentCardFetchError(f"RequestError: {str(e)}") from e
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {target_url}. Error: {str(e)}", exc_info=False)
            # raise AgentCardJSONError(f"JSONDecodeError: {str(e)}") from e
            return None
        except ValidationError as e: # Pydantic 유효성 검사 오류
            logger.error(f"Failed to validate agent card data from {target_url}. Errors: {e.errors()}", exc_info=False)
            # raise AgentCardValidationError(f"ValidationError: {str(e)}") from e
            return None
        except Exception as e: # 기타 예외
            logger.exception(f"An unexpected error occurred while fetching agent card from {target_url}.")
            # raise AgentCardError(f"Unexpected error: {str(e)}") from e
            return None

# 예제 사용법 (Streamlit 앱에서 호출 시)
# async def main_logic_in_streamlit():
#     agent_address = "some-agent.example.com" # 또는 "http://some-agent.example.com"
#     agent_card_data = await get_agent_card(agent_address)
#     if agent_card_data:
#         st.write("Agent Name:", agent_card_data.name)
#         # ... agent_card_data 사용 ...
#     else:
#         st.error(f"Failed to retrieve agent card from {agent_address}")

# # Streamlit에서 비동기 함수 실행 (예: 버튼 클릭 시)
# # import streamlit as st
# # if st.button("Fetch Agent Card"):
# #     asyncio.run(main_logic_in_streamlit()) # Streamlit은 자체적으로 비동기 지원이 제한적일 수 있어,
#                                        # 실제 사용 시에는 Streamlit의 비동기 처리 방식 확인 필요
#                                        # (예: st.experimental_singleton 또는 다른 패턴)