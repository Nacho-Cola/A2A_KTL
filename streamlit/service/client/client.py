import json
import logging # 로깅 모듈 추가
from typing import Any, Dict, Optional, Type, TypeVar # 타입 힌팅 개선

import httpx

# service.types 경로는 프로젝트 구조에 맞게 설정되어 있다고 가정합니다.
# demo/ui/service/types.py 를 사용한다고 가정
from service.types import (
    AgentClientHTTPError,
    AgentClientJSONError,
    CreateConversationRequest,
    CreateConversationResponse,
    GetEventRequest,
    GetEventResponse,
    JSONRPCRequest, # 기본 요청 모델
    ListAgentRequest,
    ListAgentResponse,
    ListConversationRequest,
    ListConversationResponse,
    ListMessageRequest,
    ListMessageResponse,
    ListTaskRequest,
    ListTaskResponse,
    PendingMessageRequest,
    PendingMessageResponse,
    RegisterAgentRequest,
    RegisterAgentResponse,
    SendMessageRequest,
    SendMessageResponse,
    # 필요한 다른 요청/응답 타입들도 여기에 추가합니다.
)

logger = logging.getLogger(__name__) # 로거 인스턴스 생성

# 응답 모델을 위한 제네릭 타입 변수
TResponse = TypeVar("TResponse")

class ConversationClient:
    """
    ConversationServer와 상호작용하기 위한 비동기 클라이언트입니다.
    """
    def __init__(self, base_url: str, timeout: float = 10.0): # 타임아웃 기본값 추가
        self.base_url = base_url.rstrip('/')
        # httpx.AsyncClient 인스턴스를 생성하여 재사용합니다.
        # base_url을 여기에 설정하면, 이후 요청 시에는 상대 경로만 사용합니다.
        self._http_client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        logger.info(f"ConversationClient initialized with base_url: {self.base_url}, timeout: {timeout}s")

    async def _send_request(
        self, request_model: JSONRPCRequest, response_model_class: Type[TResponse]
    ) -> TResponse:
        # API 접두사 추가
        api_prefix = "/api/v1" 
        method_path = api_prefix.rstrip('/') + '/' + request_model.method.lstrip('/')
        
        payload_dict = request_model.model_dump(exclude_none=True)
        """
        JSON-RPC 스타일 요청을 보내고 응답을 파싱하는 제네릭 메서드입니다.
        request_model.method는 요청 경로 (예: 'conversation/create')입니다.
        request_model.params는 실제 요청 파라미터 (Pydantic 모델)입니다.
        """
        

        try:
            logger.debug(f"→ POST to method path '{method_path}' with payload: {payload_dict}")
            # self._http_client.post는 base_url과 method_path를 조합하여 전체 URL을 만듭니다.
            response = await self._http_client.post(method_path, json=payload_dict)
            
            logger.debug(f"← Response status: {response.status_code} from '{method_path}'")
            # 응답 텍스트는 DEBUG 레벨이거나 오류 발생 시에만 로깅하여 과도한 로그 방지
            if logger.isEnabledFor(logging.DEBUG) or response.status_code >= 400:
                # 응답 텍스트가 매우 길 수 있으므로, 앞부분만 로깅하거나 요약하는 것을 고려할 수 있습니다.
                response_text_preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
                logger.debug(f"← Response text preview: {response_text_preview}")

            response.raise_for_status() # 4xx/5xx 응답에 대해 HTTPStatusError 발생
            
            response_json = response.json()
            # 응답 JSON이 response_model_class의 구조와 직접 일치한다고 가정
            return response_model_class(**response_json)

        except httpx.HTTPStatusError as e:
            error_message = f"HTTP error {e.response.status_code} for {e.request.method} {e.request.url}."
            try:
                # 서버에서 보낸 상세 오류 메시지 (JSON 형태일 경우) 추출 시도
                err_details = e.response.json()
                error_message += f" Server detail: {err_details.get('detail', e.response.text)}"
            except json.JSONDecodeError: # 서버 응답이 JSON이 아닐 경우
                error_message += f" Server response (non-JSON): {e.response.text}"
            
            logger.error(error_message) # 스택 트레이스는 httpx에서 이미 제공하므로 exc_info=False (기본값)
            raise AgentClientHTTPError(
                status_code=e.response.status_code,
                detail=error_message # 좀 더 구조화된 오류 또는 서버의 detail 필드 사용 고려
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed for response from '{method_path}'. Response text: {response.text}", exc_info=True)
            raise AgentClientJSONError(f"Failed to decode JSON response: {str(e)}. Response text: {response.text}") from e
        except httpx.RequestError as e: # 타임아웃, 네트워크 연결 오류 등 다른 요청 관련 오류 처리
            logger.error(f"Request failed for {e.request.method} {e.request.url}: {str(e)}", exc_info=True)
            # 실제 HTTP 상태 코드가 없으므로, 일반적인 서비스 불가 코드를 사용하거나 별도 처리
            raise AgentClientHTTPError(status_code=503, detail=f"Service unavailable or network error: {str(e)}") from e
        except Exception as e: # 예상치 못한 다른 예외 처리
            logger.exception(f"An unexpected error occurred in _send_request for '{method_path}'")
            raise AgentClientHTTPError(status_code=500, detail=f"An unexpected client-side error occurred: {str(e)}") from e


    async def send_message(self, payload: SendMessageRequest) -> SendMessageResponse:
        return await self._send_request(payload, SendMessageResponse)

    async def create_conversation(self, payload: CreateConversationRequest) -> CreateConversationResponse:
        return await self._send_request(payload, CreateConversationResponse)

    async def list_conversations(self, payload: ListConversationRequest) -> ListConversationResponse:
        # 메서드 이름을 list_conversations로 변경 (복수형 사용)
        return await self._send_request(payload, ListConversationResponse)

    async def get_events(self, payload: GetEventRequest) -> GetEventResponse:
        return await self._send_request(payload, GetEventResponse)

    async def list_messages(self, payload: ListMessageRequest) -> ListMessageResponse:
        return await self._send_request(payload, ListMessageResponse)

    async def get_pending_messages(self, payload: PendingMessageRequest) -> PendingMessageResponse:
        return await self._send_request(payload, PendingMessageResponse)

    async def list_tasks(self, payload: ListTaskRequest) -> ListTaskResponse:
        return await self._send_request(payload, ListTaskResponse)

    async def register_agent(self, payload: RegisterAgentRequest) -> RegisterAgentResponse:
        return await self._send_request(payload, RegisterAgentResponse)

    async def list_agents(self, payload: ListAgentRequest) -> ListAgentResponse:
        response_data = await self._send_request(payload, ListAgentResponse)
        # 원본 코드의 디버그 로깅 (필요시 유지 또는 logger.debug로 변경)
        # logger.debug(f"[list_agents] Parsed response object: {response_data}")
        # result 필드에 대한 로깅은 response_data가 ListAgentResponse 객체이므로,
        # logger.debug(f"[list_agents] Result field: {response_data.result}") 와 같이 접근해야 함.
        if response_data and hasattr(response_data, 'result'):
            logger.debug(f"[list_agents] Raw result field from server: {response_data.result}")
        return response_data

    # 클라이언트 종료 시 내부 HTTP 클라이언트를 닫는 메서드 추가
    async def aclose(self):
        """
        내부 HTTP 클라이언트를 닫습니다. 클라이언트 사용이 끝나면 호출해야 합니다.
        """
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            logger.info("ConversationClient's HTTP client closed.")

    # 비동기 컨텍스트 매니저 구현
    async def __aenter__(self):
        # httpx.AsyncClient는 __init__에서 이미 초기화되었습니다.
        # 필요하다면 여기서 클라이언트 연결 상태를 확인하거나,
        # 지연 초기화(lazy initialization)를 사용할 경우 여기서 초기화할 수 있습니다.
        if self._http_client.is_closed: # 만약 이전에 닫혔다면 재연결 시도 (또는 에러 발생)
            logger.warning("Attempting to re-enter an already closed client. Re-initializing.")
            # self._http_client = httpx.AsyncClient(base_url=self.base_url, timeout=self._http_client.timeout)
            # 위와 같이 재초기화 하거나, 이미 닫힌 클라이언트를 재사용하려는 시도에 대해 에러를 발생시킬 수 있습니다.
            # 일반적으로는 한 번 닫힌 클라이언트는 재사용하지 않습니다.
            # 여기서는 간단히 현재 인스턴스를 반환합니다.
            pass
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 컨텍스트를 나올 때 내부 HTTP 클라이언트를 안전하게 닫습니다.
        await self.aclose()