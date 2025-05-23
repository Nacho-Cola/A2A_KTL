from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, TypeAdapter

# common.types 모듈은 PYTHONPATH에 설정되어 접근 가능하다고 가정합니다.
# 이 모듈에서 AgentCard, JSONRPCRequest, JSONRPCResponse, Message, Task 등을 가져옵니다.
from common.types import (
    AgentCard,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    Task,
)


class Conversation(BaseModel):
    conversation_id: str
    is_active: bool
    name: str = Field(default='', description="대화의 이름 또는 요약")
    task_ids: List[str] = Field(default_factory=list, description="이 대화에 연관된 태스크 ID 목록")
    messages: List[Message] = Field(default_factory=list, description="이 대화에 포함된 메시지 목록")


class Event(BaseModel):
    id: str
    actor: str = Field(default='', description="이벤트를 발생시킨 주체 (예: 'user', 'agent', 'system')")
    content: Message # 이벤트 내용은 Message 객체로 표현
    timestamp: float = Field(description="이벤트 발생 Unix 타임스탬프 (UTC)")


class SendMessageRequest(JSONRPCRequest):
    method: Literal['message/send'] = 'message/send'
    params: Message # 전송할 메시지 객체


class ListMessageRequest(JSONRPCRequest):
    method: Literal['message/list'] = 'message/list'
    params: str  # 조회할 대화의 ID


class ListMessageResponse(JSONRPCResponse):
    result: Optional[List[Message]] = Field(default=None, description="메시지 목록 또는 없음")


class MessageInfo(BaseModel): # 이 모델은 SendMessageResponse에서 직접 사용되지 않을 수 있음
    message_id: str
    conversation_id: str
    task_id: Optional[str] = None # task_id를 포함할 수 있도록 확장 (선택적)


class SendMessageResponse(JSONRPCResponse):
    # server.py에서 task_id를 포함한 Dict를 반환하므로, result 타입을 Dict로 변경
    result: Optional[Dict[str, Any]] = Field(default=None, description="메시지 전송 결과 (message_id, conversation_id, task_id 포함 가능)")


class GetEventRequest(JSONRPCRequest):
    method: Literal['events/get'] = 'events/get'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict) # 필요시 추가


class GetEventResponse(JSONRPCResponse):
    result: Optional[List[Event]] = Field(default=None, description="이벤트 목록 또는 없음")


class ListConversationRequest(JSONRPCRequest):
    method: Literal['conversation/list'] = 'conversation/list'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ListConversationResponse(JSONRPCResponse):
    result: Optional[List[Conversation]] = Field(default=None, description="대화 목록 또는 없음")


class PendingMessageRequest(JSONRPCRequest):
    method: Literal['message/pending'] = 'message/pending'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PendingMessageResponse(JSONRPCResponse):
    result: Optional[List[Tuple[str, str]]] = Field(default=None, description="처리 중인 메시지 정보 목록 (message_id, status_string) 또는 없음")


class CreateConversationRequest(JSONRPCRequest):
    method: Literal['conversation/create'] = 'conversation/create'
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="새 대화 생성 시 초기 파라미터 (현재는 사용되지 않음)")


class CreateConversationResponse(JSONRPCResponse):
    result: Optional[Conversation] = Field(default=None, description="생성된 대화 객체 또는 없음")


class ListTaskRequest(JSONRPCRequest):
    method: Literal['task/list'] = 'task/list'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ListTaskResponse(JSONRPCResponse):
    result: Optional[List[Task]] = Field(default=None, description="태스크 목록 또는 없음")


class RegisterAgentRequest(JSONRPCRequest):
    method: Literal['agent/register'] = 'agent/register'
    params: Optional[str] = Field(default=None, description="등록할 에이전트 카드 정보 URL")


class RegisterAgentResponse(JSONRPCResponse):
    result: Optional[str] = Field(default=None, description="등록 결과 (예: 에이전트 ID 또는 성공 메시지) 또는 없음")


class UnregisterAgentRequest(JSONRPCRequest):
    method: Literal['agent/unregister'] = 'agent/unregister'
    params: str = Field(description="등록 해제할 에이전트의 URL")


class UnregisterAgentResponse(JSONRPCResponse):
    result: Optional[bool] = Field(default=True, description="등록 해제 성공 여부 (기본값: True)")


class ListAgentRequest(JSONRPCRequest):
    method: Literal['agent/list'] = 'agent/list'
    params: Optional[Dict[str, Any]] = Field(default_factory=dict) # params가 있을 수 있으므로 정의


class ListAgentResponse(JSONRPCResponse): # 이전에 누락되었던 ListAgentResponse 정의
    result: Optional[List[AgentCard]] = Field(default=None, description="에이전트 카드 목록 또는 없음")


# AgentRequestUnion 정의 시점에 모든 관련 모델이 위에 정의되어 있어야 함
_request_models_for_union = (
    SendMessageRequest,
    ListConversationRequest,
    ListMessageRequest,
    GetEventRequest,
    PendingMessageRequest,
    CreateConversationRequest,
    ListTaskRequest,
    RegisterAgentRequest,
    UnregisterAgentRequest,
    ListAgentRequest,
)

AgentRequestUnion = Annotated[
    Union[_request_models_for_union], # 문자열 대신 실제 타입 사용
    Field(discriminator='method'),
]
AgentRequest = TypeAdapter(AgentRequestUnion)


# 클라이언트 측에서 사용할 사용자 정의 예외 타입
class AgentClientError(Exception):
    """Agent client 관련 기본 예외"""
    pass


class AgentClientHTTPError(AgentClientError):
    """HTTP 오류 발생 시 사용될 예외"""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail 
        super().__init__(f'HTTP Error {status_code}: {detail}')


class AgentClientJSONError(AgentClientError):
    """JSON 파싱 오류 발생 시 사용될 예외"""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f'JSON Decode Error: {detail}')
