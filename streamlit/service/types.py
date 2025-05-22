from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union # Any, Dict, List, Union 추가

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
    # TODO: 모델 내부 개념(예: 함수 호출)을 지원하도록 확장 필요
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


class MessageInfo(BaseModel):
    message_id: str
    conversation_id: str


class SendMessageResponse(JSONRPCResponse):
    # 메시지 전송 결과로 전체 메시지 객체 또는 간략한 정보(ID 등)를 반환할 수 있음
    result: Optional[Union[Message, MessageInfo]] = Field(default=None)


class GetEventRequest(JSONRPCRequest):
    method: Literal['events/get'] = 'events/get'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="이벤트 필터링 조건 (예: conversation_id)")
    # 현재는 params가 없으므로, 필요시 위와 같이 추가


class GetEventResponse(JSONRPCResponse):
    result: Optional[List[Event]] = Field(default=None, description="이벤트 목록 또는 없음")


class ListConversationRequest(JSONRPCRequest):
    method: Literal['conversation/list'] = 'conversation/list'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="대화 목록 필터링 조건")


class ListConversationResponse(JSONRPCResponse):
    result: Optional[List[Conversation]] = Field(default=None, description="대화 목록 또는 없음")


class PendingMessageRequest(JSONRPCRequest):
    method: Literal['message/pending'] = 'message/pending'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="처리 중 메시지 필터링 조건")


class PendingMessageResponse(JSONRPCResponse):
    # ADKHostManager는 List[Tuple[str, str]]을, InMemoryFakeAgentManager는 List[str]을 반환.
    # ApplicationManager 인터페이스는 List[str]을 정의.
    # 여기서는 ADKHostManager의 반환 타입을 따르지만, 전반적인 일관성 확보 필요.
    result: Optional[List[Tuple[str, str]]] = Field(default=None, description="처리 중인 메시지 정보 목록 (message_id, status_string) 또는 없음")


class CreateConversationRequest(JSONRPCRequest):
    method: Literal['conversation/create'] = 'conversation/create'
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="새 대화 생성 시 초기 파라미터 (현재는 사용되지 않음)")


class CreateConversationResponse(JSONRPCResponse):
    result: Optional[Conversation] = Field(default=None, description="생성된 대화 객체 또는 없음")


class ListTaskRequest(JSONRPCRequest):
    method: Literal['task/list'] = 'task/list'
    # params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="태스크 목록 필터링 조건")


class ListTaskResponse(JSONRPCResponse):
    result: Optional[List[Task]] = Field(default=None, description="태스크 목록 또는 없음")


class RegisterAgentRequest(JSONRPCRequest):
    method: Literal['agent/register'] = 'agent/register'
    params: Optional[str] = Field(default=None, description="등록할 에이전트 카드 정보 URL") # 원본에서는 str | None


class RegisterAgentResponse(JSONRPCResponse):
    # result는 성공 여부 또는 등록된 에이전트 정보 등을 나타낼 수 있음.
    # 원본에서는 result: str | None = None. 에이전트 ID나 성공 메시지 문자열로 가정.
    result: Optional[str] = Field(default=None, description="등록 결과 (예: 에이전트 ID 또는 성공 메시지) 또는 없음")
    # error 필드는 Base JSONRPCResponse 에서 상속받아 사용 (common.types.JSONRPCResponse 정의에 따름)


# UnregisterAgentRequest: JSONRPCRequest를 상속하도록 변경
class UnregisterAgentRequest(JSONRPCRequest):
    method: Literal['agent/unregister'] = 'agent/unregister'
    params: str = Field(description="등록 해제할 에이전트의 URL")


# UnregisterAgentResponse: JSONRPCResponse를 상속하고 result 필드를 갖도록 변경
class UnregisterAgentResponse(JSONRPCResponse):
    result: Optional[bool] = Field(default=True, description="등록 해제 성공 여부 (기본값: True)")
    # error 필드는 Base JSONRPCResponse 에서 상속받아 사용 (common.types.JSONRPCResponse 정의에 따름)
    # 만약 common.types.JSONRPCResponse에 error 필드가 없다면, 여기에 직접 정의:
    # error: Optional[str] = Field(default=None, description="오류 발생 시 메시지")


class ListAgentRequest(JSONRPCRequest):
    method: Literal['agent/list'] = 'agent/list'
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

# AgentRequest TypeAdapter: 여러 요청 타입을 판별하여 처리할 때 사용 가능
# 현재 SendMessageRequest와 ListConversationRequest만 포함되어 있음. 필요시 확장.
AgentRequestUnion = Annotated[
    Union[
        'SendMessageRequest', # 문자열로 변경
        'ListConversationRequest', # 문자열로 변경
        'ListMessageRequest', # 문자열로 변경
        'GetEventRequest', # 문자열로 변경
        'PendingMessageRequest', # 문자열로 변경
        'CreateConversationRequest', # 문자열로 변경
        'ListTaskRequest', # 문자열로 변경
        'RegisterAgentRequest', # 문자열로 변경
        'UnregisterAgentRequest', # 문자열로 변경
        'ListAgentRequest', # 문자열로 변경
    ],
    Field(discriminator='method'),
]
AgentRequest = TypeAdapter(AgentRequestUnion)


# 클라이언트 측에서 사용할 사용자 정의 예외 타입
class AgentClientError(Exception):
    """Agent client 관련 기본 예외"""
    pass


class AgentClientHTTPError(AgentClientError):
    """HTTP 오류 발생 시 사용될 예외"""
    def __init__(self, status_code: int, detail: str): # message -> detail로 변경 (FastAPI HTTPException과 유사하게)
        self.status_code = status_code
        self.detail = detail # 상세 오류 메시지
        super().__init__(f'HTTP Error {status_code}: {detail}')


class AgentClientJSONError(AgentClientError):
    """JSON 파싱 오류 발생 시 사용될 예외"""
    def __init__(self, detail: str): # message -> detail로 변경
        self.detail = detail
        super().__init__(f'JSON Decode Error: {detail}')



class ListAgentResponse(JSONRPCResponse):
    result: Optional[List[AgentCard]] = Field(default=None, description="에이전트 카드 목록 또는 없음")