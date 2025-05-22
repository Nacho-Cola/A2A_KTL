import asyncio
import json
import os
import sys # traceback을 위해 유지
import traceback # traceback을 위해 유지
import logging
import uuid # ID 생성 등에 사용될 수 있음
from typing import Any, Dict, List, Optional, Union, Tuple

# streamlit 폴더 구조에 따른 임포트 경로 수정
from common.types import Message, Part, Task, AgentCard # AgentCard 추가
from service.client.client import ConversationClient
from service.types import (
    Conversation,
    CreateConversationRequest,
    Event,
    GetEventRequest,
    ListAgentRequest,
    ListAgentResponse,
    ListConversationRequest,
    # ListConversationResponse, # 클라이언트 메서드가 Conversation 리스트를 직접 반환한다고 가정
    ListMessageRequest,
    # ListMessageResponse, # 클라이언트 메서드가 Message 리스트를 직접 반환한다고 가정
    ListTaskRequest,
    # ListTaskResponse, # 클라이언트 메서드가 Task 리스트를 직접 반환한다고 가정
    PendingMessageRequest,
    # PendingMessageResponse, # 클라이언트 메서드가 적절한 타입을 반환한다고 가정 (예: Dict)
    RegisterAgentRequest,
    # RegisterAgentResponse, # 클라이언트 메서드가 bool 또는 상세 정보를 반환한다고 가정
    SendMessageRequest,
    # SendMessageResponse, # 클라이언트 메서드가 Optional[str] 또는 상세 정보를 반환한다고 가정
    UnregisterAgentRequest, 
    UnregisterAgentResponse, # service.types에 정의되어 있다고 가정
)

# streamlit/state/state.py 에서 AppState 및 State* 모델들을 가져옴
# 이 모델들은 일반 Pydantic 모델 (Mesop 의존성 없음)이라고 가정
from state.state import (
    AppState,
    SessionTask,
    StateConversation,
    StateEvent,
    StateMessage,
    StateTask,
)

logger = logging.getLogger(__name__)

# 백엔드 서버 URL 환경 변수 또는 기본값 설정
_env_url = os.environ.get("A2A_STREAMLIT_BACKEND_URL") or os.environ.get("A2A_BACKEND_URL")
server_url = _env_url if _env_url else 'http://localhost:12000' # Streamlit 앱이 통신할 백엔드 서버 주소
logger.info(f"HostAgentService (Streamlit context) using server_url: {server_url}")


# --- API 호출 서비스 함수들 ---

async def list_conversations_service() -> List[Conversation]:
    """백엔드에서 대화 목록을 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.list_conversations 가 ListConversationResponse를 반환한다고 가정
            response = await client.list_conversations(ListConversationRequest())
            conversations = response.result if response and response.result else []
            logger.debug(f"Listed conversations: {len(conversations)} found.")
            return conversations
        except Exception as e:
            logger.exception("Failed to list conversations from backend.")
            return []


async def send_message_service(message: Message) -> Optional[str]:
    """메시지를 백엔드로 전송합니다. 성공 시 메시지 ID 또는 관련 식별자 반환."""
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.send_message 가 SendMessageResponse를 반환한다고 가정
            response = await client.send_message(SendMessageRequest(params=message))
            if response and response.result:
                if hasattr(response.result, 'message_id'): # MessageInfo 타입일 경우
                    result_id = response.result.message_id # type: ignore
                    logger.info(f"Message sent successfully, received message_id: {result_id}")
                    return result_id
                # Message 객체 등 다른 타입의 result에 대한 처리는 현재 생략
                logger.warning(f"Message sent, but result format unexpected: {type(response.result)}")
            return None
        except Exception as e:
            msg_id_for_log = message.metadata.get('message_id', 'N/A') if message.metadata else 'N/A'
            logger.exception(f"Failed to send message (local ID: {msg_id_for_log}).")
            return None


async def create_conversation_service() -> Optional[Conversation]:
    """새 대화를 백엔드에 생성 요청합니다."""
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.create_conversation 가 CreateConversationResponse를 반환한다고 가정
            response = await client.create_conversation(CreateConversationRequest())
            new_conversation = response.result if response else None
            if new_conversation:
                logger.info(f"Conversation created successfully: {new_conversation.conversation_id}")
            else:
                logger.warning("Create conversation call returned no result.")
            return new_conversation
        except Exception as e:
            logger.exception("Failed to create conversation.")
            return None


async def list_remote_agents_service() -> List[AgentCard]:
    """백엔드에서 원격 에이전트 목록을 가져옵니다."""
    logger.debug(f"Listing remote agents from server_url: {server_url}")
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.list_agents 가 ListAgentResponse를 반환한다고 가정
            response = await client.list_agents(ListAgentRequest())
            agents = response.result if response and response.result else []
            logger.debug(f"ListRemoteAgents received: {len(agents)} agents.")
            return agents
        except Exception as e:
            logger.exception("Failed to list remote agents.")
            return []


async def add_remote_agent_service(path: str) -> bool:
    """원격 에이전트를 백엔드에 등록합니다."""
    logger.info(f"Registering remote agent from path: {path}")
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.register_agent 가 RegisterAgentResponse를 반환한다고 가정
            response = await client.register_agent(RegisterAgentRequest(params=path))
            if response and response.result is not None and not response.error: # common.types.JSONRPCResponse의 error 필드 확인
                logger.info(f"Agent registration successful for {path}. Result: {response.result}")
                return True
            else:
                err_msg = response.error if response and response.error else "Unknown error"
                logger.error(f"Failed to register agent at {path}. Error: {err_msg}, Result: {response.result if response else 'N/A'}")
                return False
        except Exception as e:
            logger.exception(f"Exception during agent registration for path: {path}")
            return False

async def remove_remote_agent_service(path: str) -> bool:
    """원격 에이전트를 백엔드에서 등록 해제합니다."""
    logger.info(f"Unregistering remote agent from path: {path}")
    async with ConversationClient(server_url) as client:
        try:
            # ConversationClient에 _send_request를 직접 사용하거나, unregister_agent 메서드가 있다고 가정
            # streamlit.service.types 에 UnregisterAgentRequest, UnregisterAgentResponse 가 정의되어 있어야 함
            response = await client._send_request(UnregisterAgentRequest(params=path), UnregisterAgentResponse)
            
            if response and response.result and not response.error: # JSONRPCResponse 표준 에러 필드 확인
                logger.info(f"Agent unregistration successful for {path}.")
                return True
            else:
                err_msg = response.error if response and response.error else "Unknown error during unregistration"
                logger.error(f"Failed to unregister agent at {path}. Error: {err_msg}")
                return False
        except AttributeError:
            logger.error(f"ConversationClient does not support the requested unregistration method for path: {path}.")
            return False
        except Exception as e:
            logger.exception(f"Exception during agent unregistration for path: {path}")
            return False


async def get_events_service() -> List[Event]:
    """백엔드에서 이벤트 목록을 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.get_events 가 GetEventResponse를 반환한다고 가정
            response = await client.get_events(GetEventRequest())
            events = response.result if response and response.result else []
            logger.debug(f"Retrieved {len(events)} events.")
            return events
        except Exception as e:
            logger.exception("Failed to get events.")
            return []


async def get_processing_messages_service() -> Dict[str, str]:
    """백엔드에서 처리 중인 메시지 상태를 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.get_pending_messages 가 PendingMessageResponse를 반환한다고 가정
            response = await client.get_pending_messages(PendingMessageRequest())
            # service.types.PendingMessageResponse.result 타입은 List[Tuple[str, str]] | None
            if response and response.result and isinstance(response.result, list):
                if all(isinstance(item, tuple) and len(item) == 2 and 
                       isinstance(item[0], str) and isinstance(item[1], str) for item in response.result):
                    processed_dict = dict(response.result)
                    logger.debug(f"Retrieved {len(processed_dict)} processing messages statuses.")
                    return processed_dict
                # ApplicationManager 인터페이스가 List[str]을 반환하도록 정의했고, 클라이언트가 이를 따른다면 아래 로직 필요
                # elif all(isinstance(item, str) for item in response.result):
                #     logger.warning("get_pending_messages returned list[str]. Converting to dict with dummy keys.")
                #     return {f"status_{i}": status for i, status in enumerate(response.result)}
            logger.debug("No processing messages statuses retrieved or result was not in expected format.")
            return {}
        except Exception as e:
            logger.exception("Error getting pending messages statuses.")
            return {}


def get_message_aliases_service() -> Dict[Any, Any]: # 함수명 명확화
    """메시지 별칭 정보를 반환합니다 (현재는 빈 딕셔너리)."""
    logger.debug("get_message_aliases_service called, returning empty dict.")
    return {}


async def get_tasks_service() -> List[Task]: # 함수명 명확화
    """백엔드에서 태스크 목록을 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.list_tasks 가 ListTaskResponse를 반환한다고 가정
            response = await client.list_tasks(ListTaskRequest())
            tasks = response.result if response and response.result else []
            logger.debug(f"Retrieved {len(tasks)} tasks.")
            return tasks
        except Exception as e:
            logger.exception("Failed to list tasks.")
            return []


async def list_messages_service(conversation_id: str) -> List[Message]: # 함수명 명확화
    """특정 대화의 메시지 목록을 백엔드에서 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            # streamlit.service.client.client.list_messages 가 ListMessageResponse를 반환한다고 가정
            response = await client.list_messages(
                ListMessageRequest(params=conversation_id)
            )
            messages = response.result if response and response.result else []
            logger.debug(f"Retrieved {len(messages)} messages for conv_id {conversation_id}.")
            return messages
        except Exception as e:
            logger.exception(f"Failed to list messages for conversation_id: {conversation_id}")
            return []

# --- AppState 관련 함수들 ---
# AppState 및 State* 타입은 streamlit.state.state 모듈의 Pydantic 모델이라고 가정합니다.
# 이 서비스 계층은 UI 프레임워크에 독립적으로 유지되며, Pydantic 모델을 반환합니다.
# Streamlit UI 레이어는 이 AppState 객체를 받아 st.session_state에 필요한 데이터를 채웁니다.

async def fetch_app_state_service(conversation_id: Optional[str]) -> AppState: # 함수명 명확화
    """
    여러 API 호출을 통해 전체 UI 상태를 구성하는 AppState 객체를 가져옵니다.
    (AppState는 일반 Pydantic 모델이라고 가정)
    """
    state = AppState() # AppState Pydantic 모델의 기본값으로 초기화
    logger.info(f"Fetching app state for conversation_id: {conversation_id}")

    try:
        # 독립적인 API 호출들을 병렬로 실행
        message_coro = list_messages_service(conversation_id) if conversation_id else asyncio.sleep(0, result=[]) # type: ignore
        
        results = await asyncio.gather(
            message_coro,
            list_conversations_service(),
            get_tasks_service(),
            get_processing_messages_service(),
            return_exceptions=True # 개별 호출의 예외를 반환받아 처리
        )

        # 결과 및 예외 처리
        messages_res, conversations_res, tasks_res, processing_messages_res = results

        if isinstance(messages_res, Exception):
            logger.error("Failed to fetch messages for AppState.", exc_info=messages_res)
            state.messages = []
        else:
            state.messages = [convert_message_to_state(m) for m in messages_res] if messages_res else []
        
        if conversation_id: # 제공된 경우 현재 대화 ID 설정
            state.current_conversation_id = conversation_id

        if isinstance(conversations_res, Exception):
            logger.error("Failed to fetch conversations for AppState.", exc_info=conversations_res)
            state.conversations = []
        else:
            state.conversations = [convert_conversation_to_state(c) for c in conversations_res] if conversations_res else []

        if isinstance(tasks_res, Exception):
            logger.error("Failed to fetch tasks for AppState.", exc_info=tasks_res)
            state.task_list = []
        else:
            state.task_list = [
                SessionTask(
                    session_id=extract_conversation_id_from_task(t), # 헬퍼 함수명 변경 가능성
                    task=convert_task_to_state(t)
                )
                for t in tasks_res or []
            ]
        
        if isinstance(processing_messages_res, Exception):
            logger.error("Failed to fetch processing messages for AppState.", exc_info=processing_messages_res)
            state.background_tasks = {}
        else:
            state.background_tasks = processing_messages_res or {}
        
        state.message_aliases = get_message_aliases_service() # 동기 함수 호출

        logger.info("App state fetch complete.")

    except Exception as e: # gather 또는 AppState 생성 중 예기치 않은 오류 처리
        logger.exception("Critical error during fetch_app_state_service construction.")
        # 부분적으로 채워졌거나 기본 AppState 반환 (UI 비정상 종료 방지)
    return state


async def update_api_key_on_server_service(api_key: str) -> bool: # 함수명 명확화
    """백엔드 서버의 API 키를 업데이트합니다."""
    if not api_key or not isinstance(api_key, str): # 유효성 검사
        logger.error("Invalid API key provided for update: must be a non-empty string.")
        return False
        
    logger.info(f"Attempting to update API key on server: {server_url}/api_key/update")
    # 이 함수는 ConversationClient 대신 직접 httpx.AsyncClient를 사용합니다.
    # 이 엔드포인트가 특별하거나, ConversationClient에 관련 메서드가 없는 경우일 수 있습니다.
    async with httpx.AsyncClient() as client: # 직접 httpx 클라이언트 사용
        try:
            response = await client.post(
                f'{server_url}/api_key/update', json={'api_key': api_key}
            )
            response.raise_for_status() # 4xx/5xx 상태 코드에 대해 예외 발생
            
            response_data = response.json()
            if response_data.get("status") == "success":
                logger.info("API key updated successfully on the server.")
                # 이 서비스(Streamlit 앱의 일부)가 직접 다른 Google 서비스를 호출한다면
                # 로컬 환경 변수 업데이트가 필요할 수 있습니다.
                # 서버 측 ADKHostManager는 자체적으로 환경 변수를 관리합니다.
                os.environ['GOOGLE_API_KEY'] = api_key
                logger.debug("Local GOOGLE_API_KEY environment variable also updated (if used by this service directly).")
                return True
            else:
                err_msg = response_data.get('message', 'Unknown error from server')
                logger.error(f"Server responded with an error during API key update: {err_msg}")
                return False
        except httpx.HTTPStatusError as e:
            # 응답 내용 로깅 시 주의 (민감 정보 포함 가능성)
            response_text_preview = e.response.text[:200] + "..." if e.response.text and len(e.response.text) > 200 else e.response.text
            logger.exception(f"HTTP error when updating API key on server: {e.response.status_code} - Response: {response_text_preview}")
            return False
        except Exception as e:
            logger.exception("Generic error when updating API key on server.")
            return False

# --- 데이터 변환 및 추출 헬퍼 함수들 ---
# State* 타입은 streamlit.state.state의 Pydantic 모델이라고 가정합니다.

def convert_message_to_state(message: Optional[Message]) -> StateMessage:
    """API 응답 Message 객체를 UI 상태 StateMessage 객체로 변환합니다."""
    if not message:
        logger.debug("convert_message_to_state received None, returning empty StateMessage.")
        return StateMessage() # StateMessage가 기본 초기화 값을 갖는다고 가정

    # message_id가 없으면 새로 생성하거나 오류 처리 (여기서는 UUID로 임시 ID 생성)
    message_id = extract_message_id_from_message(message) or f"generated_{uuid.uuid4()}"
    # role이 없으면 'unknown'으로 설정
    role = message.role or "unknown"
    # parts가 None이면 빈 리스트로 처리
    content_parts = extract_content_from_parts(message.parts or [])
    
    return StateMessage(
        message_id=message_id,
        role=role,
        content=content_parts,
        # StateMessage의 다른 필드들도 여기서 매핑해야 함
    )


def convert_conversation_to_state(conversation: Conversation) -> StateConversation:
    """API 응답 Conversation 객체를 UI 상태 StateConversation 객체로 변환합니다."""
    # messages가 None이면 빈 리스트로 처리하고, 각 메시지에서 ID 추출 시 None 방지
    message_ids = [
        msg_id for msg_id in (extract_message_id_from_message(x) for x in (conversation.messages or [])) if msg_id
    ]
    
    return StateConversation(
        conversation_id=conversation.conversation_id,
        conversation_name=conversation.name or "Untitled Conversation", # 이름 없으면 기본값
        is_active=conversation.is_active,
        message_ids=message_ids,
        # StateConversation의 다른 필드들도 여기서 매핑해야 함
    )


def convert_task_to_state(task: Task) -> StateTask:
    """API 응답 Task 객체를 UI 상태 StateTask 객체로 변환합니다."""
    # history가 비어있거나 None일 수 있으므로 방어 코드 추가
    message = task.history[0] if task.history else None
    last_message = task.history[-1] if task.history else None
    
    output_parts_content: List[Union[str, Dict[str, Any]]] = [] # StateTask.artifacts 타입에 맞춰야 함
    if task.artifacts:
        for artifact_item in task.artifacts:
            # extract_content_from_parts는 List[Tuple[Union[str, Dict], str]] 반환
            # StateTask.artifacts의 타입에 맞게 변환 필요
            # 예시: 첫 번째 content 항목만 문자열로 저장
            extracted_artifact_contents = extract_content_from_parts(artifact_item.parts or [])
            if extracted_artifact_contents:
                first_content_tuple = extracted_artifact_contents[0]
                # first_content_tuple[0]은 Union[str, Dict[str, Any]]
                output_parts_content.append(first_content_tuple[0]) 
    
    # 마지막 메시지가 첫 메시지와 다르고, 내용이 있으면 output_parts_content 앞에 추가
    if last_message and last_message != message and last_message.parts:
        last_message_extracted_content = extract_content_from_parts(last_message.parts)
        if last_message_extracted_content:
            first_content_tuple_last_msg = last_message_extracted_content[0]
            output_parts_content.insert(0, first_content_tuple_last_msg[0])

    return StateTask(
        task_id=task.id,
        session_id=task.sessionId or extract_conversation_id_from_task(task), # sessionId 없으면 다른 곳에서 추출
        state=str(task.status.state) if task.status else "UNKNOWN_STATE", # status 없으면 기본값
        message=convert_message_to_state(message) if message else StateMessage(), # message 없으면 빈 StateMessage
        artifacts=output_parts_content, # StateTask.artifacts 타입과 일치해야 함
        # StateTask의 다른 필드들도 여기서 매핑해야 함
    )


def convert_event_to_state(event: Event) -> StateEvent:
    """API 응답 Event 객체를 UI 상태 StateEvent 객체로 변환합니다."""
    # event.content (Message 타입)에서 conversation_id 추출
    conversation_id = extract_message_conversation_id(event.content) or "unknown_conversation"
    
    return StateEvent(
        conversation_id=conversation_id,
        actor=event.actor or "unknown_actor", # actor 없으면 기본값
        role=event.content.role or "unknown_role", # role 없으면 기본값
        id=event.id,
        content=extract_content_from_parts(event.content.parts or []), # parts 없으면 빈 리스트 전달
        # StateEvent의 다른 필드들도 여기서 매핑해야 함
    )


def extract_content_from_parts(message_parts: List[Part]) -> List[Tuple[Union[str, Dict[str, Any]], str]]:
    """Message의 Part 리스트에서 내용을 추출하여 (내용, MIME타입) 튜플 리스트로 반환합니다."""
    extracted_content_list: List[Tuple[Union[str, Dict[str, Any]], str]] = []
    if not message_parts:
        return []
    for p_item in message_parts: # 변수명 변경 (p -> p_item)
        # 각 Part 타입에 따라 안전하게 속성 접근
        part_type = getattr(p_item, 'type', None)
        if part_type == 'text' and hasattr(p_item, 'text') and p_item.text is not None:
            extracted_content_list.append((p_item.text, 'text/plain'))
        elif part_type == 'file' and hasattr(p_item, 'file') and p_item.file:
            file_obj = p_item.file # FileContent 객체
            mime_type = file_obj.mimeType or "application/octet-stream"
            if hasattr(file_obj, 'bytes') and file_obj.bytes is not None: # Base64 인코딩된 문자열
                extracted_content_list.append((file_obj.bytes, mime_type))
            elif hasattr(file_obj, 'uri') and file_obj.uri is not None:
                extracted_content_list.append((file_obj.uri, mime_type))
            else:
                logger.warning(f"FilePart (type 'file') is missing both 'bytes' and 'uri' in file object: {file_obj}")
                extracted_content_list.append(("<file_content_unavailable>", "text/plain"))
        elif part_type == 'data' and hasattr(p_item, 'data') and p_item.data is not None:
            data_content = p_item.data
            try:
                if isinstance(data_content, dict) and data_content.get('type') == 'form':
                    extracted_content_list.append((data_content, 'form')) # 딕셔너리 자체를 내용으로
                else:
                    json_data_str = json.dumps(data_content)
                    extracted_content_list.append((json_data_str, 'application/json'))
            except TypeError as e: # json.dumps 실패 시 (예: 직렬화 불가능한 객체 포함)
                logger.error(f"Failed to serialize data part to JSON: {data_content}. Error: {e}", exc_info=True)
                extracted_content_list.append((f"<unserializable_data_part: {type(data_content)}>", 'text/plain'))
        else:
            logger.warning(f"Unsupported or malformed Part encountered: type='{part_type}', content='{str(p_item)[:100]}'")
            extracted_content_list.append(("<unknown_or_malformed_part>", "text/plain"))
    return extracted_content_list


def extract_message_id_from_message(message: Optional[Message]) -> Optional[str]: # 함수명 명확화
    """Message 객체에서 message_id를 추출합니다."""
    if message and message.metadata and 'message_id' in message.metadata:
        return message.metadata['message_id']
    return None


def extract_message_conversation_id(message: Optional[Message]) -> Optional[str]:
    """Message 객체에서 conversation_id를 추출합니다."""
    if message and message.metadata and 'conversation_id' in message.metadata:
        return message.metadata['conversation_id']
    return None


def extract_conversation_id_from_task(task: Task) -> Optional[str]:
    """Task 객체에서 conversation_id (또는 sessionId)를 추출합니다."""
    if task.sessionId:
        return task.sessionId
    
    if task.status and task.status.message:
        conv_id = extract_message_conversation_id(task.status.message)
        if conv_id:
            return conv_id
            
    if task.metadata:
        conv_id = task.metadata.get('conversation_id')
        if conv_id:
            return conv_id
            
    if task.artifacts:
        for artifact_item in task.artifacts:
            if artifact_item.metadata:
                conv_id = artifact_item.metadata.get('conversation_id')
                if conv_id:
                    return conv_id
    logger.debug(f"Could not extract conversation_id from task: {task.id}")
    return None

async def get_task_details_service(task_id: str) -> Optional[Task]:
    """
    특정 태스크 ID에 대한 상세 정보(상태, 결과 등)를 백엔드에서 가져옵니다.
    백엔드에는 이 ID로 Task 객체를 반환하는 엔드포인트가 있어야 합니다.
    """
    logger.info(f"Requesting details for task_id: {task_id}")
    async with ConversationClient(server_url) as client: # ConversationClient 재사용
        try:
            # 가정: 백엔드 서버에 /api/v1/task/get 또는 /api/v1/task/list?task_id=... 와 같은 엔드포인트가 있고,
            # ConversationClient에 이를 호출하는 메서드(예: get_task_detail)가 추가되었다고 가정합니다.
            # 여기서는 ListTaskRequest를 사용하여 모든 태스크를 가져온 후 필터링하는 임시 방식을 사용합니다.
            # 실제로는 특정 태스크 ID로 직접 조회하는 API를 사용하는 것이 훨씬 효율적입니다.
            
            # 임시 로직: 모든 태스크를 가져와서 ID로 필터링 (비효율적이므로 실제로는 전용 API 사용 권장)
            list_task_req = ListTaskRequest() # params 없이 모든 태스크 요청
            response = await client.list_tasks(list_task_req) # client.list_tasks가 ListTaskResponse를 반환한다고 가정

            if response and response.result:
                for task_item in response.result:
                    if task_item.id == task_id:
                        logger.debug(f"Task {task_id} found. Status: {task_item.status.state if task_item.status else 'N/A'}")
                        return task_item
                logger.warning(f"Task {task_id} not found in the list of all tasks.")
                return None
            else:
                logger.warning(f"No tasks returned from list_tasks or response was empty while searching for task {task_id}.")
                return None
        except Exception as e:
            logger.exception(f"Failed to get details for task_id: {task_id}")
            return None
