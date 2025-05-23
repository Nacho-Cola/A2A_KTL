import asyncio
import json # noqa F401
import os
import sys # noqa F401
import traceback # noqa F401
import logging
import uuid 
from typing import Any, Dict, List, Optional, Union, Tuple

import httpx # For update_api_key_on_server_service

# streamlit 폴더 구조에 따른 임포트 경로 수정
from common.types import Message, Part, Task, AgentCard, TextPart # AgentCard, TextPart 추가
from service.client.client import ConversationClient
from service.types import (
    Conversation,
    CreateConversationRequest,
    CreateConversationResponse, # Assuming this is the wrapper response type
    Event,
    GetEventRequest,
    GetEventResponse, # Assuming this wrapper for List[Event]
    ListAgentRequest,
    ListAgentResponse, # Used by list_remote_agents_service
    ListConversationRequest,
    ListConversationResponse, # Assuming this wrapper for List[Conversation]
    ListMessageRequest,
    ListMessageResponse, # Assuming this wrapper for List[Message]
    ListTaskRequest,
    ListTaskResponse, # Assuming this wrapper for List[Task]
    PendingMessageRequest,
    PendingMessageResponse, # Assuming this wrapper for Dict[str, str]
    RegisterAgentRequest,
    RegisterAgentResponse, # Assuming this wrapper for bool or result
    SendMessageRequest,
    SendMessageResponse, # Assuming this response type will contain the task_id
    UnregisterAgentRequest, 
    UnregisterAgentResponse,
)

from state.state import (
    AppState,
    SessionTask,
    StateConversation,
    StateEvent,
    StateMessage,
    StateTask,
)

logger = logging.getLogger(__name__)

_env_url = os.environ.get("A2A_STREAMLIT_BACKEND_URL") or os.environ.get("A2A_BACKEND_URL")
server_url = _env_url if _env_url else 'http://localhost:12000'
logger.info(f"HostAgentService (Streamlit context) using server_url: {server_url}")


async def list_conversations_service() -> List[Conversation]:
    """백엔드에서 대화 목록을 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            response_wrapper: Optional[ListConversationResponse] = await client.list_conversations(ListConversationRequest())
            if response_wrapper and response_wrapper.result and isinstance(response_wrapper.result, list):
                conversations: List[Conversation] = response_wrapper.result
                logger.debug(f"Listed conversations: {len(conversations)} found.")
                return conversations
            logger.debug("No conversations found or unexpected response format from list_conversations.")
            return []
        except Exception as e:
            logger.exception("Failed to list conversations from backend.")
            return []

async def send_message_service(message: Message) -> Optional[str]:
    """
    메시지를 백엔드로 전송합니다. 
    성공 시 백엔드에서 생성/사용한 실제 Task ID를 반환합니다.
    """
    async with ConversationClient(server_url, timeout=300.0) as client: 
        try:
            logger.debug(f"Sending message to backend. Message metadata: {message.metadata}")
            api_response: Optional[SendMessageResponse] = await client.send_message(
                SendMessageRequest(params=message)
            )
            
            returned_task_id: Optional[str] = None

            if api_response:
                logger.debug(f"Received api_response from client.send_message: {type(api_response)} | Content: {api_response}")
                # 우선순위 1: 응답 객체 자체에 task_id가 있는 경우
                if hasattr(api_response, 'task_id') and getattr(api_response, 'task_id'):
                    returned_task_id = str(getattr(api_response, 'task_id'))
                    logger.info(f"Task ID found directly in response object: {returned_task_id}")
                
                # 우선순위 2: 응답 객체의 result 필드에서 task_id를 찾는 경우
                elif hasattr(api_response, 'result') and api_response.result:
                    logger.debug(f"Checking api_response.result. Type: {type(api_response.result)} | Content: {api_response.result}")
                    if isinstance(api_response.result, dict):
                        returned_task_id = api_response.result.get("task_id")
                        if returned_task_id:
                            returned_task_id = str(returned_task_id)
                            logger.info(f"Task ID found in response.result (dict): {returned_task_id}")
                        else:
                            logger.warning(f"'task_id' not found in response.result dict. Keys: {list(api_response.result.keys())}")
                    elif hasattr(api_response.result, 'task_id'): 
                        returned_task_id = str(getattr(api_response.result, 'task_id'))
                        logger.info(f"Task ID found in response.result object attribute: {returned_task_id}")
                    else:
                        logger.warning(f"response.result format unexpected or lacks task_id. Result type: {type(api_response.result)}")
                else:
                    logger.warning(f"Backend response for send_message has no 'result' field, it's empty, or no direct 'task_id' attribute on response object.")
            else:
                logger.warning(f"Backend response for send_message was None or empty.")

            if returned_task_id:
                logger.info(f"Message sent, backend processed with Task ID: {returned_task_id}")
                return returned_task_id
            else:
                logger.error(f"Failed to extract Task ID from backend response. Full API Response object type: {type(api_response)}, content: {api_response}")
                return None
                
        except Exception as e:
            msg_id_for_log = get_message_id_from_message_obj(message) or 'N/A'
            logger.exception(f"Failed to send message (local ID: {msg_id_for_log}).")
            return None


async def create_conversation_service() -> Optional[Conversation]:
    """새 대화를 백엔드에 생성 요청합니다."""
    async with ConversationClient(server_url) as client:
        try:
            response_wrapper: Optional[CreateConversationResponse] = await client.create_conversation(CreateConversationRequest())
            
            if response_wrapper and response_wrapper.result:
                new_conversation: Optional[Conversation] = response_wrapper.result 
                if new_conversation and isinstance(new_conversation, Conversation) and new_conversation.conversation_id:
                    logger.info(f"Conversation created successfully: {new_conversation.conversation_id}")
                    return new_conversation
                else:
                    logger.warning(f"Create conversation call returned a result, but it's not a valid Conversation object or lacks ID. Result type: {type(new_conversation)}, Result: {new_conversation}")
                    return None
            else:
                logger.warning(f"Create conversation call returned no result or an empty response wrapper: {response_wrapper}")
                return None
        except Exception as e:
            logger.exception("Failed to create conversation.")
            return None


async def list_remote_agents_service() -> List[AgentCard]:
    """백엔드에서 원격 에이전트 목록을 가져옵니다."""
    logger.debug(f"Listing remote agents from server_url: {server_url}")
    async with ConversationClient(server_url) as client:
        try:
            response_wrapper: Optional[ListAgentResponse] = await client.list_agents(ListAgentRequest())
            if response_wrapper and response_wrapper.result and isinstance(response_wrapper.result, list):
                agents: List[AgentCard] = response_wrapper.result
                logger.debug(f"ListRemoteAgents received: {len(agents)} agents.")
                return agents
            logger.debug("No remote agents found or unexpected response format from list_agents.")
            return []
        except Exception as e:
            logger.exception("Failed to list remote agents.")
            return []


async def add_remote_agent_service(path: str) -> bool:
    """원격 에이전트를 백엔드에 등록합니다."""
    logger.info(f"Registering remote agent from path: {path}")
    async with ConversationClient(server_url) as client:
        try:
            response: Optional[RegisterAgentResponse] = await client.register_agent(RegisterAgentRequest(params=path))
            if response and hasattr(response, 'error') and response.error is None and response.result is not None: 
                logger.info(f"Agent registration successful for {path}. Result: {response.result}")
                return True 
            elif response and hasattr(response, 'error') and response.error is not None:
                err_msg = getattr(response.error, 'message', str(response.error))
                logger.error(f"Failed to register agent at {path}. Error: {err_msg}")
                return False
            else:
                logger.error(f"Failed to register agent at {path}. Unexpected response: {response}")
                return False
        except Exception as e:
            logger.exception(f"Exception during agent registration for path: {path}")
            return False

async def remove_remote_agent_service(path: str) -> bool:
    """원격 에이전트를 백엔드에서 등록 해제합니다."""
    logger.info(f"Unregistering remote agent from path: {path}")
    async with ConversationClient(server_url) as client:
        try:
            response: Optional[UnregisterAgentResponse] = await client._send_request(
                UnregisterAgentRequest(params=path), UnregisterAgentResponse
            ) 
            
            if response and hasattr(response, 'error') and response.error is None and response.result is not None: 
                logger.info(f"Agent unregistration successful for {path}.")
                return True
            elif response and hasattr(response, 'error') and response.error is not None:
                err_msg = getattr(response.error, 'message', str(response.error))
                logger.error(f"Failed to unregister agent at {path}. Error: {err_msg}")
                return False
            else:
                logger.error(f"Failed to unregister agent at {path}. Unexpected response: {response}")
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
            response_wrapper: Optional[GetEventResponse] = await client.get_events(GetEventRequest())
            if response_wrapper and response_wrapper.result and isinstance(response_wrapper.result, list):
                events: List[Event] = response_wrapper.result
                logger.debug(f"Retrieved {len(events)} events.")
                return events
            logger.debug("No events found or unexpected response format from get_events.")
            return []
        except Exception as e:
            logger.exception("Failed to get events.")
            return []


async def get_processing_messages_service() -> Dict[str, str]:
    """백엔드에서 처리 중인 메시지 상태를 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            response_wrapper: Optional[PendingMessageResponse] = await client.get_pending_messages(PendingMessageRequest())
            if response_wrapper and response_wrapper.result and isinstance(response_wrapper.result, dict):
                statuses: Dict[str, str] = response_wrapper.result
                logger.debug(f"Retrieved {len(statuses)} processing messages statuses.")
                return statuses
            logger.debug("No processing messages statuses or unexpected format from get_pending_messages.")
            return {}
        except Exception as e:
            logger.exception("Error getting pending messages statuses.")
            return {}


def get_message_aliases_service() -> Dict[Any, Any]: 
    logger.debug("get_message_aliases_service called, returning empty dict.")
    return {}


async def get_tasks_service() -> List[Task]: 
    """백엔드에서 태스크 목록을 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            response_wrapper: Optional[ListTaskResponse] = await client.list_tasks(ListTaskRequest())
            if response_wrapper and response_wrapper.result and isinstance(response_wrapper.result, list):
                tasks: List[Task] = response_wrapper.result
                logger.debug(f"Retrieved {len(tasks)} tasks.")
                return tasks
            logger.debug("No tasks found or unexpected response format from list_tasks.")
            return []
        except Exception as e:
            logger.exception("Failed to list tasks.")
            return []


async def list_messages_service(conversation_id: str) -> List[Message]: 
    """특정 대화의 메시지 목록을 백엔드에서 가져옵니다."""
    async with ConversationClient(server_url) as client:
        try:
            response_wrapper: Optional[ListMessageResponse] = await client.list_messages(
                ListMessageRequest(params=conversation_id)
            )
            if response_wrapper and response_wrapper.result and isinstance(response_wrapper.result, list):
                messages: List[Message] = response_wrapper.result
                logger.debug(f"Retrieved {len(messages)} messages for conv_id {conversation_id}.")
                return messages
            logger.debug(f"No messages found for conv_id {conversation_id} or unexpected response format.")
            return []
        except Exception as e:
            logger.exception(f"Failed to list messages for conversation_id: {conversation_id}")
            return []


async def fetch_app_state_service(conversation_id: Optional[str]) -> AppState:
    state = AppState() 
    logger.info(f"Fetching app state for conversation_id: {conversation_id}")

    try:
        message_coro = list_messages_service(conversation_id) if conversation_id else asyncio.sleep(0, result=[]) 
        
        results = await asyncio.gather(
            message_coro,
            list_conversations_service(),
            get_tasks_service(),
            get_processing_messages_service(),
            return_exceptions=True 
        )

        messages_res, conversations_res, tasks_res, processing_messages_res = results

        if isinstance(messages_res, Exception):
            logger.error("Failed to fetch messages for AppState.", exc_info=messages_res)
            state.messages = []
        else:
            state.messages = [convert_message_to_state(m) for m in messages_res] if messages_res else []
        
        if conversation_id: 
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
                    session_id=extract_conversation_id_from_task(t), 
                    task=convert_task_to_state(t)
                )
                for t in tasks_res or []
            ]
        
        if isinstance(processing_messages_res, Exception):
            logger.error("Failed to fetch processing messages for AppState.", exc_info=processing_messages_res)
            state.background_tasks = {}
        else:
            state.background_tasks = processing_messages_res or {}
        
        state.message_aliases = get_message_aliases_service() 

        logger.info("App state fetch complete.")

    except Exception as e: 
        logger.exception("Critical error during fetch_app_state_service construction.")
    return state


async def update_api_key_on_server_service(api_key: str) -> bool: 
    if not api_key or not isinstance(api_key, str): 
        logger.error("Invalid API key provided for update: must be a non-empty string.")
        return False
        
    logger.info(f"Attempting to update API key on server: {server_url}/api_key/update")
    async with httpx.AsyncClient() as client: 
        try:
            response = await client.post(
                f'{server_url}/api_key/update', json={'api_key': api_key}
            )
            response.raise_for_status() 
            
            response_data = response.json()
            if response_data.get("status") == "success":
                logger.info("API key updated successfully on the server.")
                os.environ['GOOGLE_API_KEY'] = api_key 
                logger.debug("Local GOOGLE_API_KEY environment variable also updated.")
                return True
            else:
                err_msg = response_data.get('message', 'Unknown error from server')
                logger.error(f"Server responded with an error during API key update: {err_msg}")
                return False
        except httpx.HTTPStatusError as e:
            response_text_preview = e.response.text[:200] + "..." if e.response.text and len(e.response.text) > 200 else e.response.text
            logger.exception(f"HTTP error when updating API key on server: {e.response.status_code} - Response: {response_text_preview}")
            return False
        except Exception as e:
            logger.exception("Generic error when updating API key on server.")
            return False

# --- 데이터 변환 및 추출 헬퍼 함수들 ---

def get_message_id_from_message_obj(message: Optional[Message]) -> Optional[str]: 
    """Message 객체에서 message_id를 추출합니다."""
    if message and message.metadata and 'message_id' in message.metadata:
        return message.metadata['message_id']
    return None

def convert_message_to_state(message: Optional[Message]) -> StateMessage:
    if not message:
        logger.debug("convert_message_to_state received None, returning empty StateMessage.")
        return StateMessage()

    message_id = get_message_id_from_message_obj(message) or f"generated_{uuid.uuid4()}"
    role = message.role or "unknown"
    content_parts = extract_content_from_parts(message.parts or [])
    
    return StateMessage(message_id=message_id, role=role, content=content_parts)


def convert_conversation_to_state(conversation: Conversation) -> StateConversation:
    message_ids = [
        msg_id for msg_id in (get_message_id_from_message_obj(x) for x in (conversation.messages or [])) if msg_id
    ]
    return StateConversation(
        conversation_id=conversation.conversation_id,
        conversation_name=conversation.name or "Untitled Conversation",
        is_active=conversation.is_active,
        message_ids=message_ids,
    )


def convert_task_to_state(task: Task) -> StateTask:
    first_history_message_state: StateMessage = StateMessage()
    if task.history: 
        first_history_message_state = convert_message_to_state(task.history[0])

    output_parts_content: List[Union[str, Dict[str, Any]]] = [] 
    if task.artifacts:
        for artifact_item in task.artifacts:
            extracted_artifact_contents = extract_content_from_parts(artifact_item.parts or [])
            if extracted_artifact_contents:
                output_parts_content.append(extracted_artifact_contents[0][0]) 

    final_status_message_content_str: Optional[str] = None
    if task.status and task.status.message and task.status.message.parts:
        final_status_message_content_str = " ".join(
            p.text for p in task.status.message.parts if isinstance(p, TextPart) and p.text
        ).strip()
    
    return StateTask(
        task_id=task.id,
        session_id=task.sessionId or extract_conversation_id_from_task(task), 
        state=str(task.status.state) if task.status else "UNKNOWN_STATE", 
        message=first_history_message_state, 
        artifacts=output_parts_content, 
        final_result_text=final_status_message_content_str 
    )


def convert_event_to_state(event: Event) -> StateEvent:
    conversation_id = extract_message_conversation_id(event.content) or "unknown_conversation"
    return StateEvent(
        conversation_id=conversation_id,
        actor=event.actor or "unknown_actor", 
        role=event.content.role or "unknown_role", 
        id=event.id,
        content=extract_content_from_parts(event.content.parts or []), 
    )


def extract_content_from_parts(message_parts: List[Part]) -> List[Tuple[Union[str, Dict[str, Any]], str]]:
    extracted_content_list: List[Tuple[Union[str, Dict[str, Any]], str]] = []
    if not message_parts: return []

    for p_item in message_parts: 
        part_type = getattr(p_item, 'type', None) 
        mime_type_default = "text/plain"

        if isinstance(p_item, TextPart):
            extracted_content_list.append((p_item.text or "", 'text/plain'))
        elif isinstance(p_item, FilePart):
            mime_type = p_item.mimeType or "application/octet-stream"
            if p_item.file and p_item.file.bytes:
                extracted_content_list.append((p_item.file.bytes, mime_type))
            elif p_item.file and p_item.file.uri:
                extracted_content_list.append((p_item.file.uri, mime_type))
            elif p_item.uri: 
                 extracted_content_list.append((p_item.uri, mime_type))
            elif p_item.data: 
                 extracted_content_list.append((p_item.data, mime_type))
            else:
                logger.warning(f"FilePart is missing bytes, uri, or data: {p_item.name if hasattr(p_item, 'name') else 'Unknown File'}")
                extracted_content_list.append(("<file_content_unavailable>", mime_type_default))
        elif isinstance(p_item, DataPart):
            data_content = p_item.data
            mime_type = "application/json" 
            if p_item.metadata and p_item.metadata.get("a2a_part_type") in ["tool_call", "tool_response", "tool_output"]:
                mime_type = "application/vnd.a2a.tool_data+json" 
            
            try:
                extracted_content_list.append((data_content, mime_type)) 
            except TypeError as e: 
                logger.error(f"Failed to process data part: {data_content}. Error: {e}", exc_info=True)
                extracted_content_list.append((f"<unserializable_data_part: {type(data_content)}>", mime_type_default))
        else:
            logger.warning(f"Unsupported Part type encountered: {type(p_item)}")
            extracted_content_list.append(("<unknown_part_type>", mime_type_default))
    return extracted_content_list


def extract_message_conversation_id(message: Optional[Message]) -> Optional[str]:
    """Message 객체에서 conversation_id를 추출합니다."""
    if message and message.metadata and 'conversation_id' in message.metadata:
        return message.metadata['conversation_id']
    return None


def extract_conversation_id_from_task(task: Task) -> Optional[str]:
    """Task 객체에서 conversation_id (또는 sessionId)를 추출합니다."""
    if task.sessionId: return task.sessionId
    if task.metadata and task.metadata.get('conversation_id'): return task.metadata.get('conversation_id')
    logger.debug(f"Could not extract conversation_id from task: {task.id}")
    return None

async def get_task_details_service(task_id: str) -> Optional[Task]:
    logger.info(f"Requesting details for task_id: {task_id}")
    async with ConversationClient(server_url) as client:
        try:
            # 실제로는 client.get_task(task_id)와 같은 직접적인 API 호출이 필요합니다.
            # 아래는 임시 방편으로 list_tasks를 사용합니다.
            logger.warning(f"Client does not have a dedicated get_task method. Falling back to list_tasks for task {task_id} (highly inefficient). Consider implementing a specific backend endpoint and client method for fetching a single task by ID.")
            list_task_req = ListTaskRequest() 
            response_wrapper: Optional[ListTaskResponse] = await client.list_tasks(list_task_req) 

            if response_wrapper and response_wrapper.result and isinstance(response_wrapper.result, list):
                tasks_list: List[Task] = response_wrapper.result
                for task_item in tasks_list:
                    if task_item.id == task_id:
                        logger.debug(f"Task {task_id} found via list_tasks. Status: {task_item.status.state if task_item.status else 'N/A'}")
                        return task_item
            
            logger.warning(f"Task {task_id} not found after listing all tasks.")
            return None
        except Exception as e:
            logger.exception(f"Failed to get details for task_id: {task_id}")
            return None

