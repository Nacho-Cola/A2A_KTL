import asyncio
import base64
import os
import uuid
import logging
from typing import List, Optional, Dict, Any, Union # Union 추가

from fastapi import APIRouter, Request, Response as FastAPIResponse, HTTPException
from common.types import FileContent, FilePart, Message, Part as CommonPart, AgentCard
from service.types import (
    # Request Models - FastAPI가 request body를 이 타입으로 자동 파싱하기 위해 필요
    CreateConversationRequest,
    ListConversationRequest,
    SendMessageRequest,
    GetEventRequest,
    ListMessageRequest,
    PendingMessageRequest,
    ListTaskRequest,
    RegisterAgentRequest,
    UnregisterAgentRequest, # 정의되어 있다고 가정
    ListAgentRequest,

    # Response Models - FastAPI가 응답을 이 타입으로 자동 직렬화하기 위해 필요
    CreateConversationResponse,
    GetEventResponse,
    ListAgentResponse,
    ListConversationResponse,
    ListMessageResponse,
    ListTaskResponse,
    MessageInfo, 
    PendingMessageResponse,
    RegisterAgentResponse,
    SendMessageResponse,
    UnregisterAgentResponse # 정의되어 있다고 가정
)

from .adk_host_manager import ADKHostManager, get_message_id, AgentRegistrationError
from .application_manager import ApplicationManager
from .in_memory_manager import InMemoryFakeAgentManager

logger = logging.getLogger(__name__)

class ConversationServer:
    """
    ConversationServer는 UI에서 에이전트 상호작용을 제공하는 백엔드입니다.
    Mesop 시스템이 에이전트와 상호작용하고 실행 세부 정보를 제공하는 데 사용되는 인터페이스를 정의합니다.
    """
    manager: ApplicationManager
    _file_cache: Dict[str, FilePart] 
    _message_to_cache: Dict[str, str] 

    def __init__(self, router: APIRouter, manager: Optional[ApplicationManager] = None):
        if manager is not None:
            self.manager = manager
        else:
            agent_manager_env = os.environ.get("A2A_HOST", "ADK")
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            uses_vertex_ai = (
                os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
            )
            logger.info(f"Initializing ConversationServer with manager type: {agent_manager_env}")
            if agent_manager_env.upper() == "ADK":
                self.manager = ADKHostManager(api_key=api_key, uses_vertex_ai=uses_vertex_ai)
            else:
                self.manager = InMemoryFakeAgentManager()

        self._file_cache = {}
        self._message_to_cache = {}
        self._add_routes(router)

    def _add_routes(self, router: APIRouter):
        router.add_api_route(
            '/conversation/create', self._create_conversation, methods=['POST'], response_model=CreateConversationResponse
        )
        router.add_api_route(
            '/conversation/list', self._list_conversations, methods=['POST'], response_model=ListConversationResponse
        )
        router.add_api_route(
            '/message/send', self._send_message, methods=['POST'], response_model=SendMessageResponse
        )
        router.add_api_route(
            '/events/get', self._get_events, methods=['POST'], response_model=GetEventResponse
        )
        router.add_api_route(
            '/message/list', self._list_messages, methods=['POST'], response_model=ListMessageResponse
        )
        router.add_api_route(
            '/message/pending', self._pending_messages, methods=['POST'], response_model=PendingMessageResponse
        )
        router.add_api_route(
            '/task/list', self._list_tasks, methods=['POST'], response_model=ListTaskResponse
        )
        router.add_api_route(
            '/agent/register', self._register_agent, methods=['POST'], response_model=RegisterAgentResponse
        )
        router.add_api_route(
            '/agent/unregister', self._unregister_agent, methods=['POST'], response_model=UnregisterAgentResponse
        )
        router.add_api_route(
            '/agent/list', self._list_agents, methods=['POST'], response_model=ListAgentResponse
        )
        router.add_api_route(
            '/message/file/{file_id}', self._get_file, methods=['GET'] 
        )
        router.add_api_route(
            '/api_key/update', self._update_api_key_endpoint, methods=['POST'] 
        )

    def _update_manager_api_key(self, api_key: str):
        if isinstance(self.manager, ADKHostManager):
            logger.info("Updating API key in ADKHostManager.")
            self.manager.update_api_key(api_key)
        else:
            logger.warning(f"API key update is not supported for manager type: {type(self.manager).__name__}")

    async def _update_api_key_endpoint(self, request: Request):
        try:
            data = await request.json()
            api_key = data.get('api_key')
            if api_key is None: 
                raise HTTPException(status_code=400, detail="api_key field is required.")
            if not isinstance(api_key, str) or not api_key.strip(): 
                 raise HTTPException(status_code=400, detail="api_key must be a non-empty string.")
            self._update_manager_api_key(api_key.strip())
            return {"status": "success", "message": "API key update process initiated."}
        except HTTPException as http_exc:
            logger.warning(f"HTTPException during API key update: {http_exc.detail}")
            raise http_exc
        except Exception as e:
            logger.exception("Error updating API key via endpoint.")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    # FastAPI는 request_body의 타입을 보고 자동으로 JSON을 파싱하여 해당 Pydantic 모델로 변환해줌
    # 따라서 함수 시그니처에 request: Request 대신 request_body: CreateConversationRequest 와 같이 사용
    def _create_conversation(self, request_body: CreateConversationRequest): 
        try:
            # request_body.params 등을 사용하여 필요한 파라미터 접근 (JSON-RPC 스타일)
            # CreateConversationRequest가 params 필드를 가지고 있지 않다면, 직접 request_body 사용
            conversation = self.manager.create_conversation()
            # JSON-RPC 응답 형식에 맞춤 (id, result)
            return CreateConversationResponse(id=request_body.id, result=conversation)
        except Exception as e:
            logger.exception("Error creating conversation.")
            raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

    async def _send_message(self, request_body: SendMessageRequest): 
        try:
            message_to_process = request_body.params 
            if not isinstance(message_to_process, Message):
                raise HTTPException(status_code=400, detail="Invalid 'params' in request body, expected a Message object.")

            sanitized_message = self.manager.sanitize_message(message_to_process)
            message_id_after_sanitize = get_message_id(sanitized_message)
            conversation_id_after_sanitize = sanitized_message.metadata.get('conversation_id') if sanitized_message.metadata else None

            if not message_id_after_sanitize:
                logger.error("Message ID missing after sanitization.")
                raise HTTPException(status_code=500, detail="Failed to assign message ID during sanitization.")

            returned_task_id: Optional[str] = await self.manager.process_message(sanitized_message) 
            logger.info(f"Message {message_id_after_sanitize} processed. Associated Task ID: {returned_task_id}")
            
            response_result_data: Dict[str, Any] = {
                "message_id": message_id_after_sanitize,
                "conversation_id": conversation_id_after_sanitize or ""
            }
            if returned_task_id:
                response_result_data["task_id"] = returned_task_id
            
            return SendMessageResponse(id=request_body.id, result=response_result_data)
        except HTTPException as http_exc:
            logger.warning(f"HTTPException during send message: {http_exc.detail}")
            raise http_exc
        except Exception as e:
            logger.exception("Error sending message.")
            raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

    async def _list_messages(self, request_body: ListMessageRequest): 
        try:
            conversation_id = request_body.params 
            if not conversation_id or not isinstance(conversation_id, str):
                raise HTTPException(status_code=400, detail="Invalid or missing 'params' (expected conversation_id string).")

            conversation = self.manager.get_conversation(conversation_id)
            if conversation:
                messages_to_cache = conversation.messages if conversation.messages is not None else []
                cached_messages = self._cache_message_file_parts(messages_to_cache)
                return ListMessageResponse(id=request_body.id, result=cached_messages)
            
            logger.info(f"Conversation not found for listing messages: {conversation_id}")
            return ListMessageResponse(id=request_body.id, result=[])
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.exception(f"Error listing messages for conversation: {request_body.params}")
            raise HTTPException(status_code=500, detail=f"Failed to list messages: {str(e)}")

    def _cache_message_file_parts(self, messages: List[Message]) -> List[Message]:
        processed_messages: List[Message] = []
        for m in messages:
            if not m or not m.parts:
                processed_messages.append(m)
                continue

            message_id_from_util = get_message_id(m) 
            if not message_id_from_util:
                logger.warning("Message without ID found in cache_content. Skipping file caching for this message.")
                processed_messages.append(m)
                continue

            new_parts_for_message: List[CommonPart] = []
            modified = False
            for i, part_instance in enumerate(m.parts):
                if isinstance(part_instance, FilePart) and part_instance.file and part_instance.file.bytes:
                    mime_type = part_instance.file.mimeType or "application/octet-stream"
                    message_part_id = f'{message_id_from_util}:{i}'
                    cache_id = self._message_to_cache.get(message_part_id)
                    if not cache_id:
                        cache_id = str(uuid.uuid4())
                        self._message_to_cache[message_part_id] = cache_id
                    
                    if cache_id not in self._file_cache:
                        self._file_cache[cache_id] = part_instance 
                    
                    uri_file_part = FilePart(
                        type='file', 
                        file=FileContent(
                            mimeType=mime_type,
                            uri=f'/message/file/{cache_id}',
                        ),
                        mimeType=mime_type 
                    )
                    new_parts_for_message.append(uri_file_part)
                    modified = True
                else:
                    new_parts_for_message.append(part_instance)
            
            if modified:
                message_dict = m.model_dump(exclude_unset=True) 
                message_dict['parts'] = [p.model_dump(exclude_unset=True) for p in new_parts_for_message]
                processed_messages.append(Message.model_validate(message_dict))
            else:
                processed_messages.append(m)
        return processed_messages

    async def _pending_messages(self, request_body: PendingMessageRequest): 
        try:
            pending = self.manager.get_pending_messages()
            return PendingMessageResponse(id=request_body.id, result=pending)
        except Exception as e:
            logger.exception("Error getting pending messages.")
            raise HTTPException(status_code=500, detail=f"Failed to get pending messages: {str(e)}")

    def _list_conversations(self, request_body: ListConversationRequest): 
        try:
            return ListConversationResponse(id=request_body.id, result=self.manager.conversations)
        except Exception as e:
            logger.exception("Error listing conversations.")
            raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

    def _get_events(self, request_body: GetEventRequest): 
        try:
            return GetEventResponse(id=request_body.id, result=self.manager.events)
        except Exception as e:
            logger.exception("Error getting events.")
            raise HTTPException(status_code=500, detail=f"Failed to get events: {str(e)}")

    def _list_tasks(self, request_body: ListTaskRequest): 
        try:
            return ListTaskResponse(id=request_body.id, result=self.manager.tasks)
        except Exception as e:
            logger.exception("Error listing tasks.")
            raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

    async def _register_agent(self, request_body: RegisterAgentRequest): 
        try:
            url = request_body.params 
            if not url or not isinstance(url, str):
                raise HTTPException(status_code=400, detail="Invalid or missing 'params' (agent URL).")

            await self.manager.register_agent(url) 
            logger.info(f"Agent registration process completed for URL: {url}")
            return RegisterAgentResponse(id=request_body.id, result="Agent registration process initiated successfully.")
        except AgentRegistrationError as are: 
            logger.error(f"Error registering agent with URL: {url}. Detail: {str(are)}")
            raise HTTPException(status_code=400, detail=str(are)) 
        except HTTPException as http_exc: 
            raise http_exc
        except Exception as e: 
            logger.exception(f"Internal server error during agent registration for URL: {url}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def _list_agents(self, request_body: ListAgentRequest): 
        try:
            return ListAgentResponse(id=request_body.id, result=self.manager.agents)
        except Exception as e:
            logger.exception("Error listing agents.")
            raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

    async def _unregister_agent(self, request_body: UnregisterAgentRequest): 
        try:
            url = request_body.params 
            if not url or not isinstance(url, str):
                raise HTTPException(status_code=400, detail="Invalid or missing 'params' (expected agent URL string).")

            if not hasattr(self.manager, 'unregister_agent'):
                logger.error(f"Manager type {type(self.manager).__name__} does not support unregister_agent method.")
                raise HTTPException(status_code=501, detail="Unregister agent not supported by the current manager.")

            self.manager.unregister_agent(url) # type: ignore [attr-defined]
            logger.info(f"Agent unregistration initiated for URL: {url}")
            # service.types에 정의된 UnregisterAgentResponse 사용
            return UnregisterAgentResponse(id=request_body.id, result=True) 
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.exception(f"Error unregistering agent with URL: {request_body.params}")
            # service.types에 정의된 UnregisterAgentResponse 사용 (에러 포함)
            # error_payload = {"code": -32000, "message": str(e)} # JSON-RPC 에러 객체 예시
            # return UnregisterAgentResponse(id=request_body.id, result=False, error=error_payload)
            raise HTTPException(status_code=500, detail=f"Failed to unregister agent: {str(e)}")

    def _get_file(self, file_id: str): 
        try:
            if file_id not in self._file_cache:
                logger.warning(f"File not found in cache: {file_id}")
                raise HTTPException(status_code=404, detail="File not found.")
            
            cached_file_part = self._file_cache[file_id] 

            if not cached_file_part.file or not cached_file_part.file.bytes:
                 logger.error(f"Cached file part for ID {file_id} is missing essential content.")
                 raise HTTPException(status_code=500, detail="Cached file content is corrupted or incomplete.")

            try:
                file_bytes_content = base64.b64decode(cached_file_part.file.bytes)
            except Exception as b64_error:
                logger.error(f"Failed to decode base64 content for file ID {file_id}: {b64_error}", exc_info=True)
                raise HTTPException(status_code=500, detail="File content is not valid base64 data.")

            mime_type = cached_file_part.file.mimeType or "application/octet-stream" 
            
            return FastAPIResponse( 
                content=file_bytes_content,
                media_type=mime_type,
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.exception(f"Error retrieving file: {file_id}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")

