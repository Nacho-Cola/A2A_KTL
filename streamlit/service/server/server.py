import asyncio
import base64
import os
# import threading # _send_message 변경으로 인해 직접적인 threading 사용 제거 (FastAPI가 내부적으로 처리)
import uuid
import logging # 로깅 모듈 추가
from typing import List, Optional, Dict, Any # 타입 힌팅 추가

from fastapi import APIRouter, Request, Response as FastAPIResponse, HTTPException # HTTPException 추가
# common.types 및 service.types 경로는 프로젝트 구조에 맞게 설정되어 있다고 가정합니다.
from common.types import FileContent, FilePart, Message, Part as CommonPart # Part를 CommonPart로 별칭
from service.types import (
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
    # UnregisterAgentResponse가 service.types에 정의되어 있다고 가정
    # 필요시 아래와 같이 정의하거나, 적절한 응답 타입을 사용합니다.
    # class UnregisterAgentResponse(BaseModel): result: bool = True; error: Optional[str] = None
)

# 상대 경로 임포트는 현재 파일의 패키지 위치에 따라 달라집니다.
from .adk_host_manager import ADKHostManager, get_message_id
from .application_manager import ApplicationManager
from .in_memory_manager import InMemoryFakeAgentManager



logger = logging.getLogger(__name__) # 로거 인스턴스 생성

class ConversationServer:
    """
    ConversationServer는 UI에서 에이전트 상호작용을 제공하는 백엔드입니다.
    Mesop 시스템이 에이전트와 상호작용하고 실행 세부 정보를 제공하는 데 사용되는 인터페이스를 정의합니다.
    """
    manager: ApplicationManager
    _file_cache: Dict[str, FilePart] # 파일 ID -> FilePart 데이터 매핑
    _message_to_cache: Dict[str, str] # 메시지 파트 ID -> 캐시 ID 매핑

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

        # API 라우트 등록
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
        # UnregisterAgentResponse가 service.types에 정의되어 있다고 가정하고 response_model 추가 가능
        # from service.types import UnregisterAgentResponse # 가정
        # router.add_api_route(
        #     '/agent/unregister', self._unregister_agent, methods=['POST'], response_model=UnregisterAgentResponse
        # )
        # 임시로 generic response 사용 (아래 구현 참조)
        router.add_api_route(
            '/agent/unregister', self._unregister_agent, methods=['POST']
        )
        router.add_api_route(
            '/agent/list', self._list_agents, methods=['POST'], response_model=ListAgentResponse
        )
        router.add_api_route(
            '/message/file/{file_id}', self._get_file, methods=['GET'] # FastAPIResponse 직접 반환
        )
        router.add_api_route(
            '/api_key/update', self._update_api_key_endpoint, methods=['POST'] # JSONResponse 직접 반환
        )

    def _update_manager_api_key(self, api_key: str):
        """내부적으로 API 키를 매니저에 업데이트하는 메서드"""
        if isinstance(self.manager, ADKHostManager):
            logger.info("Updating API key in ADKHostManager.")
            self.manager.update_api_key(api_key)
        else:
            # 다른 매니저 타입에 대한 API 키 업데이트 지원이 필요하면 여기에 로직 추가
            logger.warning(f"API key update is not supported for manager type: {type(self.manager).__name__}")
            # 혹은 ApplicationManager 인터페이스에 update_api_key를 추가하는 것을 고려
            # raise NotImplementedError(f"API key update not implemented for {type(self.manager).__name__}")


    async def _update_api_key_endpoint(self, request: Request):
        """API 키 업데이트를 위한 엔드포인트"""
        try:
            data = await request.json()
            api_key = data.get('api_key')

            if api_key is None: # 키 자체가 없는 경우
                raise HTTPException(status_code=400, detail="api_key field is required.")
            if not isinstance(api_key, str) or not api_key.strip(): # 빈 문자열이거나 문자열이 아닌 경우
                 raise HTTPException(status_code=400, detail="api_key must be a non-empty string.")

            self._update_manager_api_key(api_key.strip())
            return {"status": "success", "message": "API key update process initiated."}
        except HTTPException as http_exc:
            logger.warning(f"HTTPException during API key update: {http_exc.detail}")
            raise http_exc
        except Exception as e:
            logger.exception("Error updating API key via endpoint.")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def _create_conversation(self):
        try:
            conversation = self.manager.create_conversation()
            return CreateConversationResponse(result=conversation)
        except Exception as e:
            logger.exception("Error creating conversation.")
            raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

    async def _send_message(self, request: Request):
        try:
            body = await request.json() # 요청 본문을 한 번만 읽음
            params = body.get('params')
            if not params or not isinstance(params, dict):
                raise HTTPException(status_code=400, detail="Invalid or missing 'params' in request body.")

            metadata = params.get('metadata', {}) # metadata가 없으면 빈 dict로 기본값 설정
            
            # conversation_id가 필수라면 여기서 검증
            # if not metadata.get('conversation_id'):
            #     raise HTTPException(status_code=400, detail="conversation_id is required in message metadata.")

            raw_parts = params.get('parts', [])
            message_parts: List[CommonPart] = [] # common.types.Part 사용 (Pydantic discriminated union)
            for p_data in raw_parts:
                if not isinstance(p_data, dict) or 'type' not in p_data:
                    logger.warning(f"Skipping malformed message part (missing 'type' or not a dict): {p_data}")
                    # 엄격하게 처리하려면 HTTPException 발생
                    # raise HTTPException(status_code=400, detail=f"Malformed message part: {p_data}")
                    continue 
                try:
                    # CommonPart는 TextPart, FilePart, DataPart 등의 Union이므로,
                    # p_data에 'type' 필드가 있고 해당 타입의 필드들이 있으면 Pydantic이 자동으로 변환
                    message_parts.append(CommonPart(**p_data))
                except Exception as part_exc: # Pydantic validation error 등
                    logger.warning(f"Failed to parse message part: {p_data}. Error: {part_exc}")
                    # raise HTTPException(status_code=400, detail=f"Invalid message part data: {p_data}. Error: {part_exc}")
                    continue


            message = Message(
                role=params.get('role', 'user'),
                parts=message_parts,
                metadata=metadata
            )

            message = self.manager.sanitize_message(message) # message_id 할당 등
            
            message_id_after_sanitize = message.metadata.get('message_id')
            conversation_id_after_sanitize = message.metadata.get('conversation_id')

            if not message_id_after_sanitize: # Sanitize 후에도 ID가 없으면 문제
                logger.error("Message ID missing after sanitization.")
                raise HTTPException(status_code=500, detail="Failed to assign message ID during sanitization.")

            # FastAPI의 async 지원을 활용하여 직접 await 호출 (스레드 불필요)
            await self.manager.process_message(message)
            
            logger.info(f"Message {message_id_after_sanitize} sent for processing in conversation {conversation_id_after_sanitize}.")
            return SendMessageResponse(
                result=MessageInfo(
                    message_id=message_id_after_sanitize,
                    conversation_id=conversation_id_after_sanitize or "" # None일 경우 빈 문자열
                )
            )
        except HTTPException as http_exc:
            # 이미 HTTPException이면 그대로 다시 발생시켜 FastAPI가 처리하도록 함
            logger.warning(f"HTTPException during send message: {http_exc.detail}")
            raise http_exc
        except Exception as e:
            logger.exception("Error sending message.") # 스택 트레이스 포함 로깅
            raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


    async def _list_messages(self, request: Request):
        try:
            data = await request.json()
            # 'params' 키가 conversation_id 문자열 자체라고 가정 (원본 코드 구조)
            conversation_id = data.get('params')
            if not conversation_id or not isinstance(conversation_id, str):
                raise HTTPException(status_code=400, detail="Invalid or missing 'params' (expected conversation_id string).")

            conversation = self.manager.get_conversation(conversation_id)
            if conversation:
                messages_to_cache = conversation.messages if conversation.messages is not None else []
                cached_messages = self._cache_message_file_parts(messages_to_cache)
                return ListMessageResponse(result=cached_messages)
            logger.info(f"Conversation not found for listing messages: {conversation_id}")
            return ListMessageResponse(result=[])
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            request_param_info = data.get('params') if isinstance(data, dict) else "N/A"
            logger.exception(f"Error listing messages for conversation: {request_param_info}")
            raise HTTPException(status_code=500, detail=f"Failed to list messages: {str(e)}")

    def _cache_message_file_parts(self, messages: List[Message]) -> List[Message]:
        processed_messages: List[Message] = []
        for m in messages:
            # 메시지 객체나 parts 리스트가 None일 경우 방어
            if not m or not m.parts:
                processed_messages.append(m)
                continue

            message_id_from_util = get_message_id(m) # adk_host_manager의 get_message_id 사용
            if not message_id_from_util:
                logger.warning("Message without ID found in cache_content. Skipping file caching for this message.")
                processed_messages.append(m)
                continue

            new_parts_for_message: List[CommonPart] = []
            modified = False
            for i, part_instance in enumerate(m.parts):
                # FilePart이면서 file 속성과 mimeType이 있는 경우에만 처리
                if isinstance(part_instance, FilePart) and part_instance.file and part_instance.file.bytes:
                    # part_instance.file이 None이 아님을 isinstance가 보장, part_instance.file.mimeType은 있을 수도 없을 수도 있음
                    mime_type = part_instance.file.mimeType or "application/octet-stream"

                    message_part_id = f'{message_id_from_util}:{i}' # 각 파트에 대한 고유 ID
                    cache_id = self._message_to_cache.get(message_part_id)
                    if not cache_id:
                        cache_id = str(uuid.uuid4())
                        self._message_to_cache[message_part_id] = cache_id
                    
                    # 원본 FilePart(bytes 포함)를 _file_cache에 저장
                    if cache_id not in self._file_cache:
                        self._file_cache[cache_id] = part_instance 
                    
                    # URI 참조를 포함하는 새 FilePart 생성
                    uri_file_part = FilePart(
                        type='file', # Pydantic discriminated union을 위해 명시
                        file=FileContent(
                            mimeType=mime_type,
                            uri=f'/message/file/{cache_id}',
                            # name, bytes는 URI 참조에서는 불필요
                        ),
                        mimeType=mime_type # FilePart 자체의 mimeType도 설정
                    )
                    new_parts_for_message.append(uri_file_part)
                    modified = True
                else:
                    new_parts_for_message.append(part_instance)
            
            if modified:
                # Pydantic 모델의 불변성을 고려하여 새 Message 객체 생성 또는 .copy() 사용
                # 여기서는 model_dump와 model_validate를 사용하여 안전하게 복사 및 수정
                message_dict = m.model_dump(exclude_unset=True) # 설정된 필드만 덤프
                # parts를 CommonPart의 dict 형태로 변환하여 재검증
                message_dict['parts'] = [p.model_dump(exclude_unset=True) for p in new_parts_for_message]
                processed_messages.append(Message.model_validate(message_dict))

            else:
                processed_messages.append(m)
        return processed_messages


    async def _pending_messages(self):
        try:
            pending = self.manager.get_pending_messages()
            return PendingMessageResponse(result=pending)
        except Exception as e:
            logger.exception("Error getting pending messages.")
            raise HTTPException(status_code=500, detail=f"Failed to get pending messages: {str(e)}")

    def _list_conversations(self):
        try:
            return ListConversationResponse(result=self.manager.conversations)
        except Exception as e:
            logger.exception("Error listing conversations.")
            raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

    def _get_events(self):
        try:
            return GetEventResponse(result=self.manager.events)
        except Exception as e:
            logger.exception("Error getting events.")
            raise HTTPException(status_code=500, detail=f"Failed to get events: {str(e)}")

    def _list_tasks(self):
        try:
            return ListTaskResponse(result=self.manager.tasks)
        except Exception as e:
            logger.exception("Error listing tasks.")
            raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

    async def _register_agent(self, request: Request): # 이미 async
        try:
            data = await request.json()
            url = data.get('params')
            if not url or not isinstance(url, str):
                raise HTTPException(status_code=400, detail="Invalid or missing 'params' (agent URL).")

            # self.manager.register_agent가 async로 변경되었으므로 await 사용
            await self.manager.register_agent(url) 
            logger.info(f"Agent registration process completed for URL: {url}")
            return RegisterAgentResponse(result="Agent registration process initiated successfully.") # 성공 메시지 또는 True 반환
        except AgentRegistrationError as are: # ADKHostManager에서 발생시킨 사용자 정의 예외 처리
            logger.error(f"Error registering agent with URL: {url}. Detail: {str(are)}")
            raise HTTPException(status_code=400, detail=str(are)) # 클라이언트에게 오류 전달
        except HTTPException as http_exc: # 이미 HTTPException이면 그대로 발생
            raise http_exc
        except Exception as e: # 기타 예외 처리
            logger.exception(f"Internal server error during agent registration for URL: {url}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


    async def _list_agents(self):
        try:
            return ListAgentResponse(result=self.manager.agents)
        except Exception as e:
            logger.exception("Error listing agents.")
            raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

    async def _unregister_agent(self, request: Request):
        # service.types에 UnregisterAgentResponse가 정의되어 있고 error 필드를 지원한다고 가정
        # from service.types import UnregisterAgentResponse # 가정
        class UnregisterAgentResponsePlaceholder: # 임시 플레이스홀더
            def __init__(self, result: bool = True, error: Optional[str] = None):
                self.result = result
                self.error = error

        try:
            data = await request.json()
            url = data.get('params')
            if not url or not isinstance(url, str):
                raise HTTPException(status_code=400, detail="Invalid or missing 'params' (expected agent URL string).")

            if not hasattr(self.manager, 'unregister_agent'):
                logger.error(f"Manager type {type(self.manager).__name__} does not support unregister_agent method.")
                raise HTTPException(status_code=501, detail="Unregister agent not supported by the current manager.")

            # manager에 unregister_agent가 있다고 가정하고 호출 (AttributeError는 위에서 처리)
            self.manager.unregister_agent(url) # type: ignore [attr-defined]
            logger.info(f"Agent unregistration initiated for URL: {url}")
            # 실제 UnregisterAgentResponse 사용 (service.types에 정의 필요)
            # return UnregisterAgentResponse(result=True)
            return {"status": "success", "message": f"Agent {url} unregistration process initiated."} # 일반 JSON 응답
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            request_param_info = data.get('params') if isinstance(data, dict) else "N/A"
            logger.exception(f"Error unregistering agent with URL: {request_param_info}")
            # 실제 UnregisterAgentResponse 사용 (service.types에 정의 필요)
            # return UnregisterAgentResponse(result=False, error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to unregister agent: {str(e)}")


    def _get_file(self, file_id: str): # FastAPI가 file_id를 자동으로 전달
        try:
            if file_id not in self._file_cache:
                logger.warning(f"File not found in cache: {file_id}")
                raise HTTPException(status_code=404, detail="File not found.")
            
            cached_file_part = self._file_cache[file_id] # FilePart 객체

            if not cached_file_part.file or not cached_file_part.file.bytes:
                 logger.error(f"Cached file part for ID {file_id} is missing essential content (file object or file.bytes).")
                 raise HTTPException(status_code=500, detail="Cached file content is corrupted or incomplete.")

            # file.bytes가 base64 인코딩된 문자열이라고 가정
            try:
                file_bytes_content = base64.b64decode(cached_file_part.file.bytes)
            except Exception as b64_error:
                logger.error(f"Failed to decode base64 content for file ID {file_id}: {b64_error}", exc_info=True)
                raise HTTPException(status_code=500, detail="File content is not valid base64 data.")

            mime_type = cached_file_part.file.mimeType or "application/octet-stream" # 기본 MIME 타입
            
            return FastAPIResponse( # fastapi.Response 사용
                content=file_bytes_content,
                media_type=mime_type,
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.exception(f"Error retrieving file: {file_id}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")