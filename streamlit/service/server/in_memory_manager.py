import asyncio
import datetime
import uuid
import logging # 로깅 모듈 추가
from typing import Dict, List, Optional, Tuple, Union # 타입 힌팅 개선

# common.types와 로컬 임포트는 실행 환경에서 올바르게 경로가 설정되어 있다고 가정합니다.
from common.types import (
    AgentCard,
    Artifact,
    DataPart,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
# test_image는 해당 경로에 파일이 있다고 가정합니다.
# from service.server import test_image # 원본 임포트, 실제 파일 경로에 따라 조정 필요
# demo/ui/service/server/test_image.py 를 사용한다고 가정
from service.server import test_image as demo_test_image


from service.server.application_manager import ApplicationManager
# service.types는 demo/ui/service/types.py 를 사용한다고 가정
from service.types import Conversation, Event
# utils.agent_card는 demo/ui/utils/agent_card.py 를 사용한다고 가정
from utils.agent_card import get_agent_card

logger = logging.getLogger(__name__) # 로거 인스턴스 생성

# InMemoryFakeAgentManager 클래스 외부로 이동하거나, 클래스 변수로 관리될 수 있습니다.
# 이 메시지 큐는 순서대로 반환될 미리 준비된 응답들을 나타냅니다.
# demo_test_image 사용을 위해 수정
_message_queue: List[Message] = [
    Message(role='agent', parts=[TextPart(text='Hello')]),
    Message(
        role='agent',
        parts=[
            DataPart(
                data={
                    'type': 'form',
                    'form': {
                        'type': 'object',
                        'properties': {
                            'name': {
                                'type': 'string',
                                'description': 'Enter your name',
                                'title': 'Name',
                            },
                            'date': {
                                'type': 'string',
                                'format': 'date',
                                'description': 'Birthday',
                                'title': 'Birthday',
                            },
                        },
                        'required': ['date'],
                    },
                    'form_data': {
                        'name': 'John Smith',
                    },
                    'instructions': 'Please provide your birthday and name',
                }
            ),
        ],
    ),
    Message(role='agent', parts=[TextPart(text='I like cats')]),
    demo_test_image.test_image, # service.server.test_image 대신 demo_test_image 사용
    Message(role='agent', parts=[TextPart(text='And I like dogs')]),
]

class InMemoryFakeAgentManager(ApplicationManager):
    """
    메모리 기반 관리 및 가짜 에이전트 액션을 사용하는 ApplicationManager 구현체입니다.
    AgentServer에 연결되어 프론트엔드에 정보를 제공하고 에이전트에 메시지를 보냅니다.
    """

    _conversations: Dict[str, Conversation]
    _messages: List[Message]
    _tasks: Dict[str, Task]
    _events: List[Event]
    _pending_message_ids: List[str]
    _next_message_idx: int
    _agents: List[AgentCard]
    _task_map: Dict[str, str] # message_id -> task_id 매핑

    _simulated_processing_time: float = 0.5 # 가짜 처리 시간 (초)

    def __init__(self):
        self._conversations = {}
        self._messages = []
        self._tasks = {}
        self._events = []
        self._pending_message_ids = []
        self._next_message_idx = 0
        self._agents = []
        self._task_map = {}
        logger.info("InMemoryFakeAgentManager initialized.")

    def create_conversation(self) -> Conversation:
        conversation_id = str(uuid.uuid4())
        # messages 리스트를 빈 리스트로 초기화
        c = Conversation(conversation_id=conversation_id, is_active=True, messages=[])
        self._conversations[conversation_id] = c
        logger.info(f"Conversation created: {conversation_id}")
        return c

    def sanitize_message(self, message: Message) -> Message:
        if not message.metadata:
            message.metadata = {}
        if 'message_id' not in message.metadata:
            message.metadata['message_id'] = str(uuid.uuid4())

        conversation_id = message.metadata.get('conversation_id')
        if conversation_id:
            conversation = self.get_conversation(conversation_id)
            # conversation.messages가 None이 아니고 비어있지 않은지 확인
            if conversation and conversation.messages:
                last_message_obj = conversation.messages[-1]
                # last_message_obj.metadata가 None이 아닌지 확인 후 'message_id' 접근
                last_message_id = last_message_obj.metadata.get('message_id') if last_message_obj.metadata else None
                if last_message_id:
                    message.metadata['last_message_id'] = last_message_id
        return message

    async def process_message(self, message: Message):
        message_id = message.metadata.get('message_id')
        if not message_id:
            logger.error("Message arrived at process_message without a message_id. Sanitization might have failed.")
            return

        self._messages.append(message) # 전체 메시지 목록 (선택적)
        self._pending_message_ids.append(message_id)
        
        try:
            conversation_id = message.metadata.get('conversation_id')
            conversation = self.get_conversation(conversation_id)

            if conversation:
                if conversation.messages is None: # 방어 코드
                    conversation.messages = []
                conversation.messages.append(message)
            else:
                logger.warning(f"Conversation not found for ID: {conversation_id} during message processing.")

            # 사용자 요청 이벤트 기록
            self.add_event(
                Event(
                    id=str(uuid.uuid4()),
                    actor='user',
                    content=message,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).timestamp(), # timezone.utc 사용
                )
            )

            # 가짜 작업 생성
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                sessionId=conversation_id,
                status=TaskStatus(
                    state=TaskState.SUBMITTED,
                    message=message,
                ),
                history=[message],
                artifacts=[], # 아티팩트 초기화
            )
            self._task_map[message_id] = task_id
            self.add_task(task)
            logger.info(f"Task {task_id} created and submitted for message {message_id}")

            # 가짜 처리 시간 시뮬레이션
            await asyncio.sleep(self._simulated_processing_time)

            # 다음 가짜 응답 가져오기
            response = self.next_message()
            response_message_id = str(uuid.uuid4())
            
            # 응답 메시지 메타데이터 설정
            response.metadata = {
                'conversation_id': conversation_id,
                'message_id': response_message_id,
                'last_message_id': message_id, # 현재 처리 중인 사용자 메시지 ID
            }
            # 원본 사용자 메시지의 다른 메타데이터 필드 복사 (충돌 방지)
            if message.metadata:
                for k, v in message.metadata.items():
                    if k not in response.metadata: # 주요 ID 필드 덮어쓰기 방지
                        response.metadata[k] = v
            
            self._messages.append(response) # 선택적: 전체 메시지 목록에 응답 추가
            if conversation:
                conversation.messages.append(response)

            # 에이전트 응답 이벤트 기록
            self.add_event(
                Event(
                    id=str(uuid.uuid4()),
                    actor='agent',
                    content=response,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).timestamp(), # timezone.utc 사용
                )
            )
            
            # 작업 완료 처리
            if task.status: # task.status가 None이 아닌지 확인
                task.status.state = TaskState.COMPLETED
            # response.parts가 None이 아니고 비어있지 않은지 확인
            if response.parts:
                 task.artifacts = [Artifact(name='response_artifact', parts=response.parts)]
            if task.history is None: # 방어 코드
                task.history = []
            task.history.append(response)
            self.update_task(task)
            logger.info(f"Task {task_id} completed for message {message_id}")

        except Exception as e:
            logger.exception(f"Error processing message {message_id}: {e}")
        finally:
            if message_id in self._pending_message_ids:
                self._pending_message_ids.remove(message_id)

    def add_task(self, task: Task):
        self._tasks[task.id] = task
        logger.debug(f"Task added/updated: {task.id}, Status: {task.status.state if task.status else 'N/A'}")

    def update_task(self, task: Task): # add_task와 동일하게 upsert로 동작
        self._tasks[task.id] = task
        logger.debug(f"Task explicitly updated: {task.id}, Status: {task.status.state if task.status else 'N/A'}")

    def add_event(self, event: Event):
        self._events.append(event)
        # logger.debug(f"Event added: {event.id} by {event.actor}") # 필요시 상세 로깅

    def next_message(self) -> Message:
        if not _message_queue:
            logger.warning("Message queue is empty. Returning a default message.")
            # metadata를 빈 dict로 초기화
            return Message(role='agent', parts=[TextPart(text='No messages available.')], metadata={})
        
        # Message 객체를 깊은 복사하거나, model_dump 후 다시 생성하여 원본 수정 방지
        message_template = _message_queue[self._next_message_idx]
        # pydantic 모델의 model_dump()와 model_validate()를 사용하여 복사본 생성
        message_copy = Message.model_validate(message_template.model_dump())

        self._next_message_idx = (self._next_message_idx + 1) % len(_message_queue)
        return message_copy

    def get_conversation(self, conversation_id: Optional[str]) -> Optional[Conversation]:
        if not conversation_id:
            return None
        return self._conversations.get(conversation_id)

    def get_pending_messages(self) -> List[str]: # ApplicationManager 인터페이스와 반환 타입 일치
        status_only_list: List[str] = []

        for message_id in self._pending_message_ids:
            task_id = self._task_map.get(message_id)
            status_text = "Initializing..."

            if task_id:
                task = self._tasks.get(task_id)
                if task:
                    if task.status and task.status.state == TaskState.WORKING:
                        status_text = "Working..."
                    elif task.history and task.history[-1]: # history 마지막 요소 존재 확인
                        last_history_message = task.history[-1]
                        if last_history_message.parts and last_history_message.parts[0]: # parts 및 첫번째 요소 존재 확인
                            first_part = last_history_message.parts[0]
                            if isinstance(first_part, TextPart):
                                status_text = first_part.text[:50] + "..." if len(first_part.text) > 50 else first_part.text
                            elif hasattr(first_part, 'type'): # type 속성이 있는지 확인
                                status_text = f"Processing {first_part.type}..."
                            else:
                                status_text = "Processing data..."
                        elif task.status:
                             status_text = str(task.status.state).capitalize()
                        else:
                            status_text = "Pending..."
                    elif task.status:
                        status_text = str(task.status.state).capitalize()
                    else:
                        status_text = "Task found, unknown state."
                else:
                    status_text = f"Task (ID: {task_id}) not found."
            
            status_only_list.append(status_text)
        return status_only_list

    def register_agent(self, url: str):
        try:
            agent_data = get_agent_card(url)
            if not agent_data: # get_agent_card가 None을 반환하는 경우
                logger.error(f"Failed to get agent card from URL (returned None): {url}")
                return
        except Exception as e:
            logger.exception(f"Error getting agent card from URL {url}: {e}")
            return

        # agent_data가 None이 아닐 때만 url 필드 설정
        if not agent_data.url:
            agent_data.url = url
        
        # 이미 등록된 에이전트인지 URL 기준으로 확인
        if any(ag.url == agent_data.url for ag in self._agents):
            logger.info(f"Agent with URL {agent_data.url} is already registered.")
            return

        self._agents.append(agent_data)
        logger.info(f"Agent registered: {agent_data.name} from {url}")

    @property
    def agents(self) -> List[AgentCard]:
        return self._agents

    @property
    def conversations(self) -> List[Conversation]:
        return list(self._conversations.values())

    @property
    def tasks(self) -> List[Task]:
        return list(self._tasks.values())

    @property
    def events(self) -> List[Event]:
        # timestamp 기준으로 정렬하여 반환
        return sorted(self._events, key=lambda e: e.timestamp)