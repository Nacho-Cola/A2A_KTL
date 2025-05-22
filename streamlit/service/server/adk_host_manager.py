import base64
import datetime
import json
import os
import uuid
import logging # Added
from typing import Dict, List, Optional, Any, Union # Adjusted for Dict

# Assuming common.types and other local imports are correctly resolved in the execution environment
from common.types import (
    AgentCard,
    DataPart,
    FileContent,
    FilePart,
    Message,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events.event import Event as ADKEvent
from google.adk.events.event_actions import EventActions as ADKEventActions
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types # Aliased to avoid conflict if 'types' is used locally
from hosts.multiagent.host_agent import HostAgent
from hosts.multiagent.remote_agent_connection import (
    TaskCallbackArg,
)
# These service and utils imports are assumed to be from the demo/ui structure
from service.server.application_manager import ApplicationManager
from service.types import Conversation, Event # Assuming service.types is from demo/ui/service/types.py
from utils.agent_card import get_agent_card # Assuming utils.agent_card is from demo/ui/utils/agent_card.py

logger = logging.getLogger(__name__) # Added logger

class ADKHostManager(ApplicationManager):
    """An implementation of memory based management with fake agent actions

    This implements the interface of the ApplicationManager to plug into
    the AgentServer. This acts as the service contract that the Mesop app
    uses to send messages to the agent and provide information for the frontend.
    """

    _conversations: Dict[str, Conversation] # Changed to Dict
    _messages: List[Message]
    _tasks: Dict[str, Task] # Changed to Dict
    _events: Dict[str, Event] # Kept as Dict (was already Dict)
    _pending_message_ids: List[str] # Consider set if order isn't crucial and list grows large
    _agents: List[AgentCard]
    _task_map: Dict[str, str] # Maps message_id to task_id
    _artifact_chunks: Dict[str, Dict[int, Any]] # event_id -> chunk_index -> artifact_chunk
    _next_id: Dict[str, str] # previous message to next message

    def __init__(self, api_key: str = '', uses_vertex_ai: bool = False):
        self._conversations = {} # Changed
        self._messages = []
        self._tasks = {} # Changed
        self._events = {}
        self._pending_message_ids = []
        self._agents = []
        self._artifact_chunks = {}
        self._session_service = InMemorySessionService()
        self._artifact_service = InMemoryArtifactService()
        self._memory_service = InMemoryMemoryService()
        self._host_agent = HostAgent([], self.task_callback) # Assuming HostAgent handles an empty list
        self.user_id = 'test_user'
        self.app_name = 'A2A'
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY', '')
        self.uses_vertex_ai = (
            uses_vertex_ai
            or os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'
        )

        if self.uses_vertex_ai:
            os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'TRUE'
        elif self.api_key:
            os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'FALSE'
            os.environ['GOOGLE_API_KEY'] = self.api_key
        else:
            # Handle case where no API key is provided and not using Vertex AI
            logger.warning("No API key provided and not using Vertex AI. GenAI features may not work.")


        self._initialize_host()
        self._task_map = {}
        self._next_id = {}

    def update_api_key(self, api_key: str):
        if api_key and api_key != self.api_key:
            self.api_key = api_key
            if not self.uses_vertex_ai:
                os.environ['GOOGLE_API_KEY'] = api_key
                logger.info("API key updated. Reinitializing host.")
                self._initialize_host()

    def _initialize_host(self):
        try:
            agent = self._host_agent.create_agent()
            self._host_runner = Runner(
                app_name=self.app_name,
                agent=agent,
                artifact_service=self._artifact_service,
                session_service=self._session_service,
                memory_service=self._memory_service,
            )
            logger.info("ADK Host initialized/reinitialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize ADK Host Runner.")
            # Depending on severity, might want to raise or set a faulty state

    def create_conversation(self) -> Conversation:
        session = self._session_service.create_session(
            app_name=self.app_name, user_id=self.user_id
        )
        conversation_id = session.id
        c = Conversation(conversation_id=conversation_id, is_active=True)
        self._conversations[conversation_id] = c # Changed
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
            if conversation and conversation.messages:
                last_message_obj = conversation.messages[-1]
                last_message_id = get_message_id(last_message_obj)
                if last_message_id:
                    message.metadata['last_message_id'] = last_message_id
        return message

    async def process_message(self, message: Message):
        self._messages.append(message) # Consider capping self._messages if it grows too large
        message_id = get_message_id(message)

        if not message_id:
            logger.error("Message lacks a message_id after sanitization. Aborting processing.")
            return

        self._pending_message_ids.append(message_id)
        try:
            conversation_id = message.metadata.get('conversation_id')
            conversation = self.get_conversation(conversation_id)
            if conversation:
                if conversation.messages is None: # Ensure messages list exists
                    conversation.messages = []
                conversation.messages.append(message)

            self.add_event(
                Event(
                    id=str(uuid.uuid4()),
                    actor='user',
                    content=message,
                    timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
                )
            )

            final_adk_event: Optional[ADKEvent] = None
            session = self._session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=conversation_id
            )
            
            state_update = {
                'input_message_metadata': message.metadata,
                'session_id': conversation_id,
            }
            
            last_msg_id_for_task_check = get_last_message_id(message)
            task_to_resume_id = self._task_map.get(last_msg_id_for_task_check)
            if task_to_resume_id:
                task_obj = self._tasks.get(task_to_resume_id)
                if task_still_open(task_obj):
                    state_update['task_id'] = task_to_resume_id
            
            if session: # Ensure session is not None before appending event
                self._session_service.append_event(
                    session,
                    ADKEvent(
                        id=ADKEvent.new_id(),
                        author='host_agent',
                        invocation_id=ADKEvent.new_id(),
                        actions=ADKEventActions(state_delta=state_update),
                    ),
                )
            else:
                logger.warning(f"Session not found for conversation_id: {conversation_id}. Cannot append state update event.")


            adk_input_content = self.adk_content_from_message(message)
            
            try:
                async for adk_event_item in self._host_runner.run_async(
                    user_id=self.user_id,
                    session_id=conversation_id,
                    new_message=adk_input_content,
                ):
                    if not conversation_id: # Should have conversation_id if session was found
                        logger.warning("conversation_id is missing when processing ADK event.")
                    
                    event_content_message = self.adk_content_to_message(
                        adk_event_item.content, conversation_id or "" # Pass empty string if None, though should exist
                    )
                    self.add_event(
                        Event(
                            id=adk_event_item.id,
                            actor=adk_event_item.author,
                            content=event_content_message,
                            timestamp=adk_event_item.timestamp,
                        )
                    )
                    final_adk_event = adk_event_item
            except Exception as e:
                logger.exception(f"Error during ADK host run for message {message_id} in conversation {conversation_id}")
                # Optionally, create an error response message or update task status
                # For now, error is logged, and processing continues to generate a (possibly empty) response.

            response: Optional[Message] = None
            if final_adk_event and final_adk_event.content:
                final_adk_event.content.role = 'model' # Per original logic
                response = self.adk_content_to_message(
                    final_adk_event.content, conversation_id or ""
                )
                
                # Metadata assignment logic from original, ensure clarity
                user_msg_id = get_message_id(message) # ID of the incoming user message
                agent_response_id = str(uuid.uuid4()) # Default new ID for agent's response

                # Original logic for _next_id (retained due to unclear full intent)
                if user_msg_id and user_msg_id in self._next_id:
                    agent_response_id = self._next_id[user_msg_id]
                    # If user_msg_id was in _next_id, it means this response ID was pre-determined.
                    # 'last_message_id_for_response_metadata' should still be the user_msg_id.
                    last_message_id_for_response_metadata = user_msg_id
                else:
                    # If not in _next_id, a new agent_response_id is used.
                    # The 'last_message_id' on the response should be the user_msg_id.
                    last_message_id_for_response_metadata = user_msg_id
                
                response.metadata = {
                    # Start with a clean slate for critical IDs
                    'conversation_id': conversation_id,
                    'message_id': agent_response_id,
                    'last_message_id': last_message_id_for_response_metadata,
                }
                # Copy other non-conflicting metadata from original user message
                if message.metadata:
                    for k, v in message.metadata.items():
                        if k not in response.metadata: # Avoid overwriting critical keys
                            response.metadata[k] = v
                
                self._messages.append(response)
                if conversation and response: # Ensure response is not None
                    if conversation.messages is None: conversation.messages = []
                    conversation.messages.append(response)

        except Exception as e:
            logger.exception(f"Unhandled error in process_message for message_id {message_id}")
        finally:
            if message_id in self._pending_message_ids:
                self._pending_message_ids.remove(message_id)

    def add_task(self, task: Task):
        self._tasks[task.id] = task # Changed
        logger.debug(f"Task added: {task.id}")

    def update_task(self, task: Task):
        if task.id in self._tasks:
            self._tasks[task.id] = task # Changed
            logger.debug(f"Task updated: {task.id}")
        else:
            logger.warning(f"Attempted to update non-existent task: {task.id}")


    def task_callback(self, task_arg: TaskCallbackArg, agent_card: AgentCard): # Renamed 'task' to 'task_arg'
        self.emit_event(task_arg, agent_card)
        current_task: Optional[Task] = None

        if isinstance(task_arg, TaskStatusUpdateEvent):
            current_task = self.add_or_get_task(task_arg)
            current_task.status = task_arg.status
            self.attach_message_to_task(task_arg.status.message, current_task.id)
            self.insert_message_history(current_task, task_arg.status.message)
            # update_task is called implicitly by add_or_get_task or explicitly below if needed
            self_tasks_task_id = self._tasks.get(current_task.id)
            if self_tasks_task_id:
                self_tasks_task_id.status = task_arg.status # ensure the dict entry is updated
            self.insert_id_trace(task_arg.status.message)

        elif isinstance(task_arg, TaskArtifactUpdateEvent):
            current_task = self.add_or_get_task(task_arg)
            self.process_artifact_event(current_task, task_arg)
            # update_task might be needed if process_artifact_event modified fields not covered by reference

        elif isinstance(task_arg, Task): # This is a Task object itself
            task_id = task_arg.id
            if task_id not in self._tasks:
                # This is a new task object not from an event, treat as initial Task
                current_task = task_arg # Use the passed Task object directly
                self._tasks[task_id] = current_task # Add to manager
                self.attach_message_to_task(current_task.status.message, current_task.id)
                self.insert_id_trace(current_task.status.message)
                logger.info(f"New task object registered via callback: {task_id}")
            else:
                # This is an update to an existing Task object
                current_task = self._tasks[task_id]
                # Update fields from task_arg to current_task
                current_task.status = task_arg.status 
                current_task.artifacts = task_arg.artifacts # Or merge artifacts
                current_task.history = task_arg.history # Or merge history
                current_task.metadata = task_arg.metadata # Or merge metadata
                # ... any other fields ...
                self.attach_message_to_task(current_task.status.message, current_task.id)
                self.insert_id_trace(current_task.status.message)
                logger.info(f"Task object updated via callback: {task_id}")
        else:
            logger.warning(f"Unknown type in task_callback: {type(task_arg)}")
            return None # Or raise error

        if current_task: # Ensure current_task is not None before returning
             # Ensure the version in self._tasks is the most up-to-date one.
            self._tasks[current_task.id] = current_task
            return current_task
        return None


    def emit_event(self, task_arg: TaskCallbackArg, agent_card: AgentCard):
        content: Optional[Message] = None
        conversation_id = get_conversation_id(task_arg)
        metadata = {'conversation_id': conversation_id} if conversation_id else {}

        if isinstance(task_arg, TaskStatusUpdateEvent):
            if task_arg.status.message:
                content = task_arg.status.message
            else:
                content = Message(
                    parts=[TextPart(text=str(task_arg.status.state))],
                    role='agent',
                    metadata=metadata,
                )
        elif isinstance(task_arg, TaskArtifactUpdateEvent):
            content = Message(
                parts=task_arg.artifact.parts, # Assuming parts is List[Part]
                role='agent',
                metadata=metadata,
            )
        elif isinstance(task_arg, Task) and task_arg.status and task_arg.status.message:
            content = task_arg.status.message
        elif isinstance(task_arg, Task) and task_arg.artifacts:
            parts_list: List[Part] = []
            for artifact_item in task_arg.artifacts:
                parts_list.extend(artifact_item.parts)
            content = Message(
                parts=parts_list,
                role='agent',
                metadata=metadata,
            )
        elif isinstance(task_arg, Task): # Fallback for Task if other conditions not met
             content = Message(
                parts=[TextPart(text=str(task_arg.status.state if task_arg.status else "unknown_state"))],
                role='agent',
                metadata=metadata,
            )
        else:
            # Fallback for unknown task_arg type or structure
            logger.warning(f"emit_event: Could not determine content for task_arg type {type(task_arg)}")
            content = Message(
                parts=[TextPart(text="System event update.")],
                role='agent',
                metadata=metadata,
            )
        
        if content: # Ensure content was created
            self.add_event(
                Event(
                    id=str(uuid.uuid4()),
                    actor=agent_card.name or "Unknown Agent", # Ensure agent_card.name is not None
                    content=content,
                    timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
                )
            )

    def attach_message_to_task(self, message: Optional[Message], task_id: str):
        if message and message.metadata and 'message_id' in message.metadata:
            msg_id = message.metadata['message_id']
            self._task_map[msg_id] = task_id
            logger.debug(f"Message {msg_id} attached to task {task_id}")

    def insert_id_trace(self, message: Optional[Message]):
        if not message:
            return
        message_id = get_message_id(message)
        last_message_id = get_last_message_id(message)
        if message_id and last_message_id:
            self._next_id[last_message_id] = message_id
            logger.debug(f"ID trace inserted: {last_message_id} -> {message_id}")

    def insert_message_history(self, task: Task, message: Optional[Message]):
        if not message:
            return
        if task.history is None:
            task.history = []
        
        message_id_to_add = get_message_id(message)
        if not message_id_to_add:
            logger.debug("Message has no ID, cannot add to task history.")
            return

        # Check if message with this ID already exists in history
        # Original logic: if get_message_id(task.status.message) not in [get_message_id(x) for x in task.history]:
        # This compared task.status.message's ID, not the incoming message's ID for history addition.
        # Correcting to check if the current `message` (message_id_to_add) is already in history.
        
        existing_history_ids = {get_message_id(hist_msg) for hist_msg in task.history if hist_msg}
        if message_id_to_add not in existing_history_ids:
            task.history.append(message)
            logger.debug(f"Message {message_id_to_add} added to history of task {task.id}")
        else:
            logger.debug(
                f"Message id {message_id_to_add} already in history for task {task.id}"
            )


    def add_or_get_task(self, task_arg: TaskCallbackArg) -> Task: # Changed task to task_arg
        task_id = task_arg.id
        current_task = self._tasks.get(task_id)

        if not current_task:
            conversation_id = task_arg.metadata.get('conversation_id') if task_arg.metadata else None
            
            # Determine initial state based on task_arg type
            initial_status_state = TaskState.SUBMITTED
            if isinstance(task_arg, TaskStatusUpdateEvent):
                initial_status_state = task_arg.status.state
            elif isinstance(task_arg, Task) and task_arg.status:
                 initial_status_state = task_arg.status.state


            current_task = Task(
                id=task_id,
                status=TaskStatus(state=initial_status_state),
                metadata=task_arg.metadata if task_arg.metadata else {},
                artifacts=[],
                history=[], # Initialize history
                sessionId=conversation_id,
            )
            self.add_task(current_task) # Uses the new dict-based add_task
            logger.info(f"New task created and added: {task_id} from task_arg type {type(task_arg)}")
        return current_task


    def process_artifact_event(
        self, current_task: Task, task_update_event: TaskArtifactUpdateEvent
    ):
        artifact = task_update_event.artifact
        event_id = task_update_event.id # Assuming task_update_event.id is the key for _artifact_chunks

        if not artifact.append:
            if artifact.lastChunk is None or artifact.lastChunk:
                if current_task.artifacts is None:
                    current_task.artifacts = []
                current_task.artifacts.append(artifact)
                logger.debug(f"Full artifact added to task {current_task.id}, event {event_id}")
            else:
                # First chunk of a multi-chunk artifact
                if event_id not in self._artifact_chunks:
                    self._artifact_chunks[event_id] = {}
                self._artifact_chunks[event_id][artifact.index or 0] = artifact # Ensure index is not None
                logger.debug(f"Initial artifact chunk {artifact.index or 0} stored for event {event_id}, task {current_task.id}")
        else: # append is True
            event_id_chunks = self._artifact_chunks.get(event_id)
            if not event_id_chunks:
                logger.error(f"Attempted to append chunk for event {event_id}, but no prior chunks found.")
                return

            current_temp_artifact = event_id_chunks.get(artifact.index or 0)
            if not current_temp_artifact:
                logger.error(f"Attempted to append to chunk index {artifact.index or 0} for event {event_id}, but chunk not found.")
                return
            
            if current_temp_artifact.parts is None: current_temp_artifact.parts = []
            if artifact.parts: current_temp_artifact.parts.extend(artifact.parts)
            
            logger.debug(f"Appended parts to chunk {artifact.index or 0} for event {event_id}, task {current_task.id}")

            if artifact.lastChunk:
                if current_task.artifacts is None:
                    current_task.artifacts = []
                current_task.artifacts.append(current_temp_artifact)
                event_id_chunks.pop(artifact.index or 0, None)
                if not event_id_chunks: # If no more chunks for this event_id
                    self._artifact_chunks.pop(event_id, None)
                logger.debug(f"Final appended artifact chunk {artifact.index or 0} processed for event {event_id}, task {current_task.id}")


    def add_event(self, event: Event):
        self._events[event.id] = event
        # logger.debug(f"Event added: {event.id}, actor: {event.actor}") # Can be verbose

    def get_conversation(self, conversation_id: Optional[str]) -> Optional[Conversation]:
        if not conversation_id:
            return None
        return self._conversations.get(conversation_id) # Changed

    def get_pending_messages(self) -> List[tuple[str, str]]:
        rval = []
        for message_id in self._pending_message_ids:
            task_id = self._task_map.get(message_id)
            if task_id:
                task = self._tasks.get(task_id) # Changed
                if not task:
                    rval.append((message_id, 'Task not found'))
                elif task.history and task.history[-1].parts: # Ensure history and parts exist
                    if len(task.history) == 1 and task.status.state == TaskState.SUBMITTED:
                         rval.append((message_id, 'Submitted...'))
                    elif task.status.state == TaskState.WORKING:
                         rval.append((message_id, 'Working...'))
                    else:
                        # Try to get text from the last part of the last history message
                        last_hist_msg = task.history[-1]
                        if last_hist_msg.parts:
                            part = last_hist_msg.parts[0] # Assuming first part is representative
                            if isinstance(part, TextPart): # Check if TextPart
                                rval.append((message_id, part.text))
                            else:
                                rval.append((message_id, f'{part.type.capitalize()} received...'))
                        else:
                             rval.append((message_id, 'Processing...'))
                elif task.status:
                     rval.append((message_id, str(task.status.state).capitalize() + '...'))
                else:
                    rval.append((message_id, 'Pending...')) # Fallback if task has no history/status
            else:
                rval.append((message_id, 'Initializing...')) # Message ID not yet mapped to a task
        return rval

    async def register_agent(self, url: str): # async def로 변경
        logger.info(f"ADKHostManager: Attempting to register agent from URL: {url}")
        try:
            # get_agent_card는 비동기 함수이므로 await 사용
            agent_data: Optional[AgentCard] = await get_agent_card(url)
        except Exception as e:
            # get_agent_card 내부에서 오류 로깅 및 None 반환을 할 수도 있지만,
            # 여기서도 예외를 잡아 구체적인 컨텍스트와 함께 로깅하는 것이 좋음
            logger.exception(f"ADKHostManager: Error calling get_agent_card for URL {url}")
            # 이 오류를 다시 발생시키거나, 혹은 여기서 False/None 등을 반환하여
            # ConversationServer가 적절한 HTTP 응답을 하도록 유도할 수 있습니다.
            # 여기서는 예외를 다시 발생시켜 ConversationServer의 try-except 블록에서 처리하도록 합니다.
            raise AgentRegistrationError(f"Failed to fetch agent card from {url}: {str(e)}") from e

        if not agent_data:
            logger.error(f"ADKHostManager: No agent data returned from get_agent_card for URL: {url}")
            # AgentCard를 가져오지 못했으므로 오류 처리
            raise AgentRegistrationError(f"Agent card data is missing or could not be retrieved from {url}.")

        # 이제 agent_data는 AgentCard 객체이므로 .url 등의 속성에 접근 가능
        if not agent_data.url: # AgentCard에 url 필드가 있고, 비어있을 경우 채워넣기
            agent_data.url = url

        # 중복 등록 방지 로직 (선택 사항)
        if any(ag.url == agent_data.url for ag in self._agents):
            logger.info(f"ADKHostManager: Agent with URL {agent_data.url} is already registered.")
            # 이미 등록된 경우 성공으로 간주하거나, 특정 응답을 줄 수 있음
            return # 혹은 raise DuplicateAgentError(...)

        self._agents.append(agent_data)
        # self._host_agent.register_agent_card는 AgentCard 객체를 받는다고 가정
        # 이 부분은 self._host_agent의 실제 구현에 따라 달라질 수 있음
        if hasattr(self._host_agent, 'register_agent_card'):
             self._host_agent.register_agent_card(agent_data)
        else:
            logger.warning("ADKHostManager: self._host_agent does not have 'register_agent_card' method.")


        logger.info(f"ADKHostManager: Agent '{agent_data.name}' from '{url}' successfully processed for registration. Reinitializing host.")
        self._initialize_host() 

    @property
    def agents(self) -> List[AgentCard]:
        return self._agents

    @property
    def conversations(self) -> List[Conversation]:
        return list(self._conversations.values()) # Changed

    @property
    def tasks(self) -> List[Task]:
        return list(self._tasks.values()) # Changed

    @property
    def events(self) -> List[Event]:
        return sorted(list(self._events.values()), key=lambda x: x.timestamp)

    def adk_content_from_message(self, message: Message) -> genai_types.Content:
        parts: list[genai_types.Part] = []
        for part_item in message.parts: # Renamed part to part_item to avoid conflict
            if isinstance(part_item, TextPart):
                parts.append(genai_types.Part.from_text(text=part_item.text))
            elif isinstance(part_item, DataPart):
                try:
                    json_string = json.dumps(part_item.data)
                    parts.append(genai_types.Part.from_text(text=json_string))
                except TypeError as e:
                    logger.error(f"Could not serialize DataPart data to JSON: {e}. Data: {part_item.data}")
                    parts.append(genai_types.Part.from_text(text=f"Error: Could not serialize data - {part_item.data}"))
            elif isinstance(part_item, FilePart):
                if part_item.uri:
                    parts.append(
                        genai_types.Part.from_uri(
                            file_uri=part_item.uri, mime_type=part_item.mimeType or "application/octet-stream"
                        )
                    )
                elif part_item.file and part_item.file.bytes: # Assuming file.bytes is base64 string
                    try:
                        decoded_bytes = base64.b64decode(part_item.file.bytes)
                        parts.append(
                            genai_types.Part.from_bytes(
                                data=decoded_bytes,
                                mime_type=part_item.mimeType or part_item.file.mimeType or "application/octet-stream",
                            )
                        )
                    except Exception as e:
                        logger.error(f"Failed to decode base64 bytes from FilePart.file.bytes: {e}")
                elif part_item.data: # Assuming part_item.data is base64 string for inline file
                     try:
                        decoded_bytes = base64.b64decode(part_item.data)
                        parts.append(
                            genai_types.Part.from_bytes(
                                data=decoded_bytes,
                                mime_type=part_item.mimeType or "application/octet-stream",
                            )
                        )
                     except Exception as e:
                        logger.error(f"Failed to decode base64 bytes from FilePart.data: {e}")
                else:
                    logger.warning(f"FilePart lacks URI, file.bytes, or inline data: {part_item}")
            else:
                logger.warning(f"Unsupported message part type: {type(part_item)}")
        return genai_types.Content(parts=parts, role=message.role or "user")


    def adk_content_to_message(
        self, content: genai_types.Content, conversation_id: str
    ) -> Message:
        parts: List[Part] = []
        if not content.parts:
            return Message(
                parts=[],
                role=content.role if content.role == 'user' else 'agent',
                metadata={'conversation_id': conversation_id, 'message_id': str(uuid.uuid4())},
            )
        for adk_part in content.parts: # Renamed part to adk_part
            if adk_part.text:
                try:
                    data = json.loads(adk_part.text)
                    # Heuristic: if data is a dict and has specific keys, treat as DataPart, else TextPart
                    if isinstance(data, dict) and not any(k.startswith('_') for k in data.keys()): # Simple heuristic
                        parts.append(DataPart(data=data))
                    else: # Could be simple string JSON like '"text"' or complex not meant for DataPart
                        parts.append(TextPart(text=adk_part.text))
                except json.JSONDecodeError:
                    parts.append(TextPart(text=adk_part.text))
            elif adk_part.inline_data and adk_part.inline_data.data:
                # Assuming inline_data.data is bytes, needs base64 encoding for FilePart.file.bytes or FilePart.data
                base64_encoded_data = base64.b64encode(adk_part.inline_data.data).decode('utf-8')
                parts.append(
                    FilePart(
                        # prefer using FileContent for consistency if this is meant to be like an uploaded file
                        file=FileContent(bytes=base64_encoded_data, mimeType=adk_part.inline_data.mime_type),
                        mimeType=adk_part.inline_data.mime_type
                    )
                )
            elif adk_part.file_data:
                parts.append(
                    FilePart(
                        file=FileContent(
                            uri=adk_part.file_data.file_uri,
                            mimeType=adk_part.file_data.mime_type,
                        ),
                        mimeType=adk_part.file_data.mime_type # Redundant but FilePart has it
                    )
                )
            elif adk_part.video_metadata: # Flatten to DataPart
                parts.append(DataPart(data=genai_types.VideoMetadata.to_dict(adk_part.video_metadata)))
            # 'thought' is not a standard ADK Part type. If it's custom, needs handling.
            # For 'executable_code', 'function_call', 'function_response', these are GenAI specific.
            elif adk_part.executable_code:
                parts.append(DataPart(data=genai_types.ExecutableCode.to_dict(adk_part.executable_code)))
            elif adk_part.function_call:
                parts.append(DataPart(data=genai_types.FunctionCall.to_dict(adk_part.function_call)))
            elif adk_part.function_response:
                parts.extend(
                    self._handle_function_response(adk_part, conversation_id)
                )
            else:
                logger.warning(f"ADK Part to Message: Unexpected content part type in ADK content: {type(adk_part)}")
        
        message_id = str(uuid.uuid4())
        # Attempt to find a last_message_id if possible, e.g., by looking at the conversation
        last_msg_id = None
        conv = self.get_conversation(conversation_id)
        if conv and conv.messages:
            last_msg_id = get_message_id(conv.messages[-1])

        return Message(
            role=content.role if content.role == 'user' else 'agent',
            parts=parts,
            metadata={'conversation_id': conversation_id, 'message_id': message_id, 'last_message_id': last_msg_id},
        )

    def _handle_function_response(
        self, adk_fn_response_part: genai_types.Part, conversation_id: str
    ) -> List[Part]:
        parts: List[Part] = []
        response_content = adk_fn_response_part.function_response
        if not response_content: return parts

        # Ensure response_content.response is a dict as expected by original code.
        # The genai_types.FunctionResponse has `response: dict[str, Any]`.
        data_to_process = response_content.response
        
        try:
            # Original code iterated over response['result']. Adapt if structure is different.
            # Assuming response_content.response is a dict and might contain a 'result' key.
            results_list = data_to_process.get('result', []) if isinstance(data_to_process, dict) else []
            if not isinstance(results_list, list): # If 'result' is not a list, wrap it or handle as single item
                results_list = [results_list] if results_list else []


            for p_item in results_list:
                if isinstance(p_item, str):
                    parts.append(TextPart(text=p_item))
                elif isinstance(p_item, dict):
                    if p_item.get('type') == 'file': # Check type safely
                        # Assuming p_item structure matches FilePart args
                        parts.append(FilePart(**p_item))
                    else:
                        parts.append(DataPart(data=p_item))
                # Original code had: elif isinstance(p, DataPart):
                # This is unlikely if p_item comes from JSON-like structure from genai_types.FunctionResponse
                # If it could indeed be a DataPart instance, it needs to be handled.
                else: # Fallback: convert to JSON string
                    try:
                        parts.append(TextPart(text=json.dumps(p_item)))
                    except TypeError:
                        logger.error(f"Could not JSON dump item in function response: {p_item}", exc_info=True)
                        parts.append(TextPart(text=f"[Unserializable data: {type(p_item)}]"))
        except Exception as e:
            logger.exception(f"Couldn't convert function response to message parts: {response_content.name}")
            # Fallback: include raw function response as DataPart if conversion fails
            try:
                 parts.append(DataPart(data=genai_types.FunctionResponse.to_dict(response_content)))
            except Exception as to_dict_e:
                 logger.error(f"Could not convert FunctionResponse to_dict: {to_dict_e}")
                 parts.append(DataPart(data={"error": "Could not serialize function response"}))

        return parts

    def unregister_agent(self, url: str):
        agent_to_remove = next((a for a in self._agents if (a.url or getattr(a, 'address', None)) == url), None) # getattr for safety
        if agent_to_remove:
            self._agents = [a for a in self._agents if (a.url or getattr(a, 'address', None)) != url]
            try:
                self._host_agent.unregister_agent_card(url) # Assumes unregister_agent_card takes URL
                logger.info(f"Agent {url} unregistered. Reinitializing host.")
                self._initialize_host()
            except Exception as e:
                logger.exception(f"Error during host_agent.unregister_agent_card for {url}")
        else:
            logger.warning(f"Attempted to unregister agent not found: {url}")


def get_message_id(m: Optional[Message]) -> Optional[str]:
    if m and m.metadata and 'message_id' in m.metadata:
        return m.metadata['message_id']
    return None


def get_last_message_id(m: Optional[Message]) -> Optional[str]:
    if m and m.metadata and 'last_message_id' in m.metadata:
        return m.metadata['last_message_id']
    return None


def get_conversation_id(
    item: Optional[
        Union[Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, Message] # type: ignore
    ],
) -> Optional[str]:
    if (
        item
        and hasattr(item, 'metadata')
        and item.metadata # Ensure metadata is not None
        and 'conversation_id' in item.metadata
    ):
        return item.metadata['conversation_id']
    # For Task, sessionId might be an alternative if metadata isn't consistently populated
    if isinstance(item, Task) and hasattr(item, 'sessionId') and item.sessionId:
        return item.sessionId
    return None


def task_still_open(task: Optional[Task]) -> bool:
    if not task or not task.status: # Ensure status exists
        return False
    return task.status.state in [
        TaskState.SUBMITTED,
        TaskState.WORKING,
        TaskState.INPUT_REQUIRED,
    ]