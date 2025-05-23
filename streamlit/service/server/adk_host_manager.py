import base64
import datetime
import json
import os
import uuid
import logging
from typing import Dict, List, Optional, Any, Union, Callable

# Assuming common.types and other local imports are correctly resolved
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
    # AgentRegistrationError should be defined in common.types
    # If not, define it here or import from its actual location.
)
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events.event import Event as ADKEvent
from google.adk.events.event_actions import EventActions as ADKEventActions
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types
import google.generativeai as genai # For genai.configure

from hosts.multiagent.host_agent import HostAgent
from hosts.multiagent.remote_agent_connection import (
    TaskCallbackArg,
)
from service.server.application_manager import ApplicationManager
from service.types import Conversation, Event
from utils.agent_card import get_agent_card # Assuming it's an async function
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# Define AgentRegistrationError if not available from common.types
# This is a placeholder; it should ideally be in common.types
class AgentRegistrationError(Exception):
    """Custom exception for agent registration failures."""
    pass

class ADKHostManager(ApplicationManager):
    """
    Manages ADK host operations, conversations, tasks, and agent interactions.
    This implementation uses in-memory storage for simplicity.
    """

    # --- Constants for roles and metadata keys ---
    ROLE_USER = "user"
    ROLE_AGENT = "agent"
    # ROLE_MODEL is used by genai_types.Content, typically as the string 'model'
    ROLE_GENAI_MODEL = "model" 

    METADATA_CONVERSATION_ID = "conversation_id"
    METADATA_MESSAGE_ID = "message_id"
    METADATA_LAST_MESSAGE_ID = "last_message_id"
    METADATA_INPUT_MESSAGE_METADATA = "input_message_metadata"
    METADATA_SESSION_ID = "session_id"
    METADATA_TASK_ID = "task_id"
    METADATA_A2A_PART_TYPE = "a2a_part_type" # For UI to distinguish tool calls/responses
    METADATA_TOOL_NAME = "tool_name"


    _conversations: Dict[str, Conversation]
    _messages: List[Message]
    _tasks: Dict[str, Task]
    _events: Dict[str, Event]
    _pending_message_ids: List[str]
    _agents: List[AgentCard]
    _task_map: Dict[str, str]  # message_id -> task_id (or user_message_id -> task_id)
    _artifact_chunks: Dict[str, Dict[int, Any]]
    _next_id_map: Dict[str, str] # Purpose still a bit unclear, retained for now.

    _host_agent: HostAgent
    _host_runner: Optional[Runner]
    _session_service: InMemorySessionService
    _artifact_service: InMemoryArtifactService
    _memory_service: InMemoryMemoryService


    def __init__(self, api_key: str = '', uses_vertex_ai: bool = False):
        self._conversations = {}
        self._messages = []
        self._tasks = {}
        self._events = {}
        self._pending_message_ids = []
        self._agents = []
        self._artifact_chunks = {}
        self._task_map = {}
        self._next_id_map = {}

        self._session_service = InMemorySessionService()
        self._artifact_service = InMemoryArtifactService()
        self._memory_service = InMemoryMemoryService()
        
        self._host_agent = HostAgent([], self.task_callback)
        
        self.user_id = 'test_user' 
        self.app_name = 'A2A_KTL_Host'

        logger.info(f"ADKHostManager initializing with provided: api_key {'set' if api_key else 'not set'}, uses_vertex_ai: {uses_vertex_ai}")
        
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY', '')
        env_uses_vertex_ai_str = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'FALSE').upper()
        env_uses_vertex_ai = env_uses_vertex_ai_str == 'TRUE'
        self.uses_vertex_ai = uses_vertex_ai or env_uses_vertex_ai
        
        logger.info(f"ADKHostManager effective config: api_key {'set' if self.api_key else 'not set'}, uses_vertex_ai: {self.uses_vertex_ai}")

        self._configure_genai_client()
        self._initialize_host_runner()

    def _configure_genai_client(self):
        """Configures the google-generativeai client."""
        try:
            if self.uses_vertex_ai:
                project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
                location = os.getenv('GOOGLE_CLOUD_LOCATION')
                if not project_id:
                    logger.warning("GOOGLE_CLOUD_PROJECT env var not set, might be needed for Vertex AI.")
                genai.configure(project=project_id, location=location)
                logger.info("google-generativeai configured for Vertex AI (Project: %s, Location: %s).", project_id, location)
            elif self.api_key:
                genai.configure(api_key=self.api_key)
                logger.info("google-generativeai configured with API Key.")
            else:
                logger.error("CRITICAL: Neither API Key nor Vertex AI configured. LLM ops will likely fail.")
        except Exception as e:
            logger.exception("Failed to configure google-generativeai client.")

    def _initialize_host_runner(self):
        """Initializes or re-initializes the ADK Host Runner."""
        try:
            agent = self._host_agent.create_agent()
            self._host_runner = Runner(
                app_name=self.app_name,
                agent=agent,
                artifact_service=self._artifact_service,
                session_service=self._session_service,
                memory_service=self._memory_service,
            )
            logger.info("ADK Host Runner initialized/reinitialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize ADK Host Runner.")
            self._host_runner = None

    def update_api_key(self, api_key: str):
        if api_key and api_key != self.api_key:
            logger.info("Updating API key.")
            self.api_key = api_key
            if not self.uses_vertex_ai:
                self._configure_genai_client()
                self._initialize_host_runner()
            else:
                logger.info("API key updated, but Vertex AI is active. No re-initialization by API key change.")
        elif not api_key:
            logger.warning("Attempted to update API key with an empty value.")

    def create_conversation(self) -> Conversation:
        session = self._session_service.create_session(app_name=self.app_name, user_id=self.user_id)
        conversation_id = session.id
        c = Conversation(conversation_id=conversation_id, is_active=True, messages=[])
        self._conversations[conversation_id] = c
        logger.info(f"Conversation created: {conversation_id}")
        return c

    def sanitize_message(self, message: Message) -> Message:
        if not message.metadata:
            message.metadata = {}
        if self.METADATA_MESSAGE_ID not in message.metadata:
            message.metadata[self.METADATA_MESSAGE_ID] = str(uuid.uuid4())
        
        conversation_id = message.metadata.get(self.METADATA_CONVERSATION_ID)
        if conversation_id:
            conversation = self.get_conversation(conversation_id)
            if conversation and conversation.messages:
                last_message_in_conv = conversation.messages[-1]
                last_message_id = get_message_id(last_message_in_conv)
                if last_message_id:
                    message.metadata[self.METADATA_LAST_MESSAGE_ID] = last_message_id
        return message

    async def process_message(self, message: Message) -> Optional[str]: # Return task_id if a new task is created
        """Processes an incoming message and generates a response via ADK Host Runner."""
        if not self._host_runner:
            logger.error("Host Runner is not initialized. Cannot process message.")
            error_response = Message(
                role=self.ROLE_AGENT,
                parts=[TextPart(text="System error: Host not initialized. Cannot process message.")],
                metadata={
                    self.METADATA_CONVERSATION_ID: message.metadata.get(self.METADATA_CONVERSATION_ID),
                    self.METADATA_MESSAGE_ID: str(uuid.uuid4()),
                    self.METADATA_LAST_MESSAGE_ID: get_message_id(message)
                }
            )
            self._add_message_to_conversation(error_response)
            return None

        message = self.sanitize_message(message)
        message_id = get_message_id(message)
        conversation_id = message.metadata.get(self.METADATA_CONVERSATION_ID)

        if not message_id or not conversation_id:
            logger.error(f"Message lacks essential metadata. MsgID: {message_id}, ConvID: {conversation_id}. Aborting.")
            return None

        self._add_message_to_conversation(message)
        if message_id not in self._pending_message_ids: # Avoid duplicates if called multiple times for same message
            self._pending_message_ids.append(message_id)

        current_task_for_processing: Optional[Task] = None
        active_task_id: Optional[str] = None
        
        # Try to get task_id from message metadata (set by conversation.py)
        explicit_task_id = message.metadata.get(self.METADATA_TASK_ID)
        if explicit_task_id:
            current_task_for_processing = self._tasks.get(explicit_task_id)
            if current_task_for_processing:
                active_task_id = explicit_task_id
                logger.info(f"Message {message_id} explicitly associated with existing task {active_task_id}.")
            else:
                logger.warning(f"Task ID {explicit_task_id} from message metadata not found. A new task might be created if applicable.")
        
        # If no explicit task_id, and it's a new agent execution, create a new Task
        if not current_task_for_processing:
            target_agent_url = message.metadata.get("target_agent_url")
            original_user_query = message.metadata.get("original_user_query", message.parts[0].text if message.parts and isinstance(message.parts[0], TextPart) else "N/A")

            if target_agent_url: # Indicates a new agent task is being initiated
                new_task_id = str(uuid.uuid4())
                initial_task_message_text = f"Task for '{original_user_query[:30]}...' to agent {target_agent_url} submitted."
                current_task_for_processing = Task(
                    id=new_task_id,
                    status=TaskStatus(
                        state=TaskState.SUBMITTED, 
                        message=Message(parts=[TextPart(text=initial_task_message_text)], role=self.ROLE_AGENT)
                    ),
                    metadata={
                        self.METADATA_CONVERSATION_ID: conversation_id,
                        "original_user_message_id": message_id, # The user message that triggered this task
                        "target_agent_url": target_agent_url,
                        "original_user_query": original_user_query
                    },
                    history=[], artifacts=[]
                )
                self.add_task(current_task_for_processing)
                self._task_map[message_id] = new_task_id # Link this user message to the new task
                active_task_id = new_task_id
                logger.info(f"New task {new_task_id} created for user message {message_id} targeting {target_agent_url}.")
        
        # If still no current_task (e.g. direct LLM query not for a specific agent), try to find by last_message_id
        if not current_task_for_processing:
            last_msg_id_for_task_check = get_last_message_id(message)
            task_to_resume_id = self._task_map.get(last_msg_id_for_task_check) if last_msg_id_for_task_check else None
            if task_to_resume_id:
                task_obj = self._tasks.get(task_to_resume_id)
                if task_obj and task_still_open(task_obj):
                    current_task_for_processing = task_obj
                    active_task_id = task_to_resume_id
                    logger.info(f"Resuming task {active_task_id} based on last_message_id for message {message_id}.")


        # This is the main try-block for ADK runner and response processing
        try:
            self.add_event(
                Event(
                    id=str(uuid.uuid4()), actor=self.ROLE_USER, content=message,
                    timestamp=datetime.datetime.now(datetime.UTC).timestamp()
                )
            )

            final_adk_event: Optional[ADKEvent] = None
            session = self._session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=conversation_id
            )

            if not session:
                logger.error(f"Session not found for conv_id: {conversation_id}. Cannot process.")
                if current_task_for_processing and task_still_open(current_task_for_processing):
                    current_task_for_processing.status.state = TaskState.FAILED
                    current_task_for_processing.status.message = Message(parts=[TextPart(text="System error: Session not found.")], role=self.ROLE_AGENT)
                    self.update_task(current_task_for_processing)
                return None # Indicate failure

            state_update_actions = {
                self.METADATA_INPUT_MESSAGE_METADATA: message.metadata,
                self.METADATA_SESSION_ID: conversation_id,
            }
            if active_task_id and current_task_for_processing and task_still_open(current_task_for_processing):
                state_update_actions[self.METADATA_TASK_ID] = active_task_id
                logger.info(f"Task {active_task_id} is open, adding to ADK state_delta.")
            
            self._session_service.append_event(
                session,
                ADKEvent(id=ADKEvent.new_id(), author='host_agent', invocation_id=ADKEvent.new_id(),
                         actions=ADKEventActions(state_delta=state_update_actions))
            )

            adk_input_content = self.adk_content_from_message(message)
            
            logger.info(f"Running ADK host for message {message_id} (Task: {active_task_id or 'N/A'}) in conv {conversation_id}...")
            async for adk_event_item in self._host_runner.run_async(
                user_id=self.user_id, session_id=conversation_id, new_message=adk_input_content
            ):
                logger.debug(
                    f"ADK Event: ID={adk_event_item.id}, Author={adk_event_item.author}, "
                    f"ContentParts={len(adk_event_item.content.parts) if adk_event_item.content else 0}"
                )
                event_content_message = self.adk_content_to_message(
                    adk_event_item.content, conversation_id
                )
                self.add_event(
                    Event(id=adk_event_item.id, actor=adk_event_item.author or self.ROLE_AGENT,
                          content=event_content_message, timestamp=adk_event_item.timestamp)
                )
                
                if current_task_for_processing and event_content_message.parts:
                    is_tool_event = any(
                        p.metadata and p.metadata.get(self.METADATA_A2A_PART_TYPE) in ["tool_call", "tool_response", "tool_output"]
                        for p in event_content_message.parts if isinstance(p, DataPart)
                    )
                    if is_tool_event: # Add tool-related intermediate messages to task history
                        self.insert_message_history(current_task_for_processing, event_content_message)
                        self.update_task(current_task_for_processing) # Save history update
                        logger.info(f"Tool-related ADK event msg {get_message_id(event_content_message)} added to history of task {current_task_for_processing.id}")
                
                final_adk_event = adk_event_item
            logger.info(f"ADK host run finished for message {message_id}.")

            if final_adk_event and final_adk_event.content:
                response_message = self.adk_content_to_message( # Converts role 'model' to 'agent'
                    final_adk_event.content, conversation_id
                )
                response_message.metadata = {
                    self.METADATA_CONVERSATION_ID: conversation_id,
                    self.METADATA_MESSAGE_ID: get_message_id(response_message) or str(uuid.uuid4()),
                    self.METADATA_LAST_MESSAGE_ID: message_id,
                }
                self._add_message_to_conversation(response_message)
                logger.info(f"Final response msg {get_message_id(response_message)} added to conversation.")

                if current_task_for_processing:
                    if not current_task_for_processing.status: # Should not happen if task was created properly
                        current_task_for_processing.status = TaskStatus(state=TaskState.UNKNOWN)
                    current_task_for_processing.status.message = response_message # Store final response as task result
                    current_task_for_processing.status.state = TaskState.COMPLETED
                    self.update_task(current_task_for_processing)
                    logger.info(f"Task {current_task_for_processing.id} set to COMPLETED with final message.")
                else:
                    logger.info(f"Final ADK response {get_message_id(response_message)} generated, but no active task was identified to mark as completed for this message context.")
            else:
                logger.warning(f"No final ADK event/content for message {message_id}. No explicit response by ADK run.")
                if current_task_for_processing and task_still_open(current_task_for_processing):
                    current_task_for_processing.status.state = TaskState.FAILED
                    fail_msg_parts = [TextPart(text="에이전트 실행 후 최종 응답을 받지 못했습니다.")]
                    current_task_for_processing.status.message = Message(role=self.ROLE_AGENT, parts=fail_msg_parts, metadata={
                        self.METADATA_CONVERSATION_ID: conversation_id,
                        self.METADATA_MESSAGE_ID: str(uuid.uuid4()),
                        self.METADATA_LAST_MESSAGE_ID: message_id
                    })
                    self.update_task(current_task_for_processing)
                    self._add_message_to_conversation(current_task_for_processing.status.message)
                    logger.info(f"Task {current_task_for_processing.id} marked FAILED (no final ADK response).")
        
        except Exception as e: 
            logger.exception(f"Unhandled error in process_message (ADK run or response handling) for message_id {message_id}")
            
            active_task_id_on_error = active_task_id 
            if active_task_id_on_error:
                task_to_fail = self._tasks.get(active_task_id_on_error)
                if task_to_fail and task_still_open(task_to_fail):
                    task_to_fail.status.state = TaskState.FAILED
                    error_msg_text = f"오류로 인해 작업 처리 중단: {str(e)[:100]}"
                    task_to_fail.status.message = Message(
                        role=self.ROLE_AGENT, parts=[TextPart(text=error_msg_text)],
                        metadata={
                            self.METADATA_CONVERSATION_ID: conversation_id,
                            self.METADATA_MESSAGE_ID: str(uuid.uuid4()),
                            self.METADATA_LAST_MESSAGE_ID: message_id
                        })
                    self.update_task(task_to_fail)
                    self._add_message_to_conversation(task_to_fail.status.message)
                    logger.info(f"Task {task_to_fail.id} marked FAILED due to exception: {e}")

            general_error_response = Message(
                role=self.ROLE_AGENT,
                parts=[TextPart(text=f"메시지 처리 중 시스템 오류 발생: {str(e)[:100]}")],
                metadata={
                    self.METADATA_CONVERSATION_ID: conversation_id,
                    self.METADATA_MESSAGE_ID: str(uuid.uuid4()),
                    self.METADATA_LAST_MESSAGE_ID: message_id
                }
            )
            self._add_message_to_conversation(general_error_response)
        finally:
            if message_id in self._pending_message_ids:
                self._pending_message_ids.remove(message_id)
        
        return active_task_id


    def _add_message_to_conversation(self, message: Message):
        if not message: return
        self._messages.append(message)
        conversation_id = message.metadata.get(self.METADATA_CONVERSATION_ID)
        if conversation_id:
            conversation = self.get_conversation(conversation_id)
            if conversation:
                if conversation.messages is None: conversation.messages = []
                conversation.messages.append(message)
            else:
                logger.warning(f"Attempted to add message to non-existent conversation: {conversation_id}")

    def add_task(self, task: Task):
        self._tasks[task.id] = task
        logger.debug(f"Task added: {task.id} with state {task.status.state if task.status else 'N/A'}")

    def update_task(self, task: Task):
        if task.id in self._tasks:
            self._tasks[task.id] = task
            logger.debug(f"Task updated: {task.id} to state {task.status.state if task.status else 'N/A'}")
        else:
            logger.warning(f"Attempted to update non-existent task: {task.id}")

    def task_callback(self, task_arg: TaskCallbackArg, agent_card: AgentCard) -> Optional[Task]:
        self.emit_event(task_arg, agent_card)
        current_task: Optional[Task] = None
        task_id = getattr(task_arg, 'id', None)

        if not task_id:
            logger.error(f"Task argument in callback lacks an ID. Type: {type(task_arg)}")
            return None

        current_task = self.add_or_get_task(task_arg)

        if isinstance(task_arg, TaskStatusUpdateEvent):
            logger.info(f"Task ({current_task.id}) status update via callback: {task_arg.status.state}")
            current_task.status = task_arg.status 
            if task_arg.status.message: 
                self.attach_message_to_task(task_arg.status.message, current_task.id)
                self.insert_message_history(current_task, task_arg.status.message)
                self.insert_id_trace(task_arg.status.message)

        elif isinstance(task_arg, TaskArtifactUpdateEvent):
            logger.info(f"Task ({current_task.id}) artifact update via callback: {task_arg.artifact.name}")
            self.process_artifact_event(current_task, task_arg)

        elif isinstance(task_arg, Task):
            logger.info(f"Task ({current_task.id}) object directly updated via callback. New state: {task_arg.status.state if task_arg.status else 'N/A'}")
            current_task.status = task_arg.status if task_arg.status else current_task.status
            current_task.artifacts = task_arg.artifacts if task_arg.artifacts is not None else current_task.artifacts
            current_task.history = task_arg.history if task_arg.history is not None else current_task.history
            current_task.metadata = task_arg.metadata if task_arg.metadata is not None else current_task.metadata
            if current_task.status and current_task.status.message:
                self.attach_message_to_task(current_task.status.message, current_task.id)
                self.insert_message_history(current_task, current_task.status.message)
                self.insert_id_trace(current_task.status.message)
        else:
            logger.warning(f"Unknown type in task_callback: {type(task_arg)}")
            return None

        if current_task:
            self.update_task(current_task)
        return current_task

    def emit_event(self, task_arg: TaskCallbackArg, agent_card: AgentCard):
        content: Optional[Message] = None
        conversation_id = get_conversation_id_from_item(task_arg)
        
        if not conversation_id and isinstance(task_arg, Task) and task_arg.sessionId:
            conversation_id = task_arg.sessionId
        
        metadata = {self.METADATA_CONVERSATION_ID: conversation_id} if conversation_id else {}
        actor_name = agent_card.name or "UnknownAgent"

        if isinstance(task_arg, TaskStatusUpdateEvent):
            if task_arg.status.message: 
                content = task_arg.status.message
                if conversation_id and (not content.metadata or self.METADATA_CONVERSATION_ID not in content.metadata):
                    if not content.metadata: content.metadata = {}
                    content.metadata[self.METADATA_CONVERSATION_ID] = conversation_id
            else:
                content = Message(
                    parts=[TextPart(text=f"Task status changed to: {task_arg.status.state}")],
                    role=self.ROLE_AGENT, metadata=metadata
                )
        elif isinstance(task_arg, TaskArtifactUpdateEvent):
            content = Message(
                parts=list(task_arg.artifact.parts) if task_arg.artifact.parts else [TextPart(text=f"Artifact '{task_arg.artifact.name}' updated.")],
                role=self.ROLE_AGENT, metadata=metadata
            )
        elif isinstance(task_arg, Task):
            if task_arg.status and task_arg.status.message:
                content = task_arg.status.message
                if conversation_id and (not content.metadata or self.METADATA_CONVERSATION_ID not in content.metadata):
                    if not content.metadata: content.metadata = {}
                    content.metadata[self.METADATA_CONVERSATION_ID] = conversation_id
            elif task_arg.artifacts:
                parts_list: List[Part] = []
                for art_item in task_arg.artifacts:
                    if art_item.parts: parts_list.extend(art_item.parts)
                if not parts_list: parts_list.append(TextPart(text="Task object updated with artifacts."))
                content = Message(parts=parts_list, role=self.ROLE_AGENT, metadata=metadata)
            else:
                content = Message(
                    parts=[TextPart(text=f"Task '{task_arg.id}' updated. Status: {task_arg.status.state if task_arg.status else 'Unknown'}")],
                    role=self.ROLE_AGENT, metadata=metadata
                )
        else:
            logger.warning(f"emit_event: Could not determine content for task_arg type {type(task_arg)}")
            content = Message(parts=[TextPart(text="System event occurred.")], role=self.ROLE_AGENT, metadata=metadata)
        
        if content:
            if not get_message_id(content):
                if not content.metadata: content.metadata = {}
                content.metadata[self.METADATA_MESSAGE_ID] = str(uuid.uuid4())

            self.add_event(
                Event(id=str(uuid.uuid4()), actor=actor_name, content=content,
                      timestamp=datetime.datetime.now(datetime.UTC).timestamp())
            )
            self._add_message_to_conversation(content)

    def attach_message_to_task(self, message: Optional[Message], task_id: str):
        if message:
            msg_id = get_message_id(message)
            if msg_id:
                self._task_map[msg_id] = task_id
                logger.debug(f"Message {msg_id} associated with task {task_id}")

    def insert_id_trace(self, message: Optional[Message]):
        if not message: return
        current_message_id = get_message_id(message)
        prev_message_id = get_last_message_id(message)
        if current_message_id and prev_message_id:
            self._next_id_map[prev_message_id] = current_message_id
            logger.debug(f"ID trace: Message {prev_message_id} -> {current_message_id}")

    def insert_message_history(self, task: Task, message: Optional[Message]):
        if not message: return
        if task.history is None: task.history = []
        message_id_to_add = get_message_id(message)
        if not message_id_to_add:
            logger.debug(f"Message for task {task.id} lacks ID, cannot add to history.")
            return
        if message_id_to_add not in {get_message_id(hist_msg) for hist_msg in task.history if hist_msg}:
            task.history.append(message)
            logger.debug(f"Message {message_id_to_add} added to history of task {task.id}")
        else:
            logger.debug(f"Message {message_id_to_add} already in history for task {task.id}")

    def add_or_get_task(self, task_arg: TaskCallbackArg) -> Task:
        task_id = getattr(task_arg, 'id', None)
        if not task_id:
            logger.error(f"Task argument {type(task_arg)} missing ID.")
            raise ValueError(f"Task argument {type(task_arg)} must have an ID.")

        current_task = self._tasks.get(task_id)
        if not current_task:
            logger.info(f"Task {task_id} not found. Creating new from {type(task_arg)}.")
            initial_status_state = TaskState.SUBMITTED
            task_metadata = getattr(task_arg, 'metadata', {}) or {}
            conversation_id = task_metadata.get(self.METADATA_CONVERSATION_ID)
            
            if isinstance(task_arg, Task) and task_arg.status:
                initial_status_state = task_arg.status.state
                if not conversation_id and task_arg.sessionId:
                    conversation_id = task_arg.sessionId
                    task_metadata[self.METADATA_CONVERSATION_ID] = conversation_id
            elif isinstance(task_arg, TaskStatusUpdateEvent) and task_arg.status:
                initial_status_state = task_arg.status.state
            
            current_task = Task(
                id=task_id, status=TaskStatus(state=initial_status_state),
                metadata=task_metadata, artifacts=[], history=[], sessionId=conversation_id
            )
            self.add_task(current_task)
        return current_task

    def process_artifact_event(self, current_task: Task, task_update_event: TaskArtifactUpdateEvent):
        artifact = task_update_event.artifact
        artifact_stream_id = task_update_event.id 
        chunk_index = artifact.index if artifact.index is not None else 0

        if current_task.artifacts is None: current_task.artifacts = []

        if not artifact.append:
            if artifact.lastChunk is None or artifact.lastChunk:
                current_task.artifacts.append(artifact)
                logger.debug(f"Full artifact '{artifact.name}' (stream {artifact_stream_id}) added to task {current_task.id}")
            else:
                self._artifact_chunks[artifact_stream_id] = {chunk_index: artifact}
                logger.debug(f"Initial artifact chunk {chunk_index} for '{artifact.name}' (stream {artifact_stream_id}) stored for task {current_task.id}")
        else:
            if artifact_stream_id not in self._artifact_chunks or \
               chunk_index not in self._artifact_chunks[artifact_stream_id]:
                logger.error(
                    f"Cannot append artifact chunk {chunk_index} for '{artifact.name}' (stream {artifact_stream_id}): "
                    f"Initial chunk/stream not found for task {current_task.id}."
                )
                return
            
            accumulating_artifact = self._artifact_chunks[artifact_stream_id].get(chunk_index)
            if not accumulating_artifact:
                 logger.error(f"Accumulating artifact for index {chunk_index} (stream {artifact_stream_id}) not found.")
                 return

            if accumulating_artifact.parts is None: accumulating_artifact.parts = []
            if artifact.parts: accumulating_artifact.parts.extend(artifact.parts)
            
            logger.debug(f"Appended parts to chunk {chunk_index} of '{artifact.name}' (stream {artifact_stream_id}) for task {current_task.id}")

            if artifact.lastChunk:
                current_task.artifacts.append(accumulating_artifact)
                del self._artifact_chunks[artifact_stream_id][chunk_index]
                if not self._artifact_chunks[artifact_stream_id]:
                    del self._artifact_chunks[artifact_stream_id]
                logger.debug(
                    f"Final appended artifact '{accumulating_artifact.name}' (chunk {chunk_index}, stream {artifact_stream_id}) "
                    f"processed for task {current_task.id}"
                )

    def add_event(self, event: Event):
        self._events[event.id] = event

    def get_conversation(self, conversation_id: Optional[str]) -> Optional[Conversation]:
        if not conversation_id: return None
        return self._conversations.get(conversation_id)

    def get_pending_messages(self) -> List[tuple[str, str]]:
        results = []
        for message_id in self._pending_message_ids:
            task_id = self._task_map.get(message_id)
            status_summary = 'Initializing...'
            if task_id:
                task = self._tasks.get(task_id)
                if not task: status_summary = 'Task not found'
                elif task.status:
                    state_str = str(task.status.state).replace('TaskState.', '').capitalize()
                    status_summary = f'{state_str}...'
                    if task.status.state == TaskState.WORKING and task.history:
                        last_hist_msg = task.history[-1]
                        if last_hist_msg and last_hist_msg.parts:
                            first_part = last_hist_msg.parts[0]
                            if isinstance(first_part, TextPart) and first_part.text:
                                status_summary = f'Working: {first_part.text[:30]}...'
                            # else: # This was causing an error if first_part was not TextPart
                                # status_summary = f'Working ({first_part.type.capitalize()} received)...' 
                    elif task.status.state == TaskState.SUBMITTED and len(task.history or []) <= 1: # Allow 0 or 1 history item for submitted
                        status_summary = 'Submitted...'
                else: status_summary = 'Pending (no status)...'
            results.append((message_id, status_summary))
        return results

    async def register_agent(self, url: str):
        logger.info(f"Attempting to register agent from URL: {url}")
        try:
            agent_card: Optional[AgentCard] = await get_agent_card(url)
        except Exception as e:
            logger.exception(f"Error fetching agent card from URL {url}")
            raise AgentRegistrationError(f"Failed to fetch agent card from {url}: {e}") from e

        if not agent_card:
            logger.error(f"No agent data from get_agent_card for URL: {url}")
            raise AgentRegistrationError(f"Agent card data missing/invalid from {url}.")

        if not agent_card.url: agent_card.url = url
        if any(ag.url == agent_card.url for ag in self._agents):
            logger.info(f"Agent with URL {agent_card.url} already registered. Skipping.")
            return

        self._agents.append(agent_card)
        if hasattr(self._host_agent, 'register_agent_card') and \
           isinstance(getattr(self._host_agent, 'register_agent_card', None), Callable):
            try:
                self._host_agent.register_agent_card(agent_card)
                logger.info(f"Agent '{agent_card.name}' from '{url}' passed to HostAgent. Reinitializing Runner.")
                self._initialize_host_runner()
            except Exception as e:
                logger.exception(f"Error during self._host_agent.register_agent_card for {agent_card.name}")
                self._agents.pop()
                raise AgentRegistrationError(f"Failed to register {agent_card.name} with internal HostAgent: {e}") from e
        else:
            logger.warning("HostAgent lacks 'register_agent_card' method. Registration may be incomplete.")

    def unregister_agent(self, url: str):
        agent_to_remove = next((agent for agent in self._agents if agent.url == url), None)
        if agent_to_remove:
            self._agents.remove(agent_to_remove)
            logger.info(f"Agent {url} removed from ADKHostManager list.")
            if hasattr(self._host_agent, 'unregister_agent_card') and \
               isinstance(getattr(self._host_agent, 'unregister_agent_card', None), Callable):
                try:
                    self._host_agent.unregister_agent_card(agent_to_remove)
                    logger.info(f"Agent {url} unregistered from HostAgent. Reinitializing Runner.")
                    self._initialize_host_runner()
                except Exception as e:
                    logger.exception(f"Error during self._host_agent.unregister_agent_card for {url}")
            else:
                logger.warning("HostAgent lacks 'unregister_agent_card' method. Unregistration may be incomplete.")
        else:
            logger.warning(f"Attempted to unregister agent not found: {url}")

    @property
    def agents(self) -> List[AgentCard]: return list(self._agents)
    @property
    def conversations(self) -> List[Conversation]: return list(self._conversations.values())
    @property
    def tasks(self) -> List[Task]: return list(self._tasks.values())
    @property
    def events(self) -> List[Event]: return sorted(list(self._events.values()), key=lambda e: e.timestamp)

    def adk_content_from_message(self, message: Message) -> genai_types.Content:
        genai_parts: list[genai_types.Part] = []
        if not message.parts: logger.warning(f"Message {get_message_id(message)} has no parts.")
        for app_part in message.parts:
            try:
                if isinstance(app_part, TextPart):
                    genai_parts.append(genai_types.Part.from_text(text=app_part.text or ""))
                elif isinstance(app_part, DataPart):
                    json_string = json.dumps(app_part.data)
                    genai_parts.append(genai_types.Part.from_text(text=json_string))
                elif isinstance(app_part, FilePart):
                    mime_type = app_part.mimeType or "application/octet-stream"
                    if app_part.uri:
                        genai_parts.append(genai_types.Part.from_uri(file_uri=app_part.uri, mime_type=mime_type))
                    elif app_part.file and app_part.file.bytes:
                        decoded_bytes = base64.b64decode(app_part.file.bytes)
                        genai_parts.append(genai_types.Part.from_data(data=decoded_bytes, mime_type=mime_type))
                    elif app_part.data:
                        decoded_bytes = base64.b64decode(app_part.data)
                        genai_parts.append(genai_types.Part.from_data(data=decoded_bytes, mime_type=mime_type))
                    else: logger.warning(f"FilePart in msg {get_message_id(message)} lacks URI/data: {app_part.name}")
                else: logger.warning(f"Unsupported Part type {type(app_part)} in msg {get_message_id(message)}. Skipping.")
            except Exception as e: logger.exception(f"Error converting Part {type(app_part)} to genai_types.Part: {e}")
        
        role = message.role
        # Corrected: Use string literal 'model' for genai_types.Content role
        if role not in (self.ROLE_USER, self.ROLE_GENAI_MODEL): 
            # If our internal 'agent' role is passed, convert it to 'user' for genai.Content,
            # as 'agent' is not a standard input role for genai.Content.
            # Or, if it's a response from our agent that we are feeding back, it might be 'model'.
            # For now, defaulting non-model, non-user roles to 'user'.
            logger.debug(f"Message role '{role}' adjusted to '{self.ROLE_USER}' for genai.Content input.")
            role = self.ROLE_USER 
        return genai_types.Content(parts=genai_parts, role=role)

    def _safe_model_dump_or_dict(self, obj: Any, obj_name: str) -> Dict[str, Any]:
        if obj is None:
            logger.warning(f"{obj_name} is None, cannot convert to dict.")
            return {}
        try:
            if hasattr(obj, 'model_dump'): return obj.model_dump()
            elif hasattr(obj, 'dict'): return obj.dict()
            return dict(obj)
        except Exception as e:
            logger.error(f"Failed to convert {obj_name} (type: {type(obj)}) to dict: {e}", exc_info=True)
            return {"error": f"Could not serialize {obj_name}", "type": str(type(obj))}

    def adk_content_to_message(self, content: genai_types.Content, conversation_id: str) -> Message:
        app_parts: List[Part] = []
        # Our Message model expects 'user' or 'agent'. GenAI 'model' role maps to our 'agent'.
        message_role = self.ROLE_AGENT 
        if content.role == self.ROLE_USER:
             message_role = self.ROLE_USER
        elif content.role == self.ROLE_GENAI_MODEL: # Check against the string 'model'
            message_role = self.ROLE_AGENT 
        # If content.role is something else (e.g. function), default to ROLE_AGENT
        
        if not content.parts: logger.debug(f"ADK Content for conv {conversation_id} has no parts.")
        else:
            for adk_part in content.parts:
                try:
                    part_metadata = {} 

                    if adk_part.text is not None:
                        try:
                            data = json.loads(adk_part.text)
                            if isinstance(data, dict) and not any(k.startswith('_') for k in data.keys()):
                                app_parts.append(DataPart(data=data, name="json_data_from_text"))
                            else: app_parts.append(TextPart(text=adk_part.text))
                        except json.JSONDecodeError: app_parts.append(TextPart(text=adk_part.text))
                    
                    elif adk_part.inline_data and adk_part.inline_data.data is not None:
                        base64_data = base64.b64encode(adk_part.inline_data.data).decode('utf-8')
                        app_parts.append(FilePart(
                            name=f"inline_file_{str(uuid.uuid4())[:4]}",
                            mimeType=adk_part.inline_data.mime_type or "application/octet-stream",
                            file=FileContent(bytes=base64_data, mimeType=adk_part.inline_data.mime_type)
                        ))
                    elif adk_part.file_data and adk_part.file_data.file_uri is not None:
                        app_parts.append(FilePart(
                            name=f"uri_file_{os.path.basename(adk_part.file_data.file_uri)[:10]}",
                            mimeType=adk_part.file_data.mime_type or "application/octet-stream",
                            uri=adk_part.file_data.file_uri,
                            file=FileContent(uri=adk_part.file_data.file_uri, mimeType=adk_part.file_data.mime_type)
                        ))
                    elif adk_part.function_call:
                        fc_data = self._safe_model_dump_or_dict(adk_part.function_call, "adk_part.function_call")
                        part_metadata[self.METADATA_A2A_PART_TYPE] = "tool_call"
                        part_metadata[self.METADATA_TOOL_NAME] = adk_part.function_call.name
                        app_parts.append(DataPart(name=f"tool_call_{adk_part.function_call.name}", data=fc_data, metadata=part_metadata))
                    
                    elif adk_part.function_response:
                        fn_response_parts = self._handle_function_response(adk_part.function_response, conversation_id)
                        for p in fn_response_parts:
                            if not p.metadata: p.metadata = {}
                            p.metadata[self.METADATA_A2A_PART_TYPE] = "tool_response"
                            if isinstance(p, DataPart) and not p.name: 
                                p.name = f"response_data_for_{adk_part.function_response.name}"
                        app_parts.extend(fn_response_parts)
                    
                    elif hasattr(adk_part, 'tool_code_outputs') and getattr(adk_part, 'tool_code_outputs', None):
                        if adk_part.tool_code_outputs.outputs: # Ensure outputs list is not empty
                            for tool_output in adk_part.tool_code_outputs.outputs:
                                tool_output_data = self._safe_model_dump_or_dict(tool_output, "tool_output")
                                output_name_detail = tool_output.get('name', 'unknown_tool') if isinstance(tool_output, dict) else 'unknown_tool'
                                part_metadata_tco = {
                                    self.METADATA_A2A_PART_TYPE: "tool_output",
                                    self.METADATA_TOOL_NAME: output_name_detail
                                }
                                app_parts.append(DataPart(
                                    name=f"tool_output_{output_name_detail}",
                                    data=tool_output_data, metadata=part_metadata_tco
                                ))
                    elif adk_part.executable_code:
                         app_parts.append(DataPart(name="executable_code", data=self._safe_model_dump_or_dict(adk_part.executable_code, "adk_part.executable_code")))
                    elif adk_part.video_metadata:
                         app_parts.append(DataPart(name="video_metadata", data=self._safe_model_dump_or_dict(adk_part.video_metadata, "adk_part.video_metadata")))
                    else:
                        logger.warning(f"ADK Part to Msg: Unhandled/empty ADK Part in conv {conversation_id}. Part: {adk_part}")
                except Exception as e:
                    logger.exception(f"Error converting ADK Part to app Part: {e}. ADK Part: {adk_part}")

        msg_id = str(uuid.uuid4())
        last_msg_id = None
        conv = self.get_conversation(conversation_id)
        if conv and conv.messages:
            for msg_in_hist in reversed(conv.messages):
                if msg_in_hist.role in [self.ROLE_USER, self.ROLE_AGENT]:
                    last_msg_id = get_message_id(msg_in_hist)
                    break
        
        return Message(
            role=message_role, parts=app_parts,
            metadata={
                self.METADATA_CONVERSATION_ID: conversation_id,
                self.METADATA_MESSAGE_ID: msg_id,
                self.METADATA_LAST_MESSAGE_ID: last_msg_id,
            }
        )

    def _handle_function_response(self, fn_response_content: genai_types.FunctionResponse, conversation_id: str) -> List[Part]:
        app_parts: List[Part] = []
        if not fn_response_content: return app_parts
        response_data_dict = fn_response_content.response 
        tool_name = fn_response_content.name 
        logger.debug(f"Handling function response for tool '{tool_name}'. Raw response data: {response_data_dict}")

        items_to_process = []
        if isinstance(response_data_dict, dict) and 'result' in response_data_dict:
            result_value = response_data_dict['result']
            items_to_process = result_value if isinstance(result_value, list) else ([result_value] if result_value is not None else [])
        elif response_data_dict is not None:
            items_to_process = [response_data_dict]
        
        if not items_to_process:
            logger.warning(f"Function response for '{tool_name}' has no processable items.")
            app_parts.append(TextPart(text=f"Tool '{tool_name}' executed (no specific items in result)."))

        for item_idx, item in enumerate(items_to_process):
            part_name = f"tool_response_{tool_name}_{item_idx}"
            part_metadata = {
                self.METADATA_A2A_PART_TYPE: "tool_response_item", 
                self.METADATA_TOOL_NAME: tool_name
            }
            try:
                if isinstance(item, str):
                    app_parts.append(TextPart(text=item, metadata=part_metadata))
                elif isinstance(item, dict):
                    if item.get('type') == 'file':
                        file_args = {k: v for k, v in item.items() if k != 'type'} 
                        file_args['name'] = item.get('name', f"file_from_{tool_name}_{item_idx}")
                        app_parts.append(FilePart(**file_args, metadata=part_metadata))
                    else:
                        app_parts.append(DataPart(data=item, name=part_name, metadata=part_metadata))
                else:
                    app_parts.append(TextPart(text=json.dumps(item), metadata=part_metadata))
            except Exception as e:
                logger.error(f"Could not convert item from tool '{tool_name}' response to Part: {item}", exc_info=True)
                app_parts.append(TextPart(text=f"[Unserializable data from '{tool_name}': {type(item)}]", metadata=part_metadata))
        
        if not app_parts: 
             logger.debug(f"No specific parts from items for tool '{tool_name}'. Adding raw response as DataPart.")
             app_parts.append(DataPart(
                name=f"raw_tool_response_{tool_name}",
                data=self._safe_model_dump_or_dict(fn_response_content, f"fn_response_content.{tool_name}"),
                metadata={self.METADATA_A2A_PART_TYPE: "tool_response_raw", self.METADATA_TOOL_NAME: tool_name}
            ))
        return app_parts

# --- Utility functions ---
def get_message_id(m: Optional[Message]) -> Optional[str]:
    if m and m.metadata and ADKHostManager.METADATA_MESSAGE_ID in m.metadata:
        return m.metadata[ADKHostManager.METADATA_MESSAGE_ID]
    return None

def get_last_message_id(m: Optional[Message]) -> Optional[str]:
    if m and m.metadata and ADKHostManager.METADATA_LAST_MESSAGE_ID in m.metadata:
        return m.metadata[ADKHostManager.METADATA_LAST_MESSAGE_ID]
    return None

def get_conversation_id_from_item(
    item: Optional[Union[Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, Message]],
) -> Optional[str]:
    if item and hasattr(item, 'metadata') and item.metadata:
        return item.metadata.get(ADKHostManager.METADATA_CONVERSATION_ID)
    if isinstance(item, Task) and hasattr(item, 'sessionId') and item.sessionId:
        return item.sessionId
    return None

def task_still_open(task: Optional[Task]) -> bool:
    if not task or not task.status: return False
    return task.status.state in [TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED]
