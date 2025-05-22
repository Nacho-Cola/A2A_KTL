# streamlit/state/state.py

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

# ── Conversation / Message / Task / Event 정의 ──

@dataclass
class StateConversation:
    conversation_id: str = ''
    conversation_name: str = ''
    is_active: bool = True
    message_ids: list[str] = field(default_factory=list)

@dataclass
class StateMessage:
    message_id: str = ''
    role: str = ''
    # (content, media_type) 쌍 리스트
    content: list[tuple[str | dict[str, Any], str]] = field(default_factory=list)

@dataclass
class StateTask:
    task_id: str = ''
    session_id: str | None = None
    state: str | None = None
    message: StateMessage = field(default_factory=StateMessage)
    artifacts: list[list[tuple[str | dict[str, Any], str]]] = field(default_factory=list)

@dataclass
class SessionTask:
    session_id: str = ''
    task: StateTask = field(default_factory=StateTask)

@dataclass
class StateEvent:
    conversation_id: str = ''
    actor: str = ''
    role: str = ''
    id: str = ''
    content: list[tuple[str | dict[str, Any], str]] = field(default_factory=list)


# ── AppState (모든 필드에 기본값 지정) ──

@dataclass
class AppState:
    """Streamlit 전용 전역 상태"""

    # UI 설정
    sidenav_open: bool = False
    theme_mode: Literal['system', 'light', 'dark'] = 'system'

    # 대화·메시지
    current_conversation_id: str = ''
    conversations: list[StateConversation] = field(default_factory=list)
    messages: list[StateMessage] = field(default_factory=list)

    # 작업(Task) 관리
    task_list: list[SessionTask] = field(default_factory=list)
    background_tasks: dict[str, str] = field(default_factory=dict)
    message_aliases: dict[str, str] = field(default_factory=dict)

    # form 관리
    completed_forms: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    form_responses: dict[str, str] = field(default_factory=dict)

    # 기타
    polling_interval: int = 1

    # API 키 관리
    api_key: str = ''
    uses_vertex_ai: bool = False
    api_key_dialog_open: bool = False


# ── SettingsState ──

@dataclass
class SettingsState:
    """Streamlit 전용 설정 상태"""

    output_mime_types: list[str] = field(default_factory=lambda: [
        'image/*',
        'text/plain',
    ])
