import dataclasses # dataclasses.field 등을 사용하기 위해 명시적 임포트 (선택 사항)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Union, Optional # Dict, List, Tuple, Union 추가 (Python 3.9+에서는 내장 타입 사용 가능)

# ── 대화 / 메시지 / 태스크 / 이벤트 상태 정의 ──

@dataclass
class StateConversation:
    """대화 하나의 상태를 나타냅니다."""
    conversation_id: str = ''
    conversation_name: str = ''
    is_active: bool = True # 현재 활성화된 대화인지 여부
    message_ids: List[str] = field(default_factory=list) # 이 대화에 속한 메시지 ID 목록

@dataclass
class StateMessage:
    """메시지 하나의 상태를 나타냅니다."""
    message_id: str = ''
    role: str = '' # 메시지 발신자 역할 (예: 'user', 'agent')
    # content 필드는 (콘텐츠 데이터, 미디어 타입) 형태의 튜플 리스트입니다.
    # 콘텐츠 데이터는 문자열 또는 딕셔너리 형태를 가질 수 있습니다.
    content: List[Tuple[Union[str, Dict[str, Any]], str]] = field(default_factory=list)

@dataclass
class StateTask:
    """태스크 하나의 상태를 나타냅니다."""
    task_id: str = ''
    session_id: Optional[str] = None # 태스크가 속한 세션(대화) ID (Optional 추가)
    state: Optional[str] = None # 태스크의 현재 상태 (예: 'SUBMITTED', 'WORKING', 'COMPLETED') (Optional 추가)
    message: StateMessage = field(default_factory=StateMessage) # 태스크와 연관된 (주로 시작) 메시지
    # artifacts: 태스크 결과물. 각 아티팩트는 여러 콘텐츠 부분으로 구성될 수 있고, 태스크는 여러 아티팩트를 가질 수 있습니다.
    # 예: [[(이미지_데이터, 'image/png'), (텍스트_설명, 'text/plain')], [(다른_문서_내용, 'text/markdown')]]
    artifacts: List[List[Tuple[Union[str, Dict[str, Any]], str]]] = field(default_factory=list)

@dataclass
class SessionTask:
    """세션(대화) ID와 연결된 태스크 정보를 나타냅니다."""
    session_id: str = '' # 이 태스크가 속한 세션(대화) ID
    task: StateTask = field(default_factory=StateTask) # 실제 태스크 상태 객체

@dataclass
class StateEvent:
    """이벤트 하나의 상태를 나타냅니다."""
    conversation_id: str = '' # 이벤트가 발생한 대화 ID
    actor: str = '' # 이벤트를 발생시킨 주체
    role: str = '' # 이벤트 내용과 관련된 역할
    id: str = '' # 이벤트 고유 ID
    # content 필드는 StateMessage와 동일한 구조를 가집니다.
    content: List[Tuple[Union[str, Dict[str, Any]], str]] = field(default_factory=list)


# ── AppState (Streamlit 애플리케이션 전체 상태) ──

@dataclass
class AppState:
    """Streamlit 애플리케이션의 전역 상태를 나타냅니다."""

    # UI 레이아웃 및 테마 설정
    sidenav_open: bool = False # 사이드 네비게이션 메뉴 열림 상태
    theme_mode: Literal['system', 'light', 'dark'] = 'system' # UI 테마 설정

    # 대화 및 메시지 관련 상태
    current_conversation_id: str = '' # 현재 사용자가 보고 있는 대화의 ID
    conversations: List[StateConversation] = field(default_factory=list) # 전체 대화 목록
    messages: List[StateMessage] = field(default_factory=list) # 현재 대화의 메시지 목록

    # 작업(Task) 관리 상태
    task_list: List[SessionTask] = field(default_factory=list) # 전체 태스크 목록 (세션 ID와 함께)
    background_tasks: Dict[str, str] = field(default_factory=dict) # 백그라운드 실행 중인 태스크 상태 (예: {message_id: status_string})
    message_aliases: Dict[str, str] = field(default_factory=dict) # 메시지 ID 별칭 (사용 용도에 따라 정의)

    # Form 입력 관련 상태
    completed_forms: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=dict) # 완료된 Form 데이터 (form_id: form_data)
    form_responses: Dict[str, str] = field(default_factory=dict) # Form 응답 (사용 용도에 따라 정의)

    # 기타 애플리케이션 설정
    polling_interval: int = 1 # 데이터 폴링 간격 (초)

    # API 키 및 인증 관련 상태
    api_key: str = '' # 사용자의 API 키
    uses_vertex_ai: bool = False # Vertex AI 사용 여부
    api_key_dialog_open: bool = False # API 키 입력 대화 상자 열림 상태


# ── SettingsState (애플리케이션 설정 특화 상태) ──

@dataclass
class SettingsState:
    """Streamlit 애플리케이션의 설정을 위한 상태입니다."""

    # 출력 가능한 MIME 타입 목록 기본값 설정
    output_mime_types: List[str] = field(default_factory=lambda: [
        'image/*', # 모든 이미지 타입 허용 예시
        'text/plain',
        'application/json', # JSON 타입 추가 예시
        'form', # Form 타입 추가 예시
    ])