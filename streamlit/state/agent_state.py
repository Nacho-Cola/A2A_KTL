import streamlit as st
from typing import List, Callable # 타입 힌팅을 위해 추가

# 상태 키들을 상수로 정의하여 오타 방지 및 일관성 유지 (선택 사항)
AGENT_DIALOG_OPEN = 'agent_dialog_open'
AGENT_ADDRESS = 'agent_address'
AGENT_NAME = 'agent_name'
AGENT_DESCRIPTION = 'agent_description'
INPUT_MODES = 'input_modes'
OUTPUT_MODES = 'output_modes'
STREAM_SUPPORTED = 'stream_supported'
PUSH_NOTIFICATIONS_SUPPORTED = 'push_notifications_supported'
AGENT_ERROR = 'agent_error' # 'error' 키가 다른 곳에서도 사용될 수 있으므로 좀 더 명확한 이름 사용
AGENT_FRAMEWORK_TYPE = 'agent_framework_type'

def initialize_agent_state():
    """
    Streamlit 세션 상태에 에이전트 관련 상태 변수들을 초기화합니다.
    이미 키가 존재하면 초기화하지 않습니다.
    """
    defaults = {
        AGENT_DIALOG_OPEN: False,
        AGENT_ADDRESS: '',
        AGENT_NAME: '',
        AGENT_DESCRIPTION: '',
        INPUT_MODES: [], # list 타입은 빈 리스트로 초기화
        OUTPUT_MODES: [], # list 타입은 빈 리스트로 초기화
        STREAM_SUPPORTED: False,
        PUSH_NOTIFICATIONS_SUPPORTED: False,
        AGENT_ERROR: '',
        AGENT_FRAMEWORK_TYPE: ''
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# 만약 AgentState 클래스 정의를 Pydantic 모델 등으로 남겨두어 데이터 구조를 명시하고 싶다면,
# 아래와 같이 할 수 있지만, Streamlit 세션 상태의 직접적인 대체는 아닙니다.
# from pydantic import BaseModel, Field
# from typing import List

# class AgentStateModel(BaseModel):
#     """
#     에이전트 관련 데이터 구조를 정의하는 Pydantic 모델입니다.
#     Streamlit 세션 상태의 키와 값을 나타냅니다.
#     """
#     agent_dialog_open: bool = False
#     agent_address: str = ''
#     agent_name: str = ''
#     agent_description: str = ''
#     input_modes: List[str] = Field(default_factory=list)
#     output_modes: List[str] = Field(default_factory=list)
#     stream_supported: bool = False
#     push_notifications_supported: bool = False
#     error: str = '' # AGENT_ERROR 키에 해당
#     agent_framework_type: str = ''

# Pydantic 모델을 사용하는 경우, 초기화 함수는 다음과 같을 수 있습니다:
# def initialize_agent_state_from_model():
#     if 'agent_state_data' not in st.session_state:
#         st.session_state['agent_state_data'] = AgentStateModel().model_dump()
# # 이 경우 st.session_state.agent_state_data['agent_name'] 과 같이 접근