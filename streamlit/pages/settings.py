import streamlit as st
import logging # 로깅 모듈 추가

# streamlit 폴더 구조에 따른 임포트 경로 수정
from state.state import AppState, SettingsState

logger = logging.getLogger(__name__) # 로거 인스턴스 생성

# --- 1) 세션 상태 초기화 ---
# 이 초기화 로직은 메인 앱 파일(예: home.py 또는 최상위 앱 파일)에서 한 번만 수행하는 것이 더 일반적일 수 있습니다.
# 각 페이지마다 중복으로 포함될 경우, st.session_state가 이미 초기화된 후에는 영향을 주지 않습니다.
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
    logger.info("AppState initialized in session_state for Settings page.")
if "settings_state" not in st.session_state:
    st.session_state.settings_state = SettingsState()
    logger.info("SettingsState initialized in session_state for Settings page.")

# 세션 상태에서 AppState 및 SettingsState 객체 가져오기
# 페이지 로드 시점에 state와 settings 변수에 할당
# 이 변수들은 st.session_state의 객체에 대한 참조이므로,
# 이 변수를 통해 필드를 수정하면 st.session_state도 변경됩니다.
try:
    state: AppState = st.session_state.app_state
    settings: SettingsState = st.session_state.settings_state
except AttributeError:
    # 드물지만, app_state나 settings_state가 어떤 이유로 삭제되었을 경우를 대비
    logger.error("AppState or SettingsState not found in session_state. Re-initializing.")
    st.session_state.app_state = AppState()
    st.session_state.settings_state = SettingsState()
    state = st.session_state.app_state
    settings = st.session_state.settings_state
    st.warning("애플리케이션 상태가 초기화되었습니다. 페이지를 새로고침해주세요.")
    st.stop()


# --- 페이지 설정 ---
st.set_page_config(page_title="⚙️ Settings", layout="centered", page_icon="🛠️") # 아이콘 변경 예시
st.header("⚙️ Application Settings")

# ─────────────────────────────────────────────────────────────
# 2) 출력 MIME 타입 설정
# ─────────────────────────────────────────────────────────────
st.subheader("Output MIME Types")
st.caption("애플리케이션에서 허용하거나 처리할 출력 파일의 MIME 타입들을 설정합니다.")
# 사용 가능한 전체 MIME 타입 목록 (필요에 따라 확장 가능)
all_available_mime_types = [
    "text/plain",
    "application/json",
    "image/png",
    "image/jpeg",
    "image/gif",
    "application/pdf",
    "text/markdown",
    "text/html",
    "audio/mpeg",
    "video/mp4",
    # 필요에 따라 추가 MIME 타입들
]
# settings.output_mime_types가 all_available_mime_types에 없는 값을 가질 경우를 대비
# (예: 상태 파일이 수동으로 변경되었거나, 이전 버전의 앱에서 마이그레이션된 경우)
valid_default_mime_types = [mime for mime in settings.output_mime_types if mime in all_available_mime_types]
if not valid_default_mime_types and settings.output_mime_types: # 기본값이 있지만 유효한 것이 없을 경우
    logger.warning(f"Default MIME types {settings.output_mime_types} contain invalid entries not in all_available_mime_types. Resetting to common defaults.")
    # 안전한 기본값으로 재설정 (선택적)
    # valid_default_mime_types = ["text/plain", "image/png"] 
    # 또는 사용자에게 경고만 표시

selected_mime_types = st.multiselect(
    "허용할 MIME 타입을 선택하세요:",
    options=all_available_mime_types,
    default=valid_default_mime_types, # settings.output_mime_types 대신 유효한 기본값 사용
    key="multiselect_mime_types" # 위젯 키 추가 (선택적이지만 권장)
)
# 변경 사항을 settings 객체에 즉시 반영
if settings.output_mime_types != selected_mime_types:
    logger.info(f"Output MIME types changed from {settings.output_mime_types} to {selected_mime_types}")
    settings.output_mime_types = selected_mime_types
st.caption(f"현재 선택된 MIME 타입: `{', '.join(settings.output_mime_types) if settings.output_mime_types else '선택된 타입 없음'}`")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 3) 폴링 간격 설정
# ─────────────────────────────────────────────────────────────
st.subheader("Polling Interval (seconds)")
st.caption("백그라운드 작업 상태 등을 확인하기 위한 폴링(polling) 주기를 초 단위로 설정합니다.")
new_polling_interval = st.number_input(
    "폴링 간격을 입력하세요 (최소 1초):",
    min_value=1,
    value=state.polling_interval, # AppState에서 값 가져옴
    step=1,
    key="number_input_polling_interval"
)
# 변경 사항을 state 객체에 즉시 반영
if state.polling_interval != new_polling_interval:
    logger.info(f"Polling interval changed from {state.polling_interval}s to {new_polling_interval}s")
    state.polling_interval = new_polling_interval
st.caption(f"현재 폴링 간격: `{state.polling_interval}초`")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 4) API 키 관리
# ─────────────────────────────────────────────────────────────
st.subheader("API Key Configuration")
st.caption("외부 서비스 (예: Google Generative AI) 사용을 위한 API 키를 설정합니다.")

# API 키 입력
new_api_key = st.text_input(
    "API Key:",
    value=state.api_key, # AppState에서 값 가져옴
    type="password", # 비밀번호 타입으로 입력값 가리기
    help="Google API Key 또는 Vertex AI 사용 시 관련 설정을 입력하세요.",
    key="text_input_api_key"
)
if state.api_key != new_api_key:
    # 실제 키 값을 로그에 남기지 않도록 주의. 여기서는 변경 사실만 로깅.
    logger.info(f"API Key has been modified by the user (length: {len(new_api_key)}).")
    state.api_key = new_api_key

# Vertex AI 사용 여부 체크박스
new_uses_vertex_ai = st.checkbox(
    "Vertex AI 사용",
    value=state.uses_vertex_ai, # AppState에서 값 가져옴
    help="이 옵션을 선택하면 Vertex AI를 사용하여 API를 호출합니다. 선택 해제 시 일반 API Key를 사용합니다.",
    key="checkbox_use_vertex_ai"
)
if state.uses_vertex_ai != new_uses_vertex_ai:
    logger.info(f"Vertex AI usage changed from {state.uses_vertex_ai} to {new_uses_vertex_ai}")
    state.uses_vertex_ai = new_uses_vertex_ai

# API 키가 설정되었는지 여부에 따라 간단한 상태 메시지 표시
if state.api_key:
    st.write("API Key가 설정되어 있습니다.")
else:
    st.write("API Key가 설정되지 않았습니다.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 5) 테마 모드 (AppState에 정의된 필드 사용)
# ─────────────────────────────────────────────────────────────
st.subheader("Theme Mode")
st.caption("애플리케이션의 UI 테마를 선택합니다.")
available_themes = ["system", "light", "dark"] # AppState.theme_mode의 Literal 타입과 일치해야 함
current_theme_index = available_themes.index(state.theme_mode) if state.theme_mode in available_themes else 0

selected_theme = st.selectbox(
    "테마 모드를 선택하세요:",
    options=available_themes,
    index=current_theme_index,
    key="selectbox_theme_mode"
)
# 변경 사항을 state 객체에 즉시 반영
if state.theme_mode != selected_theme:
    logger.info(f"Theme mode changed from {state.theme_mode} to {selected_theme}")
    state.theme_mode = selected_theme # type: ignore  # Literal 타입에 대한 할당이므로 실제로는 안전
st.caption(f"현재 테마 모드: `{state.theme_mode}`")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 6) 설정 저장 버튼 (현재는 UI 피드백만 제공)
# ─────────────────────────────────────────────────────────────
# Streamlit에서는 위젯 값 변경 시 st.session_state가 즉시 업데이트되므로,
# 별도의 "저장" 버튼은 실제 백엔드 저장 등의 추가 작업이 없다면 UI/UX 피드백 용도입니다.
if st.button("💾 설정 저장 확인", key="button_save_settings"):
    # 여기에 실제 설정 저장 로직(예: 파일 쓰기, API 호출 등)이 필요하다면 추가합니다.
    # 현재는 세션 상태에 이미 모든 변경사항이 반영되어 있습니다.
    logger.info("Settings 'Save' button clicked. Current settings are already in session_state.")
    st.success("설정이 현재 세션에 반영되었습니다! (별도 저장 로직 없음)")
    # 필요하다면, 여기서 백엔드로 설정을 보내는 API 호출 등을 수행할 수 있습니다.
    # 예: asyncio.run(save_settings_to_backend(st.session_state.app_state, st.session_state.settings_state))

# 페이지 하단에 현재 상태 객체 (일부)를 디버그용으로 표시 (개발 중에만 사용)
# with st.expander("Debug: Current AppState"):
#     st.json(st.session_state.app_state.model_dump_json(indent=2) if hasattr(st.session_state.app_state, 'model_dump_json') else vars(st.session_state.app_state))
# with st.expander("Debug: Current SettingsState"):
#     st.json(st.session_state.settings_state.model_dump_json(indent=2) if hasattr(st.session_state.settings_state, 'model_dump_json') else vars(st.session_state.settings_state))