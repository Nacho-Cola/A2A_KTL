import streamlit as st
# 페이지 설정은 스크립트 최상단 또는 메인 함수 시작 부분에 위치
st.set_page_config(page_title="💬 A2A Conversations", layout="wide", page_icon="🗣️")

import asyncio
import json # JSON 파싱/덤프에 사용될 수 있음 (현재 코드에서는 직접 사용 빈도 낮음)
import logging # 로깅 모듈 추가
from typing import List, Tuple, Dict, Any, Optional # 타입 힌팅

# Streamlit 프로젝트 구조에 따른 임포트 경로
from common.types import Message, TextPart, Task, Artifact # Task, Artifact 추가
from state.host_agent_service import (
    fetch_app_state_service,
    list_remote_agents_service,
    send_message_service,
    create_conversation_service,
    get_task_details_service, # 태스크 상세 정보 조회 함수 (host_agent_service.py에 구현 필요)
)
from state.state import AppState # AppState 임포트

from dotenv import load_dotenv # .env 파일 로드를 위해
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage # AIMessage 타입 힌트 추가

logger = logging.getLogger(__name__) # 로거 인스턴스 생성
load_dotenv(override=True) # OPENAI_API_KEY 등을 .env 파일에서 로드

# --- 비동기 헬퍼 함수 ---
async def _ainvoke_llm_safely(prompt_content: str) -> str:
    """LLM을 비동기적으로 호출하고 결과를 문자열로 반환합니다. 오류 발생 시 오류 메시지 반환."""
    if "chat_llm" not in st.session_state or st.session_state.chat_llm is None:
        logger.error("ChatLLM이 세션 상태에 초기화되지 않았거나 None입니다.")
        return "오류: LLM 클라이언트가 준비되지 않았습니다."
    try:
        logger.debug(f"LLM prompt (first 100 chars): {prompt_content[:100]}...")
        response_message = await st.session_state.chat_llm.ainvoke([HumanMessage(content=prompt_content)])
        if hasattr(response_message, 'content') and isinstance(response_message.content, str):
            logger.debug(f"LLM response content (first 100 chars): {response_message.content[:100]}...")
            return response_message.content
        else:
            logger.error(f"LLM 응답 객체에 'content' 속성이 없거나 문자열이 아닙니다. 응답 타입: {type(response_message)}")
            return "오류: LLM으로부터 예상치 못한 응답 형식입니다."
    except Exception as e:
        logger.exception("LLM 호출 중 오류 발생.")
        return f"LLM 통신 중 오류 발생: {str(e)}"

async def orchestrate_servers_async(question: str, available_servers: List[str]) -> Tuple[List[str], str]:
    """
    LLM에게 plan (호출할 서버 목록)과 prompt (사용자에게 보여줄 승인 요청 메시지)를
    비동기적으로 요청하고 파싱합니다.
    """
    prompt_for_orchestrator = (
        "당신은 A2A Orchestrator입니다.\n"
        "다음 질문을 처리하기 위해 호출할 서버 목록(PLAN)과, 그 계획을 사용자에게 자연어로 설명하며 승인을 요청할 메시지(PROMPT)를 "
        "다음 형식으로 정확히 출력해주세요. 각 항목은 한 줄로 명확히 구분되어야 합니다:\n\n"
        "PLAN: http://server1.example.com, http://server2.example.com\n"
        "PROMPT: [사용자에게 보여줄 자연어 승인 요청 문장]\n\n"
        "만약 질문을 처리하기 위해 호출할 서버가 필요 없거나, 직접 답변할 수 있다면 PLAN에는 빈 값을, PROMPT에는 직접적인 답변이나 설명을 포함하세요.\n"
        f"질문: {question}\n"
        f"사용 가능한 서버 목록: {', '.join(available_servers) if available_servers else '없음'}\n"
    )
    
    raw_llm_response = await _ainvoke_llm_safely(prompt_for_orchestrator)
    
    parsed_plan: List[str] = []
    parsed_user_prompt: str = raw_llm_response # 기본적으로 LLM 응답 전체를 프롬프트로 설정

    if "오류:" not in raw_llm_response: # LLM 호출 성공 시에만 파싱 시도
        prompt_found_in_response = False
        for line in raw_llm_response.splitlines():
            line_stripped = line.strip()
            if line_stripped.upper().startswith("PLAN:"):
                plan_str = line_stripped[len("PLAN:"):].strip()
                if plan_str: 
                    parsed_plan = [url.strip().rstrip("/") for url in plan_str.split(',') if url.strip()]
            elif line_stripped.upper().startswith("PROMPT:"):
                parsed_user_prompt = line_stripped[len("PROMPT:"):].strip()
                prompt_found_in_response = True
        
        if not prompt_found_in_response and not parsed_plan : # PLAN도 없고 명시적인 PROMPT도 없다면, LLM 응답 전체를 사용
             pass # parsed_user_prompt는 이미 raw_llm_response로 설정됨
        elif not prompt_found_in_response and parsed_plan: # PLAN은 있지만 PROMPT가 없다면 기본 승인 요청 문장 사용
            parsed_user_prompt = "제안된 계획을 승인하시겠습니까? (예/아니오)"
    else: # LLM 호출 실패 시
        parsed_user_prompt = "오케스트레이터 LLM 호출에 실패하여 계획을 생성할 수 없습니다. 직접 답변을 시도합니다."

    logger.info(f"Orchestrator LLM Raw Response:\n{raw_llm_response}")
    logger.info(f"Parsed Plan: {parsed_plan}")
    logger.info(f"Parsed User Prompt: {parsed_user_prompt}")
    
    return parsed_plan, parsed_user_prompt

# --- 세션 상태 초기화 함수 ---
def initialize_session_state_if_needed():
    """애플리케이션에 필요한 세션 상태 변수들을 초기화합니다."""
    if "chat_llm" not in st.session_state:
        try:
            st.session_state.chat_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
            logger.info("ChatOpenAI LLM (gpt-4o) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
            st.session_state.chat_llm = None
    if "histories" not in st.session_state:
        st.session_state.histories = {}
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None
    if "pending_plan" not in st.session_state:
        st.session_state.pending_plan = None
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None
    if "app_state_messages_loaded_for_conv" not in st.session_state:
        st.session_state.app_state_messages_loaded_for_conv = None
    
    if "active_polling_tasks" not in st.session_state:
        st.session_state.active_polling_tasks = {} 
    if "completed_task_results" not in st.session_state:
        st.session_state.completed_task_results = {}

initialize_session_state_if_needed()

# --- 데이터 로딩 함수 ---
@st.cache_data(show_spinner="대화 내용 로딩 중...", ttl=300)
def _fetch_app_state_for_conversation_cached(conv_id: str) -> Optional[AppState]:
    if not conv_id: return None
    logger.info(f"Fetching app state for conversation_id: {conv_id} (cacheable call)")
    return asyncio.run(fetch_app_state_service(conv_id))

def _load_and_prepare_chat_history(conv_id: str):
    """
    현재 대화 ID에 대한 메시지 히스토리를 st.session_state.histories에 준비합니다.
    st.rerun()을 직접 호출하지 않고, 상태 변경만 담당합니다.
    """
    if not conv_id:
        logger.warning("_load_and_prepare_chat_history called with no conv_id.")
        return False # 변경 없음

    st.session_state.histories.setdefault(conv_id, [])
    history_changed = False
    
    needs_backend_load = (st.session_state.app_state_messages_loaded_for_conv != conv_id) or \
                         (not st.session_state.histories[conv_id] and \
                          st.session_state.app_state_messages_loaded_for_conv != conv_id)

    if needs_backend_load:
        logger.info(f"Loading messages from backend for conversation_id: {conv_id}")
        app_state_data = _fetch_app_state_for_conversation_cached(conv_id)
        
        backend_messages_formatted = []
        if app_state_data and app_state_data.messages:
            for msg_obj in app_state_data.messages:
                text_content = "".join(
                    str(data) for data, mime in msg_obj.content if mime == "text/plain"
                ).strip()
                if text_content or msg_obj.role :
                    backend_messages_formatted.append({"role": msg_obj.role or "unknown", "content": text_content})
        
        if st.session_state.histories[conv_id] != backend_messages_formatted: # 실제 변경이 있을 때만
            st.session_state.histories[conv_id] = backend_messages_formatted
            history_changed = True
        st.session_state.app_state_messages_loaded_for_conv = conv_id
        logger.info(f"Messages for conversation {conv_id} loaded/updated from backend. History has {len(backend_messages_formatted)} messages. Changed: {history_changed}")
    return history_changed


# --- 폴링 관련 함수 ---
def extract_result_from_task(task: Optional[Task]) -> Optional[str]:
    """완료된 Task 객체에서 사용자가 볼 최종 결과 문자열을 추출합니다."""
    if not task or not task.status or str(task.status.state).upper() != "COMPLETED":
        return None

    results_found = []
    if task.artifacts:
        for artifact in task.artifacts:
            if artifact.parts:
                for part in artifact.parts:
                    if isinstance(part, TextPart) and hasattr(part, 'text') and part.text:
                        results_found.append(part.text)
    
    if not results_found and task.history:
        for msg in reversed(task.history):
            if msg.role == "agent" and msg.parts:
                for part in msg.parts:
                    if isinstance(part, TextPart) and hasattr(part, 'text') and part.text:
                        results_found.append(part.text)
                if results_found: break 
    
    if not results_found and task.status.message and task.status.message.parts:
        for part in task.status.message.parts:
            if isinstance(part, TextPart) and hasattr(part, 'text') and part.text:
                results_found.append(part.text)
            
    if results_found:
        return "\n".join(results_found)

    logger.warning(f"Could not extract a clear result string from completed task {task.id}")
    return "작업은 완료되었으나, 표시할 명확한 결과가 없습니다."


async def check_and_update_polling_tasks(active_conv_id: str) -> bool:
    """활성 폴링 태스크 상태 확인 및 UI 업데이트. 히스토리 업데이트 시 True 반환."""
    if not st.session_state.active_polling_tasks:
        return False

    tasks_to_remove: List[str] = []
    history_updated = False
    current_history_list = st.session_state.histories.setdefault(active_conv_id, [])

    task_ids_to_poll = list(st.session_state.active_polling_tasks.keys())
    if not task_ids_to_poll:
        return False

    logger.debug(f"Polling for tasks: {task_ids_to_poll} in conv: {active_conv_id}")
    polling_coroutines = [get_task_details_service(task_id) for task_id in task_ids_to_poll]
    results = await asyncio.gather(*polling_coroutines, return_exceptions=True)

    for i, task_id in enumerate(task_ids_to_poll):
        task_result = results[i]
        original_question = st.session_state.active_polling_tasks.get(task_id, "알 수 없는 질문")

        if isinstance(task_result, Exception):
            logger.error(f"Error polling task {task_id}: {task_result}", exc_info=task_result)
            msg = f"에이전트 작업(ID: {task_id[:8]}) 결과 조회 중 오류: {str(task_result)[:100]}"
            current_history_list.append({"role": "assistant", "content": msg})
            tasks_to_remove.append(task_id)
            history_updated = True
            continue

        task_detail: Optional[Task] = task_result
        if task_detail and task_detail.status:
            state_str = str(task_detail.status.state).upper()
            logger.debug(f"Task {task_id} (Q: '{original_question[:30]}...') polled state: {state_str}")

            if state_str == "COMPLETED":
                content = extract_result_from_task(task_detail) or "작업 완료 (내용 없음)"
                logger.info(f"Task {task_id} COMPLETED. Result: {content[:100]}...")
                current_history_list.append({"role": "assistant", "content": content})
                st.session_state.completed_task_results[task_id] = content
                tasks_to_remove.append(task_id)
                history_updated = True
            elif state_str in ["FAILED", "ERROR", "CANCELLED"]:
                msg = f"에이전트 작업(ID: {task_id[:8]})이 {state_str} 상태로 종료됨 (Q: '{original_question}')."
                logger.error(msg)
                current_history_list.append({"role": "assistant", "content": msg})
                tasks_to_remove.append(task_id)
                history_updated = True
    
    if tasks_to_remove:
        for tid in tasks_to_remove:
            if tid in st.session_state.active_polling_tasks:
                del st.session_state.active_polling_tasks[tid]
        logger.info(f"Removed tasks from polling: {tasks_to_remove}")

    return history_updated

# --- 메인 UI 렌더링 함수 ---
def render_conversation_page():
    st.title("💬 Agent-to-Agent Orchestrated Chat")

    if st.session_state.get("chat_llm") is None:
        st.error("OpenAI LLM 클라이언트를 초기화할 수 없습니다. API 키 및 설정을 확인해주세요.")

    logger.debug(f"[Page Render Start] current_conv_id: {st.session_state.get('current_conv_id')}, "
                 f"pending_plan: {st.session_state.get('pending_plan') is not None}, "
                 f"active_polling_tasks: {len(st.session_state.get('active_polling_tasks', {}))}")

    # --- 사이드바: 대화 세션 관리 ---
    with st.sidebar:
        st.header("Conversations")
        if st.button("➕ 새 대화 시작", key="new_conversation_sidebar_button_conv_page", use_container_width=True):
            logger.info("'새 대화 시작' 버튼 클릭됨 (사이드바).")
            new_conv_object = asyncio.run(create_conversation_service())
            if new_conv_object and new_conv_object.conversation_id:
                new_id = new_conv_object.conversation_id
                st.session_state.histories[new_id] = []
                st.session_state.current_conv_id = new_id
                st.session_state.app_state_messages_loaded_for_conv = None 
                st.session_state.active_polling_tasks = {} 
                st.session_state.completed_task_results = {}
                logger.info(f"New conversation '{new_id}' created and selected. Rerunning.")
                st.rerun()
            else:
                st.error("백엔드에서 새 대화 생성에 실패했습니다.")
                logger.error("Failed to create new conversation or received invalid data from service.")

        conv_ids_sorted = sorted(st.session_state.get("histories", {}).keys(), reverse=True)

        for conv_id_key in conv_ids_sorted:
            display_label = f"대화 {conv_id_key[:8]}..."
            is_current = (conv_id_key == st.session_state.get("current_conv_id"))
            button_type = "primary" if is_current else "secondary"
            
            if st.button(display_label, key=f"select_conv_button_{conv_id_key}_conv_page", disabled=is_current, use_container_width=True, type=button_type):
                logger.info(f"Conversation '{conv_id_key}' selected from sidebar.")
                if st.session_state.current_conv_id != conv_id_key: # 실제 ID 변경 시에만 상태 초기화 및 rerun
                    st.session_state.current_conv_id = conv_id_key
                    st.session_state.app_state_messages_loaded_for_conv = None
                    st.session_state.active_polling_tasks = {} 
                    st.session_state.completed_task_results = {}
                    st.rerun()

    active_conv_id = st.session_state.get("current_conv_id")
    logger.debug(f"[Main Area Start] active_conv_id = {active_conv_id}")

    if not active_conv_id:
        st.info("👈 사이드바에서 대화를 선택하거나 '새 대화 시작' 버튼을 눌러주세요.")
        st.stop()

    # --- 자동 폴링 실행 및 히스토리 로드/표시 ---
    # st.session_state에 active_polling_tasks가 있고, 내용이 있을 때만 실행
    if st.session_state.get("active_polling_tasks"): 
        logger.debug(f"Active polling tasks found: {list(st.session_state.active_polling_tasks.keys())}. Checking status.")
        history_updated_by_poll = asyncio.run(check_and_update_polling_tasks(active_conv_id))
        if history_updated_by_poll:
            logger.info("History was updated by polling. Rerunning to reflect.")
            st.rerun() # 폴링으로 히스토리 변경 시 UI 새로고침

    # 백엔드 메시지 로드 (필요시). 폴링으로 히스토리가 업데이트 되었다면, 이 함수는 추가 로드를 안할 수 있음.
    # 또는, 이 함수가 로드한 내용과 폴링 결과가 병합되어야 할 수도 있음 (현재는 교체 방식)
    history_refreshed_by_load = _load_and_prepare_chat_history(active_conv_id)
    if history_refreshed_by_load and not st.session_state.active_polling_tasks: # 폴링이 없고, 로드로 히스토리 변경 시
        logger.info("History refreshed by _load_and_prepare_chat_history. Rerunning.")
        st.rerun()
        
    active_chat_history = st.session_state.histories.get(active_conv_id, [])
    logger.debug(f"[After History Load & Poll] Messages for {active_conv_id}: {len(active_chat_history)}")

    # 채팅 메시지 표시
    for msg_entry in active_chat_history:
        role = msg_entry.get("role", "unknown")
        avatar_icon = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(name=role, avatar=avatar_icon):
            st.markdown(msg_entry.get("content", ""))

    # --- 사용자 메시지 입력 ---
    logger.debug(f"[Before Chat Input] Rendering st.chat_input for {active_conv_id}")
    user_chat_input = st.chat_input(
        "메시지를 입력하거나, 제안에 대해 '예' 또는 '아니오'로 답해주세요...", 
        key=f"chat_input_for_conv_{active_conv_id}_page" 
    )

    if user_chat_input:
        logger.info(f"User input for conv '{active_conv_id}': {user_chat_input}")
        
        current_history_list = st.session_state.histories.setdefault(active_conv_id, [])
        current_history_list.append({"role": "user", "content": user_chat_input})

        should_rerun_after_processing = False

        # --- 2단계 처리: 오케스트레이터 제안에 대한 사용자 응답 ---
        if st.session_state.get("pending_plan") is not None:
            plan_to_execute = st.session_state.pending_plan
            original_question = st.session_state.pending_input

            st.session_state.pending_plan = None
            st.session_state.pending_input = None
            should_rerun_after_processing = True

            if "예" in user_chat_input or "yes" in user_chat_input.lower():
                logger.info(f"User approved plan: {plan_to_execute} for Q: '{original_question}'")
                if not plan_to_execute:
                    msg_content = "알겠습니다. 이전 오케스트레이터의 답변을 참고해주세요."
                    current_history_list.append({"role": "assistant", "content": msg_content})
                else:
                    for agent_url_target in plan_to_execute:
                        tool_msg_content = f"💡 **Tool Executing:** A2A Agent @ `{agent_url_target}` (Q: '{original_question}')"
                        current_history_list.append({"role": "assistant", "content": tool_msg_content})
                        
                        agent_message_to_send = Message(
                            role="user", parts=[TextPart(text=original_question)],
                            metadata={"conversation_id": active_conv_id, "target_agent_url": agent_url_target}
                        )
                        # send_message_service는 태스크 ID를 반환한다고 가정
                        returned_task_id = asyncio.run(send_message_service(agent_message_to_send))
                        
                        if returned_task_id:
                            processing_feedback_msg = f"에이전트 `{agent_url_target}`에 작업 요청됨 (Task ID: {returned_task_id[:8]}...). 결과 확인 중..."
                            current_history_list.append({"role": "assistant", "content": processing_feedback_msg})
                            st.session_state.active_polling_tasks[returned_task_id] = original_question # 폴링 목록에 추가
                            logger.info(f"Task {returned_task_id} added to polling for Q: '{original_question}'")
                        else:
                            error_feedback_msg = f"에이전트 `{agent_url_target}` 요청 실패 (ID 미수신)."
                            current_history_list.append({"role": "assistant", "content": error_feedback_msg})
            else: # 거부
                logger.info(f"User rejected plan for Q: '{original_question}'. Using fallback LLM.")
                fallback_llm_response = asyncio.run(_ainvoke_llm_safely(original_question))
                current_history_list.append({"role": "assistant", "content": fallback_llm_response})
        
        # --- 1단계 처리: 새로운 사용자 질문 (오케스트레이터 호출) ---
        else:
            logger.info("Processing new user input with orchestrator.")
            should_rerun_after_processing = True
            remote_agents = asyncio.run(list_remote_agents_service())
            available_urls = [card.url.rstrip("/") for card in remote_agents if card and card.url]
            
            if not available_urls and (st.session_state.get("chat_llm") is None):
                 warning_message = "현재 사용 가능한 원격 에이전트나 기본 LLM이 없어 요청을 처리할 수 없습니다."
                 current_history_list.append({"role": "assistant", "content": warning_message})
            elif not available_urls: # 에이전트는 없지만 기본 LLM은 있는 경우
                logger.info("No remote agents available. Processing directly with default LLM.")
                direct_response = asyncio.run(_ainvoke_llm_safely(user_chat_input))
                current_history_list.append({"role": "assistant", "content": direct_response})
            else: # 오케스트레이터 호출
                plan, approval_prompt = asyncio.run(
                    orchestrate_servers_async(user_chat_input, available_urls)
                )
                current_history_list.append({"role": "assistant", "content": approval_prompt})
                if plan: # 실행할 계획이 있는 경우에만 다음 입력을 위해 저장
                    st.session_state.pending_plan = plan
                    st.session_state.pending_input = user_chat_input
                else: # 계획이 없는 경우 (LLM이 직접 답변 생성 등)
                    logger.info("Orchestrator returned no actionable plan. Approval message is considered the response.")
        
        if should_rerun_after_processing:
            st.rerun() # 모든 입력 처리 및 히스토리 업데이트 후 최종적으로 한 번 rerun

# --- __main__ ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - %(name)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Streamlit Conversations Page starting session.")
    render_conversation_page()