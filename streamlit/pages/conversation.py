import streamlit as st
# 페이지 설정은 스크립트 최상단 또는 메인 함수 시작 부분에 위치
st.set_page_config(page_title="💬 A2A Conversations", layout="wide", page_icon="🗣️")

import time 
import asyncio
import json 
import logging 
from typing import List, Tuple, Dict, Any, Optional 

# --- 로깅 설정 (스크립트 상단으로 이동) ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Streamlit 프로젝트 구조에 따른 임포트 경로
from common.types import Message, TextPart, DataPart, Task, Part, TaskState # TaskState 임포트 확인
from state.host_agent_service import (
    fetch_app_state_service,
    list_remote_agents_service,
    send_message_service, 
    create_conversation_service,
    get_task_details_service,
)
from state.state import AppState 

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

# --- 유틸리티 함수 정의 ---
METADATA_MESSAGE_ID = "message_id" # message_id 키를 상수로 정의

def get_message_id(msg_obj: Optional[Message]) -> Optional[str]:
    """Helper to safely get message_id from Message object's metadata."""
    if msg_obj and msg_obj.metadata and METADATA_MESSAGE_ID in msg_obj.metadata:
        return msg_obj.metadata[METADATA_MESSAGE_ID]
    return None

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
    parsed_user_prompt: str = raw_llm_response 

    if "오류:" not in raw_llm_response: 
        prompt_found_in_response = False
        lines = raw_llm_response.splitlines()
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("PLAN:"):
                plan_str = line_stripped[len("PLAN:"):].strip()
                if plan_str and plan_str.lower() != "none" and plan_str.lower() != "null" and plan_str.lower() != "empty":
                    parsed_plan = [url.strip().rstrip("/") for url in plan_str.split(',') if url.strip()]
            elif line_stripped.upper().startswith("PROMPT:"):
                parsed_user_prompt = line_stripped[len("PROMPT:"):].strip()
                if i + 1 < len(lines): 
                    parsed_user_prompt += "\n" + "\n".join(lines[i+1:])
                prompt_found_in_response = True
                break 
        
        if not prompt_found_in_response:
            if not parsed_plan: 
                pass 
            else: 
                parsed_user_prompt = f"다음 에이전트 실행 계획을 승인하시겠습니까?: {', '.join(parsed_plan)} (예/아니오)"
    else: 
        parsed_user_prompt = "오케스트레이터 LLM 호출에 실패하여 계획을 생성할 수 없습니다. 직접 답변을 시도합니다."
        parsed_plan = [] 

    logger.info(f"Orchestrator LLM Raw Response:\n{raw_llm_response}")
    logger.info(f"Parsed Plan: {parsed_plan}")
    logger.info(f"Parsed User Prompt: {parsed_user_prompt}")
    
    return parsed_plan, parsed_user_prompt

# --- 세션 상태 초기화 함수 ---
def initialize_session_state_if_needed():
    """애플리케이션에 필요한 세션 상태 변수들을 초기화합니다."""
    defaults = {
        "chat_llm": None,
        "histories": {}, 
        "current_conv_id": None,
        "pending_plan": None, 
        "pending_input": None, 
        "app_state_messages_loaded_for_conv": None, 
        "active_polling_tasks": {}, 
        "completed_task_results": {}, 
        "assistant_is_generating": False, 
        "last_user_input_for_processing": None, 
        "chat_llm_initialized": False, 
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if not st.session_state.chat_llm_initialized:
        st.session_state.chat_llm_initialized = True 
        try:
            st.session_state.chat_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
            logger.info("ChatOpenAI LLM (gpt-4o) initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
            st.session_state.chat_llm = None


initialize_session_state_if_needed()

# --- 데이터 로딩 함수 ---
@st.cache_data(show_spinner="대화 내용 로딩 중...", ttl=30) 
def _fetch_app_state_for_conversation_cached(conv_id: str) -> Optional[AppState]:
    if not conv_id: return None
    logger.info(f"Fetching app state for conversation_id: {conv_id} (cacheable call)")
    try:
        return asyncio.run(fetch_app_state_service(conv_id))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            logger.warning(f"asyncio.run error in _fetch_app_state_for_conversation_cached for {conv_id}. Returning None.")
            return None 
        logger.exception(f"Runtime error in _fetch_app_state_for_conversation_cached for {conv_id}")
        return None 
    except Exception as e:
        logger.exception(f"Unexpected error in _fetch_app_state_for_conversation_cached for {conv_id}")
        return None


def _process_message_parts_for_display(msg_obj: Message) -> List[Dict[str, Any]]:
    """ Message 객체의 parts를 UI 표시에 적합한 dict 리스트로 변환합니다. """
    display_parts = []
    if not msg_obj.parts:
        logger.debug(f"Message {get_message_id(msg_obj) or 'N/A'} has no parts.")
        return [] 

    for part_obj in msg_obj.parts: 
        part_info: Dict[str, Any] = {"type": "unknown", "content": str(part_obj)} 

        if isinstance(part_obj, TextPart) and hasattr(part_obj, 'text'):
            part_info = {"type": "text", "content": part_obj.text or ""}
        elif isinstance(part_obj, DataPart) and hasattr(part_obj, 'data'):
            part_info["data"] = part_obj.data 
            part_info["name"] = part_obj.name or "Data"
            
            a2a_part_type = None
            tool_name = "Unknown Tool"
            if part_obj.metadata: 
                a2a_part_type = part_obj.metadata.get("a2a_part_type")
                tool_name = part_obj.metadata.get("tool_name", tool_name) 
            
            part_info["tool_name"] = tool_name 
            if a2a_part_type == "tool_call":
                part_info["type"] = "tool_call"
                # Args will be shown in expander, content is a summary
                args_summary = ""
                if isinstance(part_obj.data, dict):
                    args_summary = ", ".join([f"{k}={str(v)[:20]}{'...' if len(str(v)) > 20 else ''}" for k,v in part_obj.data.items()])
                elif isinstance(part_obj.data, list):
                    args_summary = f"[{len(part_obj.data)} items]"
                else:
                    args_summary = str(part_obj.data)[:30] + ('...' if len(str(part_obj.data)) > 30 else '')

                part_info["content"] = f"🛠️ **Tool Call:** `{tool_name}`"
                if args_summary:
                     part_info["content"] += f" (Args: {args_summary})"
                else:
                     part_info["content"] += " (No Args)"

            elif a2a_part_type in ["tool_response", "tool_output"]:
                part_info["type"] = a2a_part_type # Keep original type for rendering logic
                part_info["content"] = f"⚙️ **Tool Output**{f' from `{tool_name}`' if tool_name and tool_name != 'Unknown Tool' else ''}"
                 # Details will be in expander based on part_info["data"]
            else: 
                part_info["type"] = "data"
                part_info["content"] = f"📊 **Data:** {part_info['name']}"
        
        display_parts.append(part_info)
    return display_parts


def _load_and_prepare_chat_history(conv_id: str) -> bool:
    if not conv_id:
        logger.warning("_load_and_prepare_chat_history called with no conv_id.")
        return False

    st.session_state.histories.setdefault(conv_id, [])
    history_changed = False
    
    needs_initial_load = st.session_state.app_state_messages_loaded_for_conv != conv_id

    if needs_initial_load:
        logger.info(f"Performing initial load of messages from backend for conversation_id: {conv_id}")
        app_state_data = _fetch_app_state_for_conversation_cached(conv_id)
        
        processed_backend_messages = []
        if app_state_data and app_state_data.messages: 
            for msg_obj in app_state_data.messages: 
                if not isinstance(msg_obj, Message):
                    logger.warning(f"Expected Message object from backend, got {type(msg_obj)}. Skipping.")
                    continue
                
                ui_display_parts = _process_message_parts_for_display(msg_obj)
                
                simple_text_summary = ""
                if ui_display_parts:
                    # Concatenate content from all parts for a better summary
                    summary_parts_content = []
                    for p in ui_display_parts:
                        content_val = p.get("content")
                        if isinstance(content_val, str):
                            summary_parts_content.append(content_val)
                        elif content_val is not None: # handle non-string content if necessary
                            summary_parts_content.append(f"[{p.get('type', 'data')}]")

                    simple_text_summary = " ".join(summary_parts_content).strip()
                    if not simple_text_summary and ui_display_parts: # Fallback if all content was empty/non-string
                         simple_text_summary = f"[{ui_display_parts[0].get('type', 'message')}]"


                processed_backend_messages.append({
                    "role": msg_obj.role or "unknown", 
                    "content": simple_text_summary[:150] + "..." if len(simple_text_summary) > 150 else simple_text_summary, 
                    "message_id": get_message_id(msg_obj) or f"backend_msg_{time.time_ns()}",
                    "ui_display_parts": ui_display_parts 
                })
        
        if st.session_state.histories.get(conv_id) != processed_backend_messages:
            st.session_state.histories[conv_id] = processed_backend_messages
            history_changed = True
        
        st.session_state.app_state_messages_loaded_for_conv = conv_id
        logger.info(f"Initial messages for conversation {conv_id} loaded. History has {len(processed_backend_messages)} messages. Changed: {history_changed}")
    
    return history_changed


# extract_result_from_task is no longer called by check_and_update_polling_tasks
# It can be kept if used elsewhere, or removed/deprecated.
# For this modification, its logic is integrated into check_and_update_polling_tasks.
# def extract_result_from_task(task: Optional[Task]) -> List[Dict[str, Any]]:
#     """
#     Task 객체에서 UI 표시에 적합한 part 리스트를 추출합니다.
#     (This function's logic is now primarily within check_and_update_polling_tasks)
#     """
#     # ... (original implementation) ...


async def check_and_update_polling_tasks(active_conv_id: str) -> bool:
    if not st.session_state.active_polling_tasks:
        return False

    tasks_to_remove_from_polling: List[str] = []
    history_updated_this_poll = False
    current_chat_history_list = st.session_state.histories.setdefault(active_conv_id, [])

    task_ids_being_polled = list(st.session_state.active_polling_tasks.keys())
    if not task_ids_being_polled: return False

    logger.debug(f"Polling for tasks: {task_ids_being_polled} in conv: {active_conv_id}")
    
    polling_coroutines = [get_task_details_service(task_id) for task_id in task_ids_being_polled]
    try:
        polled_task_results = await asyncio.gather(*polling_coroutines, return_exceptions=True)
    except RuntimeError as e:
        logger.error(f"Error in asyncio.gather for polling tasks: {e}")
        return False

    for i, task_id in enumerate(task_ids_being_polled):
        task_result_or_exception = polled_task_results[i]
        new_assistant_bubbles_from_task: List[Dict[str, Any]] = []

        if isinstance(task_result_or_exception, Exception):
            error_content = f"에이전트 작업(ID: {task_id[:8]}) 결과 조회 중 오류: {str(task_result_or_exception)[:100]}"
            logger.error(f"Error polling task {task_id}: {task_result_or_exception}", exc_info=task_result_or_exception)
            
            new_assistant_bubbles_from_task.append({
                "role": "assistant", "content": error_content,
                "message_id": f"task_poll_error_{task_id}_{int(time.time())}",
                "ui_display_parts": [{"type": "text", "content": error_content}]
            })
            tasks_to_remove_from_polling.append(task_id) # Stop polling on error
        else:
            task_detail: Optional[Task] = task_result_or_exception
            if task_detail and task_detail.status:
                logger.debug(f"Task {task_id} polled state: {task_detail.status.state}")

                is_terminal_state = task_detail.status.state in [TaskState.COMPLETED, TaskState.FAILED] # Add other terminal states if any (e.g., CANCELLED)

                if is_terminal_state:
                    # Remove the "결과 확인 중..." placeholder message for this task
                    placeholder_removed = False
                    for entry_idx, hist_entry in reversed(list(enumerate(current_chat_history_list))):
                        if hist_entry.get("role") == "assistant" and \
                           f"Task ID: {task_id[:8]}" in hist_entry.get("content","") and \
                           "결과 확인 중..." in hist_entry.get("content",""):
                            del current_chat_history_list[entry_idx]
                            placeholder_removed = True
                            logger.info(f"Removed placeholder for task {task_id}.")
                            break # Remove only one placeholder
                    
                    # Process task history messages
                    if task_detail.history:
                        logger.debug(f"Task {task_id} has {len(task_detail.history)} history messages. Processing for display.")
                        for hist_idx, hist_msg in enumerate(task_detail.history):
                            if isinstance(hist_msg, Message):
                                # Avoid duplicating the final status message if it's also the last history item
                                is_last_hist_msg = (hist_idx == len(task_detail.history) - 1)
                                is_same_as_status_msg = (task_detail.status.message and 
                                                         get_message_id(hist_msg) == get_message_id(task_detail.status.message))
                                
                                if not (is_last_hist_msg and is_same_as_status_msg and task_detail.status.state == TaskState.COMPLETED):
                                    parts_for_hist_msg = _process_message_parts_for_display(hist_msg)
                                    if parts_for_hist_msg:
                                        summary = " ".join(str(p.get("content")) for p in parts_for_hist_msg if p.get("content") and isinstance(p.get("content"), str)).strip() \
                                                    or f"[{parts_for_hist_msg[0].get('type', 'task_step')}]"
                                        new_assistant_bubbles_from_task.append({
                                            "role": "assistant", "content": summary,
                                            "message_id": f"task_{task_id}_hist_{hist_idx}_{int(time.time())}",
                                            "ui_display_parts": parts_for_hist_msg
                                        })
                                else:
                                    logger.debug(f"Skipping last history message for task {task_id} as it's the final status message.")
                            else:
                                logger.warning(f"Non-Message object found in task history for task {task_id}: {type(hist_msg)}")
                    
                    # Process final status message (for COMPLETED or FAILED with message)
                    if task_detail.status.message:
                        logger.debug(f"Processing final status message for task {task_id} (State: {task_detail.status.state})")
                        final_result_parts = _process_message_parts_for_display(task_detail.status.message)
                        if final_result_parts:
                            summary = " ".join(str(p.get("content")) for p in final_result_parts if p.get("content") and isinstance(p.get("content"), str)).strip() \
                                        or f"[{final_result_parts[0].get('type', 'task_final')}]"
                            new_assistant_bubbles_from_task.append({
                                "role": "assistant", "content": summary,
                                "message_id": f"task_{task_id}_final_status_{int(time.time())}",
                                "ui_display_parts": final_result_parts
                            })
                    elif task_detail.status.state == TaskState.COMPLETED: # Completed but no message in status
                        logger.warning(f"Task {task_id} is COMPLETED, but no message in task.status.message.")
                        completion_notice = "작업 완료 (결과 메시지 없음)"
                        new_assistant_bubbles_from_task.append({
                            "role": "assistant", "content": completion_notice,
                            "message_id": f"task_{task_id}_final_empty_{int(time.time())}",
                            "ui_display_parts": [{"type": "text", "content": completion_notice}]
                        })
                    elif task_detail.status.state == TaskState.FAILED: # Failed and no specific message in status
                         failure_notice = f"에이전트 작업(ID: {task_id[:8]})이 실패했습니다 (상세 메시지 없음)."
                         new_assistant_bubbles_from_task.append({
                            "role": "assistant", "content": failure_notice,
                            "message_id": f"task_{task_id}_failed_empty_{int(time.time())}",
                            "ui_display_parts": [{"type": "text", "content": failure_notice}]
                        })


                    # Store a simple summary for completed tasks if needed elsewhere
                    if task_detail.status.state == TaskState.COMPLETED:
                        completed_summary = "작업 완료"
                        if new_assistant_bubbles_from_task: # Get summary from the last bubble added
                            last_bubble_content = new_assistant_bubbles_from_task[-1].get("content")
                            if isinstance(last_bubble_content, str):
                                completed_summary = last_bubble_content
                        st.session_state.completed_task_results[task_id] = completed_summary
                    
                    tasks_to_remove_from_polling.append(task_id)
                    logger.info(f"Task {task_id} is terminal (State: {task_detail.status.state}). Processed for step-by-step UI display.")
                
                # If task is still running, we don't add new bubbles here.
                # Real-time updates for running tasks would require agent to push partial history
                # and get_task_details_service to support fetching it.
                # This implementation shows all history once task is terminal.

            else: # task_detail is None or task_detail.status is None
                 logger.warning(f"Polling for task {task_id} returned invalid data or no status: {task_detail}")
                 # Optionally, keep polling or remove with an error
                 # tasks_to_remove_from_polling.append(task_id) # Example: stop polling if data is bad

        if new_assistant_bubbles_from_task:
            current_chat_history_list.extend(new_assistant_bubbles_from_task)
            history_updated_this_poll = True
            
    if tasks_to_remove_from_polling:
        for tid in tasks_to_remove_from_polling:
            if tid in st.session_state.active_polling_tasks:
                del st.session_state.active_polling_tasks[tid]
        logger.info(f"Removed tasks from polling: {tasks_to_remove_from_polling}")

    return history_updated_this_poll


def process_user_input_and_generate_response(active_conv_id: str, user_input_content: str):
    current_history_list = st.session_state.histories.setdefault(active_conv_id, [])
    
    try:
        if st.session_state.get("pending_plan") is not None: 
            plan_to_execute = st.session_state.pending_plan
            original_question_for_plan = st.session_state.pending_input 

            st.session_state.pending_plan = None
            st.session_state.pending_input = None

            if "예" in user_input_content or "yes" in user_input_content.lower():
                logger.info(f"User approved plan: {plan_to_execute} for Q: '{original_question_for_plan}'")
                if not plan_to_execute: # Orchestrator decided no plan, approval_prompt was the answer
                    msg_text = "알겠습니다. 이전 오케스트레이터의 답변을 참고해주세요." # This case might need refinement based on orchestrator's behavior
                    ui_parts = [{"type": "text", "content": msg_text}]
                    current_history_list.append({"role": "assistant", "content": msg_text, "ui_display_parts": ui_parts, "message_id": f"msg_{time.time_ns()}"})
                else:
                    for agent_url_target in plan_to_execute:
                        # This message indicates the start of an agent call, before polling begins.
                        tool_exec_text = f"➡️ **에이전트 호출 시작:** `{agent_url_target}` (질문: '{original_question_for_plan[:30]}{'...' if len(original_question_for_plan)>30 else ''}')"
                        ui_parts_tool_exec = [{"type": "text", "content": tool_exec_text}]
                        current_history_list.append({"role": "assistant", "content": tool_exec_text, "ui_display_parts": ui_parts_tool_exec, "message_id": f"agent_call_start_{time.time_ns()}"})
                        
                        agent_message_to_send = Message(
                            role="user", 
                            parts=[TextPart(text=original_question_for_plan)],
                            metadata={
                                "conversation_id": active_conv_id, 
                                "target_agent_url": agent_url_target, 
                                "original_user_query": original_question_for_plan
                            }
                        )
                        
                        try:
                            returned_task_id: Optional[str] = asyncio.run(send_message_service(agent_message_to_send))
                            
                            if returned_task_id:
                                # This is the placeholder message that check_and_update_polling_tasks will replace
                                feedback_text = f"⏳ 에이전트 `{agent_url_target}` 작업 요청됨 (Task ID: {returned_task_id[:8]}...). 결과 확인 중..."
                                ui_parts_feedback = [{"type": "text", "content": feedback_text}]
                                current_history_list.append({"role": "assistant", "content": feedback_text, "ui_display_parts": ui_parts_feedback, "message_id": f"task_pending_{returned_task_id}_{time.time_ns()}"})
                                st.session_state.active_polling_tasks[returned_task_id] = original_question_for_plan
                                logger.info(f"Task {returned_task_id} (for Q: '{original_question_for_plan}') added to polling.")
                            else:
                                error_text = f"⚠️ 에이전트 `{agent_url_target}` 요청 후 Task ID를 받지 못했습니다." 
                                ui_parts_error = [{"type": "text", "content": error_text}]
                                current_history_list.append({"role": "assistant", "content": error_text, "ui_display_parts": ui_parts_error, "message_id": f"msg_err_notaskid_{time.time_ns()}"})
                                logger.error(f"Failed to get Task ID for agent {agent_url_target} and Q: '{original_question_for_plan}'. send_message_service returned None.")
                        except Exception as e:
                            logger.error(f"Error calling send_message_service for {agent_url_target}: {e}", exc_info=True)
                            error_text = f"⚠️ 에이전트 `{agent_url_target}` 호출 중 오류 발생: {str(e)[:100]}"
                            ui_parts_exception = [{"type": "text", "content": error_text}]
                            current_history_list.append({"role": "assistant", "content": error_text, "ui_display_parts": ui_parts_exception, "message_id": f"msg_err_send_{time.time_ns()}"})
            else: # User rejected plan
                logger.info(f"User rejected plan for Q: '{original_question_for_plan}'. Using fallback LLM.")
                fallback_response_text = asyncio.run(_ainvoke_llm_safely(original_question_for_plan))
                ui_parts_fallback = [{"type": "text", "content": fallback_response_text}]
                current_history_list.append({"role": "assistant", "content": fallback_response_text, "ui_display_parts": ui_parts_fallback, "message_id": f"msg_fallback_{time.time_ns()}"})
        
        else: # New user question, no pending plan
            logger.info("Processing new user input with orchestrator.")
            remote_agents = asyncio.run(list_remote_agents_service())
            available_urls = [card.url.rstrip("/") for card in remote_agents if card and card.url]
            
            if not available_urls and (st.session_state.get("chat_llm") is None):
                 warning_text = "현재 사용 가능한 원격 에이전트나 기본 LLM이 없어 요청을 처리할 수 없습니다."
                 ui_parts_warning = [{"type": "text", "content": warning_text}]
                 current_history_list.append({"role": "assistant", "content": warning_text, "ui_display_parts": ui_parts_warning, "message_id": f"msg_no_resource_{time.time_ns()}"})
            elif not available_urls: 
                logger.info("No remote agents available. Processing directly with default LLM.")
                direct_response_text = asyncio.run(_ainvoke_llm_safely(user_input_content))
                ui_parts_direct = [{"type": "text", "content": direct_response_text}]
                current_history_list.append({"role": "assistant", "content": direct_response_text, "ui_display_parts": ui_parts_direct, "message_id": f"msg_direct_llm_{time.time_ns()}"})
            else: # Orchestrate with available agents
                plan, approval_prompt_text = asyncio.run(
                    orchestrate_servers_async(user_input_content, available_urls)
                )
                ui_parts_approval = [{"type": "text", "content": approval_prompt_text}]
                current_history_list.append({"role": "assistant", "content": approval_prompt_text, "ui_display_parts": ui_parts_approval, "message_id": f"msg_approval_prompt_{time.time_ns()}"})
                if plan: # Orchestrator suggested a plan
                    st.session_state.pending_plan = plan
                    st.session_state.pending_input = user_input_content
                else: # Orchestrator provided a direct answer or no plan needed
                    logger.info("Orchestrator returned no actionable plan. Approval message is the response.")
    
    except Exception as e:
        logger.exception(f"Error in process_user_input_and_generate_response for input '{user_input_content}'")
        error_text = f"요청 처리 중 오류가 발생했습니다: {str(e)[:150]}"
        ui_parts_error_main = [{"type": "text", "content": error_text}]
        current_history_list.append({"role": "assistant", "content": error_text, "ui_display_parts": ui_parts_error_main, "message_id": f"msg_err_main_process_{time.time_ns()}"})
    finally:
        st.session_state.assistant_is_generating = False 
        logger.debug("process_user_input_and_generate_response finished. assistant_is_generating set to False.")


def render_conversation_page():
    st.title("💬 Agent-to-Agent Orchestrated Chat")

    if st.session_state.get("chat_llm") is None and st.session_state.get("chat_llm_initialized"):
        st.warning("OpenAI LLM 클라이언트를 초기화할 수 없습니다. API 키 설정을 확인하거나, 에이전트 기능만 사용 가능합니다.")

    logger.debug(f"[Page Render Start] Conv: {st.session_state.get('current_conv_id')}, "
                 f"Generating: {st.session_state.get('assistant_is_generating')}, "
                 f"Polling Tasks: {len(st.session_state.get('active_polling_tasks', {}))}, "
                 f"Last Input for Proc: {st.session_state.get('last_user_input_for_processing') is not None}")

    with st.sidebar:
        st.header("Conversations")
        if st.button("➕ 새 대화 시작", key="new_conversation_sidebar_button", use_container_width=True):
            logger.info("'새 대화 시작' 버튼 클릭됨 (사이드바).")
            try:
                new_conv_object = asyncio.run(create_conversation_service()) 
                if new_conv_object and new_conv_object.conversation_id:
                    new_id = new_conv_object.conversation_id
                    st.session_state.histories[new_id] = []
                    st.session_state.current_conv_id = new_id
                    st.session_state.app_state_messages_loaded_for_conv = None 
                    st.session_state.active_polling_tasks = {} 
                    st.session_state.completed_task_results = {}
                    st.session_state.assistant_is_generating = False
                    st.session_state.last_user_input_for_processing = None
                    st.session_state.pending_plan = None
                    st.session_state.pending_input = None
                    logger.info(f"New conversation '{new_id}' created and selected. Rerunning.")
                    st.rerun()
                else:
                    st.error("백엔드에서 새 대화 생성에 실패했습니다.")
                    logger.error("Failed to create new conversation or received invalid data from service.")
            except Exception as e: 
                st.error(f"새 대화 생성 중 오류: {str(e)[:100]}")
                logger.exception("Error during create_conversation_service call in sidebar.")


        # Ensure histories is a dict before trying to sort its keys
        histories_data = st.session_state.get("histories", {})
        if not isinstance(histories_data, dict): # Should not happen with initialize_session_state_if_needed
            logger.error(f"Session state 'histories' is not a dict: {type(histories_data)}. Resetting.")
            histories_data = {}
            st.session_state.histories = histories_data

        conv_ids_sorted = sorted(histories_data.keys(), reverse=True)

        for conv_id_key in conv_ids_sorted:
            history_for_label = histories_data.get(conv_id_key, [])
            label_content = history_for_label[0]['content'][:30] + "..." if history_for_label and history_for_label[0].get('content') else f"대화 {conv_id_key[:8]}..."
            
            is_current = (conv_id_key == st.session_state.get("current_conv_id"))
            button_type = "primary" if is_current else "secondary"
            
            if st.button(label_content, key=f"select_conv_button_{conv_id_key}", disabled=is_current, use_container_width=True, type=button_type):
                logger.info(f"Conversation '{conv_id_key}' selected from sidebar.")
                if st.session_state.current_conv_id != conv_id_key:
                    st.session_state.current_conv_id = conv_id_key
                    st.session_state.app_state_messages_loaded_for_conv = None 
                    # Reset task-related states for the new conversation
                    st.session_state.active_polling_tasks = {} 
                    st.session_state.completed_task_results = {}
                    st.session_state.assistant_is_generating = False
                    st.session_state.last_user_input_for_processing = None
                    st.session_state.pending_plan = None
                    st.session_state.pending_input = None
                    st.rerun()

    active_conv_id = st.session_state.get("current_conv_id")
    if not active_conv_id:
        st.info("👈 사이드바에서 대화를 선택하거나 '새 대화 시작' 버튼을 눌러주세요.")
        st.stop()

    # Polling logic
    if not st.session_state.assistant_is_generating and st.session_state.get("active_polling_tasks"):
        logger.debug(f"Polling check for conv: {active_conv_id}")
        try:
            history_updated_by_poll = asyncio.run(check_and_update_polling_tasks(active_conv_id))
            if history_updated_by_poll:
                logger.info("History updated by polling. Rerunning.")
                st.rerun()
        except RuntimeError as e: # Catch asyncio.run error if already in a loop
            if "cannot be called from a running event loop" in str(e):
                 logger.warning(f"Polling skipped for {active_conv_id} due to asyncio.run in existing loop.")
            else: # Other runtime errors
                logger.exception(f"Runtime error during polling for {active_conv_id}")


    history_refreshed_by_load = _load_and_prepare_chat_history(active_conv_id)
    if history_refreshed_by_load and \
       not st.session_state.assistant_is_generating and \
       not st.session_state.active_polling_tasks : 
        logger.info("History refreshed by initial load (no polling/generation). Rerunning.")
        st.rerun()
        
    active_chat_history = st.session_state.histories.get(active_conv_id, [])
    
    message_container = st.container()
    with message_container:
        for msg_idx, msg_entry in enumerate(active_chat_history):
            role = msg_entry.get("role", "unknown")
            avatar_icon = "🧑‍💻" if role == "user" else "🤖"
            
            with st.chat_message(name=role, avatar=avatar_icon):
                ui_parts_to_render = msg_entry.get("ui_display_parts")
                if ui_parts_to_render:
                    for part_idx, part_info in enumerate(ui_parts_to_render):
                        part_type = part_info.get("type")
                        content_data = part_info.get("data")
                        text_content = part_info.get("content", "")  

                        if part_type == "text":
                            st.markdown(text_content, unsafe_allow_html=True)
                        elif part_type == "tool_call":
                            tool_name = part_info.get("tool_name", "Unknown Tool")
                            st.markdown(text_content) # Content now includes summary
                            if content_data: 
                                with st.expander(f"실행 인자 보기 ({tool_name})", expanded=False):
                                    st.json(content_data)
                        elif part_type == "tool_response" or part_type == "tool_output":
                            tool_name = part_info.get("tool_name", "")
                            st.markdown(text_content) # Content is the header "Tool Output..."
                            if content_data: 
                                # Determine how to display content_data (JSON, string, etc.)
                                if isinstance(content_data, (dict, list)):
                                    with st.expander(f"상세 결과 보기 ({tool_name if tool_name else 'Output'})", expanded=True): # Expand by default for outputs
                                        st.json(content_data)
                                elif isinstance(content_data, str):
                                    # Try to parse if it looks like JSON, otherwise show as Markdown
                                    if content_data.strip().startswith(("{", "[")) and content_data.strip().endswith(("}", "]")):
                                        try:
                                            parsed_json = json.loads(content_data)
                                            with st.expander(f"상세 결과 보기 (JSON - {tool_name if tool_name else 'Output'})", expanded=True):
                                                st.json(parsed_json)
                                        except json.JSONDecodeError:
                                            st.markdown(content_data, unsafe_allow_html=True) 
                                    else:
                                        st.markdown(content_data, unsafe_allow_html=True)
                                else: # Fallback for other data types
                                    st.markdown(str(content_data), unsafe_allow_html=True)
                        elif part_type == "data": # Generic data part
                            data_name = part_info.get('name', 'Data')
                            st.markdown(text_content) # Content is "Data: {name}"
                            if content_data:
                                with st.expander(f"{data_name} 내용 보기", expanded=False):
                                    st.json(content_data) # Assuming JSON, adjust if other data types
                        else: # Unknown part type
                            st.markdown(text_content if text_content else str(part_info), unsafe_allow_html=True)
                else: # No ui_display_parts, just use content
                    st.markdown(msg_entry.get("content", ""), unsafe_allow_html=True)

        if st.session_state.assistant_is_generating:
            with st.chat_message("assistant", avatar="🤖"):
                # Use a spinner that's less intrusive or just a text message
                st.markdown("답변을 생성 중이거나 에이전트 응답을 기다리고 있습니다...")


    user_chat_input = st.chat_input(
        "메시지를 입력하거나, 제안에 대해 '예' 또는 '아니오'로 답해주세요...", 
        key=f"chat_input_for_conv_{active_conv_id}", 
        disabled=st.session_state.assistant_is_generating # Disable input while assistant is "thinking" or processing
    )

    if user_chat_input and not st.session_state.assistant_is_generating:
        logger.info(f"User input captured: '{user_chat_input}' for conv '{active_conv_id}'")
        user_message_entry = {
            "role": "user", 
            "content": user_chat_input, 
            "message_id": f"user_{int(time.time_ns())}", 
            "ui_display_parts": [{"type": "text", "content": user_chat_input}]
        }
        st.session_state.histories.setdefault(active_conv_id, []).append(user_message_entry)
        
        st.session_state.assistant_is_generating = True # Set flag
        st.session_state.last_user_input_for_processing = user_chat_input # Store input
        
        st.rerun() # Rerun to show user message and "generating" state

    # This block now runs after the rerun triggered by user input
    if st.session_state.assistant_is_generating and st.session_state.last_user_input_for_processing:
        input_to_process = st.session_state.last_user_input_for_processing
        st.session_state.last_user_input_for_processing = None # Clear after getting it
        
        logger.info(f"Calling process_user_input_and_generate_response for: '{input_to_process}'")
        
        # This function will set assistant_is_generating to False internally upon completion
        process_user_input_and_generate_response(active_conv_id, input_to_process)
        
        logger.info(f"Finished process_user_input_and_generate_response. Rerunning to display results.")
        st.rerun() # Rerun to display assistant's response


if __name__ == "__main__":
    logger.info(f"{'='*10} Conversation Page Script START {'='*10}")
    render_conversation_page()
    # logger.debug(f"{'-'*10} Conversation Page Script END for this run {'-'*10}\n")

