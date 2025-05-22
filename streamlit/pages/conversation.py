# pages/Conversations.py

import streamlit as st
st.set_page_config(page_title="💬 Conversations", layout="wide")

import asyncio
import json
from common.types import Message, TextPart
from state.host_agent_service import FetchAppState, ListRemoteAgents
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from service.client.client import ConversationClient
from service.types import SendMessageRequest
from dotenv import load_dotenv

load_dotenv(override=True)

def orchestrate_servers(question: str, servers: list[str]) -> tuple[list[str], str]:
    """
    LLM에게 plan 과 prompt 를 자연어로 뽑아오게 한 뒤, 
    'PLAN:' 라인과 'PROMPT:' 라인을 파싱합니다.
    """
    # 1) Orchestrator에게 묻기
    prompt = (
        "당신은 A2A Orchestrator입니다.\n"
        "이 질문을 처리하기 위해 호출할 서버 목록(plan)과, "
        "그 계획을 사용자에게 승인 요청할 메시지(prompt)를 다음 형식으로 출력하세요:\n\n"
        "PLAN: http://서버1, http://서버2\n"
        "PROMPT: 사용자에게 보여줄 자연어 승인 요청 문장\n\n"
        f"질문: {question}\n"
        f"서버 목록: {', '.join(servers)}\n"
    )
    raw = st.session_state.chat_llm.invoke([HumanMessage(content=prompt)]).content

    # 2) PLAN/PROMPT 라인 파싱
    plan = []
    user_prompt = "승인하시겠습니까? (예/아니오)"
    for line in raw.splitlines():
        if line.upper().startswith("PLAN:"):
            # 쉼표로 분리, 공백 제거
            plan = [u.strip().rstrip("/") for u in line[len("PLAN:"):].split(",") if u.strip()]
        elif line.upper().startswith("PROMPT:"):
            user_prompt = line[len("PROMPT:"):].strip()

    # 디버그
    st.text(f"▶ Orchestrator raw:\n{raw}")
    st.text(f"▶ Parsed plan: {plan}")
    st.text(f"▶ Parsed prompt: {user_prompt}")

    return plan, user_prompt

# ── 세션 스토어 초기화 ──
if "chat_llm" not in st.session_state:
    st.session_state.chat_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
if "histories" not in st.session_state:
    st.session_state.histories = {}
if "current_conv" not in st.session_state:
    st.session_state.current_conv = None
# 승인 대기 상태 저장
if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

@st.cache_data(show_spinner=False)
def load_conversation_state(conv_id: str):
    return asyncio.run(FetchAppState(conv_id))


def main():
    st.title("💬 Agent-to-Agent Chat")

    # ── 사이드바: 세션 목록 & 선택 ──
    with st.sidebar:
        st.header("세션 목록")
        if st.button("➕ 새 대화"):
            new_id = f"conv_{len(st.session_state.histories)+1}"
            st.session_state.histories[new_id] = []
            st.session_state.current_conv = new_id

        for cid in st.session_state.histories:
            if cid == st.session_state.current_conv:
                st.button(cid, disabled=True, use_container_width=True)
            else:
                if st.button(cid, key=f"btn_{cid}", use_container_width=True):
                    st.session_state.current_conv = cid

    conv_id = st.session_state.current_conv
    if not conv_id:
        st.error("❗ 세션을 선택하거나 ‘새 대화’를 눌러주세요.")
        return

    # ── 히스토리 로드 & 렌더링 ──
    history = st.session_state.histories.setdefault(conv_id, [])
    if not history:
        state = load_conversation_state(conv_id)
        for m in state.messages:
            txt = "".join(p for p, mime in m.content if mime=="text/plain")
            history.append({"role": m.role, "content": txt})
    for e in history:
        av = "🧑‍💻" if e["role"]=="user" else "🤖"
        with st.chat_message(e["role"], avatar=av):
            st.markdown(e["content"])

    # ── 사용자 입력 ──
    user_input = st.chat_input("메시지를 입력하세요…")
    if not user_input:
        return

    # ── (2단계) 승인을 기다리는 중? ──
    if st.session_state.pending_plan is not None:
        # 유저가 “예/아니오” 대답
        history.append({"role":"user","content":user_input})
        st.chat_message("user", avatar="🧑‍💻").markdown(user_input)

        plan = st.session_state.pending_plan
        original = st.session_state.pending_input

        if "예" in user_input or "yes" in user_input.lower():
            # ── 승인 후, plan 대로 에이전트 호출 ──
            for addr in plan:
                try:
                    # 1) “Tool: A2A@{addr}” 표시
                    st.chat_message("assistant", avatar="🤖").markdown(f"💡 Tool: A2A@{addr}")

                    # 2) Message 생성 시 metadata에 agent_url 추가
                    msg = Message(
                        role="user",
                        parts=[TextPart(text=st.session_state.pending_input)],
                        metadata={
                            "conversation_id": conv_id,
                            "agent_url": addr,          
                        }
                    )


                    resp = asyncio.run(SendMessage(msg))

                    # 4) 결과 표시
                    result = getattr(resp, "result", "<No result>")
                    st.chat_message("assistant", avatar="🤖").markdown(result)
                    history.append({"role": "assistant", "content": result})

                except Exception as e:
                    st.error(f"⛔ {addr} 연결 실패: {e}")

        else:
            # ── 거부: 기본 LLM 처리 ──
            fallback = st.session_state.chat_llm.invoke([HumanMessage(content=original)]).content
            st.chat_message("assistant", avatar="🤖").markdown(fallback)
            history.append({"role":"assistant","content":fallback})

        # 승인 절차 초기화
        st.session_state.pending_plan = None
        st.session_state.pending_input = None
        return

    # ── (1단계) 일반 질문 처리 ──
    history.append({"role":"user","content":user_input})
    st.chat_message("user", avatar="🧑‍💻").markdown(user_input)

    remotes = asyncio.run(ListRemoteAgents()) or []
    servers = [a.url.rstrip("/") for a in remotes if getattr(a,"url",None)]

    # 첫 호출: plan & prompt 생성
    plan, prompt = orchestrate_servers(user_input, servers)
    st.chat_message("assistant", avatar="🤖").markdown(prompt)
    history.append({"role":"assistant","content":prompt})

    # 대기 상태 저장 (두 번째 입력을 승인/거부로 처리)
    st.session_state.pending_plan = plan
    st.session_state.pending_input = user_input


if __name__ == "__main__":
    main()
