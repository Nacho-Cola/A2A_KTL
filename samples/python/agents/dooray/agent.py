import asyncio 
import base64 
import json 
import uuid 
import subprocess
import logging # logging 추가

from collections.abc import AsyncIterable
from typing import Any, Literal, Dict, List, Optional # Dict, List, Optional 추가
from dotenv import load_dotenv


import httpx 

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage # HumanMessage 추가
from langchain_core.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError # ValidationError 추가

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


from langchain_mcp_adapters.client import MultiServerMCPClient 
from langchain_community.tools.tavily_search import TavilySearchResults 

load_dotenv() 
memory = MemorySaver()
logger = logging.getLogger(__name__) # 모듈 레벨 로거 정의


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str

class DoorayAgent:
  SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
  SYSTEM_INSTRUCTION = ( """<ROLE>
You are a smart agent with an ability to use tools.
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question.
If you fail to answer the question with your initial knowledge, use available tools to gather context and update your answer.
Your answer should be very polite and professional

You are a Dooray API assistant. You have access to a single tool called ‘Dooray’ which lets you query or modify Dooray via its official REST API.
휴가 및 연차 = "하루 종일" and "참여자" : user
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.
- Identify if the question requires up-to-date information, external context, or further verification.
- If yes, use the most relevant tool such as the search tool to gather additional data.
- If you have multiple relevant tools, pick the one that best fits the question’s context.

Step 3: Pick the most relevant tool
- Based on the analysis, select the most appropriate tool (e.g., search, calculators, etc.) to answer the question.
- If the tool's output provides useful context, incorporate it into your answer.

Step 4: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 5: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).
- Answer the question in the same language as it was asked.
- Your answer should be clear, concise, and professional.
- If you have gathered additional information with a tool, include it to improve the answer quality.


Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source. 


Example:
1. Analyze the user’s request.
2. If the request involves anything to do with projects, members, messages, channels, etc., use the ‘Dooray’ tool to call the appropriate endpoint.
3. Do NOT use any web‐search or external search tools for Dooray‑related queries.
4. Construct your tool call parameters (e.g. name, project‑id, message text) based on the user’s intent.
5. Return the tool’s JSON response as your answer, formatted as human‑readable text or JSON as appropriate.
6. If the user’s request is unrelated to Dooray, explain that you can only handle Dooray API operations.
7. To identify a person by name:
    a. Call get_members_by_name(name) to search members.  
    b. Select the appropriate member from the results and extract their member_id.  
    c. Call get_member_detail(member_id) to retrieve details (e.g. employee number).  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)
<concise answer to the user question based on tool output>
**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""
)

  def __init__(
    self, model_type:str = "claude-3-opus-20240229", use_mcp: bool = False, mcp_config: Optional[dict] = None
  ):
    self.model_type = model_type
    self.use_mcp = use_mcp
    self.mcp_config = mcp_config
    self.model = None
    self.tools: List[Any] = [] 
    self.graph = None 
    self.client: Optional[MultiServerMCPClient] = None 
    self.output_token_info = self._initialize_output_token_info()
    logger.info(f"DoorayAgent initialized with model_type: {model_type}, use_mcp: {use_mcp}")


  async def initialize(self):
    logger.info("Initializing DoorayAgent...")
    if self.use_mcp and self.mcp_config: 
      try:
        self.client = MultiServerMCPClient(self.mcp_config)
        await self.client.__aenter__() 
        self.tools = self.client.get_tools()
        logger.info(f"MCP client initialized with {len(self.tools)} tools.")
      except Exception as e:
        logger.error(f"Failed to initialize MultiServerMCPClient: {e}", exc_info=True)
        self.client = None 
        self.tools = [] 
    
    try:
        self.model = self._load_llm(self.model_type)
        logger.info(f"LLM model '{self.model_type}' loaded.")
    except Exception as e:
        logger.error(f"Failed to load LLM model '{self.model_type}': {e}", exc_info=True)
        self.model = None 
        return 

    try:
        self.tools.append(TavilySearchResults(max_results=2))
        logger.info("TavilySearchResults tool added.")
    except Exception as e:
        logger.error(f"Failed to initialize TavilySearchResults: {e}", exc_info=True)


    if not self.tools: 
        logger.warning("No tools configured for DoorayAgent. Only LLM's direct capabilities will be available.")

    if self.model and self.tools: 
        try:
            self.graph = create_react_agent(
              self.model,
              tools=self.tools,
              checkpointer=memory,
              prompt=self.SYSTEM_INSTRUCTION,
            )
            logger.info("LangGraph ReAct agent graph initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph ReAct agent: {e}", exc_info=True)
            self.graph = None
    else:
        logger.error("Cannot initialize LangGraph ReAct agent due to missing model or tools.")
        self.graph = None


  def _load_llm(self, model_name: str):
    token_limit = self.output_token_info.get(model_name, {}).get("max_tokens", 4096) 
    logger.debug(f"Loading LLM: {model_name} with token_limit: {token_limit}")

    if model_name.startswith("claude"):
      return ChatAnthropic(model_name=model_name, temperature=0.1, max_tokens=token_limit)
    elif model_name.startswith("gpt"):
      return ChatOpenAI(model_name=model_name, temperature=0.1, max_tokens=token_limit) 
    elif model_name.startswith("gemini"): 
      return ChatGoogleGenerativeAI(model=model_name, temperature=0.1, convert_system_message_to_human=True)
    elif model_name == "ollama-auto":
      selected_model = self._get_ollama_model_or_default()
      logger.info(f"[Ollama 자동 선택] 모델: {selected_model}")
      return ChatOllama(model=selected_model, temperature=0.1) 
    else: 
      return ChatOllama(model=model_name, temperature=0.1)


  def _initialize_output_token_info(self) -> dict:
    output_info = {
      "claude-3-opus-20240229": {"max_tokens": 4096}, 
      "claude-3-sonnet-20240229": {"max_tokens": 4096},
      "claude-3-haiku-20240307": {"max_tokens": 4096},
      "claude-3-5-sonnet-20240620": {"max_tokens": 8192},
      "gpt-4o": {"max_tokens": 4096}, # Output token limit for GPT-4o is often 4096
      "gpt-4o-mini": {"max_tokens": 4096},
      "gemini-1.5-pro-latest": {"max_tokens": 8192}, 
      "gemini-1.5-flash-latest": {"max_tokens": 8192},
    }
    for model_name_prefix in ["llama3", "qwen", "mistral", "gemma"]: 
        for model_variation in ["", ":7b", ":8b", ":instruct"]: 
            model = f"{model_name_prefix}{model_variation}"
            if model not in output_info:
                 output_info[model] = {"max_tokens": 4096}
    return output_info


  def _get_all_ollama_models(self) -> list[str]:
    try:
      result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False) 
      if result.returncode != 0:
          logger.error(f"[Ollama 오류] 'ollama list' 실행 실패: {result.stderr}")
          return []
      output = result.stdout
      lines = output.strip().split("\n")
      if lines and "NAME" in lines[0].upper():
          lines = lines[1:] 
      models = [line.split()[0] for line in lines if line.strip()]
      logger.info(f"[Ollama 설치 모델]: {models}")
      return models
    except FileNotFoundError:
      logger.error("[Ollama 오류] 'ollama' 명령을 찾을 수 없습니다. PATH를 확인해주세요.")
      return []
    except Exception as e:
      logger.error(f"[Ollama 오류] 모델 목록을 불러올 수 없습니다: {e}", exc_info=True)
      return []


  def _get_ollama_model_or_default(self, default: str = "llama3") -> str:
    models = self._get_all_ollama_models()
    return models[0] if models else default


  def invoke(self, query: str, sessionId: str) -> dict[str, Any]:
    if not self.graph:
        logger.error("DoorayAgent graph not initialized. Call await agent.initialize() first.")
        return {
            'type': 'error', 'status': 'error',
            'is_task_complete': True, 
            'require_user_input': False,
            'content': '에이전트가 초기화되지 않았습니다. 관리자에게 문의하세요.',
        }
        
    config = {'configurable': {'thread_id': sessionId}}
    inputs = {'messages': [HumanMessage(content=query)]} 
    
    try:
        logger.debug(f"Invoking DoorayAgent graph with query: '{query[:50]}...' for session: {sessionId}")
        graph_output_dict = self.graph.invoke(inputs, config)
        logger.debug(f"DoorayAgent graph invoked. Output keys: {graph_output_dict.keys() if isinstance(graph_output_dict, dict) else 'Not a dict'}")
        return self.get_agent_response(config) 
    except Exception as e:
        logger.error(f"Error invoking DoorayAgent graph: {e}", exc_info=True)
        return {
            'type': 'error', 'status': 'error',
            'is_task_complete': True, 
            'require_user_input': False,
            'content': f'요청 처리 중 오류 발생: {str(e)[:100]}',
        }

  def _generate_tool_call_message(self, tool_name: str, tool_args: dict) -> str:
    description = ""
    if self.use_mcp and self.client and hasattr(self.client, "tool_registry") and self.client.tool_registry:
        mcp_tool = self.client.tool_registry.tools.get(tool_name)
        if mcp_tool and hasattr(mcp_tool, 'description'):
            description = mcp_tool.description
    
    if not description: 
        for t in self.tools: 
            if hasattr(t, 'name') and t.name == tool_name and hasattr(t, 'description'):
                description = t.description
                break
    
    try:
        args_str = json.dumps(tool_args, ensure_ascii=False, indent=2)
    except TypeError:
        args_str = str(tool_args) 

    if description:
        return f"📌 **Tool Call:** `{tool_name}`\n   - 설명: {description}\n   - 인자: \n```json\n{args_str}\n```"
    else:
        return f"🔧 **Tool Call:** `{tool_name}`\n   - 인자: \n```json\n{args_str}\n```"


  async def stream(self, query: str, sessionId: str) -> AsyncIterable[dict[str, Any]]:
    if not self.graph:
        logger.error("DoorayAgent graph not initialized for streaming. Call await agent.initialize() first.")
        yield {
            'type': 'error', 'content': '에이전트가 초기화되지 않았습니다.',
            'is_task_complete': True, 'require_user_input': False, 'status': 'error'
        }
        return

    inputs = {'messages': [HumanMessage(content=query)]} 
    config = {'configurable': {'thread_id': sessionId}}

    try:
      logger.debug(f"Streaming DoorayAgent graph with query: '{query[:50]}...' for session: {sessionId}")
      async for item in self.graph.astream(inputs, config, stream_mode='values'):
        messages = item.get('messages', [])
        if not messages:
            logger.debug(f"Stream item has no 'messages': {item}")
            continue

        last_message = messages[-1]
        logger.debug(f"Stream item - last_message type: {type(last_message)}, content snippet: {str(getattr(last_message, 'content', 'N/A'))[:70]}")

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
          tool_call = last_message.tool_calls[0] 
          tool_name = tool_call.get('name') 
          tool_args = tool_call.get('args', {})
          if not tool_name:
              logger.warning(f"Tool call in AIMessage missing name: {tool_call}")
              continue
          logger.info(f"Streaming tool_call: {tool_name} with args: {tool_args}")
          yield {
            'type': 'tool_call', 
            'tool_name': tool_name,
            'tool_args': tool_args,
            'content': self._generate_tool_call_message(tool_name, tool_args), 
            'is_task_complete': False, 'require_user_input': False, 'status': 'working'
          }
        
        elif isinstance(last_message, ToolMessage):
          tool_name = getattr(last_message, "name", None) 
          if not tool_name and hasattr(last_message, 'tool_call_id'):
              tool_name = "결과" # 임시 이름, 실제로는 tool_call_id로 원래 툴 이름 추적 필요
          elif not tool_name:
              tool_name = "알 수 없는 도구"

          logger.info(f"Streaming tool_response for: {tool_name}. Result snippet: {str(last_message.content)[:100]}...")
          yield {
            'type': 'tool_response',
            'tool_name': tool_name,
            'tool_result': last_message.content, 
            'content': f"⚙️ **Tool Output from** `{tool_name}` (Details below)", 
            'is_task_complete': False, 'require_user_input': False, 'status': 'working'
          }
        elif isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls: 
            logger.info(f"Streaming interim/final AIMessage content: {str(last_message.content)[:100]}...")
            yield {
                'type': 'interim_message', 
                'content': str(last_message.content),
                'is_task_complete': False, 
                'require_user_input': False, 'status': 'working'
            }
        elif isinstance(last_message, HumanMessage):
            logger.debug(f"Streaming HumanMessage (likely original input): {str(last_message.content)[:100]}...")
        else:
            logger.debug(f"Streaming other message type or AIMessage without content: {type(last_message)}")


      logger.info(f"Graph stream finished for session {sessionId}. Generating final response from agent state.")
      final_response_dict = self.get_agent_response(config) 
      logger.info(f"Final response for session {sessionId}: {final_response_dict}")
      
      yield {
          'type': 'final_response', 
          'content': final_response_dict.get('content', '최종 응답을 생성하지 못했습니다.'),
          'is_task_complete': final_response_dict.get('is_task_complete', True),
          'require_user_input': final_response_dict.get('require_user_input', False),
          'status': final_response_dict.get('status', 'completed' if final_response_dict.get('is_task_complete') else 'error')
      }
    except Exception as e:
        logger.error(f"Error during DoorayAgent streaming for session {sessionId}: {e}", exc_info=True)
        yield {
            'type': 'error', 'content': f'스트리밍 중 오류 발생: {str(e)[:100]}',
            'is_task_complete': True, 'require_user_input': False, 'status': 'error'
        }


  def get_agent_response(self, config: dict) -> dict[str, Any]: 
    """LangGraph의 최종 상태에서 ResponseFormat에 맞는 응답을 추출하거나, 마지막 AIMessage를 사용합니다."""
    try:
        current_state_snapshot = self.graph.get_state(config)
        if not current_state_snapshot or not current_state_snapshot.values: 
            logger.warning(f"Graph state is empty for thread_id {config.get('configurable', {}).get('thread_id', 'N/A')}.")
            return {
                'is_task_complete': True, 'require_user_input': False, 
                'content': '에이전트가 응답을 생성하지 못했습니다 (상태 비어있음).', 'status': 'error'
            }
    except Exception as e:
        logger.error(f"Error getting graph state for config {config.get('configurable', {}).get('thread_id', 'N/A')}: {e}", exc_info=True)
        return {
            'is_task_complete': True, 'require_user_input': False, 
            'content': '에이전트 상태 조회 중 오류가 발생했습니다.', 'status': 'error'
        }
    
    final_messages = current_state_snapshot.values.get('messages', [])
    final_ai_message_content: Optional[str] = None

    if final_messages and isinstance(final_messages[-1], AIMessage):
        final_ai_message = final_messages[-1]
        if not final_ai_message.tool_calls and isinstance(final_ai_message.content, str) and final_ai_message.content.strip():
            final_ai_message_content = final_ai_message.content.strip()
            logger.info(f"Attempting to use final AIMessage content: {final_ai_message_content[:100]}...")

    if final_ai_message_content:
        try:
            # LLM이 ResponseFormat 형식의 JSON을 출력했는지 확인
            # SYSTEM_INSTRUCTION에 OUTPUT_FORMAT이 명시되어 있으므로, LLM이 이를 따르도록 유도됨.
            # 하지만 항상 JSON.loads 가능한 형태로 출력된다는 보장은 없음.
            # 먼저 ResponseFormat으로 직접 파싱 시도 (LLM이 정확한 JSON 객체 문자열을 반환해야 함)
            try:
                parsed_json = json.loads(final_ai_message_content)
                # "status"와 "message" 키가 있는지 확인하여 ResponseFormat과 유사한지 검사
                if "status" in parsed_json and "message" in parsed_json:
                    structured_response = ResponseFormat.model_validate(parsed_json)
                    logger.info(f"Successfully parsed final AIMessage content into ResponseFormat: {structured_response}")
                    return {
                        'is_task_complete': structured_response.status == 'completed',
                        'require_user_input': structured_response.status == 'input_required',
                        'content': structured_response.message,
                        'status': structured_response.status
                    }
                else: # 필수 키가 없는 단순 JSON
                    logger.warning(f"Final AIMessage content is JSON but not ResponseFormat: {final_ai_message_content[:200]}")
                    # 이 경우, JSON을 문자열로 다시 사용하거나, 다른 방식으로 처리
            except json.JSONDecodeError: # JSON 파싱 실패 시, 일반 텍스트로 간주
                logger.warning(f"Final AIMessage content is not valid JSON. Treating as plain text: {final_ai_message_content[:200]}")
            
            # ResponseFormat으로 파싱되지 않았거나, JSON이 아니면 일반 텍스트로 처리
            return {
                'is_task_complete': True, 
                'require_user_input': False, 
                'content': final_ai_message_content, # 원본 content 사용
                'status': 'completed' 
            }
        except ValidationError as ve: # Pydantic 유효성 검사 오류
             logger.warning(f"Could not validate final AIMessage content as ResponseFormat. Content: {final_ai_message_content[:200]}... Error: {ve}. Using raw content.")
             return {
                'is_task_complete': True, 
                'require_user_input': False,
                'content': final_ai_message_content,
                'status': 'completed' 
            }


    logger.warning(f"Could not determine a structured (ResponseFormat) or direct final AIMessage from graph state for thread_id {config.get('configurable', {}).get('thread_id', 'N/A')}. State messages: {final_messages}")
    return {
        'is_task_complete': True, 
        'require_user_input': False, 
        'content': '요청을 처리했지만, 명확한 답변을 생성하지 못했습니다. 다시 시도해주세요.',
        'status': 'error'
    }