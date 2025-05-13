from collections.abc import AsyncIterable
from typing import Any, Literal
from dotenv import load_dotenv
import subprocess

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.tools.tavily_search import TavilySearchResults 

memory = MemorySaver()


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
    self, model_type:str = "claude-3-7-sonnet-latest", use_mcp: bool = False, mcp_config: dict = None
  ):
    """
      model_types:[
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "gpt-4o",
        "gpt-4o-mini",
        ...]
    """
    self.model_type = model_type
    self.use_mcp = use_mcp
    self.mcp_config = mcp_config
    self.model = None
    self.tools = []
    self.graph = None 
    self.client = None  
    self.output_token_info = self._initialize_output_token_info()
    response_format=ResponseFormat,


  async def initialize(self):
    if self.use_mcp :
      self.client = MultiServerMCPClient(self.mcp_config)
      await self.client.__aenter__()
      self.tools = self.client.get_tools()
    else:
      self.tools = [get_exchange_rate]

    self.model = self._load_llm(self.model_type)
    # 검색 Tool
    self.tools.append(TavilySearchResults(max_results=2))

    self.graph = create_react_agent(
      self.model,
      tools=self.tools,
      checkpointer=memory,
      prompt=self.SYSTEM_INSTRUCTION,
      response_format=ResponseFormat,
    )


  def _load_llm(self, model_name: str):
    token_limit = self.output_token_info.get(model_name, {}).get("max_tokens", 64000)

    if model_name.startswith("claude"):
      return ChatAnthropic(model=model_name, temperature=0.1, max_tokens=token_limit)
    elif model_name.startswith("gpt"):
      return ChatOpenAI(model=model_name, temperature=0.1, max_tokens=token_limit)
    elif model_name == "gemini":
      return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    elif model_name == "ollama-auto":
      selected_model = self._get_ollama_model_or_default()
      print(f"[Ollama 자동 선택] 모델: {selected_model}")
      return ChatOllama(model=selected_model, temperature=0.1, max_tokens=token_limit)
    else:
      return ChatOllama(model=model_name, temperature=0.1, max_tokens=token_limit)


  def _initialize_output_token_info(self) -> dict:
    output_info = {
      "claude-3-5-sonnet-latest": {"max_tokens": 8192},
      "claude-3-5-haiku-latest": {"max_tokens": 8192},
      "claude-3-7-sonnet-latest": {"max_tokens": 64000},
      "gpt-4o": {"max_tokens": 16000},
      "gpt-4o-mini": {"max_tokens": 16000},
      "gemini": {"max_tokens": 8192},
    }

    for model in self._get_all_ollama_models():
      if model not in output_info:
        output_info[model] = {"max_tokens": 64000}
    return output_info


  def _get_all_ollama_models(self) -> list[str]:
    try:
      output = subprocess.check_output(["ollama", "list"], universal_newlines=True)
      lines = output.strip().split("\n")
      if lines and "NAME" in lines[0].upper():
          lines = lines[1:]
      models = [line.split()[0] for line in lines if line.strip()]
      print(f"[Ollama 설치 모델]: {models}")
      return models
    except Exception as e:
      print(f"[Ollama 오류] 모델 목록을 불러올 수 없습니다: {e}")
      return []


  def _get_ollama_model_or_default(self, default: str = "llama3") -> str:
    """
    설치된 모델 중 첫 번째를 선택하거나, 없으면 기본값을 반환합니다.
    """
    models = self._get_all_ollama_models()
    return models[0] if models else default

\
  def invoke(self, query, sessionId) -> dict[str, Any]:
    """
    사용자의 쿼리를 입력받아 즉시 응답을 반환합니다.
    """
    config = {'configurable': {'thread_id': sessionId}}
    result = self.graph.invoke({'messages': [('user', query)]}, config)
    return self.get_agent_response(config)


  def _generate_tool_call_message(self, tool_name: str) -> str:
    # MCP에 등록된 도구 설명을 가져오기
    tool = getattr(self.client, "tool_registry", None)
    if tool and tool_name in tool.tools:
        description = tool.tools[tool_name].description
        return f"📌 도구 '{tool_name}' 실행 중: {description}"
    else:
        return f"🔧 '{tool_name}' 도구를 실행 중입니다..."



  async def stream(self, query, sessionId) -> AsyncIterable[dict[str, Any]]:
    """
    스트리밍 방식으로 응답을 순차적으로 반환합니다.
    """
    inputs = {'messages': [('user', query)]}
    config = {'configurable': {'thread_id': sessionId}}

    async for item in self.graph.astream(inputs, config, stream_mode='values'):
      message = item['messages'][-1]

      if isinstance(message, AIMessage) and message.tool_calls:
        tool_name = message.tool_calls[0]['name']  # MCP 또는 일반 tool name
        content = self._generate_tool_call_message(tool_name)
        yield {
          'is_task_complete': False,
          'require_user_input': False,
          'content': content
        }
      
      elif isinstance(message, ToolMessage):
        tool_name = getattr(message, "tool_name", "알 수 없는 도구")
        yield {
          'is_task_complete': False,
          'require_user_input': False,
          'content': f"🛠 '{tool_name}' 결과를 처리 중입니다...",
        }

    # 최종 응답 생성
    final = self.get_agent_response(config)
    yield final


  def get_agent_response(self, config): 
    current_state = self.graph.get_state(config)
    structured_response = current_state.values.get('structured_response')

    if structured_response and isinstance(structured_response, ResponseFormat):
        if structured_response.status in ['input_required', 'error']:
            return {
                'is_task_complete': False,
                'require_user_input': True,
                'content': structured_response.message,
            }
        elif structured_response.status == 'completed':
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': structured_response.message,
            }

    # fallback
    return {
        'is_task_complete': False,
        'require_user_input': True,
        'content': 'We are unable to process your request at the moment. Please try again',
    }


