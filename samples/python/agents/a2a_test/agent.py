from typing import Any, AsyncIterable, Dict
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent

class DockerEvalAgent:
  """An agent that handle Docker evaluation(image build, run, remove) requests."""

  SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

  def __init__(self):
    self._agent = self._build_agent()
    self._user_id = "remote_agent"

    self._runner = Runner(
      app_name = self._agent.name,
      agent = self._agent,
      artifact_service = InMemoryArtifactService(), 
      session_service = InMemorySessionService(),
      memory_service = InMemoryMemoryService(),
    )
  
  def _build_agent(self) -> LlmAgent:
    print("DockerEvalAgent : _build_agent")
    """Build the LLM agent for the docker evaluation agent."""
    GEMINI_MODEL = "gemini-1.5-flash-001"

    code_writer_agent = LlmAgent(
      name = "DockerWriterAgent",
      model = GEMINI_MODEL,
      instruction="""You are a AI agent for Docker command Automation.
      Based on the user's request, write the initial Docker command.
      Output *only* the raw code block.""",
      description="Writes inital code based on a specification.",
      output_key="generated_Docker_command"
    )


    code_reviewer_agent = LlmAgent(
      name = "DockerReviewerAgent",
      model = GEMINI_MODEL,
      instruction = """You are a Docker command Reviewer AI.
      Review the Docker command provided in the session state under the key 'generated_Docker_command'.
      Provide constructive feedback on potential errors, style issues, or improvements.
      Focus on clarity and correctness.
      Output only the review comments.
      """,
      description="Reviews code and provides feedback.",
      output_key="review_Docker_command"
    )


    code_refactorer_agent = LlmAgent(
      name = "DockerRefactorerAgent",
      model = GEMINI_MODEL,
      instruction="""You are a Docker command Refactorer AI.
      Take the original Docker command provided in the session state key 'generated_Docker_command'
      and the review comments found in the session state key 'review_Docker_command'.
      Refactor the original Docker command to address the feedback and improve its quality.
      Output *only* the final, refactored Docker command block.
      """,
      description="Refactors code based on review comments.",
      
      output_key="refactored_Docker_command"
    )

    return SequentialAgent(
    name="DockerPipelineAgent",
    sub_agents=[
        code_writer_agent, 
        code_reviewer_agent, 
        code_refactorer_agent
      ]
    )
  
  def invoke(selef, qeury:str, session_id:str) -> str:
    print("DockerEvalAgent : invoke")
    session = self._runner.session_service.get_session(
      app_name=self._agent.name, user_id=self._user_id, session_id=self.session_id
    )
    content = types.Content(
      role="user", parts=[types.Part.from_text(text=query)]
    )
    if session is None:
      sesseion = self._runner.session_service.create_session(
        app_name=self._agent.name,
        user_id=self._user_id,
        state={},
        session_id=session_id,
      )
    events = list(self._runner.run(
      user_id=self._user_id, session_id=session.id, new_message=content
    ))

    if not events or not events[-1].countent or not events[-1].content.parts:
      return ""
    return "\n".join([p.text for p in events[-1].content.parts if p.text])



  async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
    print("CodeEvalAgent: stream")
    session = self._runner.session_service.get_session(
      app_name=self._agent.name, user_id=self._user_id, session_id=session_id
    )
    content = types.Content(
      role="user", parts=[types.Part.from_text(text=query)]
    )
    
    if  session is None:
      session = self._runner.session_service.create_session(
        app_name=self._agent.name,
        user_id=self._user_id,
        state={},
        session_id=session_id,
      )
      
    async for event in self._runner.run_async(
      user_id=self._user_id, session_id=session.id, new_message=content
    ):
      if event.is_final_response():
        response = ""
        
        if (
          event.content
          and event.content.parts
          and event.content.parts[0].text
        ):
          response = "\n".join([p.text for p in event.content.parts if p.text])
        
        elif (
          event.content
          and event.content.parts
          and any([True for p in event.content.parts if p.function_response])
          ):
            response = next((p.function_response.model_dump() for p in event.content.parts))
          
        yield {
          "is_task_complete": True,
          "content": response,
        }
          
      else:
        yield {
          "is_task_complete": False,  
          "updates": "Processing the code evaluation request...",
        }












