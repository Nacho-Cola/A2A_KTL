import os
import logging
import click

import asyncio
import nest_asyncio
nest_asyncio.apply()



from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from common.utils.push_notification_auth import PushNotificationSenderAuth
from common.utils.config_loader import load_config_from_json


from agents.docker_server.agent import DockerAgent
from agents.docker_server.task_manager import AgentTaskManager

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_env_keys(required_keys: list[str]):
    for key in required_keys:
        if not os.getenv(key):
            raise MissingAPIKeyError(f"{key} 환경 변수가 설정되지 않았습니다.")

def create_skills_from_mcp(agent: DockerAgent) -> list[AgentSkill]:
    skills = []
    tool_registry = getattr(agent.client, "tool_registry", None)
    if not tool_registry:
        return []
    for tool_name, tool in tool_registry.tools.items():
        skills.append(
            AgentSkill(
                id=tool_name,
                name=tool_name.replace('_', ' ').title(),
                description=tool.description or "No description.",
                tags=["docker", "mcp"],
                examples=[f"Use {tool_name} to retrieve data."]
            )
        )
    return skills


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10031)
def main(host, port):
    asyncio.run(async_main(host, port))

async def async_main(host, port):
  try:
    validate_env_keys([
      "GOOGLE_API_KEY", 
      "TAVILY_API_KEY"
      ])

    mcp_config = load_config_from_json()
    agent = DockerAgent(
      model_type="claude-3-5-sonnet-latest",  # 명시적으로 선택
      use_mcp=True,
      mcp_config=mcp_config,
    )
    await agent.initialize()


    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    mcp_skills = create_skills_from_mcp(agent)

    agent_card = AgentCard(
      name='Docker Agent',
      description='Helps Docker-based command or monitering tasks',
      url=f'http://{host}:{port}/',
      version='1.0.0',
      defaultInputModes=DockerAgent.SUPPORTED_CONTENT_TYPES,
      defaultOutputModes=DockerAgent.SUPPORTED_CONTENT_TYPES,
      capabilities=capabilities,
      skills=mcp_skills,
    )

    notification_sender_auth = PushNotificationSenderAuth()
    notification_sender_auth.generate_jwk()
    server = A2AServer(
      agent_card=agent_card,
      task_manager=AgentTaskManager(
        agent=agent,
        notification_sender_auth=notification_sender_auth,
      ),
      host=host,
      port=port,
    )

    server.app.add_route(
      '/.well-known/jwks.json',
      notification_sender_auth.handle_jwks_endpoint,
      methods=['GET'],
    )

    logger.info(f'🚀 Starting Docker Agent server at http://{host}:{port}')
    server.start()

  except MissingAPIKeyError as e:
    logger.error(f'❌ 환경 변수 오류: {e}')
    exit(1)

  except Exception as e:
    logger.error(f'💥 서버 시작 중 오류 발생: {e}')
    exit(1)


if __name__ == '__main__':
    main()

