# server_main.py

import os
import uvicorn
from fastapi import FastAPI, APIRouter

from service.server.server import ConversationServer
from service.server.adk_host_manager import ADKHostManager

def create_app() -> FastAPI:
    router = APIRouter()

    # 환경변수에서 API 키/Vertex AI 플래그 읽기
    api_key     = os.getenv('GOOGLE_API_KEY', '')
    uses_vertex = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'

    # 전역 ADKHostManager 한 번만 생성
    adk_manager = ADKHostManager(api_key=api_key, uses_vertex_ai=uses_vertex)

    # ConversationServer 에 주입
    ConversationServer(router, manager=adk_manager)

    app = FastAPI()
    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("A2A_BACKEND_PORT", "12000"))
    uvicorn.run(
        "server_main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
