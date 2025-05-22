import os
import uvicorn
import logging # 로깅 모듈 추가
from fastapi import FastAPI, APIRouter

# streamlit 폴더 구조에 따른 임포트 경로 수정
from service.server.server import ConversationServer
# ADKHostManager 직접 임포트는 ConversationServer가 내부적으로 처리하므로 여기서는 불필요할 수 있음
# from streamlit.service.server.adk_host_manager import ADKHostManager

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """FastAPI 애플리케이션 인스턴스를 생성하고 설정합니다."""
    router = APIRouter()

    # ConversationServer는 내부적으로 A2A_HOST 환경 변수를 확인하여
    # 적절한 ApplicationManager (ADKHostManager 또는 InMemoryFakeAgentManager)를 생성하고,
    # 필요한 경우 GOOGLE_API_KEY 및 GOOGLE_GENAI_USE_VERTEXAI 환경 변수를 사용합니다.
    # 따라서 여기서는 manager 인스턴스를 직접 주입하지 않습니다.
    logger.info("Initializing ConversationServer. It will select the manager based on A2A_HOST env var.")
    ConversationServer(router) # manager 인자 없이 생성하여 내부 로직 사용

    app = FastAPI(title="A2A Conversation Service", version="1.0.0")
    app.include_router(router, prefix="/api/v1") # API 엔드포인트에 접두사 추가 (선택 사항)
    
    logger.info("FastAPI application created and router included.")
    return app

# FastAPI 애플리케이션 인스턴스 생성
app = create_app()

if __name__ == "__main__":
    # 환경 변수에서 포트 번호 읽기 (기본값: 12000)
    port = int(os.getenv("A2A_STREAMLIT_BACKEND_PORT") or os.getenv("A2A_BACKEND_PORT", "12000"))
    host = os.getenv("A2A_BACKEND_HOST", "0.0.0.0") # 호스트 설정 추가 (선택 사항)
    reload_app = os.getenv("A2A_BACKEND_RELOAD", "True").lower() == "true" # 리로드 여부 설정 (개발용)

    logger.info(f"Starting Uvicorn server on {host}:{port} (Reload: {reload_app})")
    uvicorn.run(
        "server_main:app", # 실행할 FastAPI 앱 (이 파일의 'app' 인스턴스)
        host=host,
        port=port,
        reload=reload_app, # 개발 중에는 True, 프로덕션에서는 False 권장
        log_level="info" # Uvicorn 로그 레벨 설정
    )