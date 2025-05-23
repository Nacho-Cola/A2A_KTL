# A2A-KTL Streamlit 애플리케이션


## 개요

이 Streamlit 애플리케이션은 A2A-KTL 프로젝트의 사용자 인터페이스를 제공합니다. 사용자는 이 인터페이스를 통해 [TODO: 애플리케이션의 주요 목적을 설명하세요. 예: 에이전트와 상호작용, 대화 관리, 시스템 설정 등].

## 주요 기능

* **홈 (`streamlit/home.py`):** 애플리케이션의 메인 랜딩 페이지입니다.
* **에이전트 관리 (`streamlit/pages/agent.py`):** 등록된 에이전트 확인, 관리 등의 기능을 제공
* **대화 인터페이스 (`streamlit/pages/conversation.py`):** 에이전트와의 대화형 인터페이스를 제공
* **설정 (`streamlit/pages/settings.py`):** 애플리케이션 또는 에이전트 관련 설정을 변경하는 기능을 제공


## 애플리케이션 아키텍처

본 Streamlit 애플리케이션은 다음과 같은 주요 구성 요소로 이루어져 있습니다:

* **Streamlit 프론트엔드 (`streamlit/`):** 사용자와의 상호작용을 담당하는 웹 인터페이스입니다.
    * `home.py`: 메인 애플리케이션 진입점
    * `pages/`: 멀티페이지 구성을 위한 각 페이지 스크립트
* **백엔드 서버 (`streamlit/server_main.py`):** Streamlit 애플리케이션과 통신하며 핵심 비즈니스 로직, 데이터 처리, 외부 에이전트와의 연동 등을 담당하는 별도의 Python 서버
* **공통 모듈 (`streamlit/common/`, `streamlit/service/`):** 클라이언트-서버 로직, 데이터 타입 정의, 유틸리티 함수 등 공통적으로 사용되는 코드
* **상태 관리 (`streamlit/state/`):** Streamlit 세션 상태 또는 애플리케이션 전반의 상태를 관리하는 로직이 포함

## 디렉토리 구조 (streamlit 중심)

다음은 `streamlit` 애플리케이션과 관련된 주요 디렉토리 및 파일입니다.

a2a_ktl/
├── streamlit/
│   ├── home.py                   # Streamlit 메인 앱 실행 파일
│   ├── server_main.py            # 백엔드 서버 실행 파일 
│   ├── pages/                    # Streamlit 멀티페이지
│   │   ├── agent.py
│   │   ├── conversation.py
│   │   └── settings.py
│   ├── common/                   # 공통 유틸리티, 타입, 클라이언트/서버 로직 
│   ├── service/                  # 서비스 로직, API 클라이언트 등 
│   │   ├── client/
│   │   └── server/
│   ├── state/                    # 애플리케이션 상태 관리 
│   └── utils/                    # Streamlit UI 관련 유틸리티 
├── noxfile.py                    
├── requirements.txt              
├── llms.txt                      
└── ... (기타 프로젝트 파일 및 디렉토리)