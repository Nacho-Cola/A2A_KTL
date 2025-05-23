
# A2A-KTL Streamlit 애플리케이션

## 개요

이 Streamlit 애플리케이션은 **A2A-KTL 프로젝트**의 사용자 인터페이스를 제공합니다.
사용자는 이 애플리케이션을 통해 \[TODO: 애플리케이션의 주요 목적을 명확히 기재 – 예: 에이전트와 상호작용, 대화 관리, 시스템 설정 등]할 수 있습니다.

## 🗂️ 목차

* [주요 기능](#주요-기능)
* [애플리케이션 아키텍처](#애플리케이션-아키텍처)
* [디렉토리 구조](#디렉토리-구조)
* [설치](#설치)
* [사용법](#사용법)
* [예시](#예시)
* [문제 해결](#문제-해결)
* [기여자](#기여자)
* [라이선스](#라이선스)

## 주요 기능

* **홈** (`streamlit/home.py`)
  애플리케이션의 메인 랜딩 페이지

* **에이전트 관리** (`streamlit/pages/agent.py`)
  등록된 에이전트의 목록 확인, 관리 기능 제공

* **대화 인터페이스** (`streamlit/pages/conversation.py`)
  에이전트와 실시간 대화를 위한 인터페이스

* **설정** (`streamlit/pages/settings.py`)
  시스템 또는 에이전트 관련 설정 조정 기능

## 애플리케이션 아키텍처

* **Streamlit 프론트엔드 (`streamlit/`)**
  사용자와의 상호작용을 담당하는 웹 UI

* **서버 백엔드 (`streamlit/server_main.py`)**
  핵심 로직 처리, 데이터 핸들링, 외부 연동 기능 수행

* **공통 모듈 (`streamlit/common/`, `streamlit/service/`)**
  데이터 타입 정의, 클라이언트/서버 로직, API 연동, 유틸리티 제공

* **상태 관리 (`streamlit/state/`)**
  세션 및 앱 전반의 상태 관리

## 디렉토리 구조

```
a2a_ktl/
├── streamlit/
│   ├── home.py                 # 메인 애플리케이션 실행 파일
│   ├── server_main.py          # 백엔드 서버
│   ├── pages/                  # 멀티페이지 구성
│   │   ├── agent.py
│   │   ├── conversation.py
│   │   └── settings.py
│   ├── common/                 # 공통 유틸리티 및 타입
│   ├── service/                # API 및 비즈니스 로직
│   │   ├── client/
│   │   └── server/
│   ├── state/                  # 상태 관리 모듈
│   └── utils/                  # UI 관련 도우미 함수
├── noxfile.py
├── requirements.txt
├── llms.txt
└── ...
```

## 설치

```bash
# 가상환경 생성 및 활성화 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

## 사용법

```bash
# Streamlit 앱 실행
streamlit run streamlit/home.py
```

## 예시

\[TODO: 주요 페이지 스크린샷 또는 사용 시나리오 예시가 있다면 삽입]

## 문제 해결

* 포트 충돌 시: `server_main.py` 및 `Streamlit`의 기본 포트 설정 확인
* 에이전트가 응답하지 않을 경우: API 키 및 서버 설정을 점검

## 기여자

\[TODO: 기여자 목록 또는 GitHub 프로필 링크]

## 라이선스

\[TODO: 라이선스 종류 명시 – 예: MIT, Apache 2.0 등]

