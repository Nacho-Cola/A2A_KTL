import os
import requests
from requests import get, post
from dotenv import load_dotenv
import asyncio
import httpx

from fastmcp import FastMCP


mcp = FastMCP("Dooray")

load_dotenv(override=True)


#########################################################################
#                                                                       #
#                         DOORAY 맴버 MCP Tools                          #
#                                                                       #
#########################################################################

@mcp.tool()
async def get_members_by_name(name:str,  page: int = 0, size: int = 20): # 멤버 정보를 응답답
    """
    멤버 목록을 응답하는 도구

    매개변수:
      - name: 검색할 사용자 이름 (정확한 일치를 기대)
      - page: 시작 인덱스 (기본값: 0)
      - size: 페이지당 결과 수 (기본값: 20, 최댓값: 100)
    
    Dooray API 호출 시, Authorization 헤더는 "dooray-api {TOKEN}" 형식을 사용합니다.
    """
    # API 토큰과 엔드포인트 URL 설정 (여기서는 민간 클라우드 예시)
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")
    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 환경 변수가 설정되지 않았습니다."}
    
    base_url = DOORAY_ENDPOINT + "/common/v1/members"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }

    params = {
        "name": name,
        "page": page,
        "size": size
    }

    try:
        # 동기 requests 호출을 별도 쓰레드로 실행하여 async 함수에서 기다림
        response = await asyncio.to_thread(get, base_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}



@mcp.tool()
async def get_member_detail(member_id: str):
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")

    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 누락됨"}

    url = f"{DOORAY_ENDPOINT}/common/v1/members/{member_id}"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }

    try:
        response = await asyncio.to_thread(requests.get, url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
    



#########################################################################
#                                                                       #
#                     DOORAY calendar MCP Tools                         #
#                                                                       #
#########################################################################

@mcp.tool()
async def get_calendar_list():
    """
    캘린더 목록 api
    캘린더의 목록을 가져온다.
    """
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")

    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 누락됨"}

    url = f"{DOORAY_ENDPOINT}/calendar/v1/calendars"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }

    try:
        response = await asyncio.to_thread(requests.get, url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
async def get_calendar_detail(calendar_id:str):
    """
    캘린더 상세 api
    캘린더의 상세 정보를를 가져온다.
    """
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")

    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 누락됨"}

    url = f"{DOORAY_ENDPOINT}/calendar/v1/calendars/{calendar_id}"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }

    try:
        response = await asyncio.to_thread(requests.get, url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
    




@mcp.tool()
async def get_calendar_events(
    calendar_id:str,
    time_min: str,
    time_max: str,
    post_type: str = None,
    category: str = None,
    ):
    """
    캘린더 이벤트 목록 api(최대 1년치)
    캘린더의 이벤트 목록을 가져온다.

    여러 캘린더에서 지정 기간의 이벤트 목록을 조회합니다.
    최대 1년치까지 반환됩니다.

    Args:
        calendar_ids: 조회할 캘린더 ID 
        time_min:    검색 시작 시간 (ISO 8601 형식, inclusive, ex: "2024-10-08T09:30:00+09:00")
        time_max:    검색 종료 시간 (ISO 8601 형식, exclusive, ex: "2024-10-09T09:30:00+09:00")
        post_type:   조회 대상 유형 (옵션; 
                     - "toMe": 나에게 할당된 일정  
                     - "toCcMe": 참조된 일정  
                     - "fromToCcMe": 발신 및 참조된 일정  
                     기본값: 전체)
        category:    일정 카테고리 필터 (옵션;  
                     - "general": 일반 일정  
                     - "post": 업무 일정  
                     - "milestone": 마일스톤  
                     기본값: 전체)    

    """
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")

    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 누락됨"}

    url = f"{DOORAY_ENDPOINT}/calendar/v1/calendars/*/events"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }
    params = {
        "calendars": calendar_id,
        "timeMin":   time_min,
        "timeMax":   time_max,
    }
    
    if post_type:
        params["postType"] = post_type
    if category:
        params["category"] = category


    try:
        response = await asyncio.to_thread(requests.get, url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}





@mcp.tool()
async def calendar_add_event(
    calendar_id: str,
    subject: str,
    started_at: str, 
    ended_at: str,   
    attendees_to: list = None, # 예:[member_id, member_id, member_id]
    attendees_cc: list = None, # 예:[member_id, member_id, member_id]
    body_content: str = None,
    body_mime_type: str = "text/html",
    whole_day_flag: bool = False,
    location: str = None,
    # recurrence_rule: dict = None, # 필요한 경우 추가
    # personal_settings: dict = None # 필요한 경우 추가
    ):
    """
    캘린더 일정 등록
    해당 calendar_id의 캘린더에 일정을 추가한다.

    Dooray 캘린더에 일정을 등록합니다.

    Args:
        calendar_id: 일정을 추가할 캘린더의 ID
        subject: 일정 제목
        started_at: 시작 시간 (ISO 8601 형식) 예: "2025-04-28T11:00:00+09:00" 또는 종일 일정 시 "2025-04-29+09:00"
        ended_at: 종료 시간 (ISO 8601 형식) 예: "2025-04-28T12:00:00+09:00" 또는 종일 일정 시 "2025-04-30+09:00"
        attendees_to: 참석자 (To) 목록 member_id 리스트
        attendees_cc: 참조자 (Cc) 목록 member_id 리스트
        body_content: 일정 본문 내용
        body_mime_type: 본문 MIME 타입 (기본값: "text/html")
        whole_day_flag: 종일 일정 여부 (기본값: False)
        location: 장소

    """
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")

    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 누락됨"}

    url = f"{DOORAY_ENDPOINT}/calendar/v1/calendars/{calendar_id}/events"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }

    # API 요청 본문 구성
    request_body = {
        "subject": subject,
        "startedAt": started_at,
        "endedAt": ended_at,
        "wholeDayFlag": whole_day_flag,
    }

    attendees_to_json = []
    if attendees_to:
        for member_attend in attendees_to:
            attendees = {
                "type": "member",
                "member": {
                    "organizationMemberId": member_attend
                }
            }
            attendees_to_json.append(attendees)

    attendees_cc_json = []
    if attendees_cc:
        for member_attend in attendees_cc:
            attendees = {
                "type": "member",
                "member": {
                    "organizationMemberId": member_attend
                }
            }
            attendees_cc_json.append(attendees)

    # 선택적 필드 추가
    if attendees_to_json or attendees_cc_json:
        request_body["users"] = {}
        if attendees_to_json:
            request_body["users"]["to"] = attendees_to_json
        if attendees_cc_json:
            request_body["users"]["cc"] = attendees_cc_json

    if body_content:
        request_body["body"] = {
            "mimeType": body_mime_type,
            "content": body_content
        }
    else:
        request_body["body"] = {
            "mimeType": body_mime_type,
            "content": subject
        }

    if location:
        request_body["location"] = location

    request_body["personalSettings"] = {
        "alarms": [{
            "action": "mail",
            "trigger": "TRIGGER:-PT10M",
        }],
        "busy": True,
        "class": "public",
    }


    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=request_body)
            resp.raise_for_status()
            return resp.json()
    
    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
async def calendar_del_event(
    calendar_id: str,
    event_id: str,
    deleteType:str,
    ):
    """
    캘린더 이벤트를 삭제합니다.
    해당 calendar_id의 캘린더에 이벤트를 삭제합니다.


    Args:
        calendar_id: 일정을 삭제제할 캘린더의 ID
        event_id:    삭제할 이벤트의 ID
        delete_type: 삭제 범위 지정
                     - "this": 해당 이벤트만 삭제
                     - "wholeFromThis": 이 이벤트 이후 반복 모두 삭제
                     - "whole": 전체 반복 이벤트 삭제
    """
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")

    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 누락됨"}

    url = f"{DOORAY_ENDPOINT}/calendar/v1/calendars/{calendar_id}/events/{event_id}/delete"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }

    # API 요청 본문 구성
    request_body = {}

    request_body["deleteType"] = deleteType

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=request_body)
            resp.raise_for_status()
            return resp.json()
    
    except Exception as e:
        return {"error": str(e)}




#########################################################################
#                                                                       #
#                     DOORAY 1:1 Message MCP Tools                      #
#                                                                       #
#########################################################################

@mcp.tool()
async def send_direct_message(member_id: str, message:str):
    """
    DOORAY 1:1 채널을 통해 지정한 멤버에게 직접 메시지를 전송합니다.

    Args:
        member_id:   수신자 조직 멤버의 ID (organizationMemberId)
        message:     전송할 메시지 본문 텍스트

    """
    DOORAY_API_KEY = os.environ.get("DOORAY_API_KEY")
    DOORAY_ENDPOINT = os.environ.get("DOORAY_ENDPOINT")

    if not DOORAY_API_KEY:
        return {"error": "DOORAY_API_KEY 누락됨"}

    url = f"{DOORAY_ENDPOINT}/messenger/v1/channels/direct-send"
    headers = {
        "Authorization": f"dooray-api {DOORAY_API_KEY}"
    }

    # API 요청 본문 구성
    request_body = {}

    request_body["text"] = message
    request_body["organizationMemberId"] = member_id


    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=request_body)
            resp.raise_for_status()
            return resp.json()
    
    except Exception as e:
        return {"error": str(e)}





#########################################################################
#                                                                       #
#                        DOORAY Project MCP Tools                       #
#                                                                       #
#########################################################################




if __name__ == "__main__":
    mcp.run(transport="stdio")