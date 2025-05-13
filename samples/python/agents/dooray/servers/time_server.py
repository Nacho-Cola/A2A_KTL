from datetime import datetime
from zoneinfo import ZoneInfo
import time

from fastmcp import FastMCP

mcp = FastMCP("Local_time")

@mcp.tool()
async def get_now_time():
    """South Korea 의 현재 시간을 알려주는 Tool...
    
    Keyword arguments:
    argument -- None
    Return: time of South korea:str
    """
    
    korea_time = datetime.now(ZoneInfo("Asia/Seoul"))
    return "Korea time is" + korea_time.strftime("%Y-%m-%d %H:%M:%S") 


if __name__ == "__main__":
    mcp.run(transport="stdio")