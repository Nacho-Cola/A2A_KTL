import os
import json

def load_config_from_json(file_path: str = "config.json") -> dict:
    """
    MCP tool server 설정을 로드합니다.
    기대 구조: { "ToolName": { "command": "...", "args": [...], ... } }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"⚠️ MCP 도구 구성 파일이 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        tools_config = json.load(f)

    if not isinstance(tools_config, dict):
        raise ValueError("MCP 도구 설정은 JSON 객체 형태여야 합니다.")

    return tools_config