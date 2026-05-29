import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.channel_message import create_channel_message
from core.sira_runtime import handle_message
from core.business_profile_loader import get_business_agent
from core.call_trace import create_call_trace, add_turn, add_tool_call
from core.session_manager import list_sessions, clear_session

app = FastAPI(title="Sira Channel API")


class ChannelRequest(BaseModel):
    business_profile: str
    agent_key: str
    channel: str
    user_id: str
    message_text: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Sira Channel API"
    }


@app.get("/sessions")
def sessions():
    return {
        "sessions": list_sessions()
    }

@app.delete("/sessions/{user_id}")
def delete_session(user_id: str):
    clear_session(user_id)

    return {
        "status": "cleared",
        "user_id": user_id
    }

@app.post("/message")
def message(request: ChannelRequest):
    business_agent = get_business_agent(
        request.business_profile,
        request.agent_key,
    )

    agent_config = business_agent["agent_config"]
    enabled_playbooks = agent_config.get("playbooks", [])

    channel_message = create_channel_message(
        channel=request.channel,
        user_id=request.user_id,
        message_text=request.message_text,
        metadata={
            "business": business_agent.get("business_name"),
            "agent_key": business_agent.get("agent_key"),
        },
    )

    runtime_result = handle_message(
        channel_message=channel_message,
        conversation_state=None,
        enabled_playbooks=enabled_playbooks,
    )

    return {
        "reply": runtime_result["reply"],
        "source": runtime_result["source"],
        "state": runtime_result["state"],
        "tool_call": runtime_result.get("tool_call"),
    }