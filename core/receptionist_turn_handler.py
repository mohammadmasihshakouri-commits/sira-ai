from __future__ import annotations

from typing import Any

from core.receptionist_planner import build_receptionist_plan
from core.receptionist_response_builder import build_receptionist_reply


def handle_receptionist_turn(
    user_text: str,
    conversation_state: dict[str, Any] | None = None,
    business_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conversation_state = conversation_state or {}
    business_context = business_context or {}

    plan = build_receptionist_plan(
        user_text=user_text,
        conversation_state=conversation_state,
        business_context=business_context,
    )

    reply_result = build_receptionist_reply(plan)

    return {
        "reply": reply_result["reply"],
        "action": reply_result["action"],
        "plan": plan,
        "conversation_state": conversation_state,
    }