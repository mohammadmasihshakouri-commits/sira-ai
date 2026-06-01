from __future__ import annotations

from typing import Any

from core.receptionist_action_executor import execute_receptionist_action
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

    execution_result = execute_receptionist_action(
        plan,
        conversation_state=conversation_state,
    )

    if execution_result.get("executed"):
        appointment = execution_result.get("appointment", {})
        service = appointment.get("service", "وقت مراجعه")
        start_time = appointment.get("start_time", "")

        reply = f"نوبت {service} برای زمان انتخاب‌شده ثبت شد."

        if start_time:
            reply = f"نوبت {service} برای {start_time} ثبت شد."

        return {
            "reply": reply,
            "action": plan.get("action"),
            "plan": plan,
            "execution": execution_result,
            "conversation_state": conversation_state,
        }

    reply_result = build_receptionist_reply(plan)

    return {
        "reply": reply_result["reply"],
        "action": reply_result["action"],
        "plan": plan,
        "execution": execution_result,
        "conversation_state": conversation_state,
    }
