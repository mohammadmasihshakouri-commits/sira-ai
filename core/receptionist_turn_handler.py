from __future__ import annotations

from typing import Any

from datetime import datetime, timedelta
from typing import Any

from core.receptionist_action_executor import execute_receptionist_action
from core.receptionist_availability import (
    check_requested_slot_availability,
    find_available_slots_for_day,
)
from core.receptionist_planner import build_receptionist_plan
from core.receptionist_response_builder import build_receptionist_reply

def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _format_available_slots_reply(slots: list[dict[str, Any]]) -> str:
    if not slots:
        return "متأسفانه برای این روز تایم خالی پیدا نکردم."

    displays = [slot.get("display", "") for slot in slots if slot.get("display")]

    if not displays:
        return "چند تایم خالی پیدا کردم، لطفاً زمان مدنظرتون رو بفرمایید."

    return "این تایم‌ها خالیه: " + "، ".join(displays) + ". کدومش براتون مناسب‌تره؟"


def _date_for_availability(plan: dict[str, Any]) -> str:
    datetime_result = plan.get("datetime", {}) or {}
    start_datetime = _safe_text(datetime_result.get("start_datetime"))

    if start_datetime:
        return start_datetime[:10]

    date_text = _safe_text(datetime_result.get("date_text"))

    if date_text:
        # find_available_slots_for_day can parse ISO dates only.
        # For relative Persian dates, use normalized start date when available.
        # As a temporary fallback, tomorrow is handled by datetime_parser date_text
        # but without start_datetime only in ask_availability cases.
        normalized_text = _safe_text(datetime_result.get("normalized_text"))

        if "فردا" in normalized_text or date_text == "فردا":
            return (datetime.now().astimezone() + timedelta(days=1)).date().isoformat()

        if "امروز" in normalized_text or date_text == "امروز":
            return datetime.now().astimezone().date().isoformat()

    return ""

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

    action = plan.get("action", "")
    datetime_result = plan.get("datetime", {}) or {}
    start_datetime = _safe_text(datetime_result.get("start_datetime"))

    if action == "check_calendar_availability":
        availability_date = _date_for_availability(plan)

        if not availability_date:
            return {
                "reply": "برای چه روزی تایم خالی می‌خواید؟",
                "action": action,
                "plan": plan,
                "execution": {"executed": False, "reason": "missing_availability_date"},
                "conversation_state": conversation_state,
            }

        availability = find_available_slots_for_day(availability_date)
        available_slots = availability.get("available_slots", [])

        updated_state = dict(conversation_state)
        updated_state["pending_action"] = "select_available_slot"
        updated_state["offered_slots"] = available_slots
        updated_state["availability"] = availability

        return {
            "reply": _format_available_slots_reply(available_slots),
            "action": "offer_available_slots",
            "plan": plan,
            "availability": availability,
            "conversation_state": updated_state,
        }

    if action == "collect_customer_identity" and start_datetime:
        end_datetime = (
            datetime.fromisoformat(start_datetime) + timedelta(hours=1)
        ).isoformat()

        availability = check_requested_slot_availability(
            start_time=start_datetime,
            end_time=end_datetime,
        )

        if not availability.get("is_available"):
            availability_date = start_datetime[:10]
            alternatives = find_available_slots_for_day(availability_date)
            available_slots = alternatives.get("available_slots", [])

            updated_state = dict(conversation_state)
            updated_state["pending_action"] = "select_available_slot"
            updated_state["offered_slots"] = available_slots
            updated_state["availability"] = alternatives

            return {
                "reply": "متأسفانه این زمان پر شده. " + _format_available_slots_reply(available_slots),
                "action": "offer_available_slots",
                "plan": plan,
                "availability": alternatives,
                "conversation_state": updated_state,
            }

        updated_state = dict(conversation_state)
        updated_state["selected_slot"] = {
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "display": datetime_result.get("time_text", "زمان انتخاب‌شده"),
        }

        return {
            "reply": "این تایم خالیه. برای ثبت وقت، لطفاً نام و شماره موبایلتون رو بفرمایید.",
            "action": "collect_customer_identity",
            "plan": plan,
            "availability": availability,
            "conversation_state": updated_state,
        }
    

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
