from __future__ import annotations

from typing import Any

from core.datetime_parser import parse_datetime_hint
from core.receptionist_intent_analyzer import analyze_receptionist_intent
from core.slot_selection_parser import parse_slot_selection


def _has_customer_identity(conversation_state: dict[str, Any]) -> bool:
    full_name = str(conversation_state.get("full_name", "") or "").strip()
    phone = str(conversation_state.get("phone", "") or "").strip()

    return bool(full_name and phone)


def build_receptionist_plan(
    user_text: str,
    conversation_state: dict[str, Any] | None = None,
    business_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conversation_state = conversation_state or {}
    business_context = business_context or {}
    if conversation_state.get("pending_action") == "select_available_slot":
        offered_slots = conversation_state.get("offered_slots", [])
        slot_selection = parse_slot_selection(user_text, offered_slots)

        if slot_selection.get("clarification_needed"):
            return {
                "action": "ask_slot_selection_clarification",
                "reply_hint": slot_selection.get("clarification_question", ""),
                "slot_selection": slot_selection,
                "intent": {
                    "intent": "select_offered_slot",
                    "confidence": 1.0,
                },
                "datetime": {},
                "has_customer_identity": _has_customer_identity(conversation_state),
            }

        selected_slot = slot_selection.get("selected_slot")
        has_identity = _has_customer_identity(conversation_state)

        if not has_identity:
            return {
                "action": "collect_customer_identity",
                "reply_hint": "برای ثبت این تایم، لطفاً نام و شماره موبایلتون رو بفرمایید.",
                "slot_selection": slot_selection,
                "selected_slot": selected_slot,
                "next_pending_action": "collect_identity_for_selected_slot",
                "intent": {
                    "intent": "select_offered_slot",
                    "confidence": 1.0,
                },
                "datetime": {},
                "has_customer_identity": False,
            }

        return {
            "action": "create_appointment_from_selected_slot",
            "reply_hint": "",
            "slot_selection": slot_selection,
            "selected_slot": selected_slot,
            "intent": {
                "intent": "select_offered_slot",
                "confidence": 1.0,
            },
            "datetime": {},
            "has_customer_identity": True,
        }
    intent_result = analyze_receptionist_intent(
        user_text=user_text,
        conversation_state=conversation_state,
        business_context=business_context,
    )

    datetime_text = str(intent_result.get("datetime_text", "") or "").strip()

    if datetime_text:
        datetime_input = datetime_text
    else:
        datetime_input = " ".join(
            part
            for part in [
                intent_result.get("date_text", ""),
                intent_result.get("time_text", ""),
            ]
            if part
        ).strip()

    datetime_result = parse_datetime_hint(datetime_input or user_text)

    intent = intent_result.get("intent", "unknown")
    has_identity = _has_customer_identity(conversation_state)

    action = "ask_clarification"
    reply_hint = intent_result.get(
        "clarification_question",
        "لطفاً بفرمایید دقیقاً چه کاری براتون انجام بدم؟",
    )

    if intent == "general_info":
        action = "answer_general_info"
        reply_hint = ""

    elif intent in {"greeting", "thanks"}:
        action = "smalltalk"
        reply_hint = ""

    elif intent == "ask_availability":
        if datetime_result.get("clarification_needed"):
            action = "ask_datetime_clarification"
            reply_hint = datetime_result.get("clarification_question", "")
        else:
            action = "check_calendar_availability"
            reply_hint = ""

    elif intent == "book_appointment":
        if datetime_result.get("clarification_needed"):
            action = "ask_datetime_clarification"
            reply_hint = datetime_result.get("clarification_question", "")
        elif not has_identity:
            action = "collect_customer_identity"
            reply_hint = "برای ثبت وقت، لطفاً نام و شماره موبایلتون رو بفرمایید."
        else:
            action = "check_calendar_availability"
            reply_hint = ""

    elif intent in {"change_appointment", "cancel_appointment"}:
        if not has_identity:
            action = "collect_customer_identity"
            reply_hint = "برای پیدا کردن نوبتتون، لطفاً نام و شماره موبایلتون رو بفرمایید."
        else:
            action = "lookup_existing_appointment"
            reply_hint = ""

    elif intent in {"take_message", "transfer_call"}:
        if not has_identity:
            action = "collect_customer_identity"
            reply_hint = "لطفاً نام و شماره موبایلتون رو بفرمایید تا پیامتون ثبت بشه."
        else:
            action = intent
            reply_hint = ""

    else:
        action = "ask_clarification"
        reply_hint = intent_result.get(
            "clarification_question",
            "لطفاً بفرمایید دقیقاً چه کاری براتون انجام بدم؟",
        )

    return {
        "action": action,
        "reply_hint": reply_hint,
        "intent": intent_result,
        "datetime": datetime_result,
        "has_customer_identity": has_identity,
    }
